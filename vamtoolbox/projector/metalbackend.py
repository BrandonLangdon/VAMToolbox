"""Apple Metal backend for parallel-beam 2D/3D Radon forward/back projection.

This is a standalone GPU projector for Apple Silicon (Metal), written to be a
*drop-in* replacement for the scikit-image CPU projector: the two compute
kernels reproduce ``skimage.transform.radon(circle=True)`` and
``skimage.transform.iradon(filter_name=None, circle=True)`` -- same geometry,
same scaling -- so results are interchangeable with the existing CPU/CUDA paths.

Only parallel-beam geometry is implemented (the only geometry VAMToolbox uses),
and for parallel beam the 3D problem is just a stack of independent 2D radon
transforms over z-slices, so both kernels batch over (detector, angle, slice).

Dispatch uses the lightweight ``metalcompute`` package (compiles MSL at runtime
and runs over numpy buffers).  Everything degrades gracefully: if Metal or
metalcompute is unavailable, :func:`metal_available` returns False and
:class:`MetalProjectorBackend` raises ``MetalUnavailable`` on construction, so
callers fall back to the CPU projector.
"""
from __future__ import annotations

import numpy as np

try:
    import metalcompute as _mc  # type: ignore
except Exception:  # pragma: no cover - import guard
    _mc = None


class MetalUnavailable(RuntimeError):
    """Raised when a Metal device / metalcompute cannot be used."""


# --------------------------------------------------------------------------- #
# Metal Shading Language kernels
# --------------------------------------------------------------------------- #
# Bilinear sampling treats out-of-bounds as zero (matches skimage warp with
# mode='constant', cval=0).  Pixel (row, col) is stored row-major: img[row*N+col].
_MSL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// Bilinear sample of an NxN slice at (x=col, y=row); zero outside the image.
inline float samp(const device float* img, int N, float x, float y) {
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    float fx = x - (float)x0;
    float fy = y - (float)y0;
    float v = 0.0f;
    bool xin0 = (x0 >= 0 && x0 < N);
    bool xin1 = (x0 + 1 >= 0 && x0 + 1 < N);
    bool yin0 = (y0 >= 0 && y0 < N);
    bool yin1 = (y0 + 1 >= 0 && y0 + 1 < N);
    if (yin0 && xin0) v += (1.0f - fx) * (1.0f - fy) * img[y0 * N + x0];
    if (yin0 && xin1) v += fx * (1.0f - fy) * img[y0 * N + x0 + 1];
    if (yin1 && xin0) v += (1.0f - fx) * fy * img[(y0 + 1) * N + x0];
    if (yin1 && xin1) v += fx * fy * img[(y0 + 1) * N + x0 + 1];
    return v;
}

// Forward Radon transform.  Reproduces skimage radon(circle=True):
//   x_in = cos*cc + sin*rr - center*(cos+sin-1)
//   y_in = -sin*cc + cos*rr - center*(cos-sin-1)
//   out[cc, i] = sum_rr image(x_in, y_in)
// Layout: vol (z, row, col); out (z, angle, det).
kernel void radon_fwd(const device float* vol     [[buffer(0)]],
                      const device float* cossin  [[buffer(1)]],
                      const device int*   dims     [[buffer(2)]],
                      device float*       out      [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    int N  = dims[0];
    int nA = dims[1];
    int nZ = dims[2];
    long total = (long)N * nA * nZ;
    if ((long)gid >= total) return;

    int cc = (int)(gid % (uint)N);
    int i  = (int)((gid / (uint)N) % (uint)nA);
    int z  = (int)(gid / (uint)(N * nA));

    float cosa = cossin[2 * i];
    float sina = cossin[2 * i + 1];
    float center = (float)(N / 2);
    const device float* img = vol + (long)z * N * N;

    float bx = cosa * (float)cc - center * (cosa + sina - 1.0f);
    float by = -sina * (float)cc - center * (cosa - sina - 1.0f);

    float acc = 0.0f;
    for (int rr = 0; rr < N; ++rr) {
        float x_in = bx + sina * (float)rr;
        float y_in = by + cosa * (float)rr;
        acc += samp(img, N, x_in, y_in);
    }
    out[gid] = acc;
}

// Unfiltered back projection.  Reproduces skimage iradon(filter_name=None,
// circle=True):
//   xpr = R0 - radius;  ypr = C0 - radius
//   t = ypr*cos - xpr*sin;  sample column at index t + N/2 (linear, 0 outside)
//   recon[R0, C0] = (sum_i ...) * pi / (2*nA), zeroed outside the circle.
// Layout: sino (z, angle, det); out (z, row, col).
kernel void radon_bwd(const device float* sino    [[buffer(0)]],
                      const device float* cossin  [[buffer(1)]],
                      const device int*   dims     [[buffer(2)]],
                      device float*       out      [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    int N  = dims[0];
    int nA = dims[1];
    int nZ = dims[2];
    long total = (long)N * N * nZ;
    if ((long)gid >= total) return;

    int C0 = (int)(gid % (uint)N);
    int R0 = (int)((gid / (uint)N) % (uint)N);
    int z  = (int)(gid / (uint)(N * N));

    int radius = N / 2;
    float xpr = (float)(R0 - radius);
    float ypr = (float)(C0 - radius);

    float acc = 0.0f;
    if (xpr * xpr + ypr * ypr <= (float)(radius * radius)) {
        const device float* base = sino + (long)z * nA * N;
        float center = (float)(N / 2);
        for (int i = 0; i < nA; ++i) {
            float cosa = cossin[2 * i];
            float sina = cossin[2 * i + 1];
            float t = ypr * cosa - xpr * sina;
            float pos = t + center;
            if (pos >= 0.0f && pos <= (float)(N - 1)) {
                int p0 = (int)floor(pos);
                float fp = pos - (float)p0;
                float c0v = base[i * N + p0];
                float c1v = (p0 + 1 <= N - 1) ? base[i * N + p0 + 1] : 0.0f;
                acc += (1.0f - fp) * c0v + fp * c1v;
            }
        }
        acc *= 3.14159265358979323846f / (2.0f * (float)nA);
    }
    out[gid] = acc;
}
"""


def metal_available() -> bool:
    """True if a Metal device + metalcompute are usable on this machine."""
    if _mc is None:
        return False
    try:
        _mc.Device()
        return True
    except Exception:
        return False


class MetalProjectorBackend:
    """Compiles the MSL kernels once and runs batched forward/back projection.

    A single instance owns the device + compiled functions; reuse it across
    forward()/backward() calls (one per projector object).
    """

    _shared: "MetalProjectorBackend | None" = None

    def __init__(self) -> None:
        if _mc is None:
            raise MetalUnavailable("metalcompute is not installed")
        try:
            self.dev = _mc.Device()
            lib = self.dev.kernel(_MSL_SOURCE)
            self._fwd = lib.function("radon_fwd")
            self._bwd = lib.function("radon_bwd")
        except Exception as e:  # pragma: no cover - hardware dependent
            raise MetalUnavailable(f"Metal init failed: {e}") from e

    @classmethod
    def shared(cls) -> "MetalProjectorBackend":
        """Process-wide singleton (kernel compilation is not free)."""
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    @staticmethod
    def _cossin(angles_deg: np.ndarray) -> np.ndarray:
        a = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
        cs = np.empty(2 * a.size, dtype=np.float32)
        cs[0::2] = np.cos(a)
        cs[1::2] = np.sin(a)
        return cs

    def forward(self, vol_zrc: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
        """vol (nZ, N, N) -> sinogram (nZ, nA, N).  Both float32, C-contiguous."""
        nZ, N, N2 = vol_zrc.shape
        assert N == N2, "slices must be square"
        nA = int(np.asarray(angles_deg).size)
        vol = np.ascontiguousarray(vol_zrc, dtype=np.float32)
        cossin = self._cossin(angles_deg)
        dims = np.array([N, nA, nZ], dtype=np.int32)
        out = self.dev.buffer(nZ * nA * N * 4)
        self._fwd(nZ * nA * N, vol, cossin, dims, out)
        return np.frombuffer(out, dtype=np.float32).reshape(nZ, nA, N)

    def backward(self, sino_zad: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
        """sinogram (nZ, nA, N) -> reconstruction (nZ, N, N).  float32, contiguous."""
        nZ, nA, N = sino_zad.shape
        sino = np.ascontiguousarray(sino_zad, dtype=np.float32)
        cossin = self._cossin(angles_deg)
        dims = np.array([N, nA, nZ], dtype=np.int32)
        out = self.dev.buffer(nZ * N * N * 4)
        self._bwd(nZ * N * N, sino, cossin, dims, out)
        return np.frombuffer(out, dtype=np.float32).reshape(nZ, N, N)
