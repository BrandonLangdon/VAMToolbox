"""Apple Metal parallel-beam projectors (GPU-accelerated, Apple Silicon).

Drop-in replacements for the scikit-image CPU projectors: the Metal kernels
reproduce ``skimage.transform.radon(circle=True)`` /
``iradon(filter_name=None, circle=True)`` (see :mod:`metalbackend`), so the
forward/backward pair matches the CPU and astra-CUDA conventions and results are
interchangeable.  Use these on macOS when astra+CUDA is unavailable but Metal is.

The pre/post-processing (clipToCircle, Beer-Lambert absorption mask,
zero-dose sinogram zeroing) mirrors Projector3DParallelSkimage exactly.
"""
from __future__ import annotations

import numpy as np

import vamtoolbox
from vamtoolbox.projector.metalbackend import MetalProjectorBackend, MetalUnavailable


class Projector3DParallelMetal:
    """3D parallel-beam projector running on Apple Metal (per-slice radon)."""

    def __init__(self, target_geo, proj_geo):
        self.target_geo = target_geo
        self.proj_geo = proj_geo
        self.angles_deg = np.asarray(proj_geo.angles)
        # Raises MetalUnavailable if Metal/metalcompute can't be used; the
        # projectorconstructor catches this and falls back to skimage.
        self.backend = MetalProjectorBackend.shared()

        # Occlusion: an insert sets attenuation_field; the insert casts a hard
        # shadow.  Precompute the occlusion sinogram once (it is fixed for the
        # life of the projector), matching Projector3DParallelPython.
        self.occ = None
        if proj_geo.attenuation_field is not None:
            insert = (np.asarray(proj_geo.attenuation_field) > 0).astype(np.float32)
            insert_zrc = np.ascontiguousarray(np.moveaxis(insert, 2, 0))
            self.occ = self.backend.occlusion_sinogram(insert_zrc, self.angles_deg)

    def forward(self, target):
        """Forward projector: b = Ax.  Returns (nX, n_angles, nZ)."""
        x = vamtoolbox.util.data.clipToCircle(target)
        # (nX, nY, nZ) -> (nZ, N, N) contiguous for the kernel
        vol = np.ascontiguousarray(np.moveaxis(x, 2, 0), dtype=np.float32)
        if self.occ is not None:
            # Occlusion path mirrors Projector3DParallelPython (no absorption mask).
            sino = self.backend.forward_occ(vol, self.angles_deg, self.occ)
        else:
            if self.proj_geo.absorption_coeff is not None:
                vol = np.ascontiguousarray(
                    np.moveaxis(self.proj_geo.absorption_mask * x, 2, 0),
                    dtype=np.float32,
                )
            sino = self.backend.forward(vol, self.angles_deg)   # (nZ, nA, N)
        # (nZ, nA, N) -> (nX, n_angles, nZ)
        return np.transpose(sino, (2, 1, 0))

    def backward(self, b):
        """Backward projector: x = A^T b.  Returns (nX, nY, nZ)."""
        if self.proj_geo.zero_dose_sino is not None:
            b[self.proj_geo.zero_dose_sino] = 0.0
        # (nX, n_angles, nZ) -> (nZ, nA, N) contiguous for the kernel
        sino = np.ascontiguousarray(np.transpose(b, (2, 1, 0)), dtype=np.float32)
        if self.occ is not None:
            rec = self.backend.backward_occ(sino, self.angles_deg, self.occ)
            return vamtoolbox.util.data.clipToCircle(np.moveaxis(rec, 0, 2))
        rec = self.backend.backward(sino, self.angles_deg)  # (nZ, N, N)
        x = np.moveaxis(rec, 0, 2)                           # (nX, nY, nZ)
        if self.proj_geo.absorption_coeff is not None:
            x = self.proj_geo.absorption_mask * x
        return vamtoolbox.util.data.clipToCircle(x)


class Projector2DParallelMetal:
    """2D parallel-beam projector on Apple Metal (single slice via the 3D path).

    Mirrors Projector2DParallelSkimage's interface for n_dim == 2 targets.
    """

    def __init__(self, target_geo, proj_geo):
        self.target_geo = target_geo
        self.proj_geo = proj_geo
        self.angles_deg = np.asarray(proj_geo.angles)
        self.backend = MetalProjectorBackend.shared()

    def forward(self, target):
        """Forward projector: b = Ax.  Returns (nX, n_angles)."""
        x = vamtoolbox.util.data.clipToCircle(target)
        vol = np.ascontiguousarray(x[None], dtype=np.float32)   # (1, N, N)
        sino = self.backend.forward(vol, self.angles_deg)       # (1, nA, N)
        return sino[0].T                                        # (nX, n_angles)

    def backward(self, b):
        """Backward projector: x = A^T b.  Returns (nX, nY)."""
        if self.proj_geo.zero_dose_sino is not None:
            b[self.proj_geo.zero_dose_sino] = 0.0
        sino = np.ascontiguousarray(b.T[None], dtype=np.float32)  # (1, nA, N)
        rec = self.backend.backward(sino, self.angles_deg)        # (1, N, N)
        return vamtoolbox.util.data.clipToCircle(rec[0])


__all__ = [
    "Projector3DParallelMetal",
    "Projector2DParallelMetal",
    "MetalUnavailable",
]
