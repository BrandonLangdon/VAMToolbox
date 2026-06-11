import time


# ── OSMO optimize-time estimator ──────────────────────────────────────────────
# Measured per-iteration optimize cost on this machine (RTX 4070, 32-thread CPU),
# at 360 projection angles, with the float32 + chunked projectors.  Each entry is
# (total grid voxels nX*nY*nZ, seconds per OSMO iteration).  Projection cost is
# ~linear in voxel count and ~linear in angle count, so we fit a per-voxel rate
# per backend and scale by (n_angles / 360).
_OPTIMIZE_CALIBRATION = {
    "gpu": [(3.92e6, 0.45), (31.35e6, 2.54), (131.1e6, 10.83), (1050.6e6, 103.72)],
    "sparse": [(31.35e6, 8.4)],          # astra-built CPU sparse-matrix projector
    "skimage": [(16.78e6, 18.6), (31.35e6, 41.7)],
}
# one-time per-run overhead (loky worker spawn / astra init / first-iter filter)
_OPTIMIZE_WARMUP_S = {"gpu": 3.0, "sparse": 20.0, "skimage": 20.0}


def _optimize_rate_per_voxel(backend):
    """Least-squares slope through the origin (weights the large, slow jobs that
    matter most for estimation)."""
    pts = _OPTIMIZE_CALIBRATION[backend]
    sv = sum(v * v for v, _ in pts)
    st = sum(v * t for v, t in pts)
    return st / sv


def estimateOptimizeTime(n_voxels, n_iter, backend="gpu", n_angles=360):
    """Estimate OSMO optimization wall-time from the volume size.

    Parameters
    ----------
    n_voxels : int    total grid voxels actually optimized (nX*nY*nZ at the
                      optimize resolution — i.e. after any RESOLUTION_SCALE).
    n_iter   : int    number of OSMO iterations.
    backend  : str    'gpu' (astra CUDA), 'sparse' (CPU sparse matrix), or
                      'skimage' (CPU radon).
    n_angles : int    number of projection angles (cost scales ~linearly).

    Returns
    -------
    dict with: total_s, per_iter_s, warmup_s, and a human string `pretty`.
    """
    backend = backend if backend in _OPTIMIZE_CALIBRATION else "gpu"
    rate = _optimize_rate_per_voxel(backend)
    per_iter = rate * float(n_voxels) * (n_angles / 360.0)
    warmup = _OPTIMIZE_WARMUP_S[backend]
    total = warmup + per_iter * n_iter
    return {
        "total_s": total,
        "per_iter_s": per_iter,
        "warmup_s": warmup,
        "pretty": formatDuration(total),
        "backend": backend,
    }


def formatDuration(seconds):
    """Human-friendly duration string (e.g. '2.4 min', '1 h 5 min', '8.6 s')."""
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    h = int(seconds // 3600)
    m = int(round((seconds - h * 3600) / 60))
    return f"{h} h {m} min"


def timing(func):
    """
    Decorator for timing a function
    """

    def wrap(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        print(
            "%s function took %0.4f seconds" % (func.__name__, (end_time - start_time))
        )

        return ret

    return wrap
