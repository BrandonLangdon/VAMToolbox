"""Projector correctness: forward/backward adjointness, shapes, CPU/GPU parity."""
import numpy as np

import vamtoolbox
from conftest import make_proj_geo, requires_cuda


def _projector(angles, cuda):
    N = 40
    rng = np.random.default_rng(0)
    arr = (rng.random((N, N, N)) > 0.5).astype(np.uint8)
    tg = vamtoolbox.geometry.TargetGeometry(target=arr, resolution=N)
    pg = make_proj_geo(angles, cuda=cuda)
    return vamtoolbox.projectorconstructor.projectorconstructor(tg, pg), N


def test_forward_backward_are_adjoint(angles):
    P, N = _projector(angles, cuda=False)
    rng = np.random.default_rng(1)
    x = rng.random((N, N, N)).astype(np.float32)
    y = rng.random((N, len(angles), N)).astype(np.float32)
    lhs = float(np.sum(P.forward(x) * y))
    rhs = float(np.sum(x * P.backward(y)))
    assert abs(lhs - rhs) / max(abs(lhs), 1e-9) < 1e-4


def test_forward_shape_and_nonnegativity(angles):
    P, N = _projector(angles, cuda=False)
    x = np.abs(np.random.default_rng(2).random((N, N, N))).astype(np.float32)
    sino = P.forward(x)
    assert sino.shape == (N, len(angles), N)
    assert (sino >= -1e-5).all()                 # non-neg matrix * non-neg volume


@requires_cuda
def test_cuda_forward_matches_cpu(angles):
    N = 40
    rng = np.random.default_rng(3)
    arr = (rng.random((N, N, N)) > 0.6).astype(np.uint8)
    tg = vamtoolbox.geometry.TargetGeometry(target=arr, resolution=N)
    x = arr.astype(np.float32)
    Pc = vamtoolbox.projectorconstructor.projectorconstructor(tg, make_proj_geo(angles, cuda=True))
    Ps = vamtoolbox.projectorconstructor.projectorconstructor(tg, make_proj_geo(angles, cuda=False))
    fc, fs = Pc.forward(x), Ps.forward(x)
    rel = np.linalg.norm(fc - fs) / max(np.linalg.norm(fs), 1e-9)
    # The CPU sparse projector is a calibrated approximation of the CUDA projector;
    # ~5% L2 difference is expected. This guards against gross divergence.
    assert rel < 0.10
