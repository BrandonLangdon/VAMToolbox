"""Apple Metal parallel-beam projector tests.

The Metal kernels are written to reproduce skimage radon/iradon exactly, so the
tests assert agreement against skimage (the CPU reference) and against the
VAMToolbox sinogram axis convention.  Skipped when Metal/metalcompute is
unavailable (non-Apple-Silicon, or metalcompute not installed)."""
import numpy as np
import pytest

from vamtoolbox.util import hardware

if not hardware._metal_ok():
    pytest.skip("Metal/metalcompute unavailable", allow_module_level=True)

from skimage.transform import radon, iradon  # noqa: E402
from skimage.draw import disk  # noqa: E402

import vamtoolbox as vam  # noqa: E402
from vamtoolbox.projector.metalbackend import MetalProjectorBackend  # noqa: E402
from vamtoolbox.projector.Projector3DParallelMetal import (  # noqa: E402
    Projector3DParallelMetal,
    Projector2DParallelMetal,
)
from vamtoolbox.projector.Projector3DParallel import (  # noqa: E402
    Projector3DParallelSkimage,
    Projector3DParallelPython,
)


def _phantom(N):
    img = np.zeros((N, N), np.float32)
    rr, cc = disk((N // 2 - 9, N // 2 + 7), N // 6, shape=img.shape)
    img[rr, cc] = 1.0
    rr, cc = disk((N // 2 + 14, N // 2 - 11), N // 10, shape=img.shape)
    img[rr, cc] += 0.5
    return img


# --------------------------------------------------------------------------- #
# Kernel-level agreement with skimage
# --------------------------------------------------------------------------- #
def test_backend_forward_matches_skimage():
    N = 128
    angles = np.linspace(0, 360 - 360 / 180, 180)
    img = _phantom(N)
    be = MetalProjectorBackend.shared()
    sk = radon(img, theta=angles, circle=True).astype(np.float32)        # (N, nA)
    me = be.forward(img[None], angles)[0].T                              # (N, nA)
    assert np.abs(sk - me).max() / (sk.max() + 1e-9) < 1e-4


def test_backend_backward_matches_skimage():
    N = 128
    angles = np.linspace(0, 360 - 360 / 180, 180)
    img = _phantom(N)
    sk_sino = radon(img, theta=angles, circle=True).astype(np.float32)
    be = MetalProjectorBackend.shared()
    sk = iradon(sk_sino, theta=angles, filter_name=None, circle=True).astype(np.float32)
    me = be.backward(sk_sino.T[None], angles)[0]
    assert np.abs(sk - me).max() / (np.abs(sk).max() + 1e-9) < 1e-4


# --------------------------------------------------------------------------- #
# Projector-class agreement + axis convention
# --------------------------------------------------------------------------- #
def _make_geo(N, nZ, nA):
    vol = np.zeros((N, N, nZ), np.float32)
    yy, xx = np.mgrid[:N, :N]
    vol[((xx - N // 2) ** 2 + (yy - N // 2) ** 2) < (N // 3) ** 2, :] = 1.0
    tg = vam.geometry.TargetGeometry(target=vol, resolution=N)
    angles = np.linspace(0, 360 - 360 / nA, nA)
    pg = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=False)
    return vol, tg, pg


def test_3d_projector_matches_skimage_and_convention():
    N, nZ, nA = 64, 6, 120
    vol, tg, pg = _make_geo(N, nZ, nA)
    metal = Projector3DParallelMetal(tg, pg)
    skim = Projector3DParallelSkimage(tg, pg)

    bm = metal.forward(vol)
    bs = skim.forward(vol)
    assert bm.shape == (N, nA, nZ) == bs.shape          # (nX, n_angles, nZ)
    assert np.abs(bm - bs).max() / (bs.max() + 1e-9) < 1e-3

    xm = metal.backward(bm)
    xs = skim.backward(bs)
    assert xm.shape == (N, N, nZ) == xs.shape
    assert np.abs(xm - xs).max() / (np.abs(xs).max() + 1e-9) < 1e-3


def test_2d_projector_shapes():
    N, nA = 64, 90
    img = _phantom(N)
    tg = vam.geometry.TargetGeometry(target=img, resolution=N)
    angles = np.linspace(0, 360 - 360 / nA, nA)
    pg = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=False)
    metal = Projector2DParallelMetal(tg, pg)
    b = metal.forward(img)
    assert b.shape == (N, nA)                            # (nX, n_angles)
    x = metal.backward(b)
    assert x.shape == (N, N)


# --------------------------------------------------------------------------- #
# Selection + end-to-end optimize
# --------------------------------------------------------------------------- #
def test_constructor_selects_metal():
    _, tg, pg = _make_geo(48, 4, 60)
    A = vam.projectorconstructor.projectorconstructor(tg, pg)
    assert type(A).__name__ == "Projector3DParallelMetal"


def test_constructor_metal_opt_out():
    _, tg, pg = _make_geo(48, 4, 60)
    pg.metal = False
    A = vam.projectorconstructor.projectorconstructor(tg, pg)
    assert type(A).__name__ != "Projector3DParallelMetal"


def test_optimize_runs_through_metal():
    N, nZ = 64, 8
    vol = np.zeros((N, N, nZ), np.float32)
    yy, xx = np.mgrid[:N, :N]
    vol[((xx - N // 2 - 6) ** 2 + (yy - N // 2) ** 2) < 9 ** 2, :] = 1.0
    tg = vam.geometry.TargetGeometry(target=vol, resolution=N)
    angles = np.linspace(0, 360 - 360 / 120, 120)
    pg = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=False)
    A = vam.projectorconstructor.projectorconstructor(tg, pg)
    assert type(A).__name__ == "Projector3DParallelMetal"
    opt = vam.optimize.Options(method="OSMO", n_iter=8, d_h=0.85, d_l=0.6,
                               filter="hamming", verbose=False)
    _, recon, _ = vam.optimize.optimize(tg, pg, opt)
    d = recon.array
    tgt = vol > 0.5
    # in-target dose should sit clearly above background
    assert d[tgt].mean() > d[~tgt].mean() + 0.2


# --------------------------------------------------------------------------- #
# Occlusion (insert shadowing) path vs the Python reference projector
# --------------------------------------------------------------------------- #
def _occ_geo(N=72, nZ=6, nA=90):
    vol = np.zeros((N, N, nZ), np.float32)
    yy, xx = np.mgrid[:N, :N]
    r2 = (xx - N // 2) ** 2 + (yy - N // 2) ** 2
    vol[(r2 < (N // 3) ** 2) & (r2 > (N // 6) ** 2), :] = 1.0  # ring target
    insert = np.zeros((N, N, nZ), np.float32)
    insert[((xx - N // 2 - 12) ** 2 + (yy - N // 2) ** 2) < 6 ** 2, :] = 1.0
    tg = vam.geometry.TargetGeometry(target=vol, resolution=N)
    angles = np.linspace(0, 360 - 360 / nA, nA)
    pg = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=False)
    pg.attenuation_field = np.where(insert == 1, np.inf, 0)
    return vol, tg, pg


def test_occlusion_sinogram_matches_python():
    vol, tg, pg = _occ_geo()
    met = Projector3DParallelMetal(tg, pg)
    py = Projector3DParallelPython(tg, pg)
    assert met.occ is not None                       # occlusion path active
    mo = np.transpose(met.occ, (2, 1, 0))            # -> (N, nA, nZ) like python
    mo = np.where(mo >= 0.5e9, np.nan, mo)
    po = py.occ_sinogram
    both = ~np.isnan(po) & ~np.isnan(mo)
    # identical edge depths where both see the insert, identical miss pattern
    assert np.array_equal(np.isnan(po), np.isnan(mo))
    assert np.abs(po[both] - mo[both]).max() < 1e-3


def test_occlusion_forward_matches_python():
    vol, tg, pg = _occ_geo()
    met = Projector3DParallelMetal(tg, pg)
    py = Projector3DParallelPython(tg, pg)
    bm = met.forward(vol)
    bp = py.forward(vol)
    assert np.abs(bm - bp).max() / (bp.max() + 1e-9) < 1e-3


def test_constructor_selects_metal_for_insert():
    _, tg, pg = _occ_geo()
    A = vam.projectorconstructor.projectorconstructor(tg, pg)
    assert type(A).__name__ == "Projector3DParallelMetal"


def test_optimize_with_insert_matches_python():
    vol, tg, pg = _occ_geo(nZ=8)
    angles = pg.angles
    insert_field = np.where(pg.attenuation_field > 0, np.inf, 0)

    def opt():
        return vam.optimize.Options(method="OSMO", n_iter=10, d_h=0.85,
                                    d_l=0.6, filter="hamming", verbose=False)

    pg_m = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=False)
    pg_m.attenuation_field = insert_field.copy()
    A_m = vam.projectorconstructor.projectorconstructor(tg, pg_m)
    assert type(A_m).__name__ == "Projector3DParallelMetal"
    _, recon_m, _ = vam.optimize.optimize(tg, pg_m, opt())

    pg_p = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=False)
    pg_p.attenuation_field = insert_field.copy()
    pg_p.metal = False
    A_p = vam.projectorconstructor.projectorconstructor(tg, pg_p)
    assert type(A_p).__name__ == "Projector3DParallelPython"
    _, recon_p, _ = vam.optimize.optimize(tg, pg_p, opt())

    # functionally equivalent reconstructions (boundary voxels aside)
    c = np.corrcoef(recon_m.array.ravel(), recon_p.array.ravel())[0, 1]
    assert c > 0.99
