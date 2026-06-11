"""Optimizers: OSMO / BCLP / FBP run and separate; BCLP_lowmem == BCLP."""
import numpy as np

from vamtoolbox.optimize import Options, optimize


def _sep(recon, target):
    d = recon.array
    g = target.array > 0
    return float(d[g].mean()), float(d[~g].mean())


def test_osmo_separates(cylinder_target, proj_geo):
    opt = Options(method="OSMO", n_iter=10, d_h=0.85, d_l=0.65, filter="hanning", verbose="off")
    sino, recon, _ = optimize(cylinder_target, proj_geo, opt)
    gel, void = _sep(recon, cylinder_target)
    assert gel > void
    assert sino.array.shape[0] == cylinder_target.array.shape[0]


def test_bclp_separates(cylinder_target, proj_geo):
    opt = Options(method="BCLP", n_iter=10, eps=0, weight=1, d_h=0.85, d_l=0.65,
                  learning_rate=0.005, verbose="off")
    sino, recon, _ = optimize(cylinder_target, proj_geo, opt)
    gel, void = _sep(recon, cylinder_target)
    assert gel > void


def test_fbp_runs(cylinder_target, proj_geo):
    out = optimize(cylinder_target, proj_geo, Options(method="FBP"))
    assert out[0].array.ndim == 3


def test_bclp_lowmem_matches_bclp(cylinder_target, proj_geo_factory):
    base = dict(method="BCLP", n_iter=6, eps=0, weight=1, d_h=0.85, d_l=0.65,
                learning_rate=0.005, verbose="off")
    s1, _, _ = optimize(cylinder_target, proj_geo_factory(), Options(**base))
    s2, _, _ = optimize(cylinder_target, proj_geo_factory(), Options(lowmem=True, **base))
    assert np.allclose(s1.array, s2.array, atol=1e-5, rtol=1e-4)


def test_bclp_diffusion_runs(cylinder_target, proj_geo):
    from vamtoolbox import response
    N = cylinder_target.array.shape[2]
    dk = response.blur_ker(1.0 / N, 1e-4, 10.0, 24.0)
    model = response.ResponseModel(type="analytical", form="identity", diffusion_kernel=dk)
    opt = Options(method="BCLP", response_model=model, n_iter=6, eps=0, weight=1,
                  d_h=0.85, d_l=0.65, learning_rate=0.005, verbose="off")
    sino, recon, _ = optimize(cylinder_target, proj_geo, opt)
    assert np.isfinite(sino.array).all()
    gel, void = _sep(recon, cylinder_target)
    assert gel > void
