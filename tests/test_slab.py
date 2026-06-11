"""Z-slab optimization: matches full-volume optimize and stays seam-free."""
import numpy as np

from vamtoolbox.optimize import Options, optimize, optimizeSlabbed


def test_slab_matches_full_no_seam(cone_target, proj_geo_factory):
    base = dict(method="OSMO", n_iter=12, d_h=0.85, d_l=0.65, filter="hanning", verbose="off")
    full, _, _ = optimize(cone_target, proj_geo_factory(), Options(**base))
    slab, _, _ = optimizeSlabbed(cone_target, proj_geo_factory(), Options(**base),
                                 z_slab=16, z_halo=0)
    a = full.array.astype(np.float64)
    b = slab.array.astype(np.float64)
    an, bn = a / a.max(), b / b.max()
    corr = float(np.corrcoef(an.ravel(), bn.ravel())[0, 1])
    assert corr > 0.9
    # per-z energy profile should match (no slab seams) after global equalization
    ez_a = a.sum(axis=(0, 1)); ez_b = b.sum(axis=(0, 1))
    seam = float(np.max(np.abs(ez_a / ez_a.mean() - ez_b / ez_b.mean())))
    assert seam < 0.2


def test_slab_single_block_is_valid(cone_target, proj_geo_factory):
    # z_slab larger than nZ -> one block; result must still be finite and separate
    base = dict(method="OSMO", n_iter=8, d_h=0.85, d_l=0.65, filter="hanning", verbose="off")
    sino, recon, _ = optimizeSlabbed(cone_target, proj_geo_factory(), Options(**base),
                                     z_slab=999, z_halo=0)
    assert np.isfinite(sino.array).all()
    g = cone_target.array > 0
    assert recon.array[g].mean() > recon.array[~g].mean()


def test_slab_diffusion_halo_auto(cone_target, proj_geo_factory):
    # with a diffusion kernel, optimizeSlabbed should auto-pick a z-halo and run.
    from vamtoolbox import response
    N = cone_target.array.shape[2]
    dk = response.blur_ker(1.0 / N, 1e-4, 10.0, 24.0)
    model = response.ResponseModel(type="analytical", form="identity", diffusion_kernel=dk)
    opt = Options(method="BCLP", response_model=model, n_iter=4, eps=0, weight=1,
                  d_h=0.85, d_l=0.65, learning_rate=0.005, verbose="off")
    sino, recon, _ = optimizeSlabbed(cone_target, proj_geo_factory(), opt, z_slab=16)
    assert np.isfinite(sino.array).all()
    assert sino.array.shape[2] == cone_target.array.shape[2]
