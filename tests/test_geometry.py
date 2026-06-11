"""Core geometry: target voxel grid and Beer-Lambert absorption mask."""
import numpy as np

import vamtoolbox


def test_target_geometry_from_array():
    arr = np.zeros((20, 20, 20), np.uint8)
    arr[5:15, 5:15, 5:15] = 1
    tg = vamtoolbox.geometry.TargetGeometry(target=arr, resolution=20)
    assert tg.array.shape[2] == 20
    assert np.count_nonzero(tg.array) > 0


def test_absorption_mask_radial_decay(angles):
    N = 48
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
    r = np.hypot(xx - N / 2, yy - N / 2)
    arr = (r < N * 0.3).astype(np.uint8)
    tg = vamtoolbox.geometry.TargetGeometry(target=arr, resolution=N)
    px = 0.01  # cm/voxel
    pg = vamtoolbox.geometry.ProjectionGeometry(
        angles=angles, ray_type="parallel", CUDA=False,
        absorption_coeff=2.0, container_radius=N * px * 0.8, projector_pixel_size=px)
    pg.calcAbsorptionMask(tg)
    m = np.asarray(pg.absorption_mask)
    assert m.dtype == np.float32
    assert m.max() <= 1.0 + 1e-6
    assert m.min() >= -1e-6
    assert m.min() < m.max()                      # absorption produces a gradient


def test_absorption_mask_off_when_no_coeff(angles):
    arr = np.ones((16, 16, 16), np.uint8)
    tg = vamtoolbox.geometry.TargetGeometry(target=arr, resolution=16)
    pg = vamtoolbox.geometry.ProjectionGeometry(
        angles=angles, ray_type="parallel", CUDA=False, absorption_coeff=None)
    assert pg.absorption_coeff is None


def test_rebin_serial_equals_parallel():
    # The parallel rebin path triggers at N_z >= 384; REBIN_N_JOBS toggles it.
    # Serial (1) and parallel (-1) must produce identical output.
    Nr, Na, Nz = 1388, 24, 384
    ang = np.linspace(0, 360, Na, endpoint=False)
    pg = vamtoolbox.geometry.ProjectionGeometry(angles=ang, ray_type="parallel", CUDA=False)
    arr = np.random.default_rng(0).random((Nr, Na, Nz)).astype(np.float32)
    saved = vamtoolbox.geometry.REBIN_N_JOBS
    try:
        outs = {}
        for jobs in (1, -1):
            vamtoolbox.geometry.REBIN_N_JOBS = jobs
            sino = vamtoolbox.geometry.Sinogram(arr.copy(), pg)
            out = vamtoolbox.geometry.rebinFanBeam(
                sino, vial_width=Nr, N_screen=(1080, 1920), n_write=1.51,
                throw_ratio=float("inf"))
            outs[jobs] = out.array
        assert np.allclose(outs[1], outs[-1], atol=1e-5)
    finally:
        vamtoolbox.geometry.REBIN_N_JOBS = saved
