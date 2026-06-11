"""GUI-facing pipeline API: config, progress callbacks, cancellation, run."""
import numpy as np
import pytest

from vamtoolbox.pipeline import PrintConfig, VAMPipeline, PipelineCancelled


def test_config_derived_quantities():
    cfg = PrintConfig(part_height_mm=25.6, voxel_pitch_um=80.0, resolution_scale=0.5)
    assert cfg.res_full == 320
    assert cfg.res_opt == 160
    assert cfg.pixel_size_cm > 0
    assert cfg.effective_absorption_coeff_cm > 0


def test_config_serialize_roundtrip():
    cfg = PrintConfig(method="BCLP", diffusion=True, n_iterations=7)
    cfg2 = PrintConfig.from_dict(cfg.to_dict())
    assert cfg2.method == "BCLP" and cfg2.diffusion and cfg2.n_iterations == 7


def test_config_validation():
    PrintConfig(method="OSMO").validate()
    with pytest.raises(ValueError):
        PrintConfig(method="OSMO", diffusion=True).validate()   # diffusion needs BCLP
    with pytest.raises(ValueError):
        PrintConfig(method="XYZ").validate()
    with pytest.raises(ValueError):
        PrintConfig(resolution_scale=2.0).validate()


def _cfg_for(arr, **kw):
    n = arr.shape[2]
    return PrintConfig(part_height_mm=25.4, voxel_pitch_um=25.4 / n * 1000,
                       use_cuda=False, slab="off", n_angles=60, n_iterations=6, **kw)


def test_pipeline_osmo_run(cylinder_array):
    cfg = _cfg_for(cylinder_array, method="OSMO", absorption=False)
    events = []
    pipe = VAMPipeline(cfg, on_progress=lambda s, f, m: events.append(s))
    res = pipe.run(target_array=cylinder_array, do_rebin=False)
    assert res.quality["contrast"] > 0           # gel dose > void dose
    assert "optimize" in events and "voxelize" in events
    assert res.sinogram.shape[1] == 60


def test_pipeline_iter_callback_fires(cylinder_array):
    cfg = _cfg_for(cylinder_array, method="OSMO", absorption=False)
    fracs = []
    pipe = VAMPipeline(cfg, on_progress=lambda s, f, m: s == "optimize" and fracs.append(f))
    pipe.run(target_array=cylinder_array, do_rebin=False)
    assert len(fracs) >= 3 and max(fracs) == 1.0   # per-iteration progress


def test_pipeline_bclp_diffusion(cylinder_array):
    cfg = _cfg_for(cylinder_array, method="BCLP", diffusion=True, absorption=False)
    res = VAMPipeline(cfg).run(target_array=cylinder_array, do_rebin=False)
    assert np.isfinite(res.sinogram).all()
    assert res.quality["contrast"] > 0


def test_pipeline_cancel(cylinder_array):
    cfg = _cfg_for(cylinder_array, method="OSMO", absorption=False)
    pipe = VAMPipeline(cfg)

    def cb(stage, frac, msg):
        if stage == "optimize" and frac > 0:
            pipe.cancel()
    pipe.on_progress = cb
    with pytest.raises(PipelineCancelled):
        pipe.run(target_array=cylinder_array, do_rebin=False)


def test_pipeline_rebin_produces_print_sinogram(cylinder_array):
    cfg = _cfg_for(cylinder_array, method="OSMO", absorption=False)
    res = VAMPipeline(cfg).run(target_array=cylinder_array, do_rebin=True)
    assert res.rebinned_sinogram is not None
    assert res.rebinned_sinogram.ndim == 3
