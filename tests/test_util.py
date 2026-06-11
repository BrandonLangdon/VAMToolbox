"""Utility helpers: optimize-time estimator scaling."""
from vamtoolbox.util import timing


def test_estimate_keys_and_positive():
    e = timing.estimateOptimizeTime(64 ** 3, 10, "gpu", 360)
    for k in ("per_iter_s", "total_s", "pretty"):
        assert k in e
    assert e["per_iter_s"] > 0
    assert e["total_s"] > 0


def test_estimate_scales_with_voxels():
    small = timing.estimateOptimizeTime(64 ** 3, 10, "gpu", 360)
    big = timing.estimateOptimizeTime(128 ** 3, 10, "gpu", 360)
    assert big["per_iter_s"] > small["per_iter_s"]


def test_estimate_scales_with_iterations():
    few = timing.estimateOptimizeTime(64 ** 3, 5, "gpu", 360)
    many = timing.estimateOptimizeTime(64 ** 3, 20, "gpu", 360)
    assert many["total_s"] > few["total_s"]
