"""Shared pytest fixtures for the VAMToolbox regression suite.

Tests use small SYNTHETIC targets (no STL/OpenGL) and run on the CPU sparse
projector by default, so they are fast and portable.  GPU-only tests are gated
behind `HAS_CUDA` and skipped when no CUDA device is present.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vamtoolbox  # noqa: E402


def _detect_cuda():
    try:
        return bool(vamtoolbox.util.hardware.detect_system()["cuda"])
    except Exception:
        return False


HAS_CUDA = _detect_cuda()
requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")


def make_proj_geo(angles, cuda=False, **kwargs):
    """ProjectionGeometry helper.  CPU path uses the sparse projector."""
    pg = vamtoolbox.geometry.ProjectionGeometry(
        angles=angles, ray_type="parallel", CUDA=cuda, **kwargs)
    if not cuda:
        pg.sparse = True
    return pg


@pytest.fixture
def angles():
    return np.linspace(0, 360, 60, endpoint=False)


@pytest.fixture
def proj_geo(angles):
    """Default projector geometry (CPU sparse)."""
    return make_proj_geo(angles, cuda=False)


@pytest.fixture
def proj_geo_factory(angles):
    """Returns a callable that builds a FRESH ProjectionGeometry each call
    (needed when a test runs optimize() twice and must not share projector state)."""
    def _factory(**kwargs):
        return make_proj_geo(angles, **kwargs)
    return _factory


def _cylinder(N=48):
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
    r = np.hypot(xx - N / 2, yy - N / 2)
    return ((r < N * 0.30) & (zz > N * 0.2) & (zz < N * 0.8)).astype(np.uint8)


def _cone(N=48):
    """Radius grows with z -> per-z material varies (exercises slab normalization)."""
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
    r = np.hypot(xx - N / 2, yy - N / 2)
    rad = 0.08 * N + 0.30 * N * (zz / N)
    return (r < rad).astype(np.uint8)


@pytest.fixture
def cylinder_array():
    return _cylinder()


@pytest.fixture
def cylinder_target(cylinder_array):
    return vamtoolbox.geometry.TargetGeometry(target=cylinder_array, resolution=cylinder_array.shape[2])


@pytest.fixture
def cone_target():
    arr = _cone()
    return vamtoolbox.geometry.TargetGeometry(target=arr, resolution=arr.shape[2])
