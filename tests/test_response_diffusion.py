"""Diffusion correction: kernel build, convolution paths, dmapdf caching."""
import numpy as np
import pytest
from scipy.ndimage import convolve as nd_convolve
from scipy.signal import fftconvolve, oaconvolve

from vamtoolbox import response


def test_blur_ker_normalized():
    k = response.blur_ker(0.1, 1e-4, 10.0, 24.0)
    assert k.ndim == 3
    assert k.shape == (19, 19, 19)
    assert np.isclose(k.sum(), 1.0, atol=1e-5)
    assert k.min() > -1e-6                       # PSF is non-negative
    assert np.issubdtype(k.dtype, np.floating)
    # ResponseModel stores the kernel as float32 (matches the float32 dose flow)
    m = response.ResponseModel(type="analytical", form="identity", diffusion_kernel=k)
    assert m.diffusion_kernel.dtype == np.float32


def test_identity_no_diffusion_is_passthrough():
    m = response.ResponseModel(type="analytical", form="identity")
    f = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32)
    assert np.allclose(m.map(f), f)


def test_diffusion_map_equals_reference_convolution():
    k = response.blur_ker(0.1, 1e-4, 10.0, 24.0)
    m = response.ResponseModel(type="analytical", form="identity", diffusion_kernel=k)
    f = np.zeros((24, 24, 16), np.float32)
    f[8:16, 8:16, 4:12] = 1.0
    out = m.map(f)
    ref = nd_convolve(f.astype(np.float64),
                      (k / k.sum()).astype(np.float64), mode="constant", cval=0.0)
    rel = np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-9)
    assert rel < 1e-5


def test_adaptive_convolve_fft_matches_overlap_add():
    k = response.blur_ker(0.1, 1e-4, 10.0, 24.0)
    f = np.random.default_rng(1).random((32, 32, 32)).astype(np.float32)
    a = fftconvolve(f, k, mode="same")
    b = oaconvolve(f, k, mode="same")
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1e-4


def test_diffusion_convolve_helper_threshold():
    # small volume -> fftconvolve path; both must agree with the module helper
    k = response.blur_ker(0.1, 1e-4, 10.0, 24.0)
    f = np.random.default_rng(2).random((20, 20, 20)).astype(np.float32)
    out = response._diffusion_convolve(f, k)
    ref = fftconvolve(f, k, mode="same")
    assert out.shape == f.shape
    assert np.max(np.abs(out - ref)) / np.max(np.abs(ref)) < 1e-5


def test_dmapdf_identity_diffusion_is_cached():
    k = response.blur_ker(0.1, 1e-4, 10.0, 24.0)
    m = response.ResponseModel(type="analytical", form="identity", diffusion_kernel=k)
    f = np.ones((16, 16, 12), np.float32)
    d1 = m.dmapdf(f)
    d2 = m.dmapdf(f)
    assert d1 is d2                              # constant D^T(ones) cached, not recomputed
    assert d1.shape == f.shape
