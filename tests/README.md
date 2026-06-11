# VAMToolbox regression tests

Fast, portable unit tests guarding the core pipeline and every feature added in the
diffusion / memory-scaling / hardware work. They use small **synthetic** targets
(no STL/OpenGL) and run on the **CPU sparse projector** by default, so they need no
GPU. GPU-only checks are gated behind `HAS_CUDA` and skip automatically when no CUDA
device is present.

## Run

```bash
.venv/Scripts/python.exe -m pytest          # whole suite (~30 s)
.venv/Scripts/python.exe -m pytest tests/test_optimizers.py -v   # one file
.venv/Scripts/python.exe -m pytest -k slab  # by keyword
```

## Coverage

| File | Guards |
|------|--------|
| `test_geometry.py`    | TargetGeometry from array; Beer-Lambert absorption mask; **rebin serial == parallel** (`REBIN_N_JOBS` toggle) |
| `test_projectors.py`  | forward/backward **adjointness**; shapes/non-negativity; CUDA == CPU parity *(GPU-gated)* |
| `test_response_diffusion.py` | `blur_ker` normalization; diffusion map == reference convolution; FFT vs overlap-add; **dmapdf caching** |
| `test_optimizers.py`  | OSMO / BCLP / FBP run + separate gel/void; **BCLP_lowmem == BCLP**; BCLP + diffusion |
| `test_slab.py`        | `optimizeSlabbed` matches full optimize (**no seam**); single-block; diffusion halo |
| `test_hardware.py`    | `detect_system` shape; `recommend_config` backend / rebin-jobs / VRAM logic |
| `test_util.py`        | optimize-time estimator keys + scaling |

Fixtures live in `conftest.py` (`cylinder_target`, `cone_target`, `proj_geo`,
`proj_geo_factory`, `HAS_CUDA`).
