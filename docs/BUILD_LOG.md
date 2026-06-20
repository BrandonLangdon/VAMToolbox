# Build Log

A running record of the **decisions** made while extending VAMToolbox (and its
sibling tools) — what was chosen, why, and what was rejected — so the reasoning
is recoverable later without re-reading every diff.

**Scope.** This effort spans several repos that work together:

| Repo | Role |
|---|---|
| **VAMToolbox** (this repo, GitHub) | Core tomography engine: voxelization, optimizers, projectors |
| **voxelcast** (GitHub) | PySide6 desktop GUI to drive/visualize VAMToolbox |
| **Blender3MFExporter** (Codeberg) | Blender add-on that writes role-tagged 3MF (model + lattice + inserts) |
| **VolumeFillingLattice / VFL** (Codeberg) | Blender add-on that generates beam lattices |

**How to use this file.** Append a dated entry per meaningful decision. Keep the
format: *Context → Decision → Why → Alternatives considered → Status*. Record the
*why* and the roads not taken — the code already shows the *what*. Do not put
secrets here (tokens, credentials).

---

## 2026-06 — Run VAMToolbox on macOS (CPU path, no CUDA)

**Context.** VAMToolbox's projection/reconstruction operators run on the
[ASTRA Toolbox](https://astra-toolbox.com), which is CUDA-only. On a Mac (no
NVIDIA GPU, no CUDA) even `import vamtoolbox` failed because `import astra` was
unguarded, and the OpenGL voxelizer and CUDA projectors crashed.

**Decision.** Make the whole CPU path work on macOS with graceful fallback:
- Guard `import astra` (try/except) in the projector modules so import never
  hard-fails without CUDA.
- `hardware._astra_cuda_ok()` gates the GPU branch in `projectorconstructor`;
  the default chunked CUDA projector defers astra calls, so it had to be detected
  up front rather than caught at construction.
- Added scikit-image CPU projectors (`Projector2D/3DParallelSkimage`) and an
  astra-built **sparse-matrix** CPU projector (~5× faster than skimage radon)
  as the default CPU 3D path.
- Fixed the OpenGL voxelizer on macOS: `compileProgram(..., validate=False)`
  (the validation call needs a bound VAO that macOS doesn't provide at that point).

**Why.** Let development and smaller jobs happen on a Mac without a CUDA box;
keep CUDA the fast path where present.

**Alternatives considered.** Requiring CUDA/Linux for all work (rejected — blocks
Mac development). Porting ASTRA to Metal up front (deferred — far larger; see the
Metal entry below, which we only reached once the CPU path proved the pipeline).

**Status.** Done; merged to `main`.

---

## 2026-06 — VoxelCast: a desktop GUI for VAMToolbox

**Context.** Needed to *see* 2D/3D targets, sinograms, and reconstructions, and to
import STL files and run the engine interactively.

**Decision.** A separate PySide6 app (`voxelcast`) with **dual coupling**: it can
import `vamtoolbox` directly (in-process optimize on a `QThread`) *and* load saved
result files. 2D via pyqtgraph, 3D via pyvistaqt (VTK).

**Why.** A standalone GUI keeps the engine library headless and reusable; dual
coupling supports both "run it now" and "inspect a prior run."

**macOS-specific decisions (VTK/OpenGL quirks).**
- 3D view goes blank/black on resize with MSAA → set `multi_samples=0`, and
  render on resize/show.
- Translucent volumes render empty at high res → added a **Surface threshold**
  mode alongside Volume.
- Embedded `QtInteractor` repaint is unreliable → "Open in window" uses
  `pyvistaqt.BackgroundPlotter`; switched the native plotter to BackgroundPlotter
  to fix a segfault when closing the pop-out + main window together.
- The slice plane only repaints on render churn embedded — accepted as-is.

**Status.** Done; works end-to-end. (A harmless teardown segfault on close was
later resolved — see 2026-06-20.)

---

## 2026-06 — 3MF import to carry models + lattices + roles

**Context.** Wanted to "leverage the 3MF structure for models and lattices" — a
single file describing the printed model, infill lattice, inserts, and zero-dose
regions — instead of separate STLs plus side-channel config.

**Decisions.**
- **lib3mf SDK** to parse 3MF (handles OPC/ZIP + the beam-lattice and balls
  extensions), rather than hand-rolling an XML/ZIP reader.
- **beam-lattice 3MF first** (struts/balls), the structure VFL/Blender produce.
- **Analytic capsule-SDF voxelization** for lattices (struts → capsules, balls →
  spheres), vectorized over each primitive's bounding box.
- **numpy z-ray parity scanline** voxelizer for solid meshes — chosen to **drop
  the `rtree` dependency** (`trimesh.contains` needs it; it was a hard install).
- **Object-name role convention**: a prefix tag on each 3MF object name maps it to
  a role — `print` / `insert` / `zero_dose` (with aliases like model/lattice/
  shell→print, nodose/zerodose→zero_dose; bracket form `[insert] name` also
  accepted; untagged → print). `bodies="auto"` is the default for 3MF.

**Why.** One 3MF can fully describe a VAM job. Name-tagging is exporter-agnostic
(works from Blender, others) and needs no extra metadata schema. Capsule SDF is
exact and fast for lattices; scanline fill removes a painful native dependency.

**Alternatives considered.** Custom mesh+config bundle (rejected — reinvents 3MF).
A 3MF metadata/properties schema for roles (rejected for now — names are simpler
and survive round-trips through most tools). Keeping rtree for mesh interior tests
(rejected — install friction).

**Status.** Done. `vamtoolbox/threemf.py`, wired into `TargetGeometry`
(`.3mf` auto-routes through `stlfilename`). Surfaced in VoxelCast (STL + 3MF, role
datasets shown). 24 tests.

---

## 2026-06 — Blender3MFExporter: write role tags

**Context.** To test the whole chain, the Blender exporter had to emit 3MF whose
object names carry the role convention above, plus beam lattices from a Skin
modifier.

**Decision.** Write `name` onto each 3MF object resource; document the role-tag
convention in the add-on README; add a bpy-free multi-role sample generator.

**Status.** Done; pushed to Codeberg.

---

## 2026-06-20 — Apple Metal GPU projector (Route B)

**Context.** With the CPU path working, the question returned: can the GPU work be
done on Apple Metal instead of CUDA? The user forked astra-toolbox to explore it.
Goal: make VAMToolbox fast on Mac now, and eventually contribute Metal support
upstream to ASTRA.

**Key insight (this is what made it small).** VAMToolbox only ever uses
**parallel-beam** forward/back projection, and for parallel geometry the 3D
problem is just a **stack of independent 2D Radon transforms** over z-slices
(confirmed in `Projector3DParallel`). So the Metal target is *two* of the simplest
kernels — not the ~9,600 lines / 26 `.cu` files of ASTRA's CUDA backend (fan/cone
beam, FBP, SART, FFT, all texture-based).

**Decision — Route B before Route C.** Three routes were on the table:
- **A. PyTorch-MPS** (`grid_sample` on `device="mps"`, no Metal code).
- **B. Standalone MSL projector** (custom Metal compute kernels + Python host).
- **C. Port ASTRA's CUDA backend to Metal** in the fork (upstreamable).

Chose **B first, then C later**: B gets VAMToolbox fast on Mac and produces the
kernel logic that C will reuse; C is the long-term upstream contribution. A was
rejected as the end goal because it adds a heavy torch dependency and doesn't
advance the upstream ASTRA goal (though it was noted as the quickest pure-Python
fallback if B underperformed).

**Decision — match scikit-image exactly.** The two MSL kernels reproduce
`skimage.transform.radon(circle=True)` and `iradon(filter_name=None, circle=True)`
— same rotation geometry, same `π/(2·nAngles)` scaling — so the Metal projector is
a **drop-in** for the existing CPU projector and results stay interchangeable with
the CPU/CUDA conventions. Verified to ~1e-6 relative error.

**Decision — metalcompute + manual bilinear (not textures).** Dispatch via the
lightweight `metalcompute` package over numpy buffers, doing bilinear sampling
**manually in the kernel** rather than using Metal hardware-interpolated textures.
Simpler, fully correct, no texture-binding plumbing. (The eventual Route C upstream
port should switch to `texture3d` hardware sampling for speed, mirroring how ASTRA
uses CUDA textures.)

**Decision — auto-select with opt-out.** `projectorconstructor` prefers the Metal
projector on the CPU branch when a Metal device is present and CUDA is not; set
`proj_geo.metal = False` to disable. Falls back to sparse/skimage if Metal or
metalcompute is unavailable. `metalcompute` is an **optional** macOS/arm64-only
requirement.

**Why.** ~10–13× faster than the parallelized skimage CPU path on an M1 (e.g.
151²×100, 360 angles: 0.27 s vs 2.9 s per forward+backward), with no behavior
change for existing users and no new mandatory dependency.

**Known limitation.** Metal (like ASTRA's parallel path) does **not** handle
attenuation/occlusion. A 3MF with an **insert** sets an attenuation field, which
routes to the slow `Projector3DParallelPython` (~5.7 s/iter vs ~50 ms/iter). This
is correct, not a regression. Adding a masked ray-integration variant to the Metal
kernel is the obvious follow-up to make insert-bearing parts fast too.

**Status.** Done on branch `metal-projector` (VAMToolbox), pushed to GitHub.
`metalbackend.py`, `Projector3DParallelMetal.py`, constructor wiring,
`hardware._metal_ok()`, README item 6, 8 new tests. Full suite: 67 passed,
1 skipped, 1 xfailed (the pre-existing skimage-not-exact-adjoint xfail, which
applies to Metal too since it matches skimage). **Not yet merged / PR not opened.**

---

## 2026-06-20 — VoxelCast end-to-end check with Metal

**Context.** Verify the STL and 3MF chains work through the GUI with the Metal
changes.

**Findings.**
- `metalcompute` was missing in voxelcast's venv — without it the UI silently used
  the CPU projector. Installed it; the UI now selects Metal.
- STL (TacticalBlade) → OpenGL voxelize → **Metal**, 4 ms/iter.
- 3MF lattice (no insert) → **Metal**, 47 ms/iter.
- 3MF multi-role (insert + zero_dose) → Python attenuation projector, 5.7 s/iter
  (see the known limitation above).
- voxelcast test suite: 19 passed. GUI now exits cleanly (exit 0; the earlier
  teardown segfault no longer reproduces).

**Status.** Chain verified. Open follow-ups: Metal attenuation support; open the
`metal-projector` PR.

---
```
Template for new entries:

## YYYY-MM-DD — Short title

**Context.** What problem/why now.
**Decision.** What we chose.
**Why.** The reasoning.
**Alternatives considered.** What we rejected and why.
**Status.** Done / in progress / branch / merged.
```
