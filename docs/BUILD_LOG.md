# VAMToolbox — Build Log

A running record of the **decisions** made in this repo (the core tomography
engine: voxelization, optimizers, projectors) — what was chosen, why, and what
was rejected — so the reasoning is recoverable without re-reading every diff.

**Per-repo logs.** Each repo in this effort keeps its own `BUILD_LOG.md`:
- **VAMToolbox** (this file) — engine
- **voxelcast** — PySide6 GUI that drives VAMToolbox
- **Blender3MFExporter** — Blender add-on writing role-tagged 3MF
- **VolumeFillingLattice / VFL** — Blender add-on generating beam lattices

**How to use this file.** Append a dated entry per meaningful decision. Keep the
format: *Context → Decision → Why → Alternatives considered → Status*. Record the
*why* and the roads not taken — the code already shows the *what*. Do not put
secrets here (tokens, credentials). Decisions that belong to a sibling repo go in
that repo's log.

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
(`.3mf` auto-routes through `stlfilename`). The naming-convention work is in `main`;
the rtree-drop is PR #3. 24 tests. (GUI surfacing of 3MF lives in voxelcast's log.)

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

**Status.** Done on branch `metal-projector`, PR #4. `metalbackend.py`,
`Projector3DParallelMetal.py`, constructor wiring, `hardware._metal_ok()`,
README item 6, 8 new tests. Occlusion support added next (below).

---

## 2026-06-20 — Metal occlusion (insert shadowing) support

**Context.** The one remaining slow path: a 3MF with an **insert** sets an
attenuation field, routing to `Projector3DParallelPython` (~5.7 s/iter). That
projector models an insert as a hard **occlusion** — an opaque object casts a
shadow, and everything behind it along the ray gets zero contribution. It
precomputes an "occlusion sinogram" (the insert's leading-edge depth per
detector/angle/slice) and masks both forward and backward by it.

**Decision.** Add three occlusion-aware MSL kernels — `occ_sino` (build the
occlusion sinogram), `radon_fwd_occ`, `radon_bwd_occ` — and route the
`attenuation_field is not None` 3D-parallel case to the Metal projector.
`Projector3DParallelMetal` precomputes the occlusion sinogram once in `__init__`
(it is fixed for the projector's life) and the kernels add a per-ray shadow test.

**Key decisions / subtleties.**
- **Match `Projector3DParallelPython`, not skimage.** Its backward is a *raw*
  unfiltered backprojection with **no** `π/(2·nA)` scaling (unlike skimage
  `iradon`). The occlusion backward kernel omits the scaling to be a true drop-in
  for that projector, so the insert case behaves exactly as before.
- **NaN handling.** The CPU occlusion sinogram uses `NaN` where a ray misses the
  insert, and `np.interp` propagates `NaN` (→ un-shadowed). The kernel uses a
  large sentinel and replicates the propagation at shadow boundaries.

**Why.** Makes insert-bearing parts as fast as the rest: ~41× faster end-to-end
(12 OSMO iters with an insert: 0.07 s vs 2.84 s), with recon correlation 0.99895
vs the Python projector. Forward and the occlusion sinogram match the CPU path
exactly; backward differs only on a thin shadow-edge voxel shell that washes out
over optimization.

**Alternatives considered.** Beer-Lambert continuous absorption in the kernel
(rejected for now — the codebase models inserts as binary occlusion, not
continuous attenuation; `absorption_coeff` is a separate multiplicative mask the
non-occlusion path already handles). Keeping inserts on the CPU projector
(rejected — that was the whole slow path we set out to fix).

**Status.** Done on branch `metal-projector`. Three new kernels +
`occlusion_sinogram`/`forward_occ`/`backward_occ` backend methods, projector and
constructor wiring, README item 6 updated, 4 new tests. Full suite: 70 passed,
1 skipped, 1 xfailed.

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
