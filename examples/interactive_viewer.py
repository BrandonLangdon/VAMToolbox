#!/usr/bin/env python3
"""
Interactive 3D viewer for all STL meshes — voxelizes at 1080x1920 cap,
then shows original vs voxelized side-by-side in a live PyVista window.
Times every stage from start to display.
"""
import sys, os, time, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import trimesh
import pyvista as pv
from skimage.measure import marching_cubes

import vamtoolbox as vam
from vamtoolbox.voxelize import Voxelizer

# ── Resolution cap ────────────────────────────────────────────────────────────
N_XY = 1080   # max XY pixels
N_Z  = 1920   # max Z layers
PAD  = 2

# ── Mesh list ─────────────────────────────────────────────────────────────────
RES_DIR = os.path.dirname(vam.resources.load("trifurcatedvasculature.stl"))
BUNDLED = [os.path.join(RES_DIR, f)
           for f in sorted(os.listdir(RES_DIR)) if f.endswith(".stl")]
EXTRA   = [os.path.join(os.path.dirname(__file__), "TestLattice.stl")]
ALL_STLS = BUNDLED + [p for p in EXTRA if os.path.exists(p)]


def voxelize(mesh, lt):
    centered = mesh.copy()
    bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    centered.vertices[:, :2] -= bbox_center[:2]
    centered.vertices[:, 2]  -= mesh.bounds[0][2]
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = tmp.name
    centered.export(tmp_path)
    try:
        v = Voxelizer()
        v.addMeshes({tmp_path: "body"})
        arr = v.voxelize("body", layer_thickness=lt, voxel_value=1, voxel_dtype="uint8")
    finally:
        os.unlink(tmp_path)
    return arr.astype(bool)


def make_surf(arr, lt):
    padded = np.pad(arr.astype(np.float32), PAD, mode="constant", constant_values=0)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=(lt, lt, lt))
    verts -= np.array([PAD * lt, PAD * lt, PAD * lt])
    pv_faces = np.hstack([np.full((len(faces), 1), 3, dtype=np.int32), faces])
    return pv.PolyData(verts, pv_faces)


def stl_to_pv(mesh):
    verts = np.array(mesh.vertices)
    faces = np.hstack([np.full((len(mesh.faces), 1), 3, dtype=np.int32), mesh.faces])
    return pv.PolyData(verts, faces)


# ── Main loop ─────────────────────────────────────────────────────────────────
timing_rows = []

for stl_path in ALL_STLS:
    t_start = time.perf_counter()
    model = os.path.splitext(os.path.basename(stl_path))[0]
    print(f"\n{'='*60}")
    print(f"  {model}")
    print(f"{'='*60}")

    # Load
    mesh = trimesh.load(stl_path, force="mesh")
    ext  = mesh.extents
    t_load = time.perf_counter()
    print(f"  Load      : {t_load - t_start:.2f}s")
    print(f"  Extents   : {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} mm  |  {len(mesh.faces):,} tris")

    # Resolution: benchmark uses 1080×1920 cap; vis uses adaptive (max_extent/200)
    lt_bench = max(max(ext[:2]) / N_XY, ext[2] / N_Z)
    lt_vis   = max(max(ext) / 200, lt_bench)  # never finer than bench
    nx = int(ext[0]/lt_bench); ny = int(ext[1]/lt_bench); nz = int(ext[2]/lt_bench)
    print(f"  lt_bench  : {lt_bench:.5f} mm  ->  grid ~{nx}x{ny}x{nz}")
    print(f"  lt_vis    : {lt_vis:.5f} mm")

    # Voxelize at bench resolution
    arr_b = voxelize(mesh, lt_bench)
    t_vox = time.perf_counter()
    fill  = 100.0 * arr_b.sum() / arr_b.size
    print(f"  Voxelize  : {t_vox - t_load:.2f}s  shape={arr_b.shape}  fill={fill:.2f}%")

    # Voxelize at vis resolution (may be same as bench for small meshes)
    if abs(lt_vis - lt_bench) < 1e-8:
        arr_v = arr_b
        t_vox2 = t_vox
    else:
        arr_v = voxelize(mesh, lt_vis)
        t_vox2 = time.perf_counter()
        print(f"  Vox (vis) : {t_vox2 - t_vox:.2f}s  shape={arr_v.shape}")

    # Marching cubes
    surf_vox = make_surf(arr_v, lt_vis)
    t_mc = time.perf_counter()
    print(f"  March.cubes: {t_mc - t_vox2:.2f}s  {surf_vox.n_cells:,} tris")

    # Shift voxelized surface back to original coordinate space
    surf_vox.points[:, 0] -= arr_v.shape[0] * lt_vis / 2
    surf_vox.points[:, 1] -= arr_v.shape[1] * lt_vis / 2
    surf_vox.points[:, 2] += mesh.bounds[0][2]

    orig_surf = stl_to_pv(mesh)

    # Interactive viewer
    pl = pv.Plotter(shape=(1, 2), title=f"{model}  |  lt_bench={lt_bench:.5f}mm  fill={fill:.1f}%")
    pl.subplot(0, 0)
    pl.add_mesh(orig_surf, color="#4a90d9", smooth_shading=True)
    pl.add_text("Original STL", font_size=12, color="black")

    pl.subplot(0, 1)
    pl.add_mesh(surf_vox, color="#e07040", smooth_shading=True)
    pl.add_text(
        f"Voxelized  lt={lt_vis:.5f}mm\n"
        f"grid {arr_v.shape[0]}x{arr_v.shape[1]}x{arr_v.shape[2]}  fill={fill:.1f}%",
        font_size=10, color="black")

    pl.link_views()

    t_ready = time.perf_counter()
    total   = t_ready - t_start
    print(f"  TOTAL to display: {total:.2f}s  (vox={t_vox-t_load:.2f}s  mc={t_mc-t_vox2:.2f}s)")
    timing_rows.append((model, lt_bench, arr_b.shape, fill, t_vox-t_load, t_mc-t_vox2, total))

    pl.show()   # blocks until user closes the window

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  {'Model':<26} {'lt(mm)':>8}  {'Grid':>18}  {'Fill':>6}  {'Vox(s)':>7}  {'MC(s)':>6}  {'Total':>7}")
print(f"  {'-'*26} {'-'*8}  {'-'*18}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*7}")
for model, lt, shape, fill, tv, tm, tot in sorted(timing_rows, key=lambda r: r[6]):
    g = f"{shape[0]}x{shape[1]}x{shape[2]}"
    print(f"  {model:<26} {lt:>8.5f}  {g:>18}  {fill:>5.1f}%  {tv:>7.2f}  {tm:>6.2f}  {tot:>7.2f}")
print(f"{'='*80}\n")
