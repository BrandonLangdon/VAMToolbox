#!/usr/bin/env python3
"""
OpenGL voxelizer stress test across all bundled STL meshes.

Uses a fixed voxel size (LT_MM) so comparison is fair across meshes of very
different scales (cylinder=6mm vs TestLattice=161mm).

For each mesh:
  - Voxelizes at LT_MM and measures time
  - Generates 3-D isosurface visualization (saved as PNG)
  - Shows a combined comparison grid at the end

Usage:  python opengl_stress_test.py
"""
import sys, os, time, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
import pyvista as pv
from skimage.measure import marching_cubes

import vamtoolbox as vam
from vamtoolbox.voxelize import Voxelizer

# ── Settings ──────────────────────────────────────────────────────────────────
# Max sinogram/detector size: 1920 px wide x 1080 px tall (Z).
# layer_thickness is set so neither axis exceeds these limits.
N_XY  = 1080   # max XY diameter in pixels
N_Z   = 1920   # max Z height in layers
LT_VIS = 0.3    # coarser voxel size for visualisation (faster marching cubes)
PAD    = 2      # zero-padding before marching cubes (closes boundary surfaces)

# ── Mesh list ─────────────────────────────────────────────────────────────────
RES_DIR = os.path.dirname(vam.resources.load("trifurcatedvasculature.stl"))
BUNDLED = [os.path.join(RES_DIR, f)
           for f in sorted(os.listdir(RES_DIR)) if f.endswith(".stl")]
EXTRA   = [os.path.join(os.path.dirname(__file__), "TestLattice.stl")]
ALL_STLS = BUNDLED + [p for p in EXTRA if os.path.exists(p)]

OUT_DIR = os.path.join(os.path.dirname(__file__), "stress_test_results")
os.makedirs(OUT_DIR, exist_ok=True)


def voxelize(mesh, lt):
    """Center XY, set Z_min=0, voxelize at given layer_thickness. Returns (arr, elapsed)."""
    centered = mesh.copy()
    bbox_center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    centered.vertices[:, :2] -= bbox_center[:2]   # centre XY for square_xy
    centered.vertices[:, 2]  -= mesh.bounds[0][2] # Z_min = 0 (voxeliser slices upward)

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = tmp.name
    centered.export(tmp_path)
    try:
        v = Voxelizer()
        v.addMeshes({tmp_path: "body"})
        t0 = time.perf_counter()
        arr = v.voxelize("body", layer_thickness=lt, voxel_value=1, voxel_dtype="uint8")
        elapsed = time.perf_counter() - t0
    finally:
        os.unlink(tmp_path)
    return arr.astype(bool), elapsed


def make_isosurface_surf(arr, lt):
    """Coarsen, pad, run marching cubes, return a smoothed PyVista PolyData."""
    sx, sy, sz = arr.shape
    step_xy = max(1, int(np.ceil(max(sx, sy) / 300)))
    coarse  = arr[::step_xy, ::step_xy, :]
    padded  = np.pad(coarse, PAD, mode="constant", constant_values=0)
    verts, faces, _, _ = marching_cubes(padded.astype(np.float32), level=0.5,
                                         spacing=(step_xy * lt, step_xy * lt, lt))
    verts -= np.array([PAD * step_xy * lt, PAD * step_xy * lt, PAD * lt])
    pv_faces = np.hstack([np.full((len(faces), 1), 3, dtype=np.int32), faces])
    surf = pv.PolyData(verts, pv_faces)
    return surf.smooth(n_iter=30, relaxation_factor=0.1)


def render_four_panel(surf, title, out_path):
    pv.global_theme.background = "white"
    pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1400, 1050))
    views = [
        ("Isometric", (1, 1, 0.6), (0, 0, 1)),
        ("Top",       (0, 0, 1),   (0, 1, 0)),
        ("Front",     (0, -1, 0.3),(0, 0, 1)),
        ("Side",      (1, 0, 0.3), (0, 0, 1)),
    ]
    for idx, (label, cam, up) in enumerate(views):
        row, col = divmod(idx, 2)
        pl.subplot(row, col)
        pl.add_mesh(surf, color="#4a90d9", smooth_shading=True,
                    specular=0.5, specular_power=20, ambient=0.2)
        pl.add_text(f"{title}  —  {label}", font_size=10, color="black")
        pl.view_vector(cam, viewup=up)
        pl.add_axes(color="black")
    pl.screenshot(out_path, transparent_background=False)
    pl.close()


# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  OpenGL Voxelizer Stress Test  |  max {N_XY}px XY  x  {N_Z}px Z")
print(f"{'='*72}\n")

results = []

for stl_path in ALL_STLS:
    model  = os.path.splitext(os.path.basename(stl_path))[0]
    mesh   = trimesh.load(stl_path, force="mesh")
    n_tris = len(mesh.faces)
    ext    = mesh.extents
    zmin   = mesh.bounds[0][2]

    # Choose lt so XY ≤ N_XY pixels AND Z ≤ N_Z layers
    lt_xy = max(ext[:2]) / N_XY
    lt_z  = ext[2] / N_Z
    lt_bench = max(lt_xy, lt_z)
    nx = int(ext[0] / lt_bench); ny = int(ext[1] / lt_bench); nz = int(ext[2] / lt_bench)
    print(f"[{model}]")
    print(f"  {n_tris:,} tris  |  {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm"
          f"  |  lt={lt_bench:.4f}mm  grid ~{nx}x{ny}x{nz}  |  Z_min={zmin:.2f}")

    # ── Benchmark voxelisation ────────────────────────────────────────────────
    arr_b, t_bench = voxelize(mesh, lt_bench)
    sx, sy, sz = arr_b.shape
    total = sx * sy * sz
    fill  = 100.0 * int(arr_b.sum()) / total
    print(f"  Bench: {t_bench:.2f}s  shape=({sx},{sy},{sz})  fill={fill:.2f}%")

    # ── Visualisation (adaptive voxel size: max_extent/200, floored at LT_VIS) ──
    lt_vis = max(max(ext) / 200, LT_VIS)
    arr_v, t_vis = voxelize(mesh, lt_vis)
    img_path = None
    try:
        surf = make_isosurface_surf(arr_v, lt_vis)
        n_tris_iso = surf.n_cells
        img_path = os.path.join(OUT_DIR, f"{model}.png")
        render_four_panel(surf, model, img_path)
        print(f"  Vis  ({lt_vis:.4f} mm): {t_vis:.2f}s  {n_tris_iso:,} iso-tris  -> {os.path.basename(img_path)}")
    except Exception as e:
        print(f"  Vis failed: {e}")

    results.append(dict(model=model, n_tris=n_tris, extents=ext, zmin=zmin,
                        lt=lt_bench, t_bench=t_bench, shape=(sx, sy, sz),
                        fill=fill, img=img_path))
    print()

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  RESULTS  |  max {N_XY}px XY  x  {N_Z}px Z")
print(f"{'='*72}")
print(f"  {'Model':<22} {'Tris':>8}  {'Time (s)':>9}  {'Grid (X×Y×Z)':>20}  {'Fill':>6}")
print(f"  {'-'*22} {'-'*8}  {'-'*9}  {'-'*20}  {'-'*6}")
for r in sorted(results, key=lambda x: x["t_bench"]):
    sx, sy, sz = r["shape"]
    grid = f"{sx}x{sy}x{sz}"
    print(f"  {r['model']:<22} {r['n_tris']:>8,}  {r['t_bench']:>9.2f}  {grid:>20}  {r['fill']:>5.2f}%")
print(f"{'='*72}\n")

# ── Combined comparison figure ─────────────────────────────────────────────────
imgs = [r for r in results if r["img"] and os.path.exists(r["img"])]
if imgs:
    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4.5))
    axes = np.array(axes).flatten()
    for i, r in enumerate(imgs):
        img = plt.imread(r["img"])
        axes[i].imshow(img)
        sx, sy, sz = r["shape"]
        axes[i].set_title(
            f"{r['model']}\n{r['n_tris']:,} tris  |  {r['t_bench']:.2f}s"
            f"  |  {sx}×{sy}×{sz}",
            fontsize=8)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        f"OpenGL Voxelizer — All Meshes  (bench: max {N_XY}x{N_Z}px, vis={LT_VIS}mm voxels)",
        fontsize=13)
    plt.tight_layout()
    combo = os.path.join(OUT_DIR, "_comparison.png")
    fig.savefig(combo, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison grid: {combo}")
