#!/usr/bin/env python3
"""
End-to-end VAM pipeline (headless):
  STL  ->  OpenGL voxelization  ->  astra parallel-beam projector  ->  OSMO optimization
  ->  save sinogram (.sino)  ->  CAL projection video (1080x1920 portrait MP4)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")              # headless — no GUI window
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2 as imageio
import vamtoolbox as vam

# ── Settings ──────────────────────────────────────────────────────────────────
STL        = vam.resources.load("thinker.stl")
RESOLUTION = 150
N_ANGLES   = 360
N_ITER     = 20
OUT_W, OUT_H = 1080, 1920
OUT_DIR    = os.path.dirname(__file__)
OUT_SINO   = os.path.join(OUT_DIR, "thinker_450.sino")
OUT_MP4    = os.path.join(OUT_DIR, "cal_output.mp4")
FPS        = 30

t_start = time.perf_counter()

# ── 1. Voxelize ───────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  Voxelizing  {os.path.basename(STL)}  @ {RESOLUTION} layers")
print("=" * 60)
t0 = time.perf_counter()
target_geo = vam.geometry.TargetGeometry(stlfilename=STL, resolution=RESOLUTION)
t_vox = time.perf_counter() - t0
nX, nY, nZ = target_geo.nX, target_geo.nY, target_geo.nZ
n_vox = nX * nY * nZ
print(f"  Grid : {nX} x {nY} x {nZ}  ({n_vox/1e6:.1f}M voxels)"
      f"  fill={target_geo.array.mean()*100:.2f}%   {t_vox:.2f}s")

# ── 2. Projection geometry ────────────────────────────────────────────────────
angles   = np.linspace(0, 360 - 360 / N_ANGLES, N_ANGLES)
proj_geo = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=False)

# ── 3. OSMO optimization (headless) ───────────────────────────────────────────
print(f"\n  Running OSMO ({N_ANGLES} angles, {N_ITER} iterations) headless ...")
optimizer_params = vam.optimize.Options(
    method  = "OSMO",
    n_iter  = N_ITER,
    d_h     = 0.85,
    d_l     = 0.60,
    filter  = "hamming",
    verbose = "iter",              # text-only, no plot
)

t1 = time.perf_counter()
opt_sino, opt_recon, error = vam.optimize.optimize(target_geo, proj_geo, optimizer_params)
t_opt = time.perf_counter() - t1

plt.close("all")
print(f"\n  Done in {t_opt:.1f}s   final error={error[~np.isnan(error)][-1]:.4f}")
print(f"  Sinogram : {opt_sino.array.shape}   Recon : {opt_recon.array.shape}")

# ── 4. Save sinogram ──────────────────────────────────────────────────────────
t2 = time.perf_counter()
opt_sino.save(OUT_SINO)
t_save = time.perf_counter() - t2
print(f"\n  Sinogram saved: {OUT_SINO}  ({t_save:.2f}s)")

# ── 5. Build CAL projection video ─────────────────────────────────────────────
sino      = opt_sino.array                       # (nX, n_angles, nZ)
sino_norm = sino / sino.max()
n_frames  = sino.shape[1]
vid_dur   = n_frames / FPS

print(f"\n  Writing CAL video: {n_frames} frames @ {FPS} fps = {vid_dur:.1f}s playback")
print(f"  Output: {OUT_MP4}")

def sino_frame_to_image(frame_2d):
    """(nX, nZ) float -> letterboxed 1080x1920 uint8 numpy array."""
    arr = (frame_2d.T * 255).astype(np.uint8)    # (nZ, nX)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    src_w, src_h = img.size
    scale = min(OUT_W / src_w, OUT_H / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (OUT_W, OUT_H), (0, 0, 0))
    canvas.paste(img, ((OUT_W - new_w) // 2, (OUT_H - new_h) // 2))
    return np.array(canvas)

t3 = time.perf_counter()
with imageio.get_writer(OUT_MP4, fps=FPS, macro_block_size=1) as writer:
    for i in range(n_frames):
        writer.append_data(sino_frame_to_image(sino_norm[:, i, :]))
    last = sino_frame_to_image(sino_norm[:, -1, :])
    for _ in range(FPS):
        writer.append_data(last)
t_vid = time.perf_counter() - t3

t_total = time.perf_counter() - t_start
print(f"\n  Video written in {t_vid:.1f}s  (playback duration: {vid_dur:.1f}s)")
print(f"  Saved: {OUT_MP4}")
print()
print("=" * 60)
print(f"  TIMING SUMMARY")
print(f"  Voxelization : {t_vox:.1f}s")
print(f"  OSMO ({N_ITER} iter): {t_opt:.1f}s  ({t_opt/N_ITER:.1f}s/iter)")
print(f"  Sinogram save: {t_save:.1f}s")
print(f"  Video encode : {t_vid:.1f}s  -> {vid_dur:.1f}s of {FPS}fps footage")
print(f"  TOTAL        : {t_total:.1f}s")
print("=" * 60)
