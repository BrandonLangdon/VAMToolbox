"""Headless CPU smoke test for running VAMToolbox on Apple Silicon (no CUDA/astra).

Forces the pure-CPU path (CUDA=False -> skimage projector fallback) and runs a
small 3D OSMO optimization end-to-end. Also exercises the OpenGL STL voxelizer,
which works on macOS core-profile GL once glValidateProgram is skipped. Kept
small to fit low-RAM machines.
"""
import time
import matplotlib
matplotlib.use("Agg")  # headless: no display needed

import numpy as np
import vamtoolbox as vam

print("=== VAMToolbox Mac CPU smoke test ===")
info = vam.util.hardware.detect_system()
print("hardware:", info)
print("recommended:", vam.util.hardware.recommend_config())

# --- STL voxelization via OpenGL (Apple GL-over-Metal) ---
t_vox = time.time()
stl_geo = vam.geometry.TargetGeometry(
    stlfilename=vam.resources.load("bear.stl"), resolution=40
)
n_filled = int((stl_geo.array > 0).sum())
print(f"STL voxelized: shape={stl_geo.array.shape} filled={n_filled} "
      f"({time.time()-t_vox:.1f}s)")
assert n_filled > 0, "OpenGL voxelization produced an empty volume!"

t0 = time.time()
# Build a synthetic 3D target directly (a solid cylinder) for the optimization
# leg so the tomography path is exercised independently of mesh content.
N, NZ = 60, 8
yy, xx = np.mgrid[:N, :N] - N / 2
disk = ((xx**2 + yy**2) <= (N * 0.35) ** 2).astype(np.float32)
vol = np.repeat(disk[:, :, None], NZ, axis=2)
target_geo = vam.geometry.TargetGeometry(target=vol)
print(f"target built: shape={target_geo.array.shape}  ({time.time()-t0:.1f}s)")

num_angles = 90
angles = np.linspace(0, 360 - 360 / num_angles, num_angles)
# CUDA=False -> CPU projector. With no astra installed it lands on skimage radon.
proj_geo = vam.geometry.ProjectionGeometry(angles, ray_type="parallel", CUDA=False)

optimizer_params = vam.optimize.Options(
    method="OSMO", n_iter=5, d_h=0.85, d_l=0.6, filter="hamming", verbose=True
)

t1 = time.time()
opt_sino, opt_recon, error = vam.optimize.optimize(
    target_geo, proj_geo, optimizer_params
)
print(f"\noptimize done in {time.time()-t1:.1f}s")
print("sinogram shape:", opt_sino.array.shape)
print("recon shape:   ", opt_recon.array.shape)
print("final error:   ", np.asarray(error).ravel()[-1] if np.size(error) else error)
print("\n=== SMOKE TEST PASSED ===")
