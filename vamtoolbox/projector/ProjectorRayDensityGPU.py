"""
GPU ray-density projector via Taichi + SAH-BVH ray tracing.

Works directly with the STL mesh — no voxelization step.
forward(x) ignores x and computes the sinogram from the mesh geometry.
backward(b) uses standard unfiltered iradon backprojection per z-slice.

Backend priority: Vulkan → CUDA → CPU (all handled by Taichi internally).

Requirements: taichi, trimesh
"""

import sys
import numpy as np

try:
    import taichi as ti
    import trimesh
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

import vamtoolbox

# ── Compile-time constants used inside Taichi kernels ──────────────────────
_STACK_SIZE  = 32
_RAY_EPSILON = 1e-4
_JITTER      = 1e-5
_SOURCE_SCALE = 3.0
_LEAF_SIZE   = 4

# ── Taichi one-time init ────────────────────────────────────────────────────
_taichi_initialized = False

def _ensure_taichi():
    global _taichi_initialized
    if _taichi_initialized:
        return
    try:
        for arch, label in [(ti.vulkan, "Vulkan"), (ti.cuda, "CUDA"), (ti.cpu, "CPU")]:
            try:
                ti.init(arch=arch, default_fp=ti.f32, default_ip=ti.i32)
                print(f"[RayDensityGPU] Taichi backend: {label}")
                _taichi_initialized = True
                return
            except Exception:
                continue
        raise RuntimeError("No Taichi backend available")
    except RuntimeError as e:
        # Taichi may already be initialised by another module
        if _taichi_initialized or "already" in str(e).lower():
            _taichi_initialized = True
        else:
            raise


# ── CPU BVH builder (SAH, binned) ──────────────────────────────────────────
def _build_bvh(vertices: np.ndarray, faces: np.ndarray, leaf_size: int = _LEAF_SIZE):
    """Build a flat binary BVH using binned Surface Area Heuristic (SAH)."""
    n         = len(faces)
    tri_verts = vertices[faces]
    centroids = tri_verts.mean(axis=1)
    tri_min   = tri_verts.min(axis=1)
    tri_max   = tri_verts.max(axis=1)

    MAX_NODES  = 2 * n
    bvh_min    = np.empty((MAX_NODES, 3), np.float32)
    bvh_max    = np.empty((MAX_NODES, 3), np.float32)
    bvh_left   = np.full(MAX_NODES, -1,  np.int32)
    bvh_right  = np.full(MAX_NODES, -1,  np.int32)
    bvh_tstart = np.zeros(MAX_NODES,     np.int32)
    bvh_tcount = np.zeros(MAX_NODES,     np.int32)
    tri_order  = np.empty(n,             np.int32)

    node_ptr = [0]
    tri_ptr  = [0]

    N_BINS  = 8
    C_TRAV  = 1.0
    C_ISECT = 1.5

    def _sa(mn, mx):
        d = np.maximum(mx - mn, 0.0)
        return 2.0 * float(d[0]*d[1] + d[1]*d[2] + d[0]*d[2])

    def _make_leaf(nid, idx):
        s = tri_ptr[0]
        tri_order[s: s + len(idx)] = idx
        tri_ptr[0] += len(idx)
        bvh_tstart[nid] = s
        bvh_tcount[nid] = len(idx)

    sys.setrecursionlimit(200_000)

    def _build(idx: np.ndarray) -> int:
        nid = node_ptr[0];  node_ptr[0] += 1

        lmin = tri_min[idx]
        lmax = tri_max[idx]
        nmin = lmin.min(0)
        nmax = lmax.max(0)
        bvh_min[nid] = nmin
        bvh_max[nid] = nmax

        m = len(idx)
        if m <= leaf_size:
            _make_leaf(nid, idx)
            return nid

        parent_sa  = _sa(nmin, nmax)
        best_cost  = C_ISECT * m
        best_b_idx = None
        best_k     = -1

        for axis in range(3):
            c = centroids[idx, axis]
            c_min, c_max = float(c.min()), float(c.max())
            if c_max - c_min < 1e-10:
                continue

            scale = N_BINS / (c_max - c_min)
            b_idx = np.clip(((c - c_min) * scale).astype(np.int32), 0, N_BINS - 1)

            b_min = np.full((N_BINS, 3),  np.inf, dtype=np.float32)
            b_max = np.full((N_BINS, 3), -np.inf, dtype=np.float32)
            b_cnt = np.bincount(b_idx, minlength=N_BINS).astype(np.int32)
            np.minimum.at(b_min, b_idx, lmin)
            np.maximum.at(b_max, b_idx, lmax)

            pre_min = np.minimum.accumulate(b_min, axis=0)
            pre_max = np.maximum.accumulate(b_max, axis=0)
            pre_cnt = np.cumsum(b_cnt)

            suf_min = np.minimum.accumulate(b_min[::-1], axis=0)[::-1]
            suf_max = np.maximum.accumulate(b_max[::-1], axis=0)[::-1]
            suf_cnt = np.cumsum(b_cnt[::-1])[::-1]

            for k in range(N_BINS - 1):
                n_l, n_r = int(pre_cnt[k]), int(suf_cnt[k + 1])
                if n_l == 0 or n_r == 0:
                    continue
                sa_l = _sa(pre_min[k],     pre_max[k])
                sa_r = _sa(suf_min[k + 1], suf_max[k + 1])
                cost = C_TRAV + (sa_l * n_l + sa_r * n_r) * C_ISECT / parent_sa
                if cost < best_cost:
                    best_cost  = cost
                    best_b_idx = b_idx
                    best_k     = k

        if best_b_idx is None:
            _make_leaf(nid, idx)
            return nid

        left_mask      = best_b_idx <= best_k
        bvh_left[nid]  = _build(idx[left_mask])
        bvh_right[nid] = _build(idx[~left_mask])
        return nid

    print("  [RayDensityGPU] Building BVH (SAH) …")
    _build(np.arange(n, dtype=np.int32))
    n_nodes = node_ptr[0]
    print(f"  [RayDensityGPU] BVH: {n_nodes:,} nodes")

    return (bvh_min[:n_nodes], bvh_max[:n_nodes],
            bvh_left[:n_nodes], bvh_right[:n_nodes],
            bvh_tstart[:n_nodes], bvh_tcount[:n_nodes],
            tri_order, n_nodes)


# ── Projector class ─────────────────────────────────────────────────────────
@ti.data_oriented
class ProjectorRayDensityGPU:
    """
    Mesh-based GPU ray-density projector (Taichi + SAH-BVH).

    forward(x)  – ignores x; computes the sinogram directly from the STL
                  mesh via BVH ray tracing on the GPU.
    backward(b) – standard unfiltered iradon backprojection per z-slice.

    Parameters
    ----------
    target_geo : geometry.TargetGeometry
        Must have been created with an stlfilename.
    proj_geo : geometry.ProjectionGeometry
    """

    def __init__(self, target_geo, proj_geo):
        if not _DEPS_AVAILABLE:
            raise ImportError("taichi and trimesh are required for ProjectorRayDensityGPU")
        if not hasattr(target_geo, "stlfilename"):
            raise ValueError(
                "ProjectorRayDensityGPU requires target_geo built from an STL file "
                "(target_geo.stlfilename must be set)"
            )

        _ensure_taichi()

        self.target_geo  = target_geo
        self.proj_geo    = proj_geo
        self.angles_rad  = np.deg2rad(proj_geo.angles).astype(np.float64)
        self.n_angles    = int(proj_geo.n_angles)

        nX = target_geo.nX
        nZ = target_geo.nZ if target_geo.n_dim == 3 else 1
        self.nX    = nX
        self.nZ    = nZ
        self.n_dim = target_geo.n_dim

        # ── Load mesh ──────────────────────────────────────────────────────
        print(f"  [RayDensityGPU] Loading mesh: {target_geo.stlfilename}")
        mesh = trimesh.load(target_geo.stlfilename, force="mesh")
        self._mesh_center = mesh.centroid.astype(np.float32)
        extents           = mesh.extents.astype(np.float32)
        max_extent        = float(np.max(extents))
        self._max_path    = max_extent   # physical upper bound on ray path length
        bounds            = mesh.bounds
        n_tris            = len(mesh.faces)
        print(f"  [RayDensityGPU] Mesh: {n_tris:,} triangles")

        # ── Build & upload BVH ─────────────────────────────────────────────
        (bvh_min_np, bvh_max_np,
         bvh_left_np, bvh_right_np,
         bvh_tstart_np, bvh_tcount_np,
         tri_order_np, n_nodes) = _build_bvh(
            mesh.vertices.astype(np.float32), mesh.faces
        )

        self.bvh_aabb_f   = ti.field(ti.f32, shape=(n_nodes, 6))
        self.bvh_left_f   = ti.field(ti.i32, shape=n_nodes)
        self.bvh_right_f  = ti.field(ti.i32, shape=n_nodes)
        self.bvh_tstart_f = ti.field(ti.i32, shape=n_nodes)
        self.bvh_tcount_f = ti.field(ti.i32, shape=n_nodes)
        self.tri_order_f  = ti.field(ti.i32, shape=n_tris)

        self.bvh_aabb_f.from_numpy(
            np.concatenate([bvh_min_np, bvh_max_np], axis=1)
        )
        self.bvh_left_f.from_numpy(bvh_left_np)
        self.bvh_right_f.from_numpy(bvh_right_np)
        self.bvh_tstart_f.from_numpy(bvh_tstart_np)
        self.bvh_tcount_f.from_numpy(bvh_tcount_np)
        self.tri_order_f.from_numpy(tri_order_np)

        verts_np = mesh.vertices.astype(np.float32)
        faces_np = mesh.faces
        self.tv0 = ti.field(ti.f32, shape=(n_tris, 3))
        self.te1 = ti.field(ti.f32, shape=(n_tris, 3))
        self.te2 = ti.field(ti.f32, shape=(n_tris, 3))
        self.tv0.from_numpy(verts_np[faces_np[:, 0]])
        self.te1.from_numpy(verts_np[faces_np[:, 1]] - verts_np[faces_np[:, 0]])
        self.te2.from_numpy(verts_np[faces_np[:, 2]] - verts_np[faces_np[:, 0]])

        # ── Detector ray grid ──────────────────────────────────────────────
        pad         = 0.1
        source_dist = _SOURCE_SCALE * max_extent
        self._source_dist_f32 = np.float32(source_dist)

        y_range = np.linspace(
            bounds[0, 1] - pad * extents[1],
            bounds[1, 1] + pad * extents[1], nX
        ).astype(np.float32)

        if nZ > 1:
            z_range = np.linspace(
                bounds[0, 2] - pad * extents[2],
                bounds[1, 2] + pad * extents[2], nZ
            ).astype(np.float32)
        else:
            # Single central z-slice for 2D targets
            z_range = np.array([float(self._mesh_center[2])], dtype=np.float32)

        self._y_range = y_range
        self._z_range = z_range

        grid_y, grid_z = np.meshgrid(y_range, z_range, indexing="ij")
        num_rays = int(grid_y.size)   # nX × nZ
        self._num_rays = num_rays

        rng      = np.random.default_rng(0)
        jitter_y = rng.uniform(-_JITTER, _JITTER, num_rays).astype(np.float32)
        jitter_z = rng.uniform(-_JITTER, _JITTER, num_rays).astype(np.float32)

        origins_np = np.column_stack((
            np.full(num_rays, -source_dist, dtype=np.float32),
            (grid_y.ravel() - self._mesh_center[1]).astype(np.float32) + jitter_y,
            (grid_z.ravel() - self._mesh_center[2]).astype(np.float32) + jitter_z,
        ))

        self.ray_origin_f = ti.field(ti.f32, shape=(num_rays, 3))
        self.thickness_f  = ti.field(ti.f32, shape=num_rays)
        self.ray_origin_f.from_numpy(origins_np)
        print(f"  [RayDensityGPU] Ray grid: {num_rays:,} rays ({nX}×{nZ})")

    # ── Taichi device function: BVH traversal ─────────────────────────────
    @ti.func
    def _ray_thickness(self, ro: ti.math.vec3, rd: ti.math.vec3,
                       epsilon: ti.f32, max_t: ti.f32) -> ti.f32:
        EPS   = ti.f32(1e-8)
        safe  = ti.f32(1e-12)
        inv_x = ti.f32(1.0) / rd.x if ti.abs(rd.x) > safe else ti.f32(1e12)
        inv_y = ti.f32(1.0) / rd.y if ti.abs(rd.y) > safe else ti.f32(1e12)
        inv_z = ti.f32(1.0) / rd.z if ti.abs(rd.z) > safe else ti.f32(1e12)

        total     = ti.f32(0.0)
        stack     = ti.Matrix.zero(ti.i32, _STACK_SIZE, 1)
        stack_top = ti.i32(1)
        stack[0, 0] = ti.i32(0)

        while stack_top > 0:
            stack_top -= 1
            ni = stack[stack_top, 0]

            nmin = ti.math.vec3(
                self.bvh_aabb_f[ni, 0],
                self.bvh_aabb_f[ni, 1],
                self.bvh_aabb_f[ni, 2],
            )
            nmax = ti.math.vec3(
                self.bvh_aabb_f[ni, 3],
                self.bvh_aabb_f[ni, 4],
                self.bvh_aabb_f[ni, 5],
            )

            tx1 = (nmin.x - ro.x) * inv_x
            tx2 = (nmax.x - ro.x) * inv_x
            ty1 = (nmin.y - ro.y) * inv_y
            ty2 = (nmax.y - ro.y) * inv_y
            tz1 = (nmin.z - ro.z) * inv_z
            tz2 = (nmax.z - ro.z) * inv_z

            t_near = max(max(min(tx1, tx2), min(ty1, ty2)), min(tz1, tz2))
            t_far  = min(min(max(tx1, tx2), max(ty1, ty2)), max(tz1, tz2))

            if t_far < epsilon or t_near > t_far or t_far < ti.f32(0.0):
                continue

            tc = self.bvh_tcount_f[ni]
            if tc > 0:
                ts = self.bvh_tstart_f[ni]
                for k in range(ts, ts + tc):
                    tidx = self.tri_order_f[k]
                    v0 = ti.math.vec3(
                        self.tv0[tidx, 0], self.tv0[tidx, 1], self.tv0[tidx, 2]
                    )
                    e1 = ti.math.vec3(
                        self.te1[tidx, 0], self.te1[tidx, 1], self.te1[tidx, 2]
                    )
                    e2 = ti.math.vec3(
                        self.te2[tidx, 0], self.te2[tidx, 1], self.te2[tidx, 2]
                    )
                    h = rd.cross(e2)
                    a = e1.dot(h)
                    if ti.abs(a) > EPS:
                        f  = ti.f32(1.0) / a
                        sv = ro - v0
                        u  = f * sv.dot(h)
                        if ti.f32(0.0) <= u <= ti.f32(1.0):
                            q = sv.cross(e1)
                            v = f * rd.dot(q)
                            if v >= ti.f32(0.0) and u + v <= ti.f32(1.0):
                                t = f * e2.dot(q)
                                if t > epsilon:
                                    if a > ti.f32(0.0):
                                        total -= t   # front face = entry
                                    else:
                                        total += t   # back face  = exit
            else:
                lc = self.bvh_left_f[ni]
                rc = self.bvh_right_f[ni]
                if lc >= 0 and stack_top < _STACK_SIZE - 1:
                    stack[stack_top, 0] = lc
                    stack_top += 1
                if rc >= 0 and stack_top < _STACK_SIZE - 1:
                    stack[stack_top, 0] = rc
                    stack_top += 1

        # Clamp: negative total means an exit was found without a matching entry
        # (non-manifold mesh edge). Values larger than max_t are also invalid.
        # Using max_t here as the physical path length cap (mesh max extent).
        return ti.max(ti.min(total, max_t), ti.f32(0.0))

    # ── Taichi kernel: one thread per ray ─────────────────────────────────
    @ti.kernel
    def _cast_rays(
        self,
        n_rays: int,
        cos_t: ti.f32,
        sin_t: ti.f32,
        cx: ti.f32,
        cy: ti.f32,
        cz: ti.f32,
        epsilon: ti.f32,
        max_t: ti.f32,
    ):
        for i in range(n_rays):
            o0x = self.ray_origin_f[i, 0]
            o0y = self.ray_origin_f[i, 1]
            o0z = self.ray_origin_f[i, 2]

            # Rotate origin around mesh centre (same as ray_density_gpu)
            ox = o0x * cos_t - o0y * sin_t + cx
            oy = o0x * sin_t + o0y * cos_t + cy
            oz = o0z + cz

            # Parallel-beam: all rays at this angle share the SAME direction
            # (cos_t, sin_t, 0) — exactly as ray_density_gpu.cast_parallel_rays.
            # Previously rdx/rdy were derived from the origin (cone-beam mistake)
            # and rdz = -o0z made off-centre Z rays tilt toward the origin.
            ro = ti.math.vec3(ox, oy, oz)
            rd = ti.math.vec3(cos_t, sin_t, ti.f32(0.0))
            self.thickness_f[i] = self._ray_thickness(ro, rd, epsilon, max_t)

    # ── Per-angle projection ──────────────────────────────────────────────
    def _project_angle(self, theta: float) -> np.ndarray:
        """
        Returns thickness map of shape (nX, nZ) for the given angle (radians).
        theta follows the same convention as ray_density_gpu.py:
          c = cos(theta), s = -sin(theta)  (source rotated, mesh fixed).
        """
        c  = float(np.cos(theta))
        s  = float(-np.sin(theta))
        cx, cy, cz = self._mesh_center

        self._cast_rays(
            self._num_rays,
            np.float32(c), np.float32(s),
            np.float32(cx), np.float32(cy), np.float32(cz),
            np.float32(_RAY_EPSILON),
            np.float32(self._max_path),
        )
        ti.sync()
        return self.thickness_f.to_numpy().reshape((self.nX, self.nZ))

    # ── Public API ────────────────────────────────────────────────────────
    def forward(self, x) -> np.ndarray:
        """
        Compute the sinogram from the STL mesh via GPU BVH ray tracing.

        The argument ``x`` is accepted for API compatibility but is not used —
        the projection is always computed directly from the mesh geometry.

        Returns
        -------
        np.ndarray
            Shape (nX, n_angles) for 2-D targets or
            (nX, n_angles, nZ) for 3-D targets.
        """
        if self.n_dim == 2:
            sino = np.zeros((self.nX, self.n_angles), dtype=np.float32)
            for a_i, theta in enumerate(self.angles_rad):
                tmap = self._project_angle(float(theta))
                sino[:, a_i] = tmap[:, 0]
        else:
            sino = np.zeros((self.nX, self.n_angles, self.nZ), dtype=np.float32)
            for a_i, theta in enumerate(self.angles_rad):
                sino[:, a_i, :] = self._project_angle(float(theta))
        # 1-D median filter along the detector axis (axis 0 = nX) to remove
        # hot pixel clusters from rogue rays hitting non-manifold mesh edges.
        # Kernel=11 handles clusters up to 5 consecutive bad pixels wide
        # (e.g. trifurcation branch point at z=750 in vasculature mesh).
        from scipy.ndimage import median_filter as _mf
        if self.n_dim == 3:
            sino = _mf(sino, size=(11, 1, 1))
        else:
            sino = _mf(sino, size=(11, 1))
        # Percentile clip: any value above the 99.9th percentile of nonzero
        # pixels is an outlier the median filter missed — hard-cap it.
        pos = sino[sino > 0]
        if pos.size > 0:
            clip_val = float(np.percentile(pos, 99.9))
            sino = np.clip(sino, 0.0, clip_val)
        return sino

    def backward(self, b) -> np.ndarray:
        """
        Vectorized FBP back-projection matching ray_density_gpu.py's approach,
        extended to all Z slices simultaneously.

        For each angle, the detector coordinate r = x·cos(θ) + y·sin(θ) is the
        same for every Z slice (rays are Z-parallel), so r is computed once and
        then used to interpolate across all nZ slices in a single matrix op —
        replacing nZ separate iradon() calls with one vectorized numpy operation.

        Parameters
        ----------
        b : np.ndarray
            Sinogram of shape (nX, n_angles) or (nX, n_angles, nZ).

        Returns
        -------
        np.ndarray
            Reconstruction clipped to the inscribed circle.
        """
        angles_rad = np.deg2rad(self.proj_geo.angles)

        if self.proj_geo.zero_dose_sino is not None:
            b[self.proj_geo.zero_dose_sino] = 0.0

        import time as _time; _t_back = _time.time()

        # Detector positions in world coords — same y_range used in forward
        det = self._y_range.astype(np.float64)
        N   = self.nX
        coords = np.linspace(float(det[0]), float(det[-1]), N)
        xx, yy = np.meshgrid(coords, coords, indexing="ij")  # (N, N) — same grid as ray_density_gpu

        b64 = np.asarray(b, dtype=np.float64)

        print(f"  [backward] N={N} nZ={self.nZ} n_angles={self.n_angles}")

        if self.n_dim == 2:
            recon = np.zeros((N, N), dtype=np.float64)
            for i, theta in enumerate(angles_rad):
                c, s = float(np.cos(theta)), float(np.sin(theta))
                r = xx * c + yy * s
                # Exact np.interp match to ray_density_gpu — b[:, i] = detector values at angle i
                recon += np.interp(r, det, b64[:, i], left=0.0, right=0.0)
            return vamtoolbox.util.data.clipToCircle(recon)

        # 3D: b shape (nX, n_angles, nZ)
        # Mirrors ray_density_gpu FBP loop extended to per-Z-slice.
        # r = xx*cos + yy*sin is computed once per angle (same for all Z).
        # b64[:, i, z] = detector values for angle i at z-slice z — fed into
        # np.interp exactly as sinogram_cz[i] is used in ray_density_gpu.
        recon = np.zeros((N, N, self.nZ), dtype=np.float64)
        det_span = float(det[-1] - det[0])

        _t_loop = _time.time()
        for i, theta in enumerate(angles_rad):
            c, s = float(np.cos(theta)), float(np.sin(theta))
            r = (xx * c + yy * s).ravel()   # (N*N,) — same for all Z

            # Fractional index (uniform grid → equivalent to np.interp)
            frac   = (r - det[0]) / det_span * (N - 1)
            mask   = (frac >= 0.0) & (frac <= float(N - 1))
            j      = np.clip(frac.astype(np.int32), 0, N - 2)
            w      = np.clip(frac - j, 0.0, 1.0)

            sino_row = b64[:, i, :]          # (nX, nZ)

            # (N*N, nZ) contribution — zero outside detector range
            contrib = (1.0 - w[:, None]) * sino_row[j] + w[:, None] * sino_row[j + 1]
            contrib[~mask] = 0.0

            recon += contrib.reshape(N, N, self.nZ)

        elapsed = _time.time() - _t_loop
        print(f"  [RayDensityGPU.backward] {elapsed:.1f}s  ({elapsed/self.n_angles*1000:.1f}ms/angle)")
        return vamtoolbox.util.data.clipToCircle(recon)
