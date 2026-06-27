"""3MF import for VAMToolbox, with beam-lattice support.

Unlike the STL path (trimesh -> OpenGL rasterization of triangle meshes), 3MF
can carry the **beam-lattice extension** (strut graphs: nodes + beams with
per-end radii, plus optional "ball" nodes) produced by tools like the
VolumeFillingLattice add-on + Blender 3MF Exporter. trimesh cannot read that
extension, so we use the official lib3mf SDK to read the file, then voxelize
beam lattices **analytically** (capsule signed-distance) into the voxel grid.

Public API
----------
read_3mf(path) -> list[Body]
voxelize_3mf(path, resolution, bodies="all", rot_angles=(0,0,0))
    -> (array, insert, zero_dose)   # matches voxelize.voxelizeTargetOpenGL
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# Naming convention: an object's role is taken from a leading tag in its name.
#
#   insert_handle        -> insert        zerodose_port   -> zero_dose
#   lattice_infill       -> print         [insert] handle -> insert  (bracket form)
#   my_part              -> print         (no recognized tag -> print)
#
# Matching is case-insensitive; the tag must be followed by a separator
# (_ - . : or space) or the end of the name. Aliases (model/part/body/shell/
# lattice/solid) all map to `print`; the matched keyword is preserved on Body.tag.
# --------------------------------------------------------------------------- #
ROLE_ALIASES = {
    "insert": "insert",
    "zerodose": "zero_dose", "zero_dose": "zero_dose", "zero-dose": "zero_dose",
    "nodose": "zero_dose", "no_dose": "zero_dose", "no-dose": "zero_dose",
    "print": "print", "model": "print", "part": "print", "body": "print",
    "shell": "print", "lattice": "print", "solid": "print",
}
_SEP = r"(?=$|[ _\-.:])"  # tag must end at a separator or string end


def role_from_name(name: str) -> tuple[str, str | None]:
    """Map a 3MF object name to (role, matched_tag) using the naming convention.

    Returns ("print", None) when no recognized tag is present.
    """
    if not name:
        return "print", None
    s = name.strip().lower()
    # bracketed form: [tag] rest   or   (tag) rest
    m = re.match(r"^[\[\(]\s*([a-z0-9 _\-]+?)\s*[\]\)]", s)
    if m:
        cand = m.group(1).strip()
        role = ROLE_ALIASES.get(cand) or ROLE_ALIASES.get(cand.replace(" ", "_"))
        if role:
            return role, cand
    # prefix form: longest matching keyword followed by a separator
    for kw in sorted(ROLE_ALIASES, key=len, reverse=True):
        if re.match("^" + re.escape(kw) + _SEP, s):
            return ROLE_ALIASES[kw], kw
    return "print", None


def _require_lib3mf():
    try:
        import lib3mf
    except Exception as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading 3MF files requires the lib3mf SDK. Install it with:\n"
            "    pip install lib3mf"
        ) from e
    return lib3mf


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Body:
    """One built object instance from a 3MF (geometry already world-transformed)."""

    name: str
    object_id: int
    vertices: np.ndarray                       # (N, 3) float
    triangles: np.ndarray = field(            # (M, 3) int  (solid mesh, may be empty)
        default_factory=lambda: np.empty((0, 3), dtype=np.int64))
    beam_nodes: np.ndarray = field(           # (K, 2) int  (vertex indices per beam)
        default_factory=lambda: np.empty((0, 2), dtype=np.int64))
    beam_radii: np.ndarray = field(           # (K, 2) float (radius per beam end)
        default_factory=lambda: np.empty((0, 2), dtype=float))
    ball_nodes: np.ndarray = field(           # (B,) int
        default_factory=lambda: np.empty((0,), dtype=np.int64))
    ball_radii: np.ndarray = field(           # (B,) float
        default_factory=lambda: np.empty((0,), dtype=float))
    min_length: float = 0.0
    role: str = "print"                       # print | insert | zero_dose
    tag: str | None = None                    # the matched name keyword (e.g. "lattice")

    @property
    def has_beams(self) -> bool:
        return self.beam_nodes.shape[0] > 0

    @property
    def has_mesh(self) -> bool:
        return self.triangles.shape[0] > 0

    @property
    def has_balls(self) -> bool:
        return self.ball_nodes.shape[0] > 0


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _transform_matrix(transform) -> np.ndarray:
    """lib3mf transform Fields is a 4x3 matrix (rows 0-2 = basis, row 3 = translation).
    Returns a (4, 3) numpy array; apply as  v' = v @ M[:3] + M[3]."""
    f = transform.Fields
    return np.array([[f[i][j] for j in range(3)] for i in range(4)], dtype=float)


def _apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ M[:3, :] + M[3, :]


def _read_mesh_object(model, obj, M: np.ndarray) -> Body:
    lib3mf = _require_lib3mf()
    mo = model.GetMeshObjectByID(obj.GetResourceID())

    n_v = mo.GetVertexCount()
    verts = np.empty((n_v, 3), dtype=float)
    varr = mo.GetVertices() if n_v else []
    for i in range(n_v):
        verts[i] = varr[i].Coordinates
    verts = _apply(M, verts) if n_v else verts

    n_t = mo.GetTriangleCount()
    tris = np.empty((n_t, 3), dtype=np.int64)
    if n_t:
        tarr = mo.GetTriangleIndices()
        for i in range(n_t):
            tris[i] = tarr[i].Indices

    bl = mo.BeamLattice()
    n_b = bl.GetBeamCount()
    beam_nodes = np.empty((n_b, 2), dtype=np.int64)
    beam_radii = np.empty((n_b, 2), dtype=float)
    if n_b:
        beams = bl.GetBeams()
        for i in range(n_b):
            beam_nodes[i] = (beams[i].Indices[0], beams[i].Indices[1])
            beam_radii[i] = (beams[i].Radii[0], beams[i].Radii[1])

    n_ball = bl.GetBallCount()
    ball_nodes = np.empty((n_ball,), dtype=np.int64)
    ball_radii = np.empty((n_ball,), dtype=float)
    if n_ball:
        balls = bl.GetBalls()
        for i in range(n_ball):
            ball_nodes[i] = balls[i].Index
            ball_radii[i] = balls[i].Radius

    return Body(
        name=obj.GetName() or f"object_{obj.GetResourceID()}",
        object_id=obj.GetResourceID(),
        vertices=verts, triangles=tris,
        beam_nodes=beam_nodes, beam_radii=beam_radii,
        ball_nodes=ball_nodes, ball_radii=ball_radii,
        min_length=float(bl.GetMinLength()),
    )


def _resolve_object(model, resource_id):
    """Return the base object for a resource id (mesh or components), or None."""
    for getter in ("GetMeshObjectByID", "GetComponentsObjectByID"):
        fn = getattr(model, getter, None)
        if fn is None:
            continue
        try:
            return fn(resource_id)
        except Exception:
            continue
    return None


def _collect(model, obj, M: np.ndarray, out: list[Body]) -> None:
    if obj.IsMeshObject():
        out.append(_read_mesh_object(model, obj, M))
    elif obj.IsComponentsObject():
        comps = model.GetComponentsObjectByID(obj.GetResourceID())
        for i in range(comps.GetComponentCount()):
            comp = comps.GetComponent(i)
            Mc = _transform_matrix(comp.GetTransform()) if comp.HasTransform() else _IDENT
            base = _resolve_object(model, comp.GetObjectResourceID())
            if base is not None:
                _collect(model, base, _compose(M, Mc), out)
    # other object types (level set, etc.) are ignored for now


_IDENT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)


def _compose(M_parent: np.ndarray, M_child: np.ndarray) -> np.ndarray:
    """Compose two (4,3) 3MF transforms: apply child then parent."""
    a3, at = M_child[:3], M_child[3]
    b3, bt = M_parent[:3], M_parent[3]
    out = np.empty((4, 3))
    out[:3] = a3 @ b3
    out[3] = at @ b3 + bt
    return out


def read_3mf(path: str) -> list[Body]:
    """Read a 3MF file into a list of Body instances (world-transformed)."""
    lib3mf = _require_lib3mf()
    wrapper = lib3mf.Wrapper()
    model = wrapper.CreateModel()
    model.QueryReader("3mf").ReadFromFile(str(path))

    bodies: list[Body] = []
    items = model.GetBuildItems()
    while items.MoveNext():
        item = items.GetCurrent()
        obj = item.GetObjectResource()
        M = _transform_matrix(item.GetObjectTransform()) if item.HasObjectTransform() else _IDENT
        _collect(model, obj, M, bodies)

    if not bodies:  # no build items -> fall back to every mesh object
        it = model.GetMeshObjects()
        while it.MoveNext():
            _collect(model, it.GetCurrentMeshObject(), _IDENT, bodies)
    return bodies


# --------------------------------------------------------------------------- #
# Voxelization (analytic capsule SDF for beam lattices)
# --------------------------------------------------------------------------- #
@dataclass
class _Grid:
    xs: np.ndarray      # voxel-center x coords (len nx)
    ys: np.ndarray      # voxel-center y coords (len ny)
    zs: np.ndarray      # voxel-center z coords (len nz)
    shift: np.ndarray   # world -> grid offset (subtracted from vertices)
    shape: tuple        # (ny, nx, nz)


def _rotation(rot_angles) -> np.ndarray:
    """3x3 rotation from degrees about x, then y, then z (applied as v @ R.T)."""
    rx, ry, rz = np.deg2rad(rot_angles)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _max_radius(bodies: list[Body]) -> float:
    r = 0.0
    for b in bodies:
        if b.has_beams:
            r = max(r, float(b.beam_radii.max()))
        if b.has_balls:
            r = max(r, float(b.ball_radii.max()))
    return r


def _build_grid(bodies: list[Body], resolution: int) -> _Grid:
    """Square-XY, z-layered grid matching VAMToolbox's convention: object centered
    in XY, z starting at 0, sized so all geometry fits within the inscribed
    circle (for clip_to_circle). Padded by the max strut/ball radius."""
    pts = np.vstack([b.vertices for b in bodies])
    max_r = _max_radius(bodies)
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2
    z_min = pts[:, 2].min()
    shift = np.array([cx, cy, z_min - max_r])

    z_extent = (pts[:, 2].max() - pts[:, 2].min()) + 2 * max_r
    z_extent = max(z_extent, 1e-9)
    cxy = pts[:, :2] - np.array([cx, cy])
    R = float(np.max(np.hypot(cxy[:, 0], cxy[:, 1]))) + max_r
    R = max(R, 1e-9)

    lt = z_extent / resolution
    nx = ny = max(1, int(round(2 * R / lt)))
    nz = int(resolution)
    xs = -R + (np.arange(nx) + 0.5) * (2 * R / nx)
    ys = -R + (np.arange(ny) + 0.5) * (2 * R / ny)
    zs = (np.arange(nz) + 0.5) * lt
    return _Grid(xs, ys, zs, shift, (ny, nx, nz))


def _sub_range(coords: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
    i0 = int(np.searchsorted(coords, lo, side="left"))
    i1 = int(np.searchsorted(coords, hi, side="right"))
    return max(0, i0), min(len(coords), i1)


def _fill_capsule(arr, g: _Grid, p0, p1, r0, r1) -> None:
    rmax = max(r0, r1)
    lo = np.minimum(p0, p1) - rmax
    hi = np.maximum(p0, p1) + rmax
    ix0, ix1 = _sub_range(g.xs, lo[0], hi[0])
    iy0, iy1 = _sub_range(g.ys, lo[1], hi[1])
    iz0, iz1 = _sub_range(g.zs, lo[2], hi[2])
    if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
        return
    sy, sx, sz = g.ys[iy0:iy1], g.xs[ix0:ix1], g.zs[iz0:iz1]
    YY, XX, ZZ = np.meshgrid(sy, sx, sz, indexing="ij")  # match arr[iy, ix, iz]
    P = np.stack([XX, YY, ZZ], axis=-1)
    d = p1 - p0
    L2 = float(d @ d)
    if L2 > 0:
        t = np.clip(((P - p0) @ d) / L2, 0.0, 1.0)
    else:
        t = np.zeros(P.shape[:-1])
    closest = p0 + t[..., None] * d
    dist = np.linalg.norm(P - closest, axis=-1)
    r_at = r0 + t * (r1 - r0)
    mask = dist <= r_at
    sub = arr[iy0:iy1, ix0:ix1, iz0:iz1]
    sub[mask] = 1


def _fill_sphere(arr, g: _Grid, center, radius) -> None:
    lo = center - radius
    hi = center + radius
    ix0, ix1 = _sub_range(g.xs, lo[0], hi[0])
    iy0, iy1 = _sub_range(g.ys, lo[1], hi[1])
    iz0, iz1 = _sub_range(g.zs, lo[2], hi[2])
    if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
        return
    sy, sx, sz = g.ys[iy0:iy1], g.xs[ix0:ix1], g.zs[iz0:iz1]
    YY, XX, ZZ = np.meshgrid(sy, sx, sz, indexing="ij")
    dist = np.sqrt((XX - center[0]) ** 2 + (YY - center[1]) ** 2 + (ZZ - center[2]) ** 2)
    sub = arr[iy0:iy1, ix0:ix1, iz0:iz1]
    sub[dist <= radius] = 1


def _fill_mesh(arr, g: _Grid, verts, tris) -> None:
    """Solid voxelization of a triangle mesh by z-ray parity (scanline) fill.

    For each XY voxel column, the z-heights where the surface crosses that
    column are collected (one per triangle covering the column), sorted, and the
    voxels between consecutive entry/exit pairs are filled. Pure numpy -- no
    rtree/embree/OpenGL. Best for watertight meshes (even crossing count per
    column); robust enough for the solid shells typically paired with lattices.
    """
    from collections import defaultdict

    xs, ys, zs = g.xs, g.ys, g.zs
    eps = 1e-9
    columns: dict = defaultdict(list)   # (iy, ix) -> [z crossing, ...]

    for tri in verts[tris]:
        p0, p1, p2 = tri
        xmin, xmax = min(p0[0], p1[0], p2[0]), max(p0[0], p1[0], p2[0])
        ymin, ymax = min(p0[1], p1[1], p2[1]), max(p0[1], p1[1], p2[1])
        ix0, ix1 = _sub_range(xs, xmin, xmax)
        iy0, iy1 = _sub_range(ys, ymin, ymax)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(denom) < 1e-12:           # degenerate triangle in XY
            continue
        GY, GX = np.meshgrid(ys[iy0:iy1], xs[ix0:ix1], indexing="ij")
        a = ((p1[1] - p2[1]) * (GX - p2[0]) + (p2[0] - p1[0]) * (GY - p2[1])) / denom
        b = ((p2[1] - p0[1]) * (GX - p2[0]) + (p0[0] - p2[0]) * (GY - p2[1])) / denom
        c = 1.0 - a - b
        inside = (a >= -eps) & (b >= -eps) & (c >= -eps)
        if not inside.any():
            continue
        Z = a * p0[2] + b * p1[2] + c * p2[2]
        iys, ixs = np.nonzero(inside)
        zvals = Z[iys, ixs]
        for ky, kx, z in zip(iys, ixs, zvals):
            columns[(iy0 + int(ky), ix0 + int(kx))].append(float(z))

    for (iy, ix), zlist in columns.items():
        zlist.sort()
        for k in range(0, len(zlist) - 1, 2):
            iz0 = int(np.searchsorted(zs, zlist[k], side="left"))
            iz1 = int(np.searchsorted(zs, zlist[k + 1], side="right"))
            if iz1 > iz0:
                arr[iy, ix, iz0:iz1] = 1


def _fill_body(arr, g: _Grid, body: Body) -> None:
    v = body.vertices - g.shift
    for (i, j), (r0, r1) in zip(body.beam_nodes, body.beam_radii):
        _fill_capsule(arr, g, v[i], v[j], float(r0), float(r1))
    for idx, r in zip(body.ball_nodes, body.ball_radii):
        _fill_sphere(arr, g, v[idx], float(r))
    if body.has_mesh and not body.has_beams:
        _fill_mesh(arr, g, v, body.triangles)


def _assign_groups(bodies: list[Body], bodies_map) -> dict:
    """Map bodies to print/insert/zero_dose.

    `bodies_map` is one of:
      "auto"  - assign by the object-name convention (see role_from_name)
      "all"   - everything -> print (escape hatch)
      dict    - {"print": [...], "insert": [...], "zero_dose": [...]} of
                object names or ids
    """
    groups = {"print": [], "insert": [], "zero_dose": []}
    if bodies_map == "auto":
        for b in bodies:
            b.role, b.tag = role_from_name(b.name)
            groups[b.role].append(b)
        return groups
    if bodies_map == "all" or bodies_map is None:
        for b in bodies:
            b.role, b.tag = "print", None
        groups["print"] = list(bodies)
        return groups
    by_name = {b.name: b for b in bodies}
    by_id = {b.object_id: b for b in bodies}
    for key in groups:
        for sel in bodies_map.get(key, []):
            b = by_name.get(sel) or by_id.get(sel)
            if b is not None:
                b.role = key
                groups[key].append(b)
    return groups


def _vox_group(bodies: list[Body], g: _Grid, progress=None,
               label="voxelizing") -> np.ndarray | None:
    if not bodies:
        return None
    arr = np.zeros(g.shape, dtype=np.uint8)
    for i, b in enumerate(bodies):
        _fill_body(arr, g, b)
        if progress is not None:
            progress(i + 1, len(bodies), label)
    return arr


def voxelize_3mf(path: str, resolution: int, bodies="auto",
                 rot_angles=(0, 0, 0), progress=None):
    """Voxelize a 3MF (beam lattices + solid meshes) into VAMToolbox target arrays.

    Drop-in analogue of voxelize.voxelizeTargetOpenGL: returns
    (array, insert, zero_dose) with array shape (nY, nX, nZ), dtype uint8.

    `bodies` defaults to "auto" (assign roles by the object-name convention,
    see role_from_name). Pass "all" to force every object into the print target,
    or a dict to map explicitly.

    `progress`, if given, is called as progress(done, total, label) per body so
    a GUI can report progress instead of appearing frozen.
    """
    parsed = read_3mf(path)
    if not parsed:
        raise ValueError(f"No printable objects found in 3MF: {path}")
    if np.any(rot_angles):
        R = _rotation(rot_angles)
        for b in parsed:
            b.vertices = b.vertices @ R.T
    grid = _build_grid(parsed, resolution)
    groups = _assign_groups(parsed, bodies)
    arr_print = _vox_group(groups["print"], grid, progress, "print")
    if arr_print is None:  # no object mapped to print -> empty target
        arr_print = np.zeros(grid.shape, dtype=np.uint8)
    arr_insert = _vox_group(groups["insert"], grid, progress, "insert")
    arr_zero = _vox_group(groups["zero_dose"], grid, progress, "zero-dose")
    return arr_print, arr_insert, arr_zero
