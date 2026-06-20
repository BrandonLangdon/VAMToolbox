"""3MF import tests (reader + analytic beam-lattice voxelizer).

Uses lib3mf to WRITE a small beam-lattice sample, then exercises the reader and
voxelizer on it. Skipped when lib3mf is not installed."""
import numpy as np
import pytest

pytest.importorskip("lib3mf")
import lib3mf  # noqa: E402

import vamtoolbox  # noqa: E402
from vamtoolbox import threemf  # noqa: E402


def _write_tetra_lattice(path, with_balls=True):
    """A 4-strut tetra-ish lattice: apex (node 4) connected to a square base."""
    w = lib3mf.Wrapper()
    model = w.CreateModel()
    mesh = model.AddMeshObject()
    mesh.SetName("lat")
    for (x, y, z) in [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (5, 5, 8)]:
        p = lib3mf.Position()
        p.Coordinates = (float(x), float(y), float(z))
        mesh.AddVertex(p)
    bl = mesh.BeamLattice()
    bl.SetMinLength(0.001)
    if with_balls:
        bl.SetBallOptions(lib3mf.BeamLatticeBallMode.All, 1.5)
    for (a, b) in [(4, 0), (4, 1), (4, 2), (4, 3)]:
        beam = lib3mf.Beam()
        beam.Indices = (a, b)
        beam.Radii = (1.0, 1.0)
        bl.AddBeam(beam)
    model.AddBuildItem(mesh, w.GetIdentityTransform())
    model.QueryWriter("3mf").WriteToFile(str(path))


def test_read_beam_lattice(tmp_path):
    p = tmp_path / "lat.3mf"
    _write_tetra_lattice(p, with_balls=False)
    bodies = threemf.read_3mf(str(p))
    assert len(bodies) == 1
    b = bodies[0]
    assert b.vertices.shape == (5, 3)
    assert b.has_beams and b.beam_nodes.shape == (4, 2)
    np.testing.assert_allclose(b.vertices[4], [5, 5, 8])
    assert b.beam_radii.min() == 1.0


def test_voxelize_shape_and_fill(tmp_path):
    p = tmp_path / "lat.3mf"
    _write_tetra_lattice(p)
    arr, insert, zero = threemf.voxelize_3mf(str(p), resolution=60)
    assert arr.ndim == 3 and arr.shape[2] == 60
    assert arr.shape[0] == arr.shape[1]            # square XY
    assert 0.0 < (arr > 0).mean() < 0.5            # struts present, not solid
    assert insert is None and zero is None
    assert arr.dtype == np.uint8


def test_capsule_hits_segment(tmp_path):
    """A single horizontal beam: voxels near the segment must be filled."""
    w = lib3mf.Wrapper()
    model = w.CreateModel()
    mesh = model.AddMeshObject()
    for (x, y, z) in [(-10, 0, 0), (10, 0, 0)]:
        pos = lib3mf.Position()
        pos.Coordinates = (float(x), float(y), float(z))
        mesh.AddVertex(pos)
    bl = mesh.BeamLattice()
    bl.SetMinLength(0.001)
    beam = lib3mf.Beam()
    beam.Indices = (0, 1)
    beam.Radii = (2.0, 2.0)
    bl.AddBeam(beam)
    model.AddBuildItem(mesh, w.GetIdentityTransform())
    p = tmp_path / "beam.3mf"
    model.QueryWriter("3mf").WriteToFile(str(p))
    arr, _, _ = threemf.voxelize_3mf(str(p), resolution=40)
    assert (arr > 0).any()                          # the strut got voxelized


def test_targetgeometry_threemf(tmp_path):
    p = tmp_path / "lat.3mf"
    _write_tetra_lattice(p)
    tg = vamtoolbox.geometry.TargetGeometry(threemffilename=str(p), resolution=48)
    assert tg.array.shape[2] == 48
    assert (tg.array > 0).any()
    # .3mf passed via stlfilename is auto-routed
    tg2 = vamtoolbox.geometry.TargetGeometry(stlfilename=str(p), resolution=32)
    assert (tg2.array > 0).any()


# --------------------------------------------------------------------------- #
# Naming convention
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,role", [
    ("insert_handle", "insert"),
    ("Insert-Pin", "insert"),
    ("zerodose_port", "zero_dose"),
    ("zero_dose_channel", "zero_dose"),
    ("zero-dose.x", "zero_dose"),
    ("nodose_region", "zero_dose"),
    ("lattice_infill", "print"),
    ("shell.outer", "print"),
    ("model 1", "print"),
    ("my_part", "print"),
    ("[insert] handle", "insert"),
    ("(zerodose) cooling", "zero_dose"),
    ("[lattice] core", "print"),
    ("models_x", "print"),      # 'models' is not an alias
    ("inserttab", "print"),     # no separator after 'insert'
    ("", "print"),
])
def test_role_from_name(name, role):
    assert threemf.role_from_name(name)[0] == role


def _write_named_lattices(path, names):
    w = lib3mf.Wrapper()
    model = w.CreateModel()
    for k, name in enumerate(names):
        mesh = model.AddMeshObject()
        mesh.SetName(name)
        ox = 10 * k
        for (x, y, z) in [(ox, 0, 0), (ox + 6, 0, 0), (ox + 6, 6, 0),
                          (ox, 6, 0), (ox + 3, 3, 6)]:
            pos = lib3mf.Position()
            pos.Coordinates = (float(x), float(y), float(z))
            mesh.AddVertex(pos)
        bl = mesh.BeamLattice()
        bl.SetMinLength(0.001)
        for (a, b) in [(4, 0), (4, 1), (4, 2), (4, 3)]:
            beam = lib3mf.Beam()
            beam.Indices = (a, b)
            beam.Radii = (0.8, 0.8)
            bl.AddBeam(beam)
        model.AddBuildItem(mesh, w.GetIdentityTransform())
    model.QueryWriter("3mf").WriteToFile(str(path))


def test_auto_role_assignment(tmp_path):
    p = tmp_path / "multi.3mf"
    _write_named_lattices(p, ["lattice_infill", "insert_pin", "zerodose_channel"])
    arr, insert, zero = threemf.voxelize_3mf(str(p), resolution=48, bodies="auto")
    assert (arr > 0).any()           # lattice_infill -> print
    assert insert is not None and (insert > 0).any()      # insert_pin
    assert zero is not None and (zero > 0).any()          # zerodose_channel


def test_all_mode_forces_print(tmp_path):
    p = tmp_path / "multi.3mf"
    _write_named_lattices(p, ["lattice_infill", "insert_pin", "zerodose_channel"])
    arr, insert, zero = threemf.voxelize_3mf(str(p), resolution=48, bodies="all")
    assert (arr > 0).any() and insert is None and zero is None


def test_targetgeometry_defaults_to_auto(tmp_path):
    p = tmp_path / "multi.3mf"
    _write_named_lattices(p, ["lattice_infill", "insert_pin", "zerodose_channel"])
    tg = vamtoolbox.geometry.TargetGeometry(threemffilename=str(p), resolution=40)
    assert tg.insert is not None and (tg.insert > 0).any()
    assert tg.zero_dose is not None and (tg.zero_dose > 0).any()
