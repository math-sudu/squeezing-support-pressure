"""Finite-element solver construction, chord measurements and checkpoint loading."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/vendor"))
from dp_material import DPParameters, elastic_precompression
from mixed_geometry import geometry_from_config
from mixed_reference import ReferenceSolver


def build_solver(config):
    material = DPParameters.from_young_poisson(**config["material"])
    p0 = torch.tensor(config["p0"], device=config["device"], dtype=torch.float64)
    f0, _, _ = elastic_precompression(material, p0)
    mesh = geometry_from_config(config, ROOT, f0, mesh=config["reference_mesh"])
    return ReferenceSolver(mesh, f0, material, config["p0"], config["reference"])


def observe(solver):
    arc = torch.tensor([0, 0, 0, 3], device=solver.f0.device)
    frac = torch.tensor([0., 1., .5, .5], device=solver.f0.device, dtype=torch.float64)
    x = solver.mesh.wall.sample(arc, frac)[0] @ solver.f0[:2, :2].T
    now = x + solver.wall_displacement(arc, frac)
    width0, height0 = (x[0]-x[1]).norm(), (x[2]-x[3]).norm()
    cw, ch = 1-(now[0]-now[1]).norm()/width0, 1-(now[2]-now[3]).norm()/height0
    return {"width_closure": float(cw), "height_closure": float(ch),
            "width_closure_mm": float(cw*width0*1000), "height_closure_mm": float(ch*height0*1000),
            "initial_width_m": float(width0), "initial_height_m": float(height0)}


def restore(solver, path):
    saved = torch.load(path, map_location=solver.f0.device, weights_only=True)
    if not torch.equal(saved["coordinates"], solver.mesh.volume.x) or not torch.equal(saved["connectivity"], solver.mesh.conn):
        raise ValueError("Checkpoint geometry/point identity mismatch")
    for key in ("u", "cp", "kappa"):
        setattr(solver, key, saved[key].clone())
    solver.step, solver.pressure = saved["step"], saved["pressure"]
