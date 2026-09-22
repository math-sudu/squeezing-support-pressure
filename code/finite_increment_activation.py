"""Resolve plastic activation from incoming PINN states and fixed chord kernels.

The tangent predictor uses one known state in the unenriched virtual space.
Saved endpoints are read only by the retrospective diagnostic below. No PINN
training, equilibrium-path continuation, or closure coefficient fitting occurs.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu
import torch

from directional_projection import ProjectionSpace, linearization, project_interval
from vendor.saved_pinn import dp_material as dp
from vendor.saved_pinn.horseshoe_geometry import HorseshoeWall
from vendor.saved_pinn.mixed_geometry import CurvedMesh, cofactor_normal


ROOT = Path(__file__).resolve().parents[1]


def trial_quantities(f, state, material):
    """Kirchhoff trial yield excess, cone denominator and metric flow derivative."""
    j = dp.check_deformation(f)
    elastic = .5 * dp.log_spd_plane(dp.sym(f @ torch.linalg.solve(state.cp, f.transpose(-1, -2))))
    tau, _ = dp.hencky_response(elastic, material)
    pressure, q = dp.invariants(tau)
    excess = q - material.alpha * pressure - j * material.strength(state.kappa)
    denominator = 3 * material.shear + material.alpha * material.bulk * material.beta + j * material.hardening
    # The onset expansion is for the smooth cone, away from q=0.
    dp.require(q > 0, "The smooth-cone activation expansion requires nonzero trial q")
    eye = torch.eye(3, device=f.device, dtype=f.dtype)
    flow = (3 * material.shear / q)[..., None, None] * dp.dev(elastic) + material.beta / 3 * eye
    metric_flow = 2 * dp.sym(f.transpose(-1, -2) @ dp.exp_sym_plane(-2 * elastic) @ flow @ f)
    return excess, denominator, metric_flow, q


def onset_data(f, state, direction, material):
    """Linearize the trial yield excess along a given deformation direction."""
    variable = f.detach().requires_grad_(True)
    excess, denominator, flow, q = trial_quantities(variable, state, material)
    derivative = torch.autograd.grad(excess.sum(), variable)[0]
    rate = (derivative * direction).sum((-2, -1))
    scale = q + torch.linalg.det(variable) * material.strength(state.kappa)
    dp.require(excess <= 128 * torch.finfo(f.dtype).eps * scale,
               "The onset expansion requires an admissible incoming plastic state")
    return {"margin": (-excess.detach()).clamp_min(0), "yield_excess": excess.detach(), "rate": rate.detach(),
            "denominator": denominator.detach(), "flow": flow.detach(), "q": q.detach()}


def hinge_increment(onset, step):
    """Leading cone increment with a finite unloading threshold at each point."""
    multiplier = torch.relu(step * onset["rate"] - onset["margin"]) / onset["denominator"]
    return onset["flow"] * multiplier[:, None, None], multiplier


def tensor_moments(kernel, delta_cp):
    components = torch.stack((delta_cp[:, 0, 0], delta_cp[:, 0, 1],
                              delta_cp[:, 1, 1], delta_cp[:, 2, 2]), -1)
    return torch.einsum("qjk,qk->jk", kernel, components)


def incoming_direction(mesh, fields, material, p0, pressure, gauges, f0):
    """Fixed-Cp unloading tangent, including follower traction; no future fields."""
    natural_gauges = torch.linalg.solve(f0[:2, :2], gauges.T).T
    space = ProjectionSpace(mesh, fields["node_u"][None], fields["volume_F"][None],
                            fields["wall_u"][None], fields["wall_F"][None], natural_gauges)
    tangent, _ = linearization(fields["volume_F"], fields["volume_cp"],
                               torch.zeros_like(fields["volume_cp"]), material)
    gradient = space.gradient["volume"]
    local = torch.einsum("qaij,qijkl,qbkl,q->qab", gradient, tangent, gradient, mesh.volume.weight)
    matrix = space.matrix(local, "volume") + space.follower_matrix(p0 * pressure)
    load = space.surface(p0 * cofactor_normal(fields["wall_F"], mesh.boundary.normal))
    velocity = torch.as_tensor(splu(matrix).solve(load.cpu().numpy()), device=mesh.device, dtype=mesh.dtype)
    full = velocity.new_zeros(space.nq1)
    full[space.free] = velocity
    direction = dp.plane_tensor(torch.einsum("qa,qaij->qij", full[space.connectivity["volume"]], gradient), zz=0.)
    return direction


def diagnostic_summary(delta_cp, multiplier, state, kernel, weights, lengths, plastic_threshold):
    active = multiplier > 0
    return {
        "moment_components_mm": (tensor_moments(kernel, delta_cp) * lengths[:, None]).tolist(),
        "moment_mm": (tensor_moments(kernel, delta_cp).sum(-1) * lengths).tolist(),
        "active_area_m2": float(weights[active].sum()),
        "newly_active_area_m2": float(weights[active & (state.kappa <= plastic_threshold)].sum()),
        "delta_kappa_integral_m2": float((weights * multiplier).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--construction", type=Path, default=ROOT / "runs/pinn_directional_repair")
    parser.add_argument("--branch", type=Path, default=ROOT / "runs/pinn_half_step")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/directional_closure/activation.json")
    parser.add_argument("--single-path", action="store_true", help="Diagnose only the construction path")
    args = parser.parse_args()
    torch.set_num_threads(1)
    base = args.construction
    out = ROOT / "runs/directional_closure"
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    source = json.loads((base / "source.json").read_text(encoding="utf-8"))
    material = dp.DPParameters.from_young_poisson(**config["material"])
    with np.load(out / "reference.npz") as ref:
        kernel = torch.as_tensor(ref["kernel"], device=args.device)
        reference_x = ref["x"].copy()
    with (base / "pressure_closure.csv").open(encoding="utf-8", newline="") as stream:
        observation = next(csv.DictReader(stream))
    lengths = kernel.new_tensor([1000 * float(observation["initial_" + j + "_m"]) for j in ("width", "height")])
    result = {
        "purpose": "Retrospective activation construction/diagnosis of the specified saved intervals.",
        "reference": "runs/directional_closure/reference.npz; kernel includes natural quadrature weights",
        "predictor": "Incoming fixed-Cp tangent in Q1 without snapshots; residual is not corrected or fitted.",
        "intervals": [],
    }
    paths = [(base, 0)] if args.single_path else [(base, 0), (args.branch, 1)]
    for branch, start_index in paths:
        with np.load(branch / "projection_fields.npz") as data:
            np.testing.assert_array_equal(reference_x, data["volume_x"])
            fields = {key: torch.as_tensor(data[key], device=args.device) for key in
                      ("node_u", "volume_F", "volume_cp", "volume_kappa", "wall_u", "wall_F")}
            spec = json.loads(str(data["mesh_spec"]))
            f0 = torch.as_tensor(data["f0"], device=args.device)
            pressures = data["pressures"].tolist()
            regions = torch.as_tensor(data["projection_region"], device=args.device)
            region_names = data["projection_region_names"].tolist()
        gauges = f0.new_tensor(source["observation_points_precompressed_m"])
        wall = HorseshoeWall.from_json(ROOT / "results/reused_pinn/geometry.json", device=args.device, scale=1 / f0[0, 0])
        mesh = CurvedMesh(wall, config["outer_widths"] * 13.46 / float(f0[0, 0]), **spec)
        natural_gauges = torch.linalg.solve(f0[:2, :2], gauges.T).T
        offline_space = ProjectionSpace(mesh, fields["node_u"], fields["volume_F"],
                                        fields["wall_u"], fields["wall_F"], natural_gauges)
        for n in range(start_index, len(pressures) - 1):
            incoming = {key: value[n] for key, value in fields.items()}
            state = dp.MaterialState(incoming["volume_cp"], incoming["volume_kappa"])
            f = incoming["volume_F"]
            step = pressures[n] - pressures[n + 1]
            direction = incoming_direction(mesh, incoming, material, config["p0"], pressures[n], gauges, f0)
            onset = onset_data(f, state, direction, material)
            predicted_cp, predicted_lambda = hinge_increment(onset, step)
            finite = dp.evaluate(f + step * direction, state, material)
            # The endpoint secant is diagnostic information, never a query input.
            endpoint_f = fields["volume_F"][n + 1]
            secant = (endpoint_f - f) / step
            endpoint_onset = onset_data(f, state, secant, material)
            secant_cp, secant_lambda = hinge_increment(endpoint_onset, step)
            observed = dp.evaluate(endpoint_f, state, material)
            np.testing.assert_allclose(observed.cp.cpu(), fields["volume_cp"][n + 1].cpu(), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(observed.kappa.cpu(), fields["volume_kappa"][n + 1].cpu(), rtol=1e-12, atol=1e-14)
            record = {"branch": branch.name, "pressure_start": pressures[n], "pressure_end": pressures[n + 1],
                      "incoming_yield_excess_max_p0": float(onset["yield_excess"].max() / config["p0"]),
                      "observed_apex_points": int((observed.branch == 2).sum()),
                      "tangent_predictor_apex_points": int((finite.branch == 2).sum())}
            for name, dc, dl in (("observed", observed.cp - state.cp, observed.delta_lambda),
                                  ("incoming_tangent_hinge", predicted_cp, predicted_lambda),
                                  ("incoming_tangent_return", finite.cp - state.cp, finite.delta_lambda),
                                  ("endpoint_secant_hinge", secant_cp, secant_lambda)):
                record[name] = diagnostic_summary(dc, dl, state, kernel, mesh.volume.weight, lengths, config["plastic_threshold"])
            split = project_interval(offline_space, fields, material, config["p0"], pressures, n, gauges,
                                     include_displacement_split=True)["deformation_split"]
            record["deformation_split_max_abs_error"] = float((split.sum(0) - (endpoint_f - f)).abs().max())
            # Keep the zero-residual counterfactual tied to the same endpoint operators;
            # it is an attribution within a secant identity, not a new equilibrium.
            for name, delta_f in (("offline_pressure_hinge", split[0]),
                                  ("offline_pressure_plastic_hinge", split[0] + split[1]),
                                  ("offline_all_terms_hinge", split.sum(0))):
                split_onset = onset_data(f, state, delta_f / step, material)
                dc, dl = hinge_increment(split_onset, step)
                record[name] = diagnostic_summary(dc, dl, state, kernel, mesh.volume.weight, lengths, config["plastic_threshold"])
                if name == "offline_pressure_plastic_hinge":
                    without_residual = dc
                elif name == "offline_all_terms_hinge":
                    residual_change = dc - without_residual
            record["residual_activation_change_by_region_mm"] = {
                name: (tensor_moments(kernel[regions == region], residual_change[regions == region]).sum(-1) * lengths).tolist()
                for region, name in enumerate(region_names)
            }
            np.testing.assert_allclose(np.sum(list(record["residual_activation_change_by_region_mm"].values()), axis=0),
                                       np.array(record["offline_all_terms_hinge"]["moment_mm"])
                                       - record["offline_pressure_plastic_hinge"]["moment_mm"], atol=1e-13)
            result["intervals"].append(record)
            print(json.dumps({key: value if not isinstance(value, dict) else value["moment_mm"]
                              for key, value in record.items() if key != "residual_activation_change_by_region_mm"}), flush=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
