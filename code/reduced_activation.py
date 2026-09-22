"""Three-amplitude activation model with retained material-point history.

Export mechanical coefficients from one known starting PINN field. Online
steps require this saved model and its own state; no global tangent assembly,
saved terminal field or PINN evaluation is used. Material-point history is
retained rather than inferred from the eight output moments.
"""
from dataclasses import asdict

import numpy as np
import torch

from coupled_activation import components
from finite_increment_activation import tensor_moments
from vendor.saved_pinn import dp_material as dp


def export_model(model, basis, initial_z, path):
    full = model.load.new_zeros((model.space.nq1, basis.shape[1]))
    full[model.space.free] = model.load.new_tensor(basis)
    gradient = dp.plane_tensor(torch.einsum("qam,qaij->qmij", full[model.space.connectivity["volume"]],
                                            model.gradient), zz=0.)
    feedback = torch.einsum("qmij,qkij,q->qmk", gradient[:, :, :2, :2], model.partial_cp, model.mesh.volume.weight)
    arrays = dict(
        elastic=basis.T @ (model.elastic_matrix @ basis),
        follower=basis.T @ (model.space.follower_matrix(model.p0) @ basis),
        load=basis.T @ model.load.cpu().numpy(), gradient=gradient.cpu().numpy(), feedback=feedback.cpu().numpy(),
        volume_F=model.fields["volume_F"].cpu().numpy(), cp=model.state.cp.cpu().numpy(),
        kappa=model.state.kappa.cpu().numpy(), kernel=model.kernel.cpu().numpy(),
        gauge_modes=full.reshape(-1, 2, basis.shape[1])[model.space.gauge_nodes].cpu().numpy(),
        gauge_positions=(model.gauges + model.fields["node_u"][model.space.gauge_nodes]).cpu().numpy(),
        lengths=model.lengths.cpu().numpy(), initial_z=np.asarray(initial_z), pressure=np.array(model.pressure),
    )
    arrays.update({"material_" + k: np.array(v) for k, v in asdict(model.material).items()})
    np.savez_compressed(path, **arrays)


class ReducedActivation:
    def __init__(self, path, device="cpu"):
        with np.load(path) as data:
            self.data = {key: torch.as_tensor(data[key], device=device) for key in data.files
                         if not key.startswith("material_")}
            self.material = dp.DPParameters(**{key.removeprefix("material_"): float(data[key])
                                               for key in data.files if key.startswith("material_")})

    def initial_state(self):
        d = self.data
        return {"amplitudes": torch.zeros_like(d["load"]), "cp": d["cp"].clone(), "kappa": d["kappa"].clone(),
                "pressure": float(d["pressure"])}

    def positions(self, amplitudes):
        return self.data["gauge_positions"] + torch.einsum("gim,m->gi", self.data["gauge_modes"], amplitudes)

    def advance(self, state, step):
        if step < 0 or step > state["pressure"]:
            raise ValueError("Expected a nonnegative pressure unloading increment")
        d = self.data
        previous = dp.MaterialState(state["cp"], state["kappa"])
        pressure = state["pressure"] - step
        matrix = d["elastic"] + pressure * d["follower"]
        load = d["load"] + d["follower"] @ state["amplitudes"]
        increment = torch.linalg.solve(matrix, step * load)
        f_n = d["volume_F"] + torch.einsum("qmij,m->qij", d["gradient"], state["amplitudes"])

        def evaluate(value):
            trial = dp.evaluate(f_n + torch.einsum("qmij,m->qij", d["gradient"], value), previous, self.material)
            feedback = torch.einsum("qmk,qk->m", d["feedback"], components(trial.cp - previous.cp))
            return matrix @ value + feedback - step * load, trial

        scale = max(float((step * load).norm()), torch.finfo(load.dtype).eps)
        for iteration in range(40):
            variable = increment.detach().requires_grad_(True)
            residual, trial = evaluate(variable)
            if step == 0 or float(residual.detach().norm()) <= 1e-9 * scale:
                break
            jacobian = torch.stack([torch.autograd.grad(r, variable, retain_graph=True)[0] for r in residual])
            correction = torch.linalg.solve(jacobian, -residual).detach()
            factor = 1.
            while factor >= 2**-16:
                candidate = increment + factor * correction
                if float(evaluate(candidate)[0].norm()) < float(residual.detach().norm()):
                    increment = candidate
                    break
                factor *= .5
            else:
                raise RuntimeError("Reduced activation Newton line search stalled")
        else:
            raise RuntimeError("Reduced activation Newton did not converge")
        if step == 0:
            return {key: value.clone() if torch.is_tensor(value) else value for key, value in state.items()}, {
                "moment_mm": [0., 0.], "moment_components_mm": [[0.] * 4] * 2,
                "shortening_mm": [0., 0.], "pressure_mm": [0., 0.], "plastic_feedback_mm": [0., 0.]}
        committed = dp.commit_state(previous, trial)
        new_amplitudes = (state["amplitudes"] + increment).detach()
        positions_n, positions_next = self.positions(state["amplitudes"]), self.positions(new_amplitudes)
        q = []
        shortening = []
        for a, b in ((0, 1), (2, 3)):
            before, after = positions_n[a] - positions_n[b], positions_next[a] - positions_next[b]
            direction = (before + after) / (before.norm() + after.norm())
            q.append(-direction @ (d["gauge_modes"][a] - d["gauge_modes"][b]))
            shortening.append(before.norm() - after.norm())
        q = torch.stack(q)
        pressure_increment = torch.linalg.solve(matrix, step * load)
        dz = tensor_moments(d["kernel"], committed.cp - previous.cp)
        z = d["initial_z"] + tensor_moments(d["kernel"], committed.cp - d["cp"])
        next_state = dict(amplitudes=new_amplitudes, cp=committed.cp, kappa=committed.kappa, pressure=pressure)
        output = {
            "moment_mm": (1000 * d["lengths"] * dz.sum(-1)).tolist(),
            "moment_components_mm": (1000 * d["lengths"][:, None] * dz).tolist(),
            "shortening_mm": (1000 * torch.stack(shortening)).tolist(),
            "pressure_mm": (1000 * q @ pressure_increment).tolist(),
            "plastic_feedback_mm": (1000 * q @ (increment - pressure_increment)).tolist(),
            "C_next": [float(1 - (positions_next[a] - positions_next[b]).norm() / d["lengths"][j])
                       for j, (a, b) in enumerate(((0, 1), (2, 3)))],
            "z_next": z.tolist(), "iterations": iteration, "amplitudes": new_amplitudes.tolist(),
        }
        return next_state, output
