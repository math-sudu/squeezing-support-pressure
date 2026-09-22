"""Incoming-field finite steps with material activation and plastic feedback.

The elastic and plastic stress partials are frozen at the known starting state.
The pressure follower term uses the requested endpoint pressure. Nonlinear
material returns are always evaluated from the same committed incoming history.
No endpoint snapshot, fitted label or algorithmic stress tangent is used.
"""
import numpy as np
from scipy.sparse.linalg import splu
import torch

from directional_projection import ProjectionSpace, linearization, piola
from finite_increment_activation import tensor_moments, trial_quantities
from vendor.saved_pinn import dp_material as dp
from vendor.saved_pinn.mixed_geometry import cofactor_normal


def components(tensor):
    return torch.stack((tensor[:, 0, 0], tensor[:, 0, 1], tensor[:, 1, 1], tensor[:, 2, 2]), -1)


class CoupledActivation:
    """Frozen-partial incremental balance on the incoming PINN configuration."""

    def __init__(self, mesh, fields, material, p0, pressure, gauges, f0, kernel):
        self.mesh, self.fields, self.material = mesh, fields, material
        self.p0, self.pressure, self.gauges, self.kernel = p0, pressure, gauges, kernel
        self.state = dp.MaterialState(fields["volume_cp"], fields["volume_kappa"])
        natural_gauges = torch.linalg.solve(f0[:2, :2], gauges.T).T
        self.space = ProjectionSpace(mesh, fields["node_u"][None], fields["volume_F"][None],
                                     fields["wall_u"][None], fields["wall_F"][None], natural_gauges)
        self.gradient = self.space.gradient["volume"]
        unit_cp = dp.plane_tensor(torch.ones_like(self.state.cp[:, :2, :2]))
        self.elastic, self.partial_cp = linearization(fields["volume_F"], self.state.cp, unit_cp, material)
        self.elastic_matrix = self.assemble(self.elastic)
        self.load = self.space.surface(p0 * cofactor_normal(fields["wall_F"], mesh.boundary.normal))
        self.residual = self.space.internal(piola(fields["volume_F"], self.state.cp, material)) + pressure * self.load
        variable = fields["volume_F"].detach().requires_grad_(True)
        excess, denominator, flow, q = trial_quantities(variable, self.state, material)
        scale = q + torch.linalg.det(variable) * material.strength(self.state.kappa)
        dp.require(excess <= 128 * torch.finfo(variable.dtype).eps * scale,
                   "Coupled activation requires an admissible incoming material state")
        self.yield_gradient = torch.autograd.grad(excess.sum(), variable)[0].detach()
        self.margin, self.denominator, self.flow = (-excess.detach()).clamp_min(0), denominator.detach(), flow.detach()
        self.lengths = torch.stack(((gauges[0] - gauges[1]).norm(), (gauges[2] - gauges[3]).norm()))

    def assemble(self, tangent):
        local = torch.einsum("qaij,qijkl,qbkl,q->qab", self.gradient, tangent,
                             self.gradient, self.mesh.volume.weight)
        return self.space.matrix(local, "volume")

    def full(self, displacement):
        full = self.load.new_zeros(self.space.nq1)
        full[self.space.free] = displacement
        return full

    def deformation(self, displacement, location="volume"):
        full = self.full(displacement)
        return dp.plane_tensor(torch.einsum("qa,qaij->qij", full[self.space.connectivity[location]],
                                            self.space.gradient[location]), zz=0.)

    def material_increment(self, displacement, law, derivative=False):
        delta_f = self.deformation(displacement)
        if law == "hinge":
            excess = (self.yield_gradient * delta_f).sum((-2, -1)) - self.margin
            multiplier = torch.relu(excess) / self.denominator
            delta_cp = self.flow * multiplier[:, None, None]
            dcdf = (components(self.flow)[:, :, None, None] * self.yield_gradient[:, None, :2, :2]
                    * ((excess > 0) / self.denominator)[:, None, None, None]) if derivative else None
        elif law == "return":
            variable = (self.fields["volume_F"] + delta_f).detach().requires_grad_(derivative)
            trial = dp.evaluate(variable, self.state, self.material)
            delta_cp, multiplier = trial.cp - self.state.cp, trial.delta_lambda
            dcdf = None
            if derivative:
                dcdf = torch.stack([torch.autograd.grad(value.sum(), variable, retain_graph=True)[0][:, :2, :2]
                                    for value in components(trial.cp).unbind(-1)], 1).detach()
            delta_cp, multiplier = delta_cp.detach(), multiplier.detach()
        else:
            raise ValueError("Unknown material increment law: " + law)
        stress = torch.einsum("qkij,qk->qij", self.partial_cp, components(delta_cp))
        feedback = self.space.internal(stress)
        tangent = torch.einsum("qkij,qkab->qijab", self.partial_cp, dcdf) if derivative else None
        return delta_cp, multiplier, feedback, tangent

    def matrix(self, step):
        return self.elastic_matrix + self.space.follower_matrix(self.p0 * (self.pressure - step))

    def solve(self, step, *, law="return", basis=None):
        """Preserve incoming residual: K_h d + B_n dc(d) = h f_n.

        A supplied displacement basis performs a Galerkin reduction of this
        equation. The Newton Jacobian is K_h + B_n dCp/dd, so feedback occurs
        exactly once. Solver tolerance resolves this algebraic equation only.
        """
        if step < 0 or step > self.pressure:
            raise ValueError("Expected a nonnegative unloading increment within remaining pressure")
        matrix = self.matrix(step)
        load = self.load.cpu().numpy()
        lift = (lambda x: x) if basis is None else (lambda x: basis @ x)
        restrict = (lambda x: x) if basis is None else (lambda x: basis.T @ x)
        reduced_matrix = matrix if basis is None else basis.T @ (matrix @ basis)
        if basis is not None:
            full_basis = self.load.new_zeros((self.space.nq1, basis.shape[1]))
            full_basis[self.space.free] = self.load.new_tensor(basis)
            reduced_gradient = torch.einsum("qam,qaij->qmij", full_basis[self.space.connectivity["volume"]], self.gradient)
        solve_linear = (lambda a, b: splu(a.tocsc()).solve(b)) if basis is None else np.linalg.solve
        coefficients = solve_linear(reduced_matrix, restrict(step * load))
        if step == 0:
            coefficients[:] = 0
        scale = max(np.linalg.norm(restrict(step * load)), np.finfo(float).eps)
        for iteration in range(40):
            displacement = self.load.new_tensor(lift(coefficients))
            dc, dl, feedback, tangent = self.material_increment(displacement, law, derivative=True)
            residual = restrict(matrix @ lift(coefficients) + feedback.cpu().numpy() - step * load)
            if step == 0 or np.linalg.norm(residual) <= 1e-9 * scale:
                break
            if basis is None:
                reduced_jacobian = matrix + self.assemble(tangent)
            else:
                reduced_jacobian = reduced_matrix + torch.einsum(
                    "qmij,qijkl,qnkl,q->mn", reduced_gradient, tangent, reduced_gradient,
                    self.mesh.volume.weight).cpu().numpy()
            correction = solve_linear(reduced_jacobian, -residual)
            factor = 1.
            while factor >= 2**-16:
                candidate = coefficients + factor * correction
                candidate_d = self.load.new_tensor(lift(candidate))
                new_feedback = self.material_increment(candidate_d, law)[2].cpu().numpy()
                new_residual = restrict(matrix @ lift(candidate) + new_feedback - step * load)
                if np.linalg.norm(new_residual) < np.linalg.norm(residual):
                    coefficients = candidate
                    break
                factor *= .5
            else:
                raise RuntimeError("Coupled activation Newton line search stalled")
        else:
            raise RuntimeError("Coupled activation Newton did not converge")
        if step == 0:
            dc, dl = torch.zeros_like(dc), torch.zeros_like(dl)
            residual = np.zeros_like(residual)
        return {"displacement": displacement.detach(), "delta_cp": dc, "delta_lambda": dl,
                "iterations": iteration, "basis_size": len(coefficients), "law": law,
                "equation_relative_residual": float(np.linalg.norm(residual) / scale)}

    def regional_basis(self, step, regions):
        """Pressure mode plus regional plastic-response modes from the start only.

        Each regional load uses the finite return under the uncoupled incoming
        pressure predictor. No endpoint or POD snapshot enters the span.
        """
        lu = splu(self.matrix(step))
        pressure_mode = lu.solve(self.load.cpu().numpy())
        dc = self.material_increment(self.load.new_tensor(step * pressure_mode), "return")[0]
        stress = torch.einsum("qkij,qk->qij", self.partial_cp, components(dc))
        modes = [pressure_mode]
        for region in torch.unique(regions):
            forcing = self.space.internal(stress * (regions == region)[:, None, None])
            if float(forcing.norm()) > 0:
                modes.append(lu.solve(-forcing.cpu().numpy()))
        # Normalize before rank selection: load and plastic modes have different units/scales.
        modes = np.stack(modes, -1)
        modes /= np.linalg.norm(modes, axis=0)
        u, singular, _ = np.linalg.svd(modes, full_matrices=False)
        return u[:, singular > 64 * np.finfo(float).eps * singular[0]]

    def summarize(self, solution, step):
        displacement, dc = solution["displacement"], solution["delta_cp"]
        position = self.gauges + self.fields["node_u"][self.space.gauge_nodes]
        moved = position + self.full(displacement).reshape(-1, 2)[self.space.gauge_nodes]
        q, shortening = self.space.chord_rows(position, moved)
        matrix = self.matrix(step)
        lu = splu(matrix)
        _, _, feedback, _ = self.material_increment(displacement, solution["law"])
        pressure_d = self.load.new_tensor(lu.solve((step * self.load).cpu().numpy()))
        plastic_d = self.load.new_tensor(lu.solve(-feedback.cpu().numpy()))
        delta_f = self.deformation(displacement)
        returned = dp.evaluate(self.fields["volume_F"] + delta_f, self.state, self.material)
        dc_used = returned.cp - self.state.cp
        exact_residual = self.space.internal(returned.piola) + self.space.surface(
            self.p0 * (self.pressure - step) * cofactor_normal(
                self.fields["wall_F"] + self.deformation(displacement, "wall"), self.mesh.boundary.normal))
        # Linear response to removing known incoming residual, with the coupled
        # Jacobian. This is sensitivity, not a silently applied state correction.
        _, _, _, tangent = self.material_increment(displacement, solution["law"], derivative=True)
        residual_d = self.load.new_tensor(splu((matrix + self.assemble(tangent)).tocsc()).solve(-self.residual.cpu().numpy()))
        if solution["law"] == "hinge":
            excess = (self.yield_gradient * delta_f).sum((-2, -1)) - self.margin
            rate = (self.yield_gradient * self.deformation(residual_d)).sum((-2, -1))
            sensitivity_cp = self.flow * (rate * (excess > 0) / self.denominator)[:, None, None]
            residual_moments = tensor_moments(self.kernel, sensitivity_cp).sum(-1)
        else:
            variable = (self.fields["volume_F"] + delta_f).detach().requires_grad_(True)
            trial = dp.evaluate(variable, self.state, self.material)
            residual_moments = []
            for moment in tensor_moments(self.kernel, trial.cp - self.state.cp).sum(-1):
                derivative = torch.autograd.grad(moment, variable, retain_graph=True)[0]
                residual_moments.append((derivative * self.deformation(residual_d)).sum().detach())
            residual_moments = torch.stack(residual_moments)
        defect_d = self.load.new_tensor(lu.solve((exact_residual - self.residual).detach().cpu().numpy()))
        return {
            "moment_components_mm": (tensor_moments(self.kernel, dc) * (1000 * self.lengths[:, None])).tolist(),
            "moment_mm": (tensor_moments(self.kernel, dc).sum(-1) * (1000 * self.lengths)).tolist(),
            "returned_moment_mm": (tensor_moments(self.kernel, dc_used).sum(-1) * (1000 * self.lengths)).tolist(),
            "shortening_mm": (1000 * shortening).tolist(),
            "pressure_mm": (1000 * q.T @ pressure_d).tolist(),
            "plastic_feedback_mm": (1000 * q.T @ plastic_d).tolist(),
            "balance_defect_mm": (1000 * (shortening - q.T @ (pressure_d + plastic_d))).tolist(),
            "constitutive_and_space_defect_mm": (1000 * q.T @ defect_d).tolist(),
            "incoming_residual_sensitivity_mm": (1000 * q.T @ residual_d).tolist(),
            "incoming_residual_moment_sensitivity_mm": (1000 * self.lengths * residual_moments).tolist(),
            "active_area_m2": float(self.mesh.volume.weight[solution["delta_lambda"] > 0].sum()),
            "iterations": solution["iterations"], "basis_size": solution["basis_size"],
            "equation_relative_residual": solution["equation_relative_residual"],
        }

    def advance(self, step, *, regions=None):
        """Return the predicted fields for repeated steps without teacher forcing."""
        basis = None if regions is None else self.regional_basis(step, regions)
        solution = self.solve(step, basis=basis)
        if step == 0:
            return {key: value.clone() for key, value in self.fields.items()}, solution
        displacement = solution["displacement"]
        volume_f = self.fields["volume_F"] + self.deformation(displacement)
        wall_f = self.fields["wall_F"] + self.deformation(displacement, "wall")
        trial = dp.evaluate(volume_f, self.state, self.material)
        state = dp.commit_state(self.state, trial)
        full = self.full(displacement)
        wall_u = self.fields["wall_u"] + torch.einsum("qa,qai->qi", full[self.space.connectivity["wall"]],
                                                        self.space.values["wall"])
        following = dict(node_u=self.fields["node_u"] + full.reshape(-1, 2), volume_F=volume_f,
                         volume_cp=state.cp, volume_kappa=state.kappa, wall_u=wall_u, wall_F=wall_f)
        return following, solution
