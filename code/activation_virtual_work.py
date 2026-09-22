"""Virtual fields targeting plastic-moment error in the two lower wall corners.

The local goal derivative uses the complete material return with fixed incoming
history. Its adjoint uses the matching algorithmic tangent and follower load.
No plastic term is added to this tangent. The resulting virtual displacements
are held fixed within each optimizer block and reevaluated between blocks.
"""
import numpy as np
from scipy.sparse.linalg import splu
import torch

from directional_projection import ProjectionSpace
from vendor.saved_pinn import dp_material as dp


def goal_partials(f, state, material, kernel, region_ids):
    """Return algorithmic stress tangent and paired-chord corner derivatives."""
    variable = f.detach().requires_grad_(True)
    trial = dp.evaluate(variable, state, material)
    tangent = []
    for i in range(2):
        for j in range(2):
            derivative = torch.autograd.grad(trial.piola[:, i, j].sum(), variable, retain_graph=True)[0]
            tangent.append(derivative[:, :2, :2].detach())
    delta = trial.cp - state.cp
    components = torch.stack((delta[:, 0, 0], delta[:, 0, 1], delta[:, 1, 1], delta[:, 2, 2]), -1)
    goals, derivatives = [], []
    for region in (2, 4):
        goal = (kernel.mean(1) * components * (region_ids == region)[:, None]).sum()
        derivative = torch.autograd.grad(goal, variable, retain_graph=True)[0]
        goals.append(goal.detach())
        derivatives.append(derivative[:, :2, :2].detach())
    return (torch.stack(tangent, 1).reshape(-1, 2, 2, 2, 2),
            torch.stack(derivatives, 1), torch.stack(goals))


def activation_fields(pool, base_work, data, material, gauges, retained_fields=None):
    """Build independent activation adjoints, normalized to the chord-field scale."""
    mesh, nv = pool.mesh, pool.nv
    f = data["f"][:nv].detach()
    state = dp.MaterialState(pool.history.state.cp[:nv], pool.history.state.kappa[:nv])
    natural_gauges = torch.linalg.solve(base_work.f0[:2, :2], gauges.T).T
    space = ProjectionSpace(mesh, f.new_zeros((1, mesh.nnode, 2)), f[None],
                            f.new_zeros((1, len(mesh.boundary.x), 2)), data["f"][nv:][None], natural_gauges)
    tangent, derivative, goals = goal_partials(f, state, material, base_work.reference_kernel,
                                               mesh.wall.project(mesh.volume.x).arc_index)
    gradient = space.gradient["volume"]
    local = torch.einsum("qaij,qijkl,qbkl,q->qab", gradient, tangent, gradient, mesh.volume.weight)
    matrix = space.matrix(local, "volume") + space.follower_matrix(data["pressure"])
    rhs = torch.stack([space.vector(torch.einsum("qaij,qij->qa", gradient, derivative[:, j]), "volume")
                       for j in range(2)], -1)
    psi = torch.as_tensor(splu(matrix.T.tocsc()).solve(rhs.cpu().numpy()), device=f.device, dtype=f.dtype)
    full = f.new_zeros((space.nq1, 2))
    full[space.free] = psi
    adjoint_gradient = torch.einsum("qac,qaij->qcij", full[space.connectivity["volume"]], gradient)
    wall_value = torch.einsum("qac,qai->qci", full[space.connectivity["wall"]], space.values["wall"])
    # Equilibrium test fields remain valid when the current goal derivative
    # vanishes. Retain the incoming loading predictor's activation span so an
    # inactive iterate cannot remove those local equilibrium equations.
    if retained_fields is not None:
        base_count = base_work.gradient.shape[1]
        adjoint_gradient = torch.cat((retained_fields[0][:, base_count:], adjoint_gradient), 1)
        wall_value = torch.cat((retained_fields[1][:, base_count:], wall_value), 1)
    # Remove chord components, then retain the independent local span.
    weights = mesh.volume.weight
    base_gram = torch.einsum("qcij,qdij,q->cd", base_work.gradient, base_work.gradient, weights)
    cross = torch.einsum("qcij,qdij,q->cd", base_work.gradient, adjoint_gradient, weights)
    coefficients = torch.linalg.solve(base_gram, cross)
    adjoint_gradient -= torch.einsum("qcij,cd->qdij", base_work.gradient, coefficients)
    wall_value -= torch.einsum("qci,cd->qdi", base_work.wall_value, coefficients)
    gram = torch.einsum("qcij,qdij,q->cd", adjoint_gradient, adjoint_gradient, weights)
    values, vectors = torch.linalg.eigh(gram)
    retained = values > 64 * torch.finfo(f.dtype).eps * values.max()
    scale = base_gram.diagonal().mean().sqrt()
    transform = vectors[:, retained] * (scale / values[retained].sqrt())[None]
    grad = torch.einsum("qcij,cd->qdij", adjoint_gradient, transform)
    wall = torch.einsum("qci,cd->qdi", wall_value, transform)
    info = {"corner_goals": goals.tolist(), "independent_activation_fields": int(retained.sum()),
            "adjoint_equation_max": float(np.max(np.abs(matrix.T @ psi.cpu().numpy() - rhs.cpu().numpy())))}
    return torch.cat((base_work.gradient, grad), 1), torch.cat((base_work.wall_value, wall), 1), info
