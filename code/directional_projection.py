"""Finite-increment adjoints of frozen PINN fields on an enriched weak space.

No equilibrium solution or network parameter is changed. Q1 virtual fields are
enriched by the three exact snapshot increments minus their nodal interpolants.
The same space therefore contains every analyzed displacement increment.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
import torch

from vendor.saved_pinn import dp_material as dp
from vendor.saved_pinn.mixed_geometry import cofactor_normal, gauss


def piola(f, cp, material):
    elastic = 0.5 * dp.log_spd_plane(dp.sym(f @ torch.linalg.solve(cp, f.transpose(-1, -2))))
    tau, _ = dp.hencky_response(elastic, material)
    return tau @ torch.linalg.inv(f).transpose(-1, -2)


def linearization(f, cp, delta_cp, material):
    """Partial derivative at fixed Cp and the separate Cp directional term."""
    f = f.detach().requires_grad_(True)
    cp = cp.detach().requires_grad_(True)
    stress = piola(f, cp, material)
    tangent, plastic = [], []
    for i in range(2):
        for j in range(2):
            df, dc = torch.autograd.grad(stress[:, i, j].sum(), (f, cp), retain_graph=True)
            tangent.append(df[:, :2, :2].detach())
            product = dc * delta_cp
            plastic.append(torch.stack((product[:, 0, 0], product[:, 0, 1] + product[:, 1, 0],
                                         product[:, 1, 1], product[:, 2, 2]), -1).detach())
    return (torch.stack(tangent, 1).reshape(-1, 2, 2, 2, 2),
            torch.stack(plastic, 1).transpose(1, 2).reshape(-1, 4, 2, 2))


class ProjectionSpace:
    def __init__(self, mesh, nodal_u, volume_f, wall_u, wall_f, gauges):
        self.mesh = mesh
        self.nq1 = 2 * mesh.nnode
        self.nmode = len(nodal_u) - 1
        self.free = torch.cat((torch.stack((2 * mesh.free, 2 * mesh.free + 1), -1).flatten(),
                               torch.arange(self.nq1, self.nq1 + self.nmode, device=mesh.device)))
        self.size = len(self.free)
        self.nodal_increment = nodal_u.diff(dim=0)
        identity = torch.eye(2, device=mesh.device, dtype=mesh.dtype)
        self.gradient, self.values, self.connectivity = {}, {}, {}
        residual_grad = {}
        for name, quad, f in (("volume", mesh.volume, volume_f), ("wall", mesh.boundary, wall_f)):
            interpolated = torch.einsum("mqai,qaj->qmij", self.nodal_increment[:, quad.conn], quad.grad)
            residual_grad[name] = f.diff(dim=0)[:, :, :2, :2].permute(1, 0, 2, 3) - interpolated
        self.scales = torch.einsum("qmij,qmij,q->m", residual_grad["volume"],
                                   residual_grad["volume"], mesh.volume.weight).sqrt()
        if bool((self.scales <= 0).any()):
            raise ValueError("A snapshot enrichment has zero gradient; remove that redundant mode")
        for name, quad in (("volume", mesh.volume), ("wall", mesh.boundary)):
            hats = torch.einsum("ik,qaj->qaikj", identity, quad.grad).reshape(-1, 8, 2, 2)
            self.gradient[name] = torch.cat((hats, residual_grad[name] / self.scales[None, :, None, None]), 1)
            modes = torch.arange(self.nq1, self.nq1 + self.nmode, device=mesh.device).expand(len(quad.x), -1)
            local = torch.stack((2 * quad.conn, 2 * quad.conn + 1), -1).flatten(1)
            self.connectivity[name] = torch.cat((local, modes), 1)
        quad = mesh.boundary
        hats = torch.einsum("ik,qa->qaik", identity, quad.shape).reshape(-1, 8, 2)
        interpolated = torch.einsum("mqai,qa->qmi", self.nodal_increment[:, quad.conn], quad.shape)
        residual_u = wall_u.diff(dim=0).permute(1, 0, 2) - interpolated
        self.values["wall"] = torch.cat((hats, residual_u / self.scales[None, :, None]), 1)
        distance = torch.cdist(gauges, mesh.nodes)
        if float(distance.min(dim=1).values.max()) > 1e-9:
            raise ValueError("The Q1 wall must contain all four chord endpoints")
        self.gauge_nodes = distance.argmin(dim=1)

    def matrix(self, local, name):
        """Assemble after summing quadrature within each curved element."""
        cells = len(self.mesh.conn) if name == "volume" else self.mesh.na
        per_cell = len(local) // cells
        values = local.reshape(cells, per_cell, *local.shape[1:]).sum(1).cpu().numpy()
        conn = self.connectivity[name][::per_cell].cpu().numpy()
        rows = np.broadcast_to(conn[:, :, None], values.shape).ravel()
        cols = np.broadcast_to(conn[:, None, :], values.shape).ravel()
        full = coo_matrix((values.ravel(), (rows, cols)), shape=(self.nq1 + self.nmode,) * 2).tocsc()
        free = self.free.cpu().numpy()
        return full[free][:, free]

    def vector(self, local, name):
        out = local.new_zeros(self.nq1 + self.nmode)
        out.index_add_(0, self.connectivity[name].flatten(), local.flatten())
        return out[self.free]

    def internal(self, stress):
        local = torch.einsum("qaij,qij,q->qa", self.gradient["volume"],
                             stress[:, :2, :2], self.mesh.volume.weight)
        return self.vector(local, "volume")

    def surface(self, traction):
        local = torch.einsum("qai,qi,q->qa", self.values["wall"],
                             traction[:, :2], self.mesh.boundary.weight)
        return self.vector(local, "wall")

    def follower_matrix(self, pressure):
        g = self.gradient["wall"]
        n = self.mesh.boundary.normal
        dload = torch.stack((g[:, :, 1, 1] * n[:, None, 0] - g[:, :, 1, 0] * n[:, None, 1],
                            -g[:, :, 0, 1] * n[:, None, 0] + g[:, :, 0, 0] * n[:, None, 1]), -1)
        local = torch.einsum("qai,qbi,q->qab", self.values["wall"], dload,
                             pressure * self.mesh.boundary.weight)
        return self.matrix(local, "wall")

    def increment(self, index):
        modes = self.scales.new_zeros(self.nmode)
        modes[index] = self.scales[index]
        full = torch.cat((self.nodal_increment[index].flatten(), modes))
        return full[self.free]

    def chord_rows(self, positions_before, positions_after):
        full = self.scales.new_zeros((self.nq1 + self.nmode, 2))
        lengths = []
        for j, (a, b) in enumerate(((0, 1), (2, 3))):
            d0 = positions_before[a] - positions_before[b]
            d1 = positions_after[a] - positions_after[b]
            # Normalize to closure after applying the exact chord secant.
            direction = (d0 + d1) / (d0.norm() + d1.norm())
            ia, ib = int(self.gauge_nodes[a]), int(self.gauge_nodes[b])
            full[2 * ia:2 * ia + 2, j] = -direction
            full[2 * ib:2 * ib + 2, j] = direction
            lengths.append(d0.norm() - d1.norm())
        return full[self.free], torch.stack(lengths)


def project_interval(space, fields, material, p0, pressures, index, gauges_precompressed, path_order=3,
                     include_displacement_split=False):
    mesh = space.mesh
    f0, f1 = fields["volume_F"][index:index + 2]
    cp0, cp1 = fields["volume_cp"][index:index + 2]
    fw0, fw1 = fields["wall_F"][index:index + 2]
    r0, r1 = pressures[index:index + 2]
    delta_ell = r0 - r1
    matrix = space.follower_matrix(p0 * (r0 + r1) / 2)
    plastic_components = f0.new_zeros((len(f0), 4, 2, 2))
    points, weights = gauss(path_order, mesh.device)
    gradient = space.gradient["volume"]
    local_matrix = f0.new_zeros((len(f0), gradient.shape[1], gradient.shape[1]))
    for t, weight in zip(points, weights):
        f = f0 + t * (f1 - f0)
        cp = cp0 + t * (cp1 - cp0)
        dp.check_deformation(f)
        dp.check_block(cp, "Interpolated Cp", symmetric=True, positive=True)
        tangent, plastic = linearization(f, cp, cp1 - cp0, material)
        local_matrix += weight * torch.einsum("qaij,qijkl,qbkl,q->qab", gradient, tangent,
                                               gradient, mesh.volume.weight)
        plastic_components += weight * plastic
    matrix += space.matrix(local_matrix, "volume")
    load = space.surface(p0 * cofactor_normal((fw0 + fw1) / 2, mesh.boundary.normal))
    plastic = space.internal(plastic_components.sum(1))
    residuals = []
    for f, cp, fw, pressure in ((f0, cp0, fw0, r0), (f1, cp1, fw1, r1)):
        residuals.append(space.internal(piola(f, cp, material)) +
                         space.surface(p0 * pressure * cofactor_normal(fw, mesh.boundary.normal)))
    delta_residual = residuals[1] - residuals[0]
    positions = gauges_precompressed[None] + fields["node_u"][:, space.gauge_nodes]
    q, shortening = space.chord_rows(positions[index], positions[index + 1])
    initial_lengths = torch.stack(((gauges_precompressed[0] - gauges_precompressed[1]).norm(),
                                    (gauges_precompressed[2] - gauges_precompressed[3]).norm()))
    q /= initial_lengths[None]
    adjoint = splu(matrix.T.tocsc()).solve(q.cpu().numpy())
    psi = torch.as_tensor(adjoint, device=mesh.device, dtype=mesh.dtype)
    chi = psi.T @ load
    pi = -psi.T @ plastic
    epsilon = psi.T @ delta_residual
    increment = shortening / initial_lengths
    full_psi = f0.new_zeros((space.nq1 + space.nmode, 2))
    full_psi[space.free] = psi
    psi_grad = torch.einsum("qac,qaij->qcij", full_psi[space.connectivity["volume"]], gradient)
    component_density = -torch.einsum("qcij,qkij,q->qck", psi_grad, plastic_components, mesh.volume.weight)
    density = component_density.sum(-1)
    # These identities detect implementation/line-integration defects. They
    # do not estimate the accuracy of the frozen physical solution.
    identity = (matrix @ space.increment(index).cpu().numpy() + plastic.cpu().numpy()
                - load.cpu().numpy() * delta_ell - delta_residual.cpu().numpy())
    result = dict(chi=chi, plastic=pi, residual=epsilon, increment=increment,
                pressure=chi * delta_ell, plastic_density=density,
                gradient=psi_grad, plastic_component_density=component_density, initial_lengths=initial_lengths,
                closure_identity_error=increment - chi * delta_ell - pi - epsilon,
                residual_identity_max=float(np.max(np.abs(identity))),
                adjoint_equation_max=float(np.max(np.abs(matrix.T @ adjoint - q.cpu().numpy()))))
    if include_displacement_split:
        # Offline decomposition: these operators depend on both saved endpoints.
        rhs = torch.stack((load * delta_ell, -plastic, delta_residual), -1)
        displacement = torch.as_tensor(splu(matrix).solve(rhs.cpu().numpy()), device=mesh.device, dtype=mesh.dtype)
        full = f0.new_zeros((space.nq1 + space.nmode, 3))
        full[space.free] = displacement
        gradients = torch.einsum("qas,qaij->sqij", full[space.connectivity["volume"]], gradient)
        result["deformation_split"] = dp.plane_tensor(gradients, zz=0.)
    return result


def reference_adjoint(space, fields, material, p0, pressure, gauges_precompressed):
    """Return the two reference virtual fields and constitutive partials."""
    mesh = space.mesh
    f, cp = fields["volume_F"][0], fields["volume_cp"][0]
    unit_cp = dp.plane_tensor(torch.ones_like(cp[:, :2, :2]))
    tangent, component_partials = linearization(f, cp, unit_cp, material)
    gradient = space.gradient["volume"]
    local = torch.einsum("qaij,qijkl,qbkl,q->qab", gradient, tangent, gradient, mesh.volume.weight)
    matrix = space.matrix(local, "volume") + space.follower_matrix(p0 * pressure)
    position = gauges_precompressed + fields["node_u"][0, space.gauge_nodes]
    q, _ = space.chord_rows(position, position)
    lengths = torch.stack(((gauges_precompressed[0] - gauges_precompressed[1]).norm(),
                           (gauges_precompressed[2] - gauges_precompressed[3]).norm()))
    q /= lengths[None]
    psi = torch.as_tensor(splu(matrix.T.tocsc()).solve(q.cpu().numpy()), device=mesh.device, dtype=mesh.dtype)
    full = f.new_zeros((space.nq1 + space.nmode, 2))
    full[space.free] = psi
    psi_grad = torch.einsum("qac,qaij->qcij", full[space.connectivity["volume"]], gradient)
    psi_wall = torch.einsum("qac,qai->qci", full[space.connectivity["wall"]], space.values["wall"])
    return psi_grad, psi_wall, component_partials


def reference_moments(space, fields, material, p0, pressure, gauges_precompressed):
    """Fixed kernels at the first saved state; later z reads only its own Cp.

    The moment vector is a candidate state representation, not a learned
    evolution rule. Its finite differences approximate the path-weighted Pi.
    """
    psi_grad, _, component_partials = reference_adjoint(
        space, fields, material, p0, pressure, gauges_precompressed)
    mesh = space.mesh
    kernel = -torch.einsum("qcij,qkij,q->qck", psi_grad, component_partials, mesh.volume.weight)
    return kernel, plastic_moments(kernel, fields["volume_cp"], fields["volume_cp"][0])


def plastic_moments(kernel, cp, reference_cp):
    """Integrate tensor states with a kernel that already includes quadrature weights."""
    delta_cp = cp - reference_cp
    components = torch.stack((delta_cp[:, :, 0, 0], delta_cp[:, :, 0, 1],
                              delta_cp[:, :, 1, 1], delta_cp[:, :, 2, 2]), -1)
    return torch.einsum("qck,nqk->nck", kernel, components)


def initial_reference_kernel(mesh, fields, material, p0, pressure, gauges_precompressed, f0):
    """Build a Q1 reference kernel using only the first known state.

    Snapshot enrichment is appropriate for retrospective interval identities,
    but would use future displacements when defining a predictive state.
    """
    initial = {key: fields[key][:1] for key in
               ("node_u", "volume_F", "volume_cp", "wall_u", "wall_F")}
    natural_gauges = torch.linalg.solve(f0[:2, :2], gauges_precompressed.T).T
    space = ProjectionSpace(mesh, initial["node_u"], initial["volume_F"],
                            initial["wall_u"], initial["wall_F"], natural_gauges)
    kernel, _ = reference_moments(space, initial, material, p0, pressure, gauges_precompressed)
    return kernel
