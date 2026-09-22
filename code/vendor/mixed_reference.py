"""Independent displacement FEM on an exact curved reference mesh, GPU only.

Nonassociated residual Newton with an unsymmetric tangent; no energy
minimization, Burgers kernel, projected stress, or PINN labels are involved.
"""
import torch
import dp_reference
from mixed_geometry import cofactor_normal, wall_geometry_checks


def embed(a):
    f = torch.zeros((*a.shape[:-2], 3, 3), device=a.device, dtype=a.dtype)
    f[..., :2, :2], f[..., 2, 2] = a, 1
    return f


class ReferenceSolver:
    def __init__(self, mesh, f0, material, p0, controls):
        self.mesh, self.f0, self.m, self.p0, self.controls = mesh, f0, material, p0, controls
        self.u = torch.zeros_like(mesh.nodes)
        self.cp = torch.eye(3, device=f0.device, dtype=f0.dtype).expand(len(mesh.volume.x), 3, 3).clone()
        self.kappa = torch.zeros(len(mesh.volume.x), device=f0.device, dtype=f0.dtype)
        self.pinit = dp_reference.update(f0[None], self.cp[:1], self.kappa[:1], material).piola[0]
        self.step, self.pressure = 0, p0

    def deformation(self, u, quad):
        return embed(self.f0[:2, :2]+torch.einsum("qai,qaj->qij", u[quad.conn], quad.grad))

    def residual(self, u, pressure):
        v, b = self.mesh.volume, self.mesh.boundary
        f, fw = self.deformation(u, v), self.deformation(u, b)
        tr = dp_reference.update(f, self.cp, self.kappa, self.m)
        r = self.mesh.weak(tr.piola, fw, pressure, self.p0, self.f0, self.pinit)
        return r, tr, f, fw

    def tangent(self, u, pressure, f):
        mesh, v, b = self.mesh, self.mesh.volume, self.mesh.boundary
        eps = self.controls["tangent_difference"]
        columns = []
        # Central material differences, all quadrature points batched on GPU.
        for k in range(2):
            for l in range(2):
                df = torch.zeros_like(f)
                df[:, k, l] = eps
                pp = dp_reference.update(f+df, self.cp, self.kappa, self.m).piola[:, :2, :2]
                pm = dp_reference.update(f-df, self.cp, self.kappa, self.m).piola[:, :2, :2]
                columns.append((pp-pm)/(2*eps))
        a = torch.stack(columns, -1).reshape(-1, 2, 2, 2, 2)
        ke = torch.einsum("qaj,qijkl,qbl,q->qaibk", v.grad, a, v.grad, v.weight)
        nq = len(v.x)//len(mesh.conn)
        local_dofs = 2*mesh.conn.shape[1]
        ke = ke.reshape(len(mesh.conn), nq, local_dofs, local_dofs).sum(1)
        # Exact derivative of pressure*cof(F)*N, including follower stiffness.
        follower = torch.zeros((len(b.x), 2, 2, 2), device=f.device, dtype=f.dtype)
        n = b.normal
        follower[:, 0, 1, 1], follower[:, 0, 1, 0] = n[:, 0], -n[:, 1]
        follower[:, 1, 0, 1], follower[:, 1, 0, 0] = -n[:, 0], n[:, 1]
        kw = pressure*torch.einsum("qa,qikl,qbl,q->qaibk", b.shape, follower, b.grad, b.weight)
        ke[:mesh.na] += kw.reshape(mesh.na, -1, local_dofs, local_dofs).sum(1)
        dofs = (2*mesh.conn[:, :, None]+torch.arange(2, device=f.device)).reshape(-1, local_dofs)
        nfree = 2*len(mesh.free)
        scale = (self.p0*mesh.test_scale[mesh.free]).repeat_interleave(2)
        if self.controls.get("linear_solver") == "bicgstab":
            from mixed_krylov import SparseTangent
            rows = dofs[:, :, None].expand_as(ke).flatten()
            columns = dofs[:, None, :].expand_as(ke).flatten()
            keep = (rows < nfree) & (columns < nfree)
            return SparseTangent(rows[keep], columns[keep], ke.flatten()[keep]/scale[rows[keep]], nfree)
        ndof = 2*mesh.nnode
        ids = dofs[:, :, None]*ndof+dofs[:, None, :]
        matrix = torch.zeros(ndof*ndof, device=f.device, dtype=f.dtype)
        matrix.index_add_(0, ids.flatten(), ke.flatten())
        matrix = matrix.reshape(ndof, ndof)[:2*len(mesh.free), :2*len(mesh.free)]
        return matrix/scale[:, None]

    def solve(self, pressure):
        old_u = self.u.clone()
        u, records, linear_records = old_u.clone(), [], []
        for iteration in range(self.controls["newton_iterations"]):
            r, trial, f, fw = self.residual(u, pressure)
            norm = float(r.abs().max())
            records.append(norm)
            if norm < self.controls["newton_tolerance"]:
                break
            matrix = self.tangent(u, pressure, f)
            if self.controls.get("linear_solver") == "bicgstab":
                from mixed_krylov import bicgstab
                direction, report = bicgstab(matrix, -r.flatten())
                linear_records.append(report)
                du = direction.reshape(-1, 2)
            else:
                du = torch.linalg.solve(matrix, -r.flatten()).reshape(-1, 2)
            accepted = False
            for backtrack in range(14):
                candidate = u.clone()
                candidate[self.mesh.free] += (0.5**backtrack)*du
                try:
                    rr = self.residual(candidate, pressure)[0]
                    if float(rr.norm()) < float(r.norm()):
                        u, accepted = candidate, True
                        break
                except ValueError:
                    pass
            if not accepted:
                return {"accepted": False, "reason": "Newton line search failed", "residual_history": records}
        r, trial, f, fw = self.residual(u, pressure)
        nw = getattr(self.mesh, "wall_node_count", self.mesh.na)
        geo = wall_geometry_checks(self.mesh.nodes[:nw] @ self.f0[:2, :2].T+u[:nw])
        metrics = {"pressure_ratio": pressure/self.p0, "residual_max": float(r.abs().max()),
            "j_min": float(torch.linalg.det(f).min()), "yield_max": float(trial.yield_value.max()),
            "plastic_volume_max": float((.5*torch.linalg.slogdet(trial.cp)[1]-self.m.beta*trial.kappa).abs().max()),
            "kappa_max": float(trial.kappa.max()), "delta_lambda_max": float(trial.dl.max()),
            "cumulative_plastic_area_m2": float((self.mesh.volume.weight*(trial.kappa > 1e-7)).sum()),
            "active_plastic_area_m2": float((self.mesh.volume.weight*(trial.dl > 1e-7)).sum()),
            "residual_history": records, "linear_solver": self.controls.get("linear_solver", "dense"),
            "linear_solve_history": linear_records, **geo}
        ok = (metrics["residual_max"] < self.controls["newton_tolerance"] and metrics["j_min"] > .5
              and metrics["yield_max"] < 1e-8 and metrics["plastic_volume_max"] < 1e-10
              and geo["wall_self_intersections"] == 0)
        metrics["accepted"] = ok
        if ok:
            self.u, self.cp, self.kappa = u.detach(), trial.cp.detach(), trial.kappa.detach()
            self.pressure, self.step = pressure, self.step+1
        return metrics

    def wall_displacement(self, arc, fraction):
        if getattr(self.mesh, "degree", 1) == 2:
            from mixed_reference_sampling import locate
            points = self.mesh.wall.sample(arc, fraction)[0]
            quad = locate(self.mesh, points)
            return (quad.shape[:, :, None]*self.u[quad.conn]).sum(1)
        counts = torch.tensor(self.mesh.counts, device=arc.device)
        offsets = torch.cat((torch.zeros_like(counts[:1]), counts.cumsum(0)[:-1]))
        local = fraction*counts[arc]
        left = local.floor().long().clamp_min(0)
        left = torch.minimum(left, counts[arc]-1)
        r = local-left
        first = offsets[arc]+left
        return (1-r[:, None])*self.u[first]+r[:, None]*self.u[(first+1) % self.mesh.na]

    def checkpoint(self):
        return {"u": self.u, "cp": self.cp, "kappa": self.kappa,
                "step": self.step, "pressure": self.pressure,
                "coordinates": self.mesh.volume.x, "connectivity": self.mesh.conn,
                "material_point_ids": torch.arange(len(self.kappa), device=self.kappa.device)}
