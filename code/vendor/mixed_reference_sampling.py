"""Reconstruct reference FE gradients and history at independent material points.

The point locator intersects rays with the six finite arcs, independently of
the nearest-normal projection used by the PINN collar. No plastic tensor is
interpolated: every accepted displacement is replayed at the requested X.
"""
import torch
from mixed_reference import embed
from mixed_geometry import Quadrature
import dp_reference


def locate(mesh, x):
    radius = x.norm(dim=-1)
    direction = x/radius[:, None]
    centers, radii = mesh.wall.centers, mesh.wall.radii
    dot = (direction[:, None, :]*centers).sum(-1)
    disc = radii.square()-centers.square().sum(-1)+dot.square()
    root = dot+disc.clamp_min(0).sqrt()
    w = root[:, :, None]*direction[:, None, :]
    rel = torch.remainder(torch.atan2(w[..., 1]-centers[:, 1], w[..., 0]-centers[:, 0])-mesh.wall.starts, 2*torch.pi)
    rel = torch.where(rel > 2*torch.pi-1e-10, torch.zeros_like(rel), rel)
    valid = (disc >= 0) & (root > 0) & (rel <= mesh.wall.sweeps+1e-10)
    if not bool(valid.any(-1).all()):
        raise ValueError("Could not locate a reference point on a horseshoe ray")
    arc = valid.to(torch.long).argmax(-1)
    index = torch.arange(len(x), device=x.device)
    s = (rel[index, arc]/mesh.wall.sweeps[arc]).clamp(0, 1)
    wallr = root[index, arc]
    t = torch.log(radius/wallr)/torch.log(mesh.radius/wallr)
    if not bool(((t >= -1e-9) & (t <= 1+1e-9)).all()):
        raise ValueError("Reference interpolation point lies outside the domain")
    t = t.clamp(0, 1)
    counts = torch.tensor(mesh.counts, device=x.device)
    offsets = torch.cat((counts[:1]*0, counts.cumsum(0)[:-1]))
    ia = torch.minimum((s*counts[arc]).floor().long(), counts[arc]-1)
    ir = (torch.searchsorted(mesh.levels, t.contiguous(), right=True)-1).clamp(0, mesh.nr-1)
    xi = s*counts[arc]-ia
    dt = mesh.levels[ir+1]-mesh.levels[ir]
    eta = (t-mesh.levels[ir])/dt
    conn = mesh.conn[ir*mesh.na+offsets[arc]+ia]
    if getattr(mesh, "degree", 1) == 2:
        from mixed_q2 import shape_q2
        shape, dn = shape_q2(xi, eta)
    else:
        shape = torch.stack(((1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta), -1)
        dn = torch.stack((torch.stack((-(1-eta), -(1-xi)), -1), torch.stack((1-eta, -xi), -1),
                          torch.stack((eta, xi), -1), torch.stack((-eta, 1-xi), -1)), -2)
    reconstructed, jac, _ = mesh.mapping(arc, s, t)
    if float((reconstructed-x).abs().max()) > 1e-8:
        raise ValueError("Inverse geometry map failed reconstruction")
    jac = jac*torch.stack((1/counts[arc].to(x.dtype), dt), -1)[:, None, :]
    grad = dn @ torch.linalg.inv(jac)
    return Quadrature(x, torch.ones_like(t), shape, grad, conn)


class ReferenceProbe:
    def __init__(self, source_mesh, validation_mesh, f0, material):
        self.mesh, self.f0, self.m = validation_mesh, f0, material
        self.nv = len(validation_mesh.volume.x)
        self.x = torch.cat((validation_mesh.volume.x, validation_mesh.boundary.x))
        self.interpolation = locate(source_mesh, self.x)
        self.cp = torch.eye(3, device=f0.device, dtype=f0.dtype).expand(len(self.x), 3, 3).clone()
        self.kappa = torch.zeros(len(self.x), device=f0.device, dtype=f0.dtype)
        self.step = 0

    def evaluate(self, u):
        q = self.interpolation
        f = embed(self.f0[:2, :2]+torch.einsum("qai,qaj->qij", u[q.conn], q.grad))
        displacement = (q.shape[:, :, None]*u[q.conn]).sum(1)
        trial = dp_reference.update(f, self.cp, self.kappa, self.m)
        return f, displacement, trial

    def commit(self, trial):
        self.cp, self.kappa = trial.cp.detach().clone(), trial.kappa.detach().clone()
        self.step += 1

    def checkpoint(self, f, u, trial):
        return {"coordinates": self.x, "cp": trial.cp, "kappa": trial.kappa, "sigma": trial.sigma,
                "delta_lambda": trial.dl, "f": f, "u": u, "volume_weights": self.mesh.volume.weight,
                "volume_count": self.nv, "step": self.step}
