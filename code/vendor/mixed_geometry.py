"""Exact six-arc geometry, curved Q1 test functions and GPU quadrature.

X(s,t)=w(s)*exp(t*log(R/|w(s)|)), with arc fraction s and log radius t.
Only displacement/test fields are Q1; the wall geometry is exact. All
coordinates, weights, gradients, assembly and integration use CUDA float64.
"""
from dataclasses import dataclass
import math
import torch
from horseshoe_geometry import HorseshoeWall


def gauss(order, device):
    """Golub-Welsch on GPU, without NumPy/SciPy quadrature."""
    k = torch.arange(1, order, device=device, dtype=torch.float64)
    off = k / torch.sqrt(4 * k.square() - 1)
    x, v = torch.linalg.eigh(torch.diag(off, 1) + torch.diag(off, -1))
    return (x + 1) / 2, v[0].square()


@dataclass
class Quadrature:
    x: torch.Tensor
    weight: torch.Tensor
    shape: torch.Tensor
    grad: torch.Tensor
    conn: torch.Tensor
    normal: torch.Tensor | None = None


class CurvedMesh:
    def __init__(self, wall, radius, angular=64, radial=16, quadrature=3):
        self.wall, self.radius = wall, radius
        self.device, self.dtype = wall.radii.device, wall.radii.dtype
        lengths = wall.radii * wall.sweeps
        counts = [max(2, 2 * math.ceil(float(v) * angular / float(lengths.sum()) / 2)) for v in lengths]
        self.counts, self.na, self.nr = counts, sum(counts), radial
        self.arc = torch.cat([torch.full((n,), i, device=self.device, dtype=torch.long) for i, n in enumerate(counts)])
        self.left = torch.cat([torch.arange(n, device=self.device, dtype=self.dtype) / n for n in counts])
        self.ds = torch.cat([torch.full((n,), 1/n, device=self.device, dtype=self.dtype) for n in counts])
        self.levels = torch.linspace(0, 1, radial+1, device=self.device, dtype=self.dtype).pow(1.5)
        a = torch.arange(self.na, device=self.device)
        i = torch.arange(radial, device=self.device)[:, None]
        nxt = (a+1) % self.na
        self.conn = torch.stack((i*self.na+a, i*self.na+nxt, (i+1)*self.na+nxt, (i+1)*self.na+a), -1).reshape(-1, 4)
        self.nodes = self.mapping(self.arc.repeat(radial+1), self.left.repeat(radial+1), self.levels.repeat_interleave(self.na))[0]
        self.nnode = len(self.nodes)
        self.free = torch.arange(self.na*radial, device=self.device)
        self.volume, self.boundary = self.quadratures(quadrature)
        v = self.volume
        support = torch.zeros(self.nnode, device=self.device, dtype=self.dtype)
        norm = torch.zeros_like(support)
        support.index_add_(0, v.conn.flatten(), v.weight[:, None].expand(-1, 4).flatten())
        norm.index_add_(0, v.conn.flatten(), (v.weight[:, None]*v.grad.square().sum(-1)).flatten())
        self.test_scale = torch.sqrt(support*norm).clamp_min(1e-14)

    def mapping(self, arc, s, t):
        w, normal = self.wall.sample(arc, s)
        angle = self.wall.starts[arc] + self.wall.sweeps[arc]*s
        ws = self.wall.radii[arc, None]*self.wall.sweeps[arc, None]*torch.stack((-angle.sin(), angle.cos()), -1)
        r = w.norm(dim=-1)
        logr = torch.log(self.radius/r)
        scale = torch.exp(t*logr)
        x = scale[:, None]*w
        xs = scale[:, None]*(ws - t[:, None]*w*(w*ws).sum(-1)[:, None]/r.square()[:, None])
        xt = x*logr[:, None]
        return x, torch.stack((xs, xt), -1), normal

    def quadratures(self, order):
        g, gw = gauss(order, self.device)
        xi, eta = torch.meshgrid(g, g, indexing="ij")
        wg = (gw[:, None]*gw[None, :]).flatten()
        xi, eta = xi.flatten(), eta.flatten()
        n = torch.stack(((1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta), -1)
        dn = torch.stack((torch.stack((-(1-eta), -(1-xi)), -1), torch.stack((1-eta, -xi), -1),
                          torch.stack((eta, xi), -1), torch.stack((-eta, 1-xi), -1)), -2)
        ne, nq = len(self.conn), len(xi)
        arc = self.arc.repeat(self.nr)
        s0, ds = self.left.repeat(self.nr), self.ds.repeat(self.nr)
        t0 = self.levels[:-1].repeat_interleave(self.na)
        dt = self.levels.diff().repeat_interleave(self.na)
        x, jac, _ = self.mapping(arc.repeat_interleave(nq), (s0[:, None]+ds[:, None]*xi).flatten(),
                                 (t0[:, None]+dt[:, None]*eta).flatten())
        jac = jac*torch.stack((ds, dt), -1).repeat_interleave(nq, 0)[:, None, :]
        grad = dn.repeat(ne, 1, 1) @ torch.linalg.inv(jac)
        weight = torch.linalg.det(jac).abs()*wg.repeat(ne)
        volume = Quadrature(x, weight, n.repeat(ne, 1), grad, self.conn.repeat_interleave(nq, 0))
        # Only the actual wall has a natural load; outer v=0 by DOF elimination.
        arcw = self.arc.repeat_interleave(order)
        sw = (self.left[:, None]+self.ds[:, None]*g).flatten()
        xw, jw, nw = self.mapping(arcw, sw, torch.zeros_like(sw))
        nsh = torch.stack((1-g, g, g*0, g*0), -1).repeat(self.na, 1)
        dnw = torch.stack((torch.stack((-torch.ones_like(g), -(1-g)), -1),
                           torch.stack((torch.ones_like(g), -g), -1), torch.stack((g*0, g), -1),
                           torch.stack((g*0, 1-g), -1)), -2).repeat(self.na, 1, 1)
        jw = jw*torch.stack((self.ds, self.levels[1].expand_as(self.ds)), -1).repeat_interleave(order, 0)[:, None, :]
        gradw = dnw @ torch.linalg.inv(jw)
        weightw = self.wall.radii[arcw]*self.wall.sweeps[arcw]*self.ds.repeat_interleave(order)*gw.repeat(self.na)
        boundary = Quadrature(xw, weightw, nsh, gradw, self.conn[:self.na].repeat_interleave(order, 0), nw)
        return volume, boundary

    def assemble(self, local, conn):
        out = torch.zeros((self.nnode, *local.shape[2:]), device=self.device, dtype=local.dtype)
        return out.index_add(0, conn.flatten(), local.flatten(0, 1))

    def weak(self, piola, fwall, pressure, p0, f0, initial_piola, *, subtract_initial=True):
        v, b = self.volume, self.boundary
        p = piola[:, :2, :2]-initial_piola[:2, :2] if subtract_initial else piola[:, :2, :2]
        internal = self.assemble(torch.einsum("qij,qaj,q->qai", p, v.grad, v.weight), v.conn)
        load = pressure*cofactor_normal(fwall, b.normal)
        if subtract_initial:
            load = load-p0*cofactor_normal(f0.expand_as(fwall), b.normal)
        boundary = self.assemble(b.shape[:, :, None]*load[:, None, :2]*b.weight[:, None, None], b.conn)
        return (internal+boundary)[self.free]/(p0*self.test_scale[self.free, None])


def cofactor_normal(f, n):
    """Plane-strain polynomial, preserving the current F derivative chain."""
    return torch.stack((f[:, 1, 1]*n[:, 0]-f[:, 1, 0]*n[:, 1],
                        -f[:, 0, 1]*n[:, 0]+f[:, 0, 0]*n[:, 1], n[:, 2]*0), -1)


def geometry_from_config(config, root, f0, *, outer=None, mesh=None):
    wall = HorseshoeWall.from_json(root/config["geometry"], device=f0.device, scale=1/f0[0, 0])
    radius = (outer or config["outer_widths"])*13.46/f0[0, 0]
    specification = dict(mesh or config["train_mesh"])
    degree = specification.pop("degree", 1)
    if degree == 2:
        from mixed_q2 import CurvedQ2Mesh
        return CurvedQ2Mesh(wall, radius, **specification)
    if degree != 1:
        raise ValueError("Only Q1 and Q2 field/test spaces are supported")
    return CurvedMesh(wall, radius, **specification)


def wall_geometry_checks(x):
    """Discrete segment intersection audit, not a global injectivity proof."""
    x = x.detach()
    y = x.roll(-1, 0)
    def cross(a, b):
        return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    a, b, c, d = x[:, None], y[:, None], x[None, :], y[None, :]
    hit = (cross(b-a, c-a)*cross(b-a, d-a) < -1e-20) & (cross(d-c, a-c)*cross(d-c, b-c) < -1e-20)
    return {"wall_self_intersections": int(hit.sum().item()//2), "wall_polygon_area_m2": float(.5*cross(x, y).sum()),
            "wall_min_segment_m": float((y-x).norm(dim=-1).min())}


class MultiScaleTests:
    """Compact smooth virtual fields at cavity, intermediate and outer scales.

    Normalized nodal hats alone lose sensitivity to long-wave equilibrium
    errors as their supports shrink. These fields explicitly retain it.
    Validation uses shifted radii, so its virtual fields are not training tests.
    """
    def __init__(self, mesh, length, shift=1.):
        self.mesh = mesh
        self.scales = [min(s*length*shift, float(mesh.radius)*.95) for s in (.75, 1.5, 3., 6.)]
        x = mesh.volume.x.detach().clone().requires_grad_(True)
        self.values, masks = self.values_at(x)
        self.grad = torch.stack([torch.autograd.grad(value.sum(), x, retain_graph=True)[0]
                                 for value in self.values]).detach()
        self.values = self.values.detach()
        self.wall = self.values_at(mesh.boundary.x)[0].detach()
        support = (masks*mesh.volume.weight[None, :]).sum(-1)
        norm = (self.grad.square().sum(-1)*mesh.volume.weight[None, :]).sum(-1)
        self.scale = (support*norm).sqrt().clamp_min(1e-14)

    def values_at(self, x):
        values, masks = [], []
        for scale in self.scales:
            xx, yy = (x/scale).unbind(-1)
            rr = xx.square()+yy.square()
            cut = (1-rr).clamp_min(0).pow(3)
            terms = (torch.ones_like(xx), xx, yy, xx.square()-yy.square(), 2*xx*yy,
                     xx.pow(3), yy.pow(3), xx.square()*yy, xx*yy.square())
            values.extend(cut*term for term in terms)
            masks.extend(rr < 1 for _ in terms)
        return torch.stack(values), torch.stack(masks)

    def residual(self, piola, fwall, pressure, p0, f0, initial_piola, *, subtract_initial=True):
        p = piola[:, :2, :2]-initial_piola[:2, :2] if subtract_initial else piola[:, :2, :2]
        internal = torch.einsum("qij,aqj,q->ai", p, self.grad, self.mesh.volume.weight)
        load = pressure*cofactor_normal(fwall, self.mesh.boundary.normal)
        if subtract_initial:
            load = load-p0*cofactor_normal(f0.expand_as(fwall), self.mesh.boundary.normal)
        external = torch.einsum("qi,aq,q->ai", load[:, :2], self.wall, self.mesh.boundary.weight)
        return (internal+external)/(p0*self.scale[:, None])
