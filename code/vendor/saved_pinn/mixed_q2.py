"""Quadratic displacement reference on the exact same curved geometry.

Q1 remains the PINN's local test space. Q2 adds field resolution for the
independent FEM reference without changing the material or loading model.
"""
import math
import torch
from .mixed_geometry import CurvedMesh, Quadrature, gauss


def shape_q2(x, y):
    def line(z):
        return ((1-z)*(1-2*z), 4*z*(1-z), z*(2*z-1)), (4*z-3, 4-8*z, 4*z-1)
    lx, dx = line(x)
    ly, dy = line(y)
    n = torch.stack([ly[j]*lx[i] for j in range(3) for i in range(3)], -1)
    dn = torch.stack([torch.stack((ly[j]*dx[i], dy[j]*lx[i]), -1) for j in range(3) for i in range(3)], -2)
    return n, dn


class CurvedQ2Mesh(CurvedMesh):
    degree = 2

    def __init__(self, wall, radius, angular=64, radial=16, quadrature=4, radial_exponent=1.5):
        self.wall, self.radius = wall, radius
        self.device, self.dtype = wall.radii.device, wall.radii.dtype
        lengths = wall.radii*wall.sweeps
        self.counts = [max(2, 2*math.ceil(float(v)*angular/float(lengths.sum())/2)) for v in lengths]
        self.na, self.nr = sum(self.counts), radial
        self.arc = torch.cat([torch.full((n,), i, device=self.device, dtype=torch.long) for i, n in enumerate(self.counts)])
        self.left = torch.cat([torch.arange(n, device=self.device, dtype=self.dtype)/n for n in self.counts])
        self.ds = torch.cat([torch.full((n,), 1/n, device=self.device, dtype=self.dtype) for n in self.counts])
        if radial_exponent <= 0:
            raise ValueError("Radial grading exponent must be positive")
        self.levels = torch.linspace(0, 1, radial+1, device=self.device, dtype=self.dtype).pow(radial_exponent)
        angular_nodes = torch.stack((self.left, self.left+self.ds/2), -1).flatten()
        radial_nodes = torch.cat((torch.stack((self.levels[:-1], (self.levels[:-1]+self.levels[1:])/2), -1).flatten(), self.levels[-1:]))
        self.wall_node_count = 2*self.na
        self.nodes = self.mapping(self.arc.repeat_interleave(2).repeat(2*radial+1),
                                  angular_nodes.repeat(2*radial+1), radial_nodes.repeat_interleave(2*self.na))[0]
        a = torch.arange(self.na, device=self.device)[None, :]
        r = torch.arange(radial, device=self.device)[:, None]
        self.conn = torch.stack([(2*r+j)*self.wall_node_count+(2*a+i) % self.wall_node_count
                                  for j in range(3) for i in range(3)], -1).reshape(-1, 9)
        self.nnode = len(self.nodes)
        self.free = torch.arange(self.nnode-self.wall_node_count, device=self.device)
        self.volume, self.boundary = self.quadratures(quadrature)
        v = self.volume
        support = torch.zeros(self.nnode, device=self.device, dtype=self.dtype)
        norm = torch.zeros_like(support)
        support.index_add_(0, v.conn.flatten(), v.weight[:, None].expand(-1, 9).flatten())
        norm.index_add_(0, v.conn.flatten(), (v.weight[:, None]*v.grad.square().sum(-1)).flatten())
        self.test_scale = (support*norm).sqrt().clamp_min(1e-14)

    def quadratures(self, order):
        g, w = gauss(order, self.device)
        xi, eta = torch.meshgrid(g, g, indexing="ij")
        weights = (w[:, None]*w[None, :]).flatten()
        xi, eta = xi.flatten(), eta.flatten()
        n, dn = shape_q2(xi, eta)
        ne, nq = len(self.conn), len(xi)
        ds, s0 = self.ds.repeat(self.nr), self.left.repeat(self.nr)
        dt = self.levels.diff().repeat_interleave(self.na)
        t0 = self.levels[:-1].repeat_interleave(self.na)
        x, jac, _ = self.mapping(self.arc.repeat(self.nr).repeat_interleave(nq),
            (s0[:, None]+ds[:, None]*xi).flatten(), (t0[:, None]+dt[:, None]*eta).flatten())
        jac = jac*torch.stack((ds, dt), -1).repeat_interleave(nq, 0)[:, None, :]
        volume = Quadrature(x, torch.linalg.det(jac).abs()*weights.repeat(ne), n.repeat(ne, 1),
                             dn.repeat(ne, 1, 1) @ torch.linalg.inv(jac), self.conn.repeat_interleave(nq, 0))
        sw = (self.left[:, None]+self.ds[:, None]*g).flatten()
        arc = self.arc.repeat_interleave(order)
        xw, jw, nw = self.mapping(arc, sw, torch.zeros_like(sw))
        jw = jw*torch.stack((self.ds, self.levels[1].expand_as(self.ds)), -1).repeat_interleave(order, 0)[:, None, :]
        nsh, dnw = shape_q2(g, torch.zeros_like(g))
        boundary = Quadrature(xw, self.wall.radii[arc]*self.wall.sweeps[arc]*self.ds.repeat_interleave(order)*w.repeat(self.na),
            nsh.repeat(self.na, 1), dnw.repeat(self.na, 1, 1) @ torch.linalg.inv(jw),
            self.conn[:self.na].repeat_interleave(order, 0), nw)
        return volume, boundary
