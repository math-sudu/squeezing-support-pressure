"""GPU geometry for the existing six-arc convex horseshoe, independent of FEM."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from .dp_material import require


@dataclass(frozen=True)
class WallProjection:
    point: torch.Tensor
    normal: torch.Tensor  # Outward from rock: into the cavity.
    distance: torch.Tensor  # Positive into rock, in the natural reference.
    arc_index: torch.Tensor


class HorseshoeWall:
    """Closed CCW C1 convex arc boundary; curvature is piecewise continuous.

    Projection is onto the exterior of the cavity. No nearest-wall coordinates
    are assigned to its interior. Derivatives at arc joins are one-sided.
    scale=1/a converts the documented precompressed contour to natural reference.
    """

    def __init__(self, centers, radii, starts, sweeps):
        self.centers, self.radii = centers, radii
        self.starts, self.sweeps = starts, sweeps
        require((radii > 0) & (sweeps > 0) & (sweeps <= torch.pi), "Invalid convex arcs")
        require((sweeps.sum() - 2 * torch.pi).abs() < 1e-12, "Need one complete tangent rotation")
        start, start_n = self.sample(torch.arange(len(radii), device=radii.device),
                                     torch.zeros_like(radii))
        end, end_n = self.sample(torch.arange(len(radii), device=radii.device),
                                 torch.ones_like(radii))
        require(torch.linalg.vector_norm(end - start.roll(-1, 0), dim=-1) < 1e-10,
                "Arc contour must close")
        require(torch.linalg.vector_norm(end_n - start_n.roll(-1, 0), dim=-1) < 1e-12,
                "Arc contour must have continuous tangents")

    @classmethod
    def from_json(cls, path, *, device, dtype=torch.float64, scale=1.0):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        boundary = next(b for b in data["boundaries"] if b["role"] == "initial_support_outer")
        arcs = boundary["circular_arcs"]
        def tensor(values):
            return torch.tensor(values, device=device, dtype=dtype)
        factor = torch.as_tensor(scale, device=device, dtype=dtype)
        require(torch.isfinite(factor) & (factor > 0), "Geometry scale must be positive")
        return cls(tensor([a["center"] for a in arcs]) * factor,
                   tensor([a["radius"] for a in arcs]) * factor,
                   tensor([a["start_rad"] for a in arcs]), tensor([a["sweep_rad"] for a in arcs]))

    def sample(self, arc_index, fraction):
        angle = self.starts[arc_index] + self.sweeps[arc_index] * fraction
        radial = torch.stack((torch.cos(angle), torch.sin(angle)), -1)
        point = self.centers[arc_index] + self.radii[arc_index, None] * radial
        normal = torch.cat((-radial, torch.zeros_like(radial[..., :1])), -1)
        return point, normal

    def project(self, points):
        """Analytic finite-arc projection, batched on the input device.

        The signed normal distance has the correct one-sided derivative at
        d=0; a norm of (X-pi) would give an artificial zero derivative there.
        """
        if points.shape[-1] != 2 or points.device != self.radii.device or points.dtype != self.radii.dtype:
            raise ValueError("Points must match the contour device/dtype and have two coordinates")
        v = points[..., None, :] - self.centers
        angle = torch.atan2(v[..., 1], v[..., 0])
        rel = torch.remainder(angle - self.starts, 2 * torch.pi)
        on_arc = rel <= self.sweeps
        radial = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(torch.finfo(v.dtype).tiny)
        radial_point = self.centers + self.radii[:, None] * radial
        ids = torch.arange(len(self.radii), device=points.device)
        first, _ = self.sample(ids, torch.zeros_like(self.radii))
        last, _ = self.sample(ids, torch.ones_like(self.radii))
        choose_first = (points[..., None, :] - first).square().sum(-1) <= (
            points[..., None, :] - last).square().sum(-1)
        endpoint = torch.where(choose_first[..., None], first, last)
        candidates = torch.where(on_arc[..., None], radial_point, endpoint)
        best = (points[..., None, :] - candidates).square().sum(-1).argmin(-1)
        nearest = torch.gather(candidates, -2, best[..., None, None].expand(*best.shape, 1, 2)).squeeze(-2)
        outward = (nearest - self.centers[best]) / self.radii[best, None]
        gap = points - nearest
        distance = (gap * outward).sum(-1)
        tol = 1e-10 * self.radii.max()
        require(distance >= -tol, "Nearest-wall coordinates are restricted to the rock exterior")
        require(torch.linalg.vector_norm(gap - distance[..., None] * outward, dim=-1) < tol,
                "Projection is not on a unique smooth exterior normal ray")
        normal = torch.cat((-outward, torch.zeros_like(outward[..., :1])), -1)
        # Roundoff-sized negative values at an exact wall point only.
        distance = torch.where(distance < 0, distance - distance.detach(), distance)
        return WallProjection(nearest, normal, distance, best)
