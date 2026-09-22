"""Current-normal stress heads with a shared Cartesian traction projection."""

import torch

from .dp_material import check_block, check_deformation, plane_tensor, require


def stress_from_four(a):
    """Four tensor components (11,12,22,33), without engineering shear factors."""
    if a.shape[-1] != 4:
        raise ValueError("A plane-strain stress head must have four components")
    return plane_tensor(torch.stack((torch.stack((a[..., 0], a[..., 1]), -1),
                                     torch.stack((a[..., 1], a[..., 2]), -1)), -2), a[..., 3])


def current_frame(f, normal):
    """Return current unit n, Q=[n,e3 cross n,e3], and Nanson area factor."""
    j = check_deformation(f)
    if normal.shape != f.shape[:-1] or normal.device != f.device or normal.dtype != f.dtype:
        raise ValueError("Reference normals must match F batch/device/dtype")
    tol = 64 * torch.finfo(f.dtype).eps
    require(torch.isfinite(normal), "Nonfinite reference normal")
    require((normal.square().sum(-1) - 1).abs() <= tol, "Reference normals must be unit length")
    require(normal[..., 2] == 0, "Wall normals must be in-plane")
    pushed = torch.linalg.solve(f.transpose(-1, -2), normal[..., None]).squeeze(-1)
    length = torch.linalg.vector_norm(pushed, dim=-1)
    n = pushed / length[..., None]
    z = torch.zeros_like(n[..., 0])
    t = torch.stack((-n[..., 1], n[..., 0], z), -1)
    e3 = torch.stack((z, z, torch.ones_like(z)), -1)
    return n, torch.stack((n, t, e3), -1), j * length


def traction_projection(b, n, pressure):
    """Frobenius orthogonal projection of symmetric B onto sigma*n=-p*n."""
    check_block(b, "Free stress", symmetric=True)
    require((n.square().sum(-1) - 1).abs() < 64 * torch.finfo(b.dtype).eps,
            "Projection needs unit normals")
    nn = n[..., :, None] * n[..., None, :]
    tangent = torch.eye(3, device=b.device, dtype=b.dtype) - nn
    p = torch.as_tensor(pressure, device=b.device, dtype=b.dtype)
    return tangent @ b @ tangent - p[..., None, None] * nn


def blended_stress(f, wall_head, outer_head, distance, reference_normal, pressure, delta,
                   *, basis="current", hard=True, initial_f=None, constitutive_increment=None):
    """Build current (L), Cartesian (G), or initial-basis (R) free fields.

    Head shapes are (...,4), distance (...), and N (...,3). Only d<delta
    entries use wall heads, normals or F0; outside these may be NaN sentinels.
    The hard projection ALWAYS uses the current F, including basis='initial'.
    chi is C1 at both ends; eta is C1 at the exterior and has nonzero wall
    slope. Reference distance/projection data can be cached at fixed points.
    An optional Cartesian Cauchy increment is added with chi before projection.
    """
    if not delta > 0 or basis not in ("current", "cartesian", "initial"):
        raise ValueError("Need delta>0 and basis=current/cartesian/initial")
    shape = f.shape[:-2]
    if (wall_head.shape != (*shape, 4) or outer_head.shape != (*shape, 4)
            or distance.shape != shape or reference_normal.shape != (*shape, 3)):
        raise ValueError("Incompatible stress-layer batch shapes")
    for a in (wall_head, outer_head, distance, reference_normal):
        if a.device != f.device or a.dtype != f.dtype:
            raise ValueError("Stress-layer inputs must share device and dtype")
    if constitutive_increment is not None:
        if (constitutive_increment.shape != f.shape or constitutive_increment.device != f.device
                or constitutive_increment.dtype != f.dtype):
            raise ValueError("Constitutive increment must match F shape/device/dtype")
    require(torch.isfinite(distance) & (distance >= 0), "Distances must be finite and nonnegative")
    outer = stress_from_four(outer_head)
    check_block(outer, "Outer stress", symmetric=True)
    active = distance < delta
    out = outer.clone()
    # Gather first: undefined exterior normals are never inverted/normalized.
    if bool(active.any()):
        fa, na = f[active], reference_normal[active]
        aw = stress_from_four(wall_head[active])
        check_block(aw, "Wall stress", symmetric=True)
        if basis == "current" or hard:
            n, q, _ = current_frame(fa, na)
        if basis == "current":
            aw = q @ aw @ q.transpose(-1, -2)
        elif basis == "initial":
            if initial_f is None or initial_f.shape != f.shape:
                raise ValueError("Initial-basis mode requires matching fixed F0")
            _, q0, _ = current_frame(initial_f[active], na)
            aw = q0 @ aw @ q0.transpose(-1, -2)
        r = distance[active] / delta
        chi = 1 - 3 * r.square() + 2 * r.pow(3)
        b = (1 - chi)[..., None, None] * outer[active] + chi[..., None, None] * aw
        if constitutive_increment is not None:
            increment = constitutive_increment[active]
            check_block(increment, "Constitutive Cauchy increment", symmetric=True)
            b = b + chi[..., None, None] * increment
        if hard:
            p = torch.as_tensor(pressure, device=f.device, dtype=f.dtype).expand(shape)[active]
            eta = (1 - r).square()
            b = (1 - eta)[..., None, None] * b + eta[..., None, None] * traction_projection(b, n, p)
        out[active] = b
    return out
