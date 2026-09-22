"""Independent finite-strain Cauchy-DP material for 3D plane strain.

Energy is per unit stress-free reference volume. The update is an elastic
predictor followed by an exponential corrector at fixed total F, not a
backward-Euler update of b_e. History is immutable by convention; only
commit_state detaches accepted history. No legacy material/solver is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def sym(a):
    return 0.5 * (a + a.transpose(-1, -2))


def trace(a):
    return a.diagonal(dim1=-2, dim2=-1).sum(-1)


def dev(a):
    return a - trace(a)[..., None, None] * torch.eye(3, device=a.device, dtype=a.dtype) / 3


def plane_tensor(a, zz=1.0):
    """Embed (...,2,2) tensors, retaining the prescribed 33 component."""
    z = torch.zeros_like(a[..., 0, 0])
    c = torch.as_tensor(zz, device=a.device, dtype=a.dtype) + z
    return torch.stack((torch.stack((a[..., 0, 0], a[..., 0, 1], z), -1),
                        torch.stack((a[..., 1, 0], a[..., 1, 1], z), -1),
                        torch.stack((z, z, c), -1)), -2)


def require(condition, message):
    """Validate on the tensor's device; transfer only one Boolean to the host."""
    if not bool(torch.all(condition)):
        raise ValueError(message)


def check_block(a, name, *, symmetric=False, positive=False):
    if a.shape[-2:] != (3, 3) or a.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must be a float32/float64 (...,3,3) tensor")
    require(torch.isfinite(a), f"{name} contains nonfinite values")
    # Zero transverse shear is an input contract, not a silent truncation.
    require((a[..., :2, 2] == 0) & (a[..., 2, :2] == 0),
            f"{name} must have zero 13/23/31/32 components")
    if symmetric:
        tol = 32 * torch.finfo(a.dtype).eps * a.abs().amax(dim=(-2, -1))
        require((a - a.transpose(-1, -2)).abs().amax(dim=(-2, -1)) <= tol,
                f"{name} must be symmetric")
    if positive:
        det2 = a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
        require((a[..., 0, 0] > 0) & (det2 > 0) & (a[..., 2, 2] > 0),
                f"{name} must be positive definite")


def check_deformation(f):
    check_block(f, "F")
    require(f[..., 2, 2] == 1, "Total F33 must equal one in plane strain")
    j = torch.linalg.det(f)
    require(j > 0, "F must preserve orientation before inverse/log evaluation")
    return j


def log_spd_plane(b):
    """Matrix logarithm of a plane block, with a smooth repeated-root limit.

    log(B2) = log(det(B2))/2 I + atanh(sqrt(z))/(m sqrt(z)) dev2(B2).
    A 12-term analytic series at z <= .01 avoids eigenvector derivatives.
    Its first omitted term is at most .01**12/25 in the scalar coefficient.
    The 33 eigenvalue is independent, so crossings with it need no branch.
    Input SPD/domain validation belongs to the calling material interface.
    """
    b = sym(b)
    m = (b[..., 0, 0] + b[..., 1, 1]) / 2
    h = (b[..., 0, 0] - b[..., 1, 1]) / 2
    z = (h.square() + b[..., 0, 1].square()) / m.square()
    small = z <= 0.01
    zs = torch.where(small, z, torch.zeros_like(z))
    series = torch.ones_like(z)
    power = torch.ones_like(z)
    for k in range(1, 12):
        power = power * zs
        series = series + power / (2 * k + 1)
    root = torch.sqrt(torch.where(small, torch.full_like(z, 0.25), z))
    closed = torch.atanh(root) / root
    coef = torch.where(small, series, closed) / m
    logdet = (torch.log(b[..., 0, 0])
              + torch.log(b[..., 1, 1] - b[..., 0, 1].square() / b[..., 0, 0]))
    a = torch.stack((torch.stack((0.5 * logdet + coef * h, coef * b[..., 0, 1]), -1),
                     torch.stack((coef * b[..., 0, 1], 0.5 * logdet - coef * h), -1)), -2)
    return plane_tensor(a, torch.log(b[..., 2, 2]))


def exp_sym_plane(a):
    """Symmetric plane-block exponential, including the repeated-root limit.

    exp(A2)=exp(m)[cosh(r) I + sinh(r)/r dev2(A2)]. The local power series
    differentiates the same analytic matrix function; no spectral vectors or
    host numerical calculation enter the forward or backward evaluation.
    """
    a = sym(a)
    m = (a[..., 0, 0] + a[..., 1, 1]) / 2
    h = (a[..., 0, 0] - a[..., 1, 1]) / 2
    r2 = h.square() + a[..., 0, 1].square()
    small = r2 <= .01
    z = torch.where(small, r2, torch.zeros_like(r2))
    cosh_series, sinhc_series = torch.ones_like(z), torch.ones_like(z)
    cterm, sterm = torch.ones_like(z), torch.ones_like(z)
    for k in range(1, 8):
        cterm = cterm * z / ((2 * k - 1) * (2 * k))
        sterm = sterm * z / ((2 * k) * (2 * k + 1))
        cosh_series, sinhc_series = cosh_series + cterm, sinhc_series + sterm
    r = torch.sqrt(torch.where(small, torch.ones_like(r2), r2))
    c = torch.where(small, cosh_series, torch.cosh(r))
    s = torch.where(small, sinhc_series, torch.sinh(r) / r)
    scale = m.exp()
    block = torch.stack((torch.stack((scale * (c + s * h), scale * s * a[..., 0, 1]), -1),
                         torch.stack((scale * s * a[..., 0, 1], scale * (c - s * h)), -1)), -2)
    return plane_tensor(block, a[..., 2, 2].exp())


def invariants(stress):
    s = dev(stress)
    q2 = 1.5 * s.square().sum(dim=(-2, -1))
    nonzero = q2 > 0
    q = torch.where(nonzero, torch.sqrt(torch.where(nonzero, q2, torch.ones_like(q2))),
                    torch.zeros_like(q2))
    return -trace(stress) / 3, q


@dataclass(frozen=True)
class DPParameters:
    shear: float
    bulk: float
    alpha: float
    beta: float
    k0: float
    hardening: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Material parameters must be finite")
        if not (self.shear > 0 and self.bulk > 0 and self.k0 > 0
                and self.hardening >= 0 and 0 <= self.beta <= self.alpha):
            raise ValueError("Need G,K,k0>0, H>=0 and 0<=beta<=alpha")

    @classmethod
    def from_young_poisson(cls, young, poisson, alpha, beta, k0, hardening):
        if not (-1 < poisson < 0.5):
            raise ValueError("Poisson ratio must lie in (-1, .5)")
        return cls(young / (2 * (1 + poisson)), young / (3 * (1 - 2 * poisson)),
                   alpha, beta, k0, hardening)

    def strength(self, kappa):
        return self.k0 + self.hardening * kappa


@dataclass(frozen=True)
class MaterialState:
    cp: torch.Tensor
    kappa: torch.Tensor
    step: int = 0

    @classmethod
    def virgin(cls, batch_shape=(), *, device, dtype=torch.float64):
        return cls(torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, 3, 3).clone(),
                   torch.zeros(batch_shape, device=device, dtype=dtype))


@dataclass(frozen=True)
class MaterialTrial:
    source: MaterialState
    cp: torch.Tensor
    kappa: torch.Tensor
    sigma: torch.Tensor
    tau: torch.Tensor
    piola: torch.Tensor
    elastic_log: torch.Tensor
    trial_log: torch.Tensor
    delta_lambda: torch.Tensor
    branch: torch.Tensor  # 0: elastic, 1: smooth cone, 2: apex
    yield_value: torch.Tensor
    energy: torch.Tensor


def hencky_response(e, material):
    tau = 2 * material.shear * dev(e) + material.bulk * trace(e)[..., None, None] * torch.eye(
        3, device=e.device, dtype=e.dtype)
    energy = (material.shear * dev(e).square().sum(dim=(-2, -1))
              + 0.5 * material.bulk * trace(e).square())
    return tau, energy


def cauchy_to_piola(f, sigma):
    """P = J sigma F^{-T}; P itself is generally nonsymmetric."""
    j = check_deformation(f)
    return j[..., None, None] * torch.linalg.solve(f, sigma.transpose(-1, -2)).transpose(-1, -2)


def evaluate(f, state, material):
    """Pure batched trial evaluation from one fixed, committed history.

    All material tensors must share shape, device and dtype. First and higher
    derivatives within a smooth branch use native PyTorch operations. At a
    switch AD selects the active branch; it is not a global smooth tangent.
    beta=0 DP is supported only away from its degenerate tensile apex. The
    alpha=beta=0 J2 limit has its own smooth return and no apex denominator.
    """
    j = check_deformation(f)
    check_block(state.cp, "Cp", symmetric=True, positive=True)
    if (state.cp.shape != f.shape or state.kappa.shape != f.shape[:-2]
            or state.cp.device != f.device or state.kappa.device != f.device
            or state.cp.dtype != f.dtype or state.kappa.dtype != f.dtype):
        raise ValueError("F, Cp and kappa must have matching batch shape/device/dtype")
    if state.cp.requires_grad or state.kappa.requires_grad:
        raise ValueError("Old history must be committed (fixed during this load step)")
    require(torch.isfinite(state.kappa) & (state.kappa >= 0), "Invalid accumulated multiplier")
    btr = sym(f @ torch.linalg.solve(state.cp, f.transpose(-1, -2)))
    check_block(btr, "Elastic trial metric", symmetric=True, positive=True)
    etr = 0.5 * log_spd_plane(btr)
    ttr, _ = hencky_response(etr, material)
    ptr, qtr = invariants(ttr)
    ftr = qtr - material.alpha * ptr - j * material.strength(state.kappa)
    plastic = ftr > 0
    ds = ftr / (3 * material.shear + material.alpha * material.bulk * material.beta
                + j * material.hardening)
    if material.alpha == 0:
        apex = torch.zeros_like(plastic)
        da = torch.zeros_like(ds)
    else:
        apex = plastic & (ds >= qtr / (3 * material.shear))
        if material.beta == 0:
            require(~apex, "Zero-dilatancy DP tensile apex is outside this implementation's domain")
            da = torch.zeros_like(ds)
        else:
            da = (-material.alpha * ptr - j * material.strength(state.kappa)) / (
                material.alpha * material.bulk * material.beta + j * material.hardening)
    cone = plastic & ~apex
    dl = torch.where(plastic, torch.where(apex, da, ds), torch.zeros_like(ds))
    safe_q = torch.where(cone, qtr, torch.ones_like(qtr))
    factor = torch.where(cone, 1 - 3 * material.shear * dl / safe_q,
                         torch.where(apex, torch.zeros_like(dl), torch.ones_like(dl)))
    eye = torch.eye(3, dtype=f.dtype, device=f.device)
    enew = factor[..., None, None] * dev(etr) + (
        (trace(etr) - material.beta * dl) / 3)[..., None, None] * eye
    tau, energy = hencky_response(enew, material)
    # Computing exp(-2e) directly avoids inverting a freshly exponentiated SPD tensor.
    cp_candidate = sym(f.transpose(-1, -2) @ exp_sym_plane(-2 * enew) @ f)
    cp = torch.where(plastic[..., None, None], cp_candidate, state.cp)
    kappa = state.kappa + dl
    sigma = tau / j[..., None, None]
    p, q = invariants(sigma)
    check_block(cp, "Updated Cp", symmetric=True, positive=True)
    require(torch.isfinite(sigma), "Nonfinite material stress")
    return MaterialTrial(state, cp, kappa, sigma, tau, cauchy_to_piola(f, sigma),
                         enew, etr, dl, cone.to(torch.int64) + 2 * apex.to(torch.int64),
                         q - material.alpha * p - material.strength(kappa), energy)


def commit_state(state, trial):
    """Accept a trial once against its source; reject a stale trial on a new state.

    This creates an owned copy and breaks the graph only at acceptance. Calling
    evaluate repeatedly never changes state. Reaccepting against the same old
    object is idempotent in value and cannot accumulate a second increment.
    The caller must perform load-step equilibrium/consistency checks first.
    """
    if trial.source is not state:
        raise ValueError("Stale trial: commit requires the exact source state")
    return MaterialState(trial.cp.detach().clone(), trial.kappa.detach().clone(), state.step + 1)


def elastic_precompression(material, pressure):
    """Return F0=diag(a,a,1) from given positive in-plane Cauchy pressure.

    pressure is a device tensor. The scalar root is bracketed on the GPU in
    b=-log(a), solving C*b*exp(2*b)=p0. Initial data are fixed, not trainable.
    A strictly elastic initial state is checked, including sigma33.
    """
    if pressure.requires_grad:
        raise ValueError("Precompression is fixed initial data")
    require(torch.isfinite(pressure) & (pressure > 0), "Need finite positive p0")
    c = 2 * material.bulk + 2 * material.shear / 3
    lo, hi = torch.zeros_like(pressure), pressure / c
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        # Logarithmic residual avoids overflow for large initial pressures.
        below = torch.log(mid) + 2 * mid < torch.log(pressure / c)
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    a = torch.exp(-0.5 * (lo + hi))
    f0 = plane_tensor(torch.diag_embed(torch.stack((a, a), -1)))
    state = MaterialState.virgin(pressure.shape, device=pressure.device, dtype=pressure.dtype)
    initial = evaluate(f0, state, material)
    require((initial.branch == 0) & (initial.yield_value < 0),
            "Requested precompression is not strictly elastic")
    return f0, state, initial
