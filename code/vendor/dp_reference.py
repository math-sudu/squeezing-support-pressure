"""Independent CUDA spectral/implicit reference for the same Cauchy-DP model.

No dp_material/current_traction import. Principal elastic logarithms and the
plastic multiplier solve endpoint flow/yield equations. A separate 3-variable
linear solve selects cone versus apex; no closed-form multiplier is used.
This routine is numerical-reference code; global tangents use GPU differences.
"""
from dataclasses import dataclass
import torch


@dataclass
class ReferenceTrial:
    sigma: torch.Tensor
    piola: torch.Tensor
    cp: torch.Tensor
    kappa: torch.Tensor
    dl: torch.Tensor
    branch: torch.Tensor
    yield_value: torch.Tensor
    flow_error: torch.Tensor


def update(f, cp, kappa, m):
    # Bound batched eigensolver workspace without moving arithmetic to CPU.
    if len(f) > 4096:
        pieces = [_update_batch(f[i:i+4096], cp[i:i+4096], kappa[i:i+4096], m)
                  for i in range(0, len(f), 4096)]
        return ReferenceTrial(**{name: torch.cat([getattr(p, name) for p in pieces])
                                 for name in ReferenceTrial.__dataclass_fields__})
    return _update_batch(f, cp, kappa, m)


def _update_batch(f, cp, kappa, m):
    if not (f.is_cuda and f.dtype == torch.float64):
        raise ValueError("Reference arithmetic requires GPU float64")
    j = torch.linalg.det(f)
    if not bool((j > 0).all()):
        raise ValueError("Reference deformation lost orientation")
    btr = f @ torch.linalg.inv(cp) @ f.transpose(-1, -2)
    vals, vec = torch.linalg.eigh(btr)
    if not bool((vals > 0).all()):
        raise ValueError("Invalid elastic trial spectrum")
    etr = .5*vals.log()
    g, bulk, alpha, beta, h = m.shear, m.bulk, m.alpha, m.beta, m.hardening
    tau_tr = 2*g*etr + (bulk-2*g/3)*etr.sum(-1, keepdim=True)
    mean_tr = tau_tr.mean(-1)
    strial = tau_tr - mean_tr[:, None]
    qt = (1.5*strial.square().sum(-1)).sqrt()
    strength = m.k0 + h*kappa
    plastic = (qt+alpha*mean_tr)/j-strength > 0
    e, dl = etr.clone(), torch.zeros_like(j)
    branch = torch.zeros_like(j, dtype=torch.long)
    if bool(plastic.any()):
        ids = torch.where(plastic)[0]
        jp, kp = j[ids], kappa[ids]
        # Independent solve of deviatoric, volumetric and yield equations.
        mat = torch.zeros((len(ids), 3, 3), device=f.device, dtype=f.dtype)
        mat[:, 0, 0], mat[:, 0, 2] = 1, 3*g
        mat[:, 1, 1], mat[:, 1, 2] = 1, bulk*beta
        mat[:, 2, 0], mat[:, 2, 1], mat[:, 2, 2] = 1/jp, alpha/jp, -h
        rhs = torch.stack((qt[ids], mean_tr[ids], strength[ids]), -1)
        candidate = torch.linalg.solve(mat, rhs)
        cone = candidate[:, 0] > 1e-13
        smooth = ids[cone]
        apex = ids[~cone]
        if len(smooth):
            # Four unknowns: three principal elastic logarithms and lambda.
            ee = etr[smooth].clone()
            lam = torch.zeros_like(j[smooth])
            ident = torch.eye(3, device=f.device, dtype=f.dtype)
            devmap = ident-torch.ones_like(ident)/3
            for _ in range(8):
                tr = ee.sum(-1)
                s = 2*g*(ee-tr[:, None]/3)
                q = (1.5*s.square().sum(-1)).sqrt()
                direction = 1.5*s/q[:, None] + beta/3
                flow = ee-etr[smooth]+lam[:, None]*direction
                yf = (q+alpha*bulk*tr)/j[smooth]-m.k0-h*(kappa[smooth]+lam)
                residual = torch.cat((flow, yf[:, None]), -1)
                if bool(residual.abs().max() < 2e-13):
                    break
                dq = 3*g*s/q[:, None]
                da = 3*g*devmap/q[:, None, None]-1.5*s[:, :, None]*dq[:, None, :]/q.square()[:, None, None]
                jac = torch.zeros((len(smooth), 4, 4), device=f.device, dtype=f.dtype)
                jac[:, :3, :3] = ident+lam[:, None, None]*da
                jac[:, :3, 3] = direction
                jac[:, 3, :3] = (dq+alpha*bulk)/j[smooth, None]
                jac[:, 3, 3] = -h
                correction = torch.linalg.solve(jac, residual)
                ee, lam = ee-correction[:, :3], lam-correction[:, 3]
            tr = ee.sum(-1)
            s = 2*g*(ee-tr[:, None]/3)
            q = (1.5*s.square().sum(-1)).sqrt()
            flow = ee-etr[smooth]+lam[:, None]*(1.5*s/q[:, None]+beta/3)
            yf = (q+alpha*bulk*tr)/j[smooth]-m.k0-h*(kappa[smooth]+lam)
            if not bool((flow.abs().amax(-1) < 1e-11).all() and (yf.abs() < 1e-11).all() and (lam >= -1e-13).all()):
                raise ValueError("Independent principal flow/yield equations did not converge")
            e[smooth], dl[smooth], branch[smooth] = ee, lam, 1
        if len(apex):
            if beta <= 0:
                raise ValueError("Degenerate zero-dilatancy apex")
            mat = torch.zeros((len(apex), 2, 2), device=f.device, dtype=f.dtype)
            mat[:, 0, 0], mat[:, 0, 1] = 1, beta
            mat[:, 1, 0], mat[:, 1, 1] = alpha*bulk/j[apex], -h
            sol = torch.linalg.solve(mat, torch.stack((etr[apex].sum(-1), strength[apex]), -1))
            e[apex], dl[apex], branch[apex] = sol[:, :1]/3, sol[:, 1], 2
    tau_principal = 2*g*e + (bulk-2*g/3)*e.sum(-1, keepdim=True)
    tau = (vec*tau_principal[:, None, :]) @ vec.transpose(-1, -2)
    sigma = tau/j[:, None, None]
    invbe = (vec*torch.exp(-2*e)[:, None, :]) @ vec.transpose(-1, -2)
    cpp = f.transpose(-1, -2) @ invbe @ f
    cpnew = torch.where(plastic[:, None, None], .5*(cpp+cpp.transpose(-1, -2)), cp)
    mean = tau_principal.mean(-1)
    deviator = tau_principal-mean[:, None]
    q = (1.5*deviator.square().sum(-1)).sqrt()
    yf = (q+alpha*mean)/j-m.k0-h*(kappa+dl)
    err = (e.sum(-1)-etr.sum(-1)+beta*dl).abs()
    piola = tau @ torch.linalg.inv(f).transpose(-1, -2)
    return ReferenceTrial(sigma, piola, cpnew, kappa+dl, dl, branch, yf, err)
