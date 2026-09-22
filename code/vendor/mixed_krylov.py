"""GPU sparse BiCGSTAB for the unsymmetric reference Newton equations.

No CPU matrix or parallel worker is used. Convergence always checks the true
assembled residual; a failed iterative solve cannot become a physical step.
"""
import torch


class SparseTangent:
    def __init__(self, rows, columns, values, size):
        if not (values.is_cuda and values.dtype == torch.float64):
            raise ValueError("Sparse reference algebra requires CUDA float64")
        self.matrix = torch.sparse_coo_tensor(torch.stack((rows, columns)), values,
            (size, size), device=values.device, dtype=values.dtype).coalesce().to_sparse_csr()
        diagonal = rows == columns
        self.diagonal = torch.zeros(size, device=values.device, dtype=values.dtype)
        self.diagonal.index_add_(0, rows[diagonal], values[diagonal])
        if not bool((self.diagonal.abs() > 1e-15).all()):
            raise ValueError("Singular Jacobi preconditioner in reference tangent")
        self.shape = (size, size)

    def __matmul__(self, vector):
        return torch.sparse.mm(self.matrix, vector[:, None]).squeeze(-1)


def bicgstab(operator, rhs, *, rtol=1e-10, maxiter=8000):
    if not (rhs.is_cuda and rhs.dtype == torch.float64):
        raise ValueError("Krylov vectors must stay on GPU in float64")
    norm = rhs.norm()
    if float(norm) == 0:
        return torch.zeros_like(rhs), {"iterations": 0, "relative_residual": 0.}
    # Normalize the linear problem before scalar breakdown tests. Late Newton
    # right-hand sides can be tiny without the tangent being singular.
    original_norm = norm
    rhs = rhs/norm
    norm = torch.ones_like(norm)
    target = rtol*norm
    x = torch.zeros_like(rhs)
    r, shadow = rhs.clone(), rhs.clone()
    p, v = torch.zeros_like(rhs), torch.zeros_like(rhs)
    previous, alpha, omega = (torch.ones((), device=rhs.device, dtype=rhs.dtype) for _ in range(3))
    restarts = 0
    for iteration in range(1, maxiter+1):
        rho = shadow @ r
        if float(rho.abs()) < 1e-40 or float(omega.abs()) < 1e-40:
            r = rhs-operator @ x
            shadow, p, v = r.clone(), torch.zeros_like(r), torch.zeros_like(r)
            previous, alpha, omega = (torch.ones_like(rho) for _ in range(3))
            rho = shadow @ r
            restarts += 1
        beta = (rho/previous)*(alpha/omega)
        p = r+beta*(p-omega*v)
        phat = p/operator.diagonal
        v = operator @ phat
        denominator = shadow @ v
        if float(denominator.abs()) < 1e-40:
            raise RuntimeError("BiCGSTAB shadow-product breakdown")
        alpha = rho/denominator
        s = r-alpha*v
        if float(s.norm()) <= float(target):
            x = x+alpha*phat
            true = rhs-operator @ x
            if float(true.norm()) <= float(target):
                return x*original_norm, {"iterations": iteration, "relative_residual": float(true.norm()/norm), "restarts": restarts}
            r, shadow = true, true.clone()
            p, v = torch.zeros_like(r), torch.zeros_like(r)
            previous, alpha, omega = (torch.ones_like(rho) for _ in range(3))
            restarts += 1
            continue
        shat = s/operator.diagonal
        t = operator @ shat
        tt = t @ t
        if float(tt.abs()) < 1e-40:
            raise RuntimeError("BiCGSTAB stabilization breakdown")
        omega = (t @ s)/tt
        x = x+alpha*phat+omega*shat
        r = s-omega*t
        previous = rho
        if iteration % 25 == 0 or float(r.norm()) <= float(target):
            true = rhs-operator @ x
            if not bool(torch.isfinite(true).all()):
                raise RuntimeError("Nonfinite sparse Newton direction")
            if float(true.norm()) <= float(target):
                return x*original_norm, {"iterations": iteration, "relative_residual": float(true.norm()/norm), "restarts": restarts}
            if float((r-true).norm()) > .1*float(true.norm()):
                r, shadow = true, true.clone()
                p, v = torch.zeros_like(r), torch.zeros_like(r)
                previous, alpha, omega = (torch.ones_like(rho) for _ in range(3))
                restarts += 1
    raise RuntimeError(f"Sparse Newton solve did not converge: relative residual={float((rhs-operator @ x).norm()/norm):.3e}")
