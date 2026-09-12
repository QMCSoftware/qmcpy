import torch

def _check_inputs(x, gamma=None):
    """
    x: (B, N, d) in [0,1]
    gamma: (d,) nonnegative weights (optional)
    """
    if x.dim() != 3:
        raise ValueError(f"x must be (batch,N,d); got {tuple(x.shape)}")
    B, N, d = x.shape
    if gamma is not None:
        if gamma.dim() != 1 or gamma.shape[0] != d:
            raise ValueError(f"gamma must be (d,) with d={d}; got {tuple(gamma.shape)}")
    return B, N, d

def _pairwise(x):
    # x_i: (B, N, 1, d), x_j: (B, 1, N, d)
    return x.unsqueeze(2), x.unsqueeze(1)

def _sqrt_safe(v):
    # numeric guard for small negatives from fp error
    return torch.sqrt(torch.clamp_min(v, 0.0))

# ----------------------------
# L2 STAR (Warnock)
# ----------------------------
def L2star(x: torch.Tensor) -> torch.Tensor:
    _, N, d = _check_inputs(x)
    t1 = (1.0 / 3.0) ** d
    p = torch.prod(1.0 - x**2, dim=2)
    t2 = (2.0 / N) * (2.0 ** (-d)) * torch.sum(p, dim=1)
    xi, xj = _pairwise(x)
    prod_ij = torch.prod(1.0 - torch.maximum(xi, xj), dim=3)
    t3 = (1.0 / (N * N)) * torch.sum(prod_ij, dim=(1, 2))
    return _sqrt_safe(t1 - t2 + t3)

def L2star_weighted(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    _, N, d = _check_inputs(x, gamma)
    g = gamma
    t1 = torch.prod(1.0 + g / 3.0)
    p = torch.prod(1.0 + (g.view(1, 1, d) / 2.0) * (1.0 - x**2), dim=2)
    t2 = (2.0 / N) * torch.sum(p, dim=1)
    xi, xj = _pairwise(x)
    q = torch.prod(1.0 + g.view(1, 1, 1, d) * (1.0 - torch.maximum(xi, xj)), dim=3)
    t3 = (1.0 / (N * N)) * torch.sum(q, dim=(1, 2))
    return _sqrt_safe(t1 - t2 + t3)

# -----------------------------------------
# L2 EXTREME
# -----------------------------------------
def L2ext(x: torch.Tensor) -> torch.Tensor:
    _, N, d = _check_inputs(x)
    t1 = (1.0 / 12.0) ** d
    p = torch.prod(0.5 * (x - x**2), dim=2)
    t2 = (2.0 / N) * torch.sum(p, dim=1)
    xi, xj = _pairwise(x)
    q = torch.prod(torch.minimum(xi, xj) - xi * xj, dim=3)
    t3 = (1.0 / (N * N)) * torch.sum(q, dim=(1, 2))
    return _sqrt_safe(t1 - t2 + t3)

def L2ext_weighted(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    _, N, d = _check_inputs(x, gamma)
    g = gamma
    t1 = torch.prod(1.0 + g / 12.0)
    p = torch.prod(1.0 + g.view(1, 1, d) * 0.5 * (x - x**2), dim=2)
    t2 = (2.0 / N) * torch.sum(p, dim=1)
    xi, xj = _pairwise(x)
    q = torch.prod(1.0 + g.view(1, 1, 1, d) * (torch.minimum(xi, xj) - xi * xj), dim=3)
    t3 = (1.0 / (N * N)) * torch.sum(q, dim=(1, 2))
    return _sqrt_safe(t1 - t2 + t3)

# -----------------------------------------
# L2 PERIODIC
# -----------------------------------------
def L2per(x: torch.Tensor) -> torch.Tensor:
    _, N, d = _check_inputs(x)
    t1 = (1.0 / 3.0) ** d
    xi, xj = _pairwise(x)
    Δ = xi - xj
    q = torch.prod(0.5 - torch.abs(Δ) + Δ**2, dim=3)
    t3 = (1.0 / (N * N)) * torch.sum(q, dim=(1, 2))
    return _sqrt_safe(-t1 + t3)

def L2per_weighted(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    _, N, d = _check_inputs(x, gamma)
    g = gamma
    t1 = torch.prod(1.0 + g / 3.0)
    xi, xj = _pairwise(x)
    Δ = xi - xj
    q = torch.prod(1.0 + g.view(1, 1, 1, d) * (0.5 - torch.abs(Δ) + Δ**2), dim=3)
    t3 = (1.0 / (N * N)) * torch.sum(q, dim=(1, 2))
    return _sqrt_safe(-t1 + t3)

# -----------------------------------------
# L2 CENTERED
# -----------------------------------------
def L2ctr(x: torch.Tensor) -> torch.Tensor:
    _, N, d = _check_inputs(x)
    t1 = (1.0 / 12.0) ** d
    u = torch.abs(x - 0.5)
    p = torch.prod(0.5 * (u - u**2), dim=2)
    t2 = (2.0 / N) * torch.sum(p, dim=1)
    xi, xj = _pairwise(x)
    q = torch.prod(0.5 * (torch.abs(xi - 0.5) + torch.abs(xj - 0.5) - torch.abs(xi - xj)), dim=3)
    t3 = (1.0 / (N * N)) * torch.sum(q, dim=(1, 2))
    return _sqrt_safe(t1 - t2 + t3)

def L2ctr_weighted(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    _, N, d = _check_inputs(x, gamma)
    g = gamma
    t1 = torch.prod(1.0 + g / 12.0)
    u = torch.abs(x - 0.5)
    p = torch.prod(1.0 + (g.view(1, 1, d) / 2.0) * (u - u**2), dim=2)
    t2 = (2.0 / N) * torch.sum(p, dim=1)
    xi, xj = _pairwise(x)
    q = torch.prod(1.0 + (g.view(1, 1, 1, d) / 2.0) * (torch.abs(xi - 0.5) + torch.abs(xj - 0.5) - torch.abs(xi - xj)), dim=3)
    t3 = (1.0 / (N * N)) * torch.sum(q, dim=(1, 2))
    return _sqrt_safe(t1 - t2 + t3)

# -----------------------------------------
# L2 SYMMETRIC
# -----------------------------------------
def L2sym(x: torch.Tensor) -> torch.Tensor:
    _, N, d = _check_inputs(x)
    t1 = (1.0 / 12.0) ** d
    p = torch.prod(0.5 * (x - x**2), dim=2)
    t2 = (2.0 / N) * torch.sum(p, dim=1)
    xi, xj = _pairwise(x)
    q = torch.prod(0.25 * (1.0 - 2.0 * torch.abs(xi - xj)), dim=3)
    t3 = (1.0 / (N * N)) * torch.sum(q, dim=(1, 2))
    return _sqrt_safe(t1 - t2 + t3)

def L2sym_weighted(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    _, N, d = _check_inputs(x, gamma)
    g = gamma
    t1 = torch.prod(1.0 + g / 12.0)
    p = torch.prod(1.0 + (g.view(1, 1, d) / 2.0) * (x - x**2), dim=2)
    t2 = (2.0 / N) * torch.sum(p, dim=1)
    xi, xj = _pairwise(x)
    q = torch.prod(1.0 + (g.view(1, 1, 1, d) / 4.0) * (1.0 - 2.0 * torch.abs(xi - xj)), dim=3)
    t3 = (1.0 / (N * N)) * torch.sum(q, dim=(1, 2))
    return _sqrt_safe(t1 - t2 + t3)

# -----------------------------------------
# L2 MIXTURE
# -----------------------------------------
def L2mix(x: torch.Tensor) -> torch.Tensor:
    _, N, d = _check_inputs(x)
    t1 = (7.0 / 12.0) ** d
    u = x - 0.5
    p = torch.prod(2.0 / 3.0 - 0.25 * torch.abs(u) - 0.25 * (u**2), dim=2)
    t2 = (2.0 / N) * torch.sum(p, dim=1)
    xi, xj = _pairwise(x)
    ui, uj = xi - 0.5, xj - 0.5
    Δ = xi - xj
    q = torch.prod(7.0 / 8.0 - 0.25 * torch.abs(ui) - 0.25 * torch.abs(uj) - 0.75 * torch.abs(Δ) + 0.5 * (Δ**2), dim=3)
    t3 = (1.0 / (N * N)) * torch.sum(q, dim=(1, 2))
    return _sqrt_safe(t1 - t2 + t3)

def L2mix_weighted(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    _, N, d = _check_inputs(x, gamma)
    g = gamma
    t1 = torch.prod(1.0 + (7.0 / 12.0) * g)
    u = x - 0.5
    p = torch.prod(1.0 + g.view(1, 1, d) * (2.0 / 3.0 - 0.25 * torch.abs(u) - 0.25 * (u**2)), dim=2)
    t2 = (2.0 / N) * torch.sum(p, dim=1)
    xi, xj = _pairwise(x)
    ui, uj = xi - 0.5, xj - 0.5
    Δ = xi - xj
    q = torch.prod(1.0 + g.view(1, 1, 1, d) * (7.0 / 8.0 - 0.25 * torch.abs(ui) - 0.25 * torch.abs(uj) - 0.75 * torch.abs(Δ) + 0.5 * (Δ**2)), dim=3)
    t3 = (1.0 / (N * N)) * torch.sum(q, dim=(1, 2))
    return _sqrt_safe(t1 - t2 + t3)

