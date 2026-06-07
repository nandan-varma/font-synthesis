"""Bivariate Gaussian MDN negative log-likelihood loss."""

import math
import torch


def mdn_loss(
    params: torch.Tensor,    # (B, T, 6M+1)
    targets: torch.Tensor,   # (B, T, 3)
    lengths: torch.Tensor,   # (B,) valid timesteps per sample
) -> torch.Tensor:
    B, T, _ = params.shape
    M = (params.shape[-1] - 1) // 6
    eps = 1e-8

    pis    = params[..., :M]                               # (B, T, M)
    mus    = params[..., M:3*M].reshape(B, T, M, 2)        # (B, T, M, 2)
    sigmas = params[..., 3*M:5*M].reshape(B, T, M, 2)     # (B, T, M, 2)
    rhos   = params[..., 5*M:6*M]                          # (B, T, M)
    eos_logit = params[..., 6*M]                           # (B, T)

    y1 = targets[..., 0].unsqueeze(-1)  # (B, T, 1)
    y2 = targets[..., 1].unsqueeze(-1)
    y3 = targets[..., 2]                # (B, T)

    pis    = torch.softmax(pis, dim=-1)
    sigmas = torch.clamp(torch.exp(sigmas), min=1e-4)
    rhos   = torch.clamp(torch.tanh(rhos), min=eps - 1.0, max=1.0 - eps)
    eos_prob = torch.clamp(torch.sigmoid(eos_logit), min=eps, max=1.0 - eps)

    mu1, mu2 = mus[..., 0], mus[..., 1]
    s1, s2   = sigmas[..., 0], sigmas[..., 1]

    norm = 1.0 - rhos ** 2
    z = (((y1 - mu1) / s1) ** 2
         + ((y2 - mu2) / s2) ** 2
         - 2.0 * rhos * (y1 - mu1) * (y2 - mu2) / (s1 * s2))

    log_gauss = (-0.5 * z / norm
                 - torch.log(2.0 * math.pi * s1 * s2 * torch.sqrt(norm).clamp(min=eps)))

    log_gmm = torch.logsumexp(torch.log(pis + eps) + log_gauss, dim=-1)  # (B, T)

    log_bern = torch.where(
        y3 == 1,
        torch.log(eos_prob + eps),
        torch.log(1.0 - eos_prob + eps),
    )

    nll = -(log_gmm + log_bern)
    mask = torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    nll = torch.nan_to_num(nll, nan=0.0, posinf=0.0, neginf=0.0) * mask.float()

    return nll.sum() / mask.float().sum().clamp(min=1.0)
