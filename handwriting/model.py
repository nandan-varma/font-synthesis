"""Graves 2013 LSTM + Window Attention + MDN handwriting synthesis model."""

from collections import namedtuple
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alphabet import alphabet

State = namedtuple('State', ['h1', 'c1', 'h2', 'c2', 'h3', 'c3', 'kappa', 'w'])


class HandwritingRNN(nn.Module):
    def __init__(
        self,
        lstm_size: int = 512,
        output_mixtures: int = 30,
        attention_mixtures: int = 10,
        vocab_size: int = 65,
    ):
        super().__init__()
        self.lstm_size = lstm_size
        self.output_mixtures = output_mixtures
        self.attention_mixtures = attention_mixtures
        self.vocab_size = vocab_size
        self.output_units = 6 * output_mixtures + 1

        # LSTM1 input: [w (vocab_size) | x_t (3)]
        self.lstm1 = nn.LSTMCell(vocab_size + 3, lstm_size)
        # LSTM2/3 input: [x_t (3) | h_prev (lstm_size) | w (vocab_size)]
        self.lstm2 = nn.LSTMCell(3 + lstm_size + vocab_size, lstm_size)
        self.lstm3 = nn.LSTMCell(3 + lstm_size + vocab_size, lstm_size)

        # Window attention: h1 → (alpha, beta, kappa_delta) × K
        self.attention = nn.Linear(lstm_size, 3 * attention_mixtures)

        # LayerNorm after each LSTM
        self.ln1 = nn.LayerNorm(lstm_size)
        self.ln2 = nn.LayerNorm(lstm_size)
        self.ln3 = nn.LayerNorm(lstm_size)

        # MDN head: h3 → output_units
        self.output_head = nn.Linear(lstm_size, self.output_units)

    def zero_state(self, batch_size: int, device: torch.device) -> State:
        z = lambda: torch.zeros(batch_size, self.lstm_size, device=device)
        return State(
            h1=z(), c1=z(), h2=z(), c2=z(), h3=z(), c3=z(),
            kappa=torch.zeros(batch_size, self.attention_mixtures, device=device),
            w=torch.zeros(batch_size, self.vocab_size, device=device),
        )

    def _attend(
        self,
        h1: torch.Tensor,         # (B, lstm_size)
        kappa: torch.Tensor,      # (B, K)
        c_oh: torch.Tensor,       # (B, T_c, vocab_size)
        c_len: torch.Tensor,      # (B,) float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T_c = c_oh.shape[1]
        raw = self.attention(h1)                                        # (B, 3K)
        alpha_h, beta_h, kappa_h = raw.chunk(3, dim=-1)
        alpha = F.softplus(alpha_h)                                     # (B, K) > 0
        beta  = F.softplus(beta_h).clamp(min=0.01)                     # (B, K) > 0
        kappa = kappa + F.softplus(kappa_h) / 25.0                     # (B, K) monotone

        u = torch.arange(T_c, device=h1.device, dtype=h1.dtype)        # (T_c,)
        phi = (alpha.unsqueeze(2)
               * torch.exp(-beta.unsqueeze(2) * (kappa.unsqueeze(2) - u) ** 2)
               ).sum(dim=1)                                             # (B, T_c)

        # mask positions beyond character length
        mask = torch.arange(T_c, device=h1.device) < c_len.unsqueeze(1)
        phi = phi * mask.float()

        w = (phi.unsqueeze(2) * c_oh).sum(dim=1)                       # (B, vocab_size)
        return w, kappa, phi

    def _step(
        self,
        x_t: torch.Tensor,    # (B, 3)
        state: State,
        c_oh: torch.Tensor,   # (B, T_c, vocab_size)
        c_len: torch.Tensor,  # (B,) float
    ) -> Tuple[torch.Tensor, State, torch.Tensor]:
        h1, c1 = self.lstm1(torch.cat([state.w, x_t], dim=1), (state.h1, state.c1))
        h1 = self.ln1(h1)

        w, kappa, phi = self._attend(h1, state.kappa, c_oh, c_len)

        h2, c2 = self.lstm2(torch.cat([x_t, h1, w], dim=1), (state.h2, state.c2))
        h2 = self.ln2(h2)

        h3, c3 = self.lstm3(torch.cat([x_t, h2, w], dim=1), (state.h3, state.c3))
        h3 = self.ln3(h3)

        out = self.output_head(h3)
        return out, State(h1=h1, c1=c1, h2=h2, c2=c2, h3=h3, c3=c3, kappa=kappa, w=w), phi

    def forward(
        self,
        x: torch.Tensor,      # (B, T, 3)
        c: torch.Tensor,      # (B, T_c)
        c_len: torch.Tensor,  # (B,) int
        x_len: torch.Tensor,  # (B,) int  (unused in forward but kept for API symmetry)
    ) -> torch.Tensor:
        """Teacher-forcing pass. Returns MDN params (B, T, output_units)."""
        B, T, _ = x.shape
        c_oh = F.one_hot(c, self.vocab_size).float()      # (B, T_c, V)
        c_len_f = c_len.float()
        state = self.zero_state(B, x.device)
        outs = []
        for t in range(T):
            out, state, _ = self._step(x[:, t], state, c_oh, c_len_f)
            outs.append(out)
        return torch.stack(outs, dim=1)                   # (B, T, output_units)

    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor,                           # (1, T_c)
        c_len: int,
        bias: float = 0.5,
        max_steps: int = 2400,
        style_strokes: Optional[torch.Tensor] = None,  # (1, T_s, 3)
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        if device is None:
            device = next(self.parameters()).device
        self.eval()

        c = c.to(device)
        c_oh = F.one_hot(c, self.vocab_size).float()       # (1, T_c, V)
        c_len_t = torch.tensor([c_len], device=device, dtype=torch.float)
        state = self.zero_state(1, device)

        if style_strokes is not None:
            style_strokes = style_strokes.to(device)
            for t in range(style_strokes.shape[1]):
                _, state, _ = self._step(style_strokes[:, t], state, c_oh, c_len_t)

        x_t = torch.zeros(1, 3, device=device)
        x_t[0, 2] = 1.0
        strokes = []

        for _ in range(max_steps):
            params, state, phi = self._step(x_t, state, c_oh, c_len_t)
            x_t = self._sample_params(params.squeeze(0), bias).unsqueeze(0)
            strokes.append(x_t.squeeze(0).cpu().numpy())

            char_pos = int(phi.squeeze(0).argmax().item())
            if char_pos >= c_len - 1 and x_t[0, 2].item() > 0.5:
                break
            if char_pos >= c_len:
                break

        return np.array(strokes)  # (T, 3)

    def _sample_params(self, params: torch.Tensor, bias: float) -> torch.Tensor:
        M = self.output_mixtures
        eps = 1e-8

        pis      = params[:M] * (1 + bias)
        sigmas   = params[3*M:5*M].reshape(M, 2) - bias
        mus      = params[M:3*M].reshape(M, 2)
        rhos     = params[5*M:6*M]
        eos_logit = params[6*M]

        pis    = torch.softmax(pis, dim=0)
        pis    = torch.where(pis < 0.01, torch.zeros_like(pis), pis)
        sigmas = torch.clamp(torch.exp(sigmas), min=eps)
        rhos   = torch.clamp(torch.tanh(rhos), min=eps - 1.0, max=1.0 - eps)
        eos_p  = torch.clamp(torch.sigmoid(eos_logit), min=eps, max=1.0 - eps)

        idx = torch.multinomial(pis + eps, 1).item()
        mu   = mus[idx]
        sx, sy = sigmas[idx]
        rho  = rhos[idx]

        z1 = torch.randn((), device=params.device)
        z2 = torch.randn((), device=params.device)
        dx = mu[0] + sx * z1
        dy = mu[1] + sy * (rho * z1 + torch.sqrt(1.0 - rho ** 2) * z2)
        eos = torch.bernoulli(eos_p)

        return torch.stack([dx, dy, eos])
