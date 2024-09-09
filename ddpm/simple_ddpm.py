"""
Adapted from: https://github.com/cloneofsimo/minDiffusion
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class DummyEpsModel(nn.Module):
    """
    This should be unet-like, but let's don't think about the model too much :P
    Basically, any universal R^n -> R^n model should work.
    """

    def __init__(self, input_dim: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x, t) -> torch.Tensor:
        x = self.fc(x)
        return x  # Remove the channel dimension

class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        _ts = torch.randint(1, self.n_T, (batch_size,)).to(x.device)
        eps = torch.randn_like(x)

        sqrtab_t = self.sqrtab[_ts]  # Shape: [batch_size]
        sqrtmab_t = self.sqrtmab[_ts]  # Shape: [batch_size]

        x_t = sqrtab_t.unsqueeze(1) * x + sqrtmab_t.unsqueeze(1) * eps
        
        return self.criterion(eps, self.eps_model(x_t, _ts.float() / self.n_T))

    def sample(self, n_sample: int, size: Tuple[int, int], device: torch.device) -> torch.Tensor:
        x_i = torch.randn(n_sample, size[1]).to(device)  # x_T ~ N(0, 1)

        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, size[1]).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, torch.full((n_sample,), i / self.n_T, device=device))
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i