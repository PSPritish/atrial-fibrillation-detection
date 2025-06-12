import torch
import torch.nn as nn


class ModReLU(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, z):
        # z: (B, C, H, W) — complex tensor
        magnitude = torch.abs(z)  # |z|, shape: (B, C, H, W)
        phase = z / (magnitude + 1e-8)  # z / |z|, avoid div-by-zero
        bias = self.bias.view(1, -1, 1, 1)  # reshape bias for broadcasting

        # relu(|z| + b)
        mod_relu_out = torch.relu(magnitude + bias) * phase
        return mod_relu_out


class GeneralizedReLU(nn.Module):
    def __init__(self, max_value=None, threshold=0.0, alpha=0.0):
        super().__init__()
        self.max_value = max_value
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, x):
        x = torch.where(x >= self.threshold, x, self.alpha * (x - self.threshold))
        if self.max_value is not None:
            x = torch.clamp(x, max=self.max_value)
        return x


class zReLU(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        phase = torch.angle(z)  # φ(z) ∈ [-π, π]
        mask = (phase >= 0) & (phase <= torch.pi / 2)
        return z * mask


class ComplexCardioid(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        phase = torch.angle(z)
        scale = 0.5 * (1 + torch.cos(phase))  # scalar in [0, 1]
        return scale * z
