import torch
import torch.nn as nn
import torch.nn.functional as F


class STESign(torch.autograd.Function):
    """Straight-Through Estimator: forward Sign(x)→{-1,+1}, backward pasa sin modificarse."""

    @staticmethod
    def forward(ctx, weight):
        return torch.where(weight > 0, torch.ones_like(weight), -torch.ones_like(weight))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def weight_quant(W):
    """Cuantización 1-bit: W̃ = Sign(W − α), β = mean(|W|)."""
    alpha = W.mean()
    beta  = W.abs().mean().clamp(min=1e-8)
    W_binary = STESign.apply(W - alpha)
    return W_binary, beta


def activation_quant(x, num_bits=8):
    """Cuantización absmax b-bit: x̃ = Clip(x × Qb/γ, −Qb+ε, Qb−ε), γ = ||x||∞."""
    Qb = 2 ** (num_bits - 1) - 1
    eps = torch.finfo(x.dtype).eps

    gamma = x.abs().max().clamp(min=1e-8)
    x_scaled = x * (Qb / gamma)
    x_quant  = x_scaled.clamp(-Qb + eps, Qb - eps).round()

    # STE via detach trick: gradiente pasa por x original
    x_quant = x + (x_quant - x).detach()

    return x_quant, gamma


class BitLinear(nn.Linear):
    """Capa BitLinear: y = W̃ · Quant(LN(x)) × βγ/Qb."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.layer_norm = nn.LayerNorm(in_features, elementwise_affine=False)
        self.num_bits = 8
        self.Qb = 2 ** (self.num_bits - 1) - 1

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x_quant, gamma = activation_quant(x_norm, self.num_bits)
        w_quant, beta = weight_quant(self.weight)
        out = F.linear(x_quant, w_quant, self.bias)
        out = out * (beta * gamma / self.Qb)
        return out


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    layer = BitLinear(64, 32)
    x = torch.randn(4, 64, requires_grad=True)

    y = layer(x)
    loss = y.sum()
    loss.backward()

    print("Output shape:", y.shape)
    print("Grad en x shape:", x.grad.shape)
    print("Grad en W shape:", layer.weight.grad.shape)
    print("STE + BitLinear funcionando correctamente!")