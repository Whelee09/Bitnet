import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. STRAIGHT-THROUGH ESTIMATOR (STE)
#    Forward:  Sign(x) → {-1, +1} 
#    Backward: gradiente pasa sin modificarse
# ─────────────────────────────────────────────
class STESign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):  # Cambiamos x por weight
        return torch.where(weight > 0, torch.ones_like(weight), -torch.ones_like(weight))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # El gradiente pasa igual, sin modificar


# ─────────────────────────────────────────────
# 2. CUANTIZACIÓN DE PESOS  (1-bit)
#    W̃ = Sign(W − α)        
#    α  = (1/nm) Σ Wij       
#    β  = (1/nm) ||W||₁    
# ─────────────────────────────────────────────
def weight_quant(W):
    alpha = W.mean()                         
    beta  = W.abs().mean().clamp(min=1e-8) 
    W_binary = STESign.apply(W - alpha)     

    return W_binary, beta


# ─────────────────────────────────────────────
# 3. CUANTIZACIÓN DE ACTIVACIONES  (absmax, b-bit)
#
#    x̃ = Quant(x) = Clip(x × Qb/γ, −Qb+ε, Qb−ε)  
#    γ  = ||x||∞  (valor absoluto máximo)
#    Qb = 2^(b-1) − 1
#
#    Corrección vs versión anterior:
#    se usa γ = abs_max (no abs_mean) y el clip incluye ε
# ─────────────────────────────────────────────
def activation_quant(x, num_bits=8):
    Qb = 2 ** (num_bits - 1) - 1             # e.g. 127 para 8-bit
    eps = torch.finfo(x.dtype).eps           # ε pequeño para evitar overflow

    gamma = x.abs().max().clamp(min=1e-8)    # γ = ||x||∞  

    x_scaled = x * (Qb / gamma)
    x_quant  = x_scaled.clamp(-Qb + eps, Qb - eps).round()

    # STE via detach trick: gradiente pasa por x original
    x_quant = x + (x_quant - x).detach()

    return x_quant, gamma


# ─────────────────────────────────────────────
# 4. CAPA BITLINEAR COMPLETA
#
#    y = W̃ · Quant(LN(x)) × βγ/Qb
#
#    Flujo:
#    x → LN(x) → Quant → W̃ · x̃ → × βγ/Qb → y
# ─────────────────────────────────────────────
class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.layer_norm = nn.LayerNorm(in_features, elementwise_affine=False)
        self.num_bits = 8
        self.Qb = 2 ** (self.num_bits - 1) - 1  # 127

    def forward(self, x):
        # 1. LayerNorm antes de cuantizar 
        x_norm = self.layer_norm(x)

        # 2. Cuantiza activaciones → x̃, γ
        x_quant, gamma = activation_quant(x_norm, self.num_bits)

        # 3. Cuantiza pesos → W̃, β
        w_quant, beta = weight_quant(self.weight)

        # 4. Multiplicación lineal con pesos 1-bit
        out = F.linear(x_quant, w_quant, self.bias)

        # 5. Dequantización: × βγ/Qb  (eq. 11)
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

    print("Output shape:      ", y.shape)              # [4, 32]
    print("Grad en x shape:   ", x.grad.shape)         # [4, 64]
    print("Grad en W shape:   ", layer.weight.grad.shape)  # [32, 64]
    print("Pesos cuantizados únicos:", 
          layer.weight.data.sign().unique().tolist())   # [-1, 1]
    print("\n STE + BitLinear funcionando correctamente!")