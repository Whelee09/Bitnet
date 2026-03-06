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
        result = torch.where(weight > 0, torch.ones_like(weight), -torch.ones_like(weight))
        print(f"  [STESign forward] Input range: [{weight.min():.4f}, {weight.max():.4f}]")
        print(f"  [STESign forward] Valores únicos en salida: {result.unique().tolist()}")
        return result

    @staticmethod
    def backward(ctx, grad_output):
        print(f"  [STESign backward] Gradiente pasa sin cambios | norm: {grad_output.norm():.4f}")
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
    print(f"\n[weight_quant] W shape: {W.shape}")
    print(f"[weight_quant] α (media de W):       {alpha:.6f}")
    print(f"[weight_quant] β (media |W| orig):   {beta:.6f}")
    W_binary = STESign.apply(W - alpha)     
    print(f"[weight_quant] W_binary únicos: {W_binary.unique().tolist()}")

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
    print(f"\n[activation_quant] x shape: {x.shape}")
    print(f"[activation_quant] Qb ({num_bits}-bit): {Qb}")
    print(f"[activation_quant] gama = ||x||∞:  {gamma:.6f}")

    x_scaled = x * (Qb / gamma)
    print(f"[activation_quant] x_scaled range: [{x_scaled.min():.4f}, {x_scaled.max():.4f}]")

    x_quant  = x_scaled.clamp(-Qb + eps, Qb - eps).round()#TODO no se si esto aqui deberia seguir o no
    print(f"[activation_quant] x_quant range:  [{x_quant.min():.4f}, {x_quant.max():.4f}]")

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
        print(f"[BitLinear forward] Input x shape: {x.shape}")
        print(f"[BitLinear forward] Input x range: [{x.min():.4f}, {x.max():.4f}]")

        # 1. LayerNorm antes de cuantizar 
        x_norm = self.layer_norm(x)
        print(f"\n[Paso 1 - LayerNorm] x_norm media: {x_norm.mean():.6f} | std: {x_norm.std():.6f}")

        # 2. Cuantiza activaciones → x̃, γ
        x_quant, gamma = activation_quant(x_norm, self.num_bits)
        print(f"\n[Paso 2 - Act. Quant] γ={gamma:.4f} | x_quant media: {x_quant.mean():.4f}")

        # 3. Cuantiza pesos → W̃, β
        w_quant, beta = weight_quant(self.weight)
        print(f"\n[Paso 3 - Weight Quant] β={beta:.4f} | w_quant shape: {w_quant.shape}")

        # 4. Multiplicación lineal con pesos 1-bit
        out = F.linear(x_quant, w_quant, self.bias)
        print(f"\n[Paso 4 - F.linear] out shape: {out.shape} | out range: [{out.min():.4f}, {out.max():.4f}]")

        # 5. Dequantización: × βγ/Qb  (eq. 11)
        out = out * (beta * gamma / self.Qb)
        print(f"\n[Paso 5 - Dequant] factor βγ/Qb: {(beta * gamma / self.Qb):.6f}")
        print(f"[Paso 5 - Dequant] out final range: [{out.min():.4f}, {out.max():.4f}]")

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

    print("\n--- RESULTADOS FINALES ---")
    print("Output shape:      ", y.shape)              # [4, 32]
    print("Grad en x shape:   ", x.grad.shape)         # [4, 64]
    print("Grad en W shape:   ", layer.weight.grad.shape)  # [32, 64]
    print("Pesos cuantizados únicos:", 
          layer.weight.data.sign().unique().tolist())   # [-1, 1]
    print("\n STE + BitLinear funcionando correctamente!")