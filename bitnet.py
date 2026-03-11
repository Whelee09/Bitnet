import torch
import torch.nn as nn
import torch.nn.functional as F


class STESign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        return torch.where(weight > 0, torch.ones_like(weight), -torch.ones_like(weight))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def weight_quant(W):
    alpha = W.mean()
    beta  = W.abs().mean().clamp(min=1e-8)
    W_binary = STESign.apply(W - alpha)
    return W_binary, beta


def activation_quant(x, num_bits=8):
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


class BitNet(nn.Module):
    """MLP fully connected usando capas BitLinear."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden=1):
        super().__init__()
        layers = [BitLinear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden):
            layers += [BitLinear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(BitLinear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# ENTRENAMIENTO CON MNIST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import wandb

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Hiperparámetros ---
    INPUT_DIM   = 28 * 28       # MNIST: imágenes 28×28 aplanadas
    HIDDEN_DIM  = 256
    NUM_CLASSES = 10
    NUM_HIDDEN  = 4             # 4 capas ocultas
    EPOCHS      = 10
    BATCH_SIZE  = 128
    LR          = 1e-3

    # --- Dataset MNIST ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # media y std de MNIST
    ])

    train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # --- Modelo: entrada(784) → 4 capas ocultas(256) → salida(10) ---
    model     = BitNet(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, num_hidden=NUM_HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())

    # --- Wandb ---
    wandb.init(
        project="bitnet-mnist",
        config={
            "input_dim":   INPUT_DIM,
            "hidden_dim":  HIDDEN_DIM,
            "num_classes": NUM_CLASSES,
            "num_hidden":  NUM_HIDDEN,
            "epochs":      EPOCHS,
            "batch_size":  BATCH_SIZE,
            "lr":          LR,
            "total_params": total_params,
            "device":      str(device),
        },
    )
    wandb.watch(model, log="all", log_freq=100)  # loguea gradientes y pesos

    print(f"Arquitectura: {INPUT_DIM} → {HIDDEN_DIM}×{NUM_HIDDEN} → {NUM_CLASSES}")
    print(f"Parámetros totales: {total_params:,}")
    print(f"Dispositivo: {device}\n")

    # --- Entrenamiento ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        correct    = 0
        total      = 0

        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)  # aplanar 28×28 → 784
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += images.size(0)

        train_acc = 100.0 * correct / total

        # --- Evaluación en test ---
        model.eval()
        test_correct = 0
        test_total   = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)
                logits = model(images)
                test_correct += (logits.argmax(dim=1) == labels).sum().item()
                test_total   += images.size(0)

        test_acc = 100.0 * test_correct / test_total

        # Log a wandb
        log_dict = {
            "epoch":      epoch,
            "train/loss":  train_loss / total,
            "train/acc":   train_acc,
            "test/acc":    test_acc,
        }

        # Loguear histogramas de pesos latentes vs cuantizados por capa
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, BitLinear):
                    w = module.weight
                    w_q, beta = weight_quant(w)
                    tag = name.replace(".", "/")
                    log_dict[f"weights/{tag}/latent"]    = wandb.Histogram(w.cpu().numpy())
                    log_dict[f"weights/{tag}/quantized"] = wandb.Histogram(w_q.cpu().numpy())
                    log_dict[f"weights/{tag}/beta"]      = beta.item()
                    # % de pesos que son +1 vs -1
                    pct_pos = (w_q == 1).float().mean().item() * 100
                    log_dict[f"weights/{tag}/pct_positive"] = pct_pos

        wandb.log(log_dict)

        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={train_loss/total:.4f}  "
              f"train_acc={train_acc:.1f}%  "
              f"test_acc={test_acc:.1f}%")

    wandb.finish()
    print("\nEntrenamiento completado.")