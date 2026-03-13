import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

from bitnet import BitLinear, BitNet, weight_quant


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Hiperparámetros ---
    INPUT_DIM   = 3 * 32 * 32    # CIFAR-10: imágenes 32×32 RGB aplanadas
    HIDDEN_DIMS = [128, 256, 512, 1024, 1024,2048]
    NUM_CLASSES = 10
    EPOCHS      = 30
    BATCH_SIZE  = 128
    LR          = 1e-3

    # --- Dataset CIFAR-10 ---
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform_train)
    test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = BitNet(INPUT_DIM, HIDDEN_DIMS, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())

    # --- Wandb ---
    wandb.init(
        project="bitnet-cifar10",
        config={
            "input_dim":   INPUT_DIM,
            "hidden_dims": HIDDEN_DIMS,
            "num_classes": NUM_CLASSES,
            "epochs":      EPOCHS,
            "batch_size":  BATCH_SIZE,
            "lr":          LR,
            "scheduler":   "CosineAnnealingLR",
            "total_params": total_params,
            "device":      str(device),
            "dataset":     "CIFAR-10",
        },
    )
    wandb.watch(model, log="all", log_freq=200)

    arch_str = " → ".join(str(d) for d in [INPUT_DIM] + HIDDEN_DIMS + [NUM_CLASSES])
    print(f"Arquitectura: {arch_str}")
    print(f"Parámetros totales: {total_params:,}")
    print(f"Dispositivo: {device}\n")

    # --- Entrenamiento ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        correct    = 0
        total      = 0

        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)  # aplanar 32×32×3 → 3072
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += images.size(0)

        scheduler.step()
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
            "epoch":       epoch,
            "train/loss":  train_loss / total,
            "train/acc":   train_acc,
            "test/acc":    test_acc,
            "lr":          scheduler.get_last_lr()[0],
        }

        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, BitLinear):
                    w = module.weight
                    w_q, beta = weight_quant(w)
                    tag = name.replace(".", "/")
                    log_dict[f"weights/{tag}/latent"]       = wandb.Histogram(w.cpu().numpy())
                    log_dict[f"weights/{tag}/quantized"]    = wandb.Histogram(w_q.cpu().numpy())
                    log_dict[f"weights/{tag}/beta"]         = beta.item()
                    pct_pos = (w_q == 1).float().mean().item() * 100
                    log_dict[f"weights/{tag}/pct_positive"] = pct_pos

        wandb.log(log_dict)

        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={train_loss/total:.4f}  "
              f"train_acc={train_acc:.1f}%  "
              f"test_acc={test_acc:.1f}%  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

    wandb.finish()
    print("\nEntrenamiento completado.")
