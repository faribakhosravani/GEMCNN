import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .masking import random_modality_mask


def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    uv_mean:    torch.Tensor,
    ftir_mean:  torch.Tensor,
    elem_mean:  torch.Tensor,
    mask_prob:  float = 0.7,
) -> float:
    """
    One training epoch with random modality masking (p=0.7).
    Expects each batch to be a dict with keys:
      'uv', 'ftir', 'xrf', 'icpms' (optional), 'label'
    Returns mean loss over the epoch.
    """
    model.train()
    criterion = nn.NLLLoss()
    total_loss = 0.0

    for batch in loader:
        uv    = batch['uv'].to(device)
        ftir  = batch['ftir'].to(device)
        xrf   = batch['xrf'].to(device)
        icpms = batch.get('icpms')
        if icpms is not None:
            icpms = icpms.to(device)
        labels = batch['label'].to(device)

        # random modality masking
        uv, ftir, xrf = random_modality_mask(
            [uv, ftir, xrf],
            [uv_mean.to(device), ftir_mean.to(device), elem_mean.to(device)],
            p=mask_prob,
        )

        optimizer.zero_grad()
        log_probs = model(uv, ftir, xrf, icpms)
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (mean_loss, accuracy)."""
    model.eval()
    criterion  = nn.NLLLoss()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch in loader:
        uv    = batch['uv'].to(device)
        ftir  = batch['ftir'].to(device)
        xrf   = batch['xrf'].to(device)
        icpms = batch.get('icpms')
        if icpms is not None:
            icpms = icpms.to(device)
        labels = batch['label'].to(device)

        log_probs = model(uv, ftir, xrf, icpms)
        total_loss += criterion(log_probs, labels).item()
        correct    += (log_probs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def fit(
    model:      nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:     torch.device,
    uv_mean:    torch.Tensor,
    ftir_mean:  torch.Tensor,
    elem_mean:  torch.Tensor,
    epochs:     int   = 250,
    lr:         float = 1e-4,
    lr_patience: int  = 10,
    lr_factor:  float = 0.1,
    ckpt_every: int   = 5,
    ckpt_path:  str   = "gemcnn_best.pt",
) -> dict:
    """
    Full training loop matching paper spec:
      - 250 epochs, lr=1e-4, decay ×0.1 after >10 epochs no improvement
      - checkpoint every 5 epochs, best model by val accuracy
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=lr_factor,
        patience=lr_patience, verbose=True,
    )

    best_acc  = 0.0
    history   = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            uv_mean, ftir_mean, elem_mean,
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:03d} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f}")

        # periodic checkpoint
        if epoch % ckpt_every == 0:
            torch.save(model.state_dict(), f"gemcnn_epoch{epoch:03d}.pt")

        # best-model checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ new best ({best_acc:.4f}) saved to {ckpt_path}")

    return history
