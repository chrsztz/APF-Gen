import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.data.features import FINGER_VALUES
from src.models.model import FingeringModel
from src.models.transformer_model import TransformerFingering
from src.models.ar_models import ArLSTM, ArGNN
from src.utils.config import load_config
from src.utils.metrics import evaluate_metrics

# Finger class values as a tensor (for ordinal MSE loss)
_FINGER_T = torch.tensor(FINGER_VALUES, dtype=torch.float32)


def compute_loss(main_logits, phys_logits, labels, mask, num_classes, phys_lambda):
    """Ordinal MSE loss on expected finger value.

    Instead of treating fingers as unordered classes (cross-entropy),
    we compute the expected finger value from the softmax distribution
    and penalise deviation from the true finger value with MSE.
    This gives a smooth gradient that penalises "far-off" predictions
    more than "close" ones  (e.g. predicting finger 2 when true is 3
    costs less than predicting 5).
    """
    finger_vals = _FINGER_T.to(main_logits.device)

    main_prob = F.softmax(main_logits, dim=-1)
    phys_prob = F.softmax(phys_logits, dim=-1)

    # Expected finger value per position: sum(prob_i * finger_i)
    main_expected = (main_prob * finger_vals).sum(dim=-1)   # (B, T)
    phys_expected = (phys_prob * finger_vals).sum(dim=-1)

    # True finger values (padding labels → clamp to valid range)
    labels_safe = labels.clamp(min=0, max=num_classes - 1)
    true_fingers = finger_vals[labels_safe]                 # (B, T)

    # Mask: exclude padding (-100) positions
    valid = mask & (labels >= 0)
    n_valid = valid.sum().clamp(min=1)

    main_mse = ((main_expected - true_fingers) ** 2 * valid.float()).sum() / n_valid
    phys_mse = ((phys_expected - true_fingers) ** 2 * valid.float()).sum() / n_valid

    loss = (1 - phys_lambda) * main_mse + phys_lambda * phys_mse
    return loss, main_mse.item(), phys_mse.item()


def _resolve_device(cfg_device: str) -> torch.device:
    """Pick the best available device respecting the config hint."""
    if cfg_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(config_path: str):
    cfg = load_config(config_path)
    device = _resolve_device(cfg["train"]["device"])
    print(f"Using device: {device}")
    os.makedirs("outputs/checkpoints", exist_ok=True)

    # TensorBoard
    log_dir = cfg["train"].get("tb_log_dir", "runs")
    writer = SummaryWriter(log_dir=log_dir)

    train_loader, val_loader, _, builder, _ = create_dataloaders(
        root=cfg["data"]["root"],
        feature_type=cfg["data"]["feature_type"],
        word2vec_dim=cfg["data"]["word2vec_dim"],
        velocity_threshold=cfg["data"]["velocity_threshold"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        seed=cfg["data"]["split_seed"],
        augment_shifts=cfg["data"].get("augment_shifts", []),
        augment_speeds=cfg["data"].get("augment_speeds", []),
    )

    sample_batch = next(iter(train_loader))
    input_dim = sample_batch["main"].shape[-1]
    phys_dim = sample_batch["phys"].shape[-1]
    print(f"input_dim={input_dim}, phys_dim={phys_dim}, "
          f"train_samples={len(train_loader.dataset)}, val_samples={len(val_loader.dataset)}")

    arch = cfg["model"].get("arch", "cnn_bilstm")
    if arch == "transformer":
        model = TransformerFingering(
            input_dim=input_dim,
            phys_dim=phys_dim,
            d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"],
            num_layers=cfg["model"]["tf_layers"],
            dropout=cfg["model"]["dropout"],
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif arch == "arlstm":
        model = ArLSTM(
            input_dim=input_dim,
            phys_dim=phys_dim,
            hidden_size=cfg["model"]["hidden_size"],
            num_layers=cfg["model"]["lstm_layers"],
            num_classes=cfg["model"]["num_classes"],
            dropout=cfg["model"]["dropout"],
        ).to(device)
    elif arch == "argnn":
        model = ArGNN(
            input_dim=input_dim,
            phys_dim=phys_dim,
            hidden_size=cfg["model"]["hidden_size"],
            num_classes=cfg["model"]["num_classes"],
            dropout=cfg["model"]["dropout"],
        ).to(device)
    else:
        model = FingeringModel(
            input_dim=input_dim,
            phys_dim=phys_dim,
            hidden_size=cfg["model"]["hidden_size"],
            cnn_channels=cfg["model"]["cnn_channels"],
            cnn_layers=cfg["model"]["cnn_layers"],
            lstm_layers=cfg["model"]["lstm_layers"],
            dropout=cfg["model"]["dropout"],
            num_classes=cfg["model"]["num_classes"],
            use_attention=cfg["model"]["attention"],
        ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=cfg["train"]["scheduler_factor"],
        patience=cfg["train"]["scheduler_patience"],
    )

    best_val = float("inf")
    patience = cfg["train"]["early_stop_patience"]
    patience_ctr = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            main = batch["main"].to(device)
            phys = batch["phys"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad()
            main_logits, phys_logits, _ = model(main, phys, mask)
            loss, _, _ = compute_loss(
                main_logits, phys_logits, labels, mask, cfg["model"]["num_classes"], cfg["model"]["phys_lambda"]
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (batch_idx + 1) % cfg["train"]["log_interval"] == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_train = train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        metrics_accum = {"M_gen": 0.0, "M_high": 0.0, "M_soft": 0.0, "M_cp": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                main = batch["main"].to(device)
                phys = batch["phys"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["mask"].to(device)
                main_logits, phys_logits, _ = model(main, phys, mask)
                loss, _, _ = compute_loss(
                    main_logits, phys_logits, labels, mask, cfg["model"]["num_classes"], cfg["model"]["phys_lambda"]
                )
                val_loss += loss.item()
                metrics = evaluate_metrics(main_logits, labels, mask)
                for k in metrics_accum:
                    metrics_accum[k] += metrics[k]
        avg_val = val_loss / max(1, len(val_loader))
        metrics_mean = {k: v / max(1, len(val_loader)) for k, v in metrics_accum.items()}
        scheduler.step(avg_val)

        # TensorBoard logging
        writer.add_scalar("Loss/train", avg_train, epoch)
        writer.add_scalar("Loss/val", avg_val, epoch)
        writer.add_scalar("Metrics/M_gen", metrics_mean["M_gen"], epoch)
        writer.add_scalar("Metrics/M_soft", metrics_mean["M_soft"], epoch)
        writer.add_scalar("Metrics/M_cp", metrics_mean["M_cp"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch+1}: train_loss={avg_train:.4f} val_loss={avg_val:.4f} "
            f"M_gen={metrics_mean['M_gen']:.2f} M_cp={metrics_mean['M_cp']:.3f}"
        )

        if avg_val < best_val:
            best_val = avg_val
            patience_ctr = 0
            ckpt_path = "outputs/checkpoints/best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "input_dim": input_dim,
                    "phys_dim": phys_dim,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping triggered.")
                break

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
