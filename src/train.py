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
from src.models.model import FingeringModel
from src.models.transformer_model import TransformerFingering
from src.models.ar_models import ArLSTM, ArGNN
from src.utils.config import load_config
from src.utils.metrics import evaluate_metrics, evaluate_preds


def compute_loss(main_logits, phys_logits, labels, mask, num_classes, phys_lambda, crf=None):
    b, t, _ = main_logits.shape

    if crf is not None:
        loss_main = crf(main_logits, labels, mask)
    else:
        main_flat = main_logits.reshape(b * t, num_classes)
        labels_flat = labels.view(-1)
        loss_main = nn.CrossEntropyLoss(ignore_index=-100)(main_flat, labels_flat)

    phys_flat = phys_logits.reshape(b * t, num_classes)
    labels_flat = labels.view(-1)
    loss_phys = nn.CrossEntropyLoss(ignore_index=-100)(phys_flat, labels_flat)

    loss = (1 - phys_lambda) * loss_main + phys_lambda * loss_phys
    return loss, loss_main.item(), loss_phys.item()


def compute_transition_loss(main_logits, main_feats, mask):
    """Penalize predicting the same finger on consecutive different-pitch, same-hand notes.

    Returns the mean probability of same-finger predictions at adjacent positions
    where the pitch changes and the hand stays the same.
    """
    probs = F.softmax(main_logits, dim=-1)  # (B, T, C)

    # P(same finger at t and t+1) = sum_c P(c|t) * P(c|t+1)
    same_prob = (probs[:, :-1, :] * probs[:, 1:, :]).sum(dim=-1)  # (B, T-1)

    # Only penalize when pitch changes (feature 0 = midi/127)
    midi_vals = main_feats[:, :, 0]
    pitch_changed = (midi_vals[:, :-1] - midi_vals[:, 1:]).abs() > 1e-4  # (B, T-1)

    # Only penalize same hand (feature 3 = channel)
    channel_vals = main_feats[:, :, 3]
    same_hand = (channel_vals[:, :-1] - channel_vals[:, 1:]).abs() < 0.5  # (B, T-1)

    # Both positions must be non-padded
    valid = mask[:, :-1] & mask[:, 1:]  # (B, T-1)

    condition = pitch_changed & same_hand & valid  # (B, T-1)

    if condition.sum() > 0:
        return (same_prob * condition.float()).sum() / condition.sum().float()
    return torch.tensor(0.0, device=main_logits.device)


def train(config_path: str):
    cfg = load_config(config_path)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs/checkpoints", exist_ok=True)

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
            attn_heads=cfg["model"].get("attn_heads", 4),
            attn_window=cfg["model"].get("attn_window", 10),
            use_crf=cfg["model"].get("use_crf", False),
            use_ar=cfg["model"].get("use_ar", False),
            ar_embed_dim=cfg["model"].get("ar_embed_dim", 16),
        ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=cfg["train"]["scheduler_factor"],
        patience=cfg["train"]["scheduler_patience"],
    )

    run_name = f"{arch}_{cfg['data']['feature_type']}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    best_val = float("inf")
    patience = cfg["train"]["early_stop_patience"]
    patience_ctr = 0
    global_step = 0

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
            train_labels = labels if getattr(model, "use_ar", False) else None
            main_logits, phys_logits, _ = model(main, phys, mask, labels=train_labels)
            crf = getattr(model, "crf", None)
            loss, _, _ = compute_loss(
                main_logits, phys_logits, labels, mask, cfg["model"]["num_classes"], cfg["model"]["phys_lambda"], crf=crf
            )
            trans_lambda = cfg["model"].get("transition_lambda", 0.0)
            if trans_lambda > 0 and crf is None:
                trans_loss = compute_transition_loss(main_logits, main, mask)
                loss = loss + trans_lambda * trans_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            global_step += 1
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            if (batch_idx + 1) % cfg["train"]["log_interval"] == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_train = train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        metrics_accum = {"M_gen": 0.0, "M_high": 0.0, "M_soft": 0.0, "M_cp": 0.0}
        crf = getattr(model, "crf", None)
        with torch.no_grad():
            for batch in val_loader:
                main = batch["main"].to(device)
                phys = batch["phys"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["mask"].to(device)
                main_logits, phys_logits, _ = model(main, phys, mask)
                loss, _, _ = compute_loss(
                    main_logits, phys_logits, labels, mask, cfg["model"]["num_classes"], cfg["model"]["phys_lambda"], crf=crf
                )
                val_loss += loss.item()
                if crf is not None:
                    # Combine emissions then Viterbi decode for metrics
                    combined_emit = (1 - cfg["model"]["phys_lambda"]) * main_logits + cfg["model"]["phys_lambda"] * phys_logits
                    decoded = crf.decode(combined_emit, mask)
                    preds = torch.full_like(labels, -100)
                    for bi, seq in enumerate(decoded):
                        preds[bi, :len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)
                    metrics = evaluate_preds(preds, labels, mask)
                else:
                    metrics = evaluate_metrics(main_logits, labels, mask)
                for k in metrics_accum:
                    metrics_accum[k] += metrics[k]
        avg_val = val_loss / max(1, len(val_loader))
        metrics_mean = {k: v / max(1, len(val_loader)) for k, v in metrics_accum.items()}
        scheduler.step(avg_val)

        writer.add_scalar("train/loss_epoch", avg_train, epoch + 1)
        writer.add_scalar("val/loss", avg_val, epoch + 1)
        writer.add_scalar("val/M_gen", metrics_mean["M_gen"], epoch + 1)
        writer.add_scalar("val/M_high", metrics_mean["M_high"], epoch + 1)
        writer.add_scalar("val/M_soft", metrics_mean["M_soft"], epoch + 1)
        writer.add_scalar("val/M_cp", metrics_mean["M_cp"], epoch + 1)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch + 1)

        print(
            f"Epoch {epoch+1}: train_loss={avg_train:.4f} val_loss={avg_val:.4f} "
            f"M_gen={metrics_mean['M_gen']:.2f} M_cp={metrics_mean['M_cp']:.3f}"
        )

        if avg_val < best_val:
            best_val = avg_val
            patience_ctr = 0
            ckpt_path = f"outputs/checkpoints/best.pt"
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

