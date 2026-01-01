import argparse
import os
import torch
import torch.nn.functional as F

from src.data.dataset import create_dataloaders
from src.models.model import FingeringModel
from src.models.transformer_model import TransformerFingering
from src.models.ar_models import ArLSTM, ArGNN
from src.utils.config import load_config
from src.utils.decoder import beam_search_decode
from src.utils.metrics import evaluate_preds


def evaluate(config_path: str, checkpoint: str):
    cfg = load_config(config_path)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, builder, pieces = create_dataloaders(
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
    )
    sample = next(iter(train_loader))
    input_dim = sample["main"].shape[-1]
    phys_dim = sample["phys"].shape[-1]

    ckpt = torch.load(checkpoint, map_location=device)
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
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def base_id(pid: str) -> str:
        return pid.split("-")[0] if "-" in pid else pid

    # build grouped references
    gt_groups = {}
    for pid, events in pieces.items():
        main_feats, phys_feats, labels = builder.build(events)
        gt_groups.setdefault(base_id(pid), []).append(torch.tensor(labels, dtype=torch.long))

    metrics_cfg = cfg.get("metrics", {})
    soft_tol = metrics_cfg.get("soft_tolerance", 0)
    report_conf = metrics_cfg.get("report_confusion", False)
    dump_limit = metrics_cfg.get("error_dump_limit", 0)
    dump_path = metrics_cfg.get("error_dump_path", "outputs/eval_errors.csv")

    def run(loader, split_name: str):
        metrics_tot = {"M_gen": 0.0, "M_high": 0.0, "M_soft": 0.0, "M_cp": 0.0}
        num_cls = cfg["model"]["num_classes"]
        conf_mat = torch.zeros((num_cls, num_cls), dtype=torch.long)
        error_rows = []
        sample_count = 0
        with torch.no_grad():
            for batch in loader:
                main = batch["main"].to(device)
                phys = batch["phys"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["mask"].to(device)
                midi = batch["midi"]
                channel = batch["channel"]
                main_logits, phys_logits, _ = model(main, phys, mask)
                main_prob = F.softmax(main_logits, dim=-1)
                phys_prob = F.softmax(phys_logits, dim=-1)
                combined = (1 - cfg["model"]["phys_lambda"]) * main_prob + cfg["model"]["phys_lambda"] * phys_prob
                piece_ids = batch["piece_ids"]
                if cfg.get("decoder", {}).get("use_beam_eval", False):
                    preds_list = []
                    for i in range(main.shape[0]):
                        L = int(batch["lengths"][i].item())
                        probs_i = combined[i, :L].cpu()
                        midi_i = midi[i, :L].cpu().tolist()
                        channel_i = channel[i, :L].cpu().tolist()
                        seq = beam_search_decode(
                            probs_i,
                            midi_i,
                            channel_i,
                            beam_size=cfg["decoder"]["beam_size"],
                            alpha=cfg["decoder"]["alpha"],
                            beta=cfg["decoder"]["beta"],
                        )
                        preds = torch.tensor(seq, device=device, dtype=torch.long)
                        if L < labels.shape[1]:
                            pad = torch.full((labels.shape[1] - L,), -100, device=device, dtype=torch.long)
                            preds = torch.cat([preds, pad], dim=0)
                        preds_list.append(preds)
                    preds_tensor = torch.stack(preds_list, dim=0)
                    preds_for_metrics = preds_tensor
                else:
                    preds_for_metrics = combined.argmax(dim=-1)

                for bi in range(preds_for_metrics.shape[0]):
                    pid = piece_ids[bi]
                    base = base_id(pid)
                    refs = gt_groups.get(base, [labels[bi].cpu()])
                    seq_len = int(batch["lengths"][bi].item())
                    pred_seq = preds_for_metrics[bi][:seq_len]
                    mask_seq = mask[bi][:seq_len]

                    m_gen = 0.0
                    m_high = -1e9
                    m_soft = 0.0
                    m_cp = 0.0
                    for ref in refs:
                        if isinstance(ref, torch.Tensor):
                            ref_t = ref.to(pred_seq.device)[:seq_len]
                        else:
                            ref_t = torch.tensor(ref, device=pred_seq.device, dtype=torch.long)[:seq_len]
                        m = evaluate_preds(pred_seq.unsqueeze(0), ref_t.unsqueeze(0), mask_seq.unsqueeze(0), soft_tol)
                        m_gen += m["M_gen"]
                        m_soft += m["M_soft"]
                        m_cp += m["M_cp"]
                        m_high = max(m_high, m["M_gen"])
                    if refs:
                        m_gen /= len(refs)
                        m_soft /= len(refs)
                        m_cp /= len(refs)
                    metrics_tot["M_gen"] += m_gen
                    metrics_tot["M_high"] += m_high
                    metrics_tot["M_soft"] += m_soft
                    metrics_tot["M_cp"] += m_cp
                    sample_count += 1

                    if report_conf or dump_limit > 0:
                        first_ref = refs[0]
                        if isinstance(first_ref, torch.Tensor):
                            ref_t = first_ref.to(pred_seq.device)[:seq_len]
                        else:
                            ref_t = torch.tensor(first_ref, device=pred_seq.device, dtype=torch.long)[:seq_len]
                        valid_mask = mask_seq
                        p_flat = pred_seq[valid_mask]
                        l_flat = ref_t[valid_mask]
                        for p, l in zip(p_flat.tolist(), l_flat.tolist()):
                            conf_mat[l, p] += 1
                    if dump_limit > 0 and len(error_rows) < dump_limit:
                        first_ref = refs[0]
                        if isinstance(first_ref, torch.Tensor):
                            ref_t = first_ref.to(pred_seq.device)[:seq_len]
                        else:
                            ref_t = torch.tensor(first_ref, device=pred_seq.device, dtype=torch.long)[:seq_len]
                        for t in range(seq_len):
                            if pred_seq[t].item() != ref_t[t].item():
                                error_rows.append(
                                    [split_name, pid, t, int(midi[bi, t]), int(channel[bi, t]), ref_t[t].item(), pred_seq[t].item()]
                                )
                                if len(error_rows) >= dump_limit:
                                    break
        metrics_mean = {k: v / max(1, sample_count) for k, v in metrics_tot.items()}
        return metrics_mean, conf_mat, error_rows

    val_metrics, val_conf, val_err = run(val_loader, "val")
    test_metrics, test_conf, test_err = run(test_loader, "test")
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    if metrics_cfg.get("report_confusion", False):
        os.makedirs(os.path.dirname(metrics_cfg.get("error_dump_path", "outputs/eval_errors.csv")), exist_ok=True)
        torch.save({"val": val_conf, "test": test_conf}, "outputs/confusion.pt")
    if metrics_cfg.get("error_dump_limit", 0) > 0:
        import csv
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "piece_id", "t", "midi", "channel", "label_idx", "pred_idx"])
            for row in val_err + test_err:
                writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)

