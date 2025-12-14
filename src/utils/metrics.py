from typing import Dict

import torch

from src.data.features import FINGER_CLASSES

import torch


def _masked(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    preds = preds[mask]
    labels = labels[mask]
    return preds, labels


def matching_rate(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    preds, labels = _masked(preds, labels, mask)
    if preds.numel() == 0:
        return 0.0
    return (preds == labels).float().mean().item() * 100


def change_positions(seq: torch.Tensor) -> int:
    seq = seq.tolist()
    prev = None
    changes = 0
    for f in seq:
        if f is None:
            continue
        if prev is not None and f != prev:
            changes += 1
        prev = f
    return changes


def change_position_rate(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    batch = preds.shape[0]
    ratios = []
    for i in range(batch):
        m = mask[i]
        p = preds[i][m]
        l = labels[i][m]
        pred_changes = change_positions(p)
        gt_changes = max(1, change_positions(l))
        ratios.append(pred_changes / gt_changes)
    return float(sum(ratios) / len(ratios))


def _soft_match_rate(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, tol: int) -> float:
    if tol <= 0:
        return matching_rate(preds, labels, mask)
    preds_masked, labels_masked = _masked(preds, labels, mask)
    if preds_masked.numel() == 0:
        return 0.0
    pred_f = torch.tensor(FINGER_CLASSES, device=preds.device)[preds_masked]
    label_f = torch.tensor(FINGER_CLASSES, device=preds.device)[labels_masked]
    ok = (pred_f == label_f) | (pred_f - label_f).abs() <= tol
    return ok.float().mean().item() * 100


def evaluate_metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, soft_tol: int = 0) -> Dict[str, float]:
    preds = logits.argmax(dim=-1)
    m_gen = matching_rate(preds, labels, mask)
    m_high = m_gen
    m_soft = _soft_match_rate(preds, labels, mask, soft_tol)
    m_cp = change_position_rate(preds, labels, mask)
    return {
        "M_gen": m_gen,
        "M_high": m_high,
        "M_soft": m_soft,
        "M_cp": m_cp,
    }


def evaluate_preds(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, soft_tol: int = 0) -> Dict[str, float]:
    m_gen = matching_rate(preds, labels, mask)
    m_high = m_gen
    m_soft = _soft_match_rate(preds, labels, mask, soft_tol)
    m_cp = change_position_rate(preds, labels, mask)
    return {
        "M_gen": m_gen,
        "M_high": m_high,
        "M_soft": m_soft,
        "M_cp": m_cp,
    }

