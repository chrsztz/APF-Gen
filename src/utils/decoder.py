import math
from typing import List, Sequence

import torch

from src.data.features import FINGER_CLASSES


def finger_candidates(channel: int) -> List[int]:
    if channel == 0:
        return [f for f in FINGER_CLASSES if f > 0]
    if channel == 1:
        return [f for f in FINGER_CLASSES if f < 0]
    return [f for f in FINGER_CLASSES if f != 0]


def transition_cost(prev_f: int, prev_midi: int, f: int, midi: int, alpha: float, beta: float) -> float:
    if prev_f is None:
        return 0.0
    stretch = abs(midi - prev_midi) / max(1.0, abs(f - prev_f))
    crossing = (
        abs(midi - prev_midi)
        if (prev_f < f and prev_midi > midi) or (prev_f > f and prev_midi < midi)
        else 0.0
    )
    return alpha * stretch + beta * crossing


def beam_search_decode(
    probs: torch.Tensor,
    midis: Sequence[int],
    channels: Sequence[int],
    beam_size: int = 5,
    alpha: float = 0.1,
    beta: float = 0.05,
) -> List[int]:
    """
    probs: (T, num_classes) probabilities
    midis: list of midi ints length T
    channels: list of channel ints length T (0 right,1 left)
    """
    logp = torch.log(torch.clamp(probs, min=1e-8))
    T, C = logp.shape
    beam = [ (0.0, []) ]  # list of (score, seq)
    for t in range(T):
        midi_t = int(midis[t])
        ch_t = int(channels[t])
        cand_f = finger_candidates(ch_t)
        step = []
        for score, seq in beam:
            prev_f = seq[-1] if seq else None
            prev_m = midis[t - 1] if t > 0 else midi_t
            for f in cand_f:
                f_idx = FINGER_CLASSES.index(f)
                lp = float(logp[t, f_idx])
                cost = transition_cost(prev_f, prev_m, f, midi_t, alpha, beta)
                step.append((score + lp - cost, seq + [f_idx]))
        step.sort(key=lambda x: x[0], reverse=True)
        beam = step[:beam_size]
    best_seq = max(beam, key=lambda x: x[0])[1]
    return best_seq




