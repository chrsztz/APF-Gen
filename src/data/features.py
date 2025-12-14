import math
from typing import List, Dict, Tuple, Optional

import numpy as np
from gensim.models import Word2Vec

from .parser import NoteEvent

FINGER_CLASSES = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
FINGER_TO_IDX = {f: i for i, f in enumerate(FINGER_CLASSES)}
IDX_TO_FINGER = {i: f for f, i in FINGER_TO_IDX.items()}


def is_black_key(midi: int) -> int:
    if midi < 0:
        return 0
    return 1 if midi % 12 in {1, 3, 6, 8, 10} else 0


def detect_chords(events: List[NoteEvent]) -> Tuple[List[int], List[int]]:
    flags = [0] * len(events)
    sizes = [1] * len(events)
    for i, ev in enumerate(events):
        overlap = 1
        for j, other in enumerate(events):
            if i == j:
                continue
            if ev.channel == other.channel and other.onset < ev.offset and ev.onset < other.offset:
                flags[i] = 1
                overlap += 1
        sizes[i] = overlap
    return flags, sizes


def compute_physical_features(
    events: List[NoteEvent],
    chord_flags: List[int],
    velocity_threshold: int,
) -> np.ndarray:
    phys = []
    for i, ev in enumerate(events):
        prev_same_hand = None
        for j in range(i - 1, -1, -1):
            if events[j].channel == ev.channel:
                prev_same_hand = events[j]
                break
        stretch = 0.0
        crossing = 0.0
        if prev_same_hand and prev_same_hand.finger != 0 and ev.finger != 0:
            stretch = abs(ev.midi - prev_same_hand.midi) / max(1.0, abs(ev.finger - prev_same_hand.finger))
            crossing = (
                abs(ev.midi - prev_same_hand.midi)
                if (prev_same_hand.finger < ev.finger and prev_same_hand.midi > ev.midi)
                or (prev_same_hand.finger > ev.finger and prev_same_hand.midi < ev.midi)
                else 0.0
            )
        active_same_time = [
            other.midi
            for other in events
            if other.channel == ev.channel and other.onset <= ev.onset < other.offset
        ]
        hand_pos = float(np.mean(active_same_time)) if active_same_time else float(ev.midi)
        natural_violation = 1.0 if (abs(ev.finger) == 1 and is_black_key(ev.midi)) else 0.0
        strength_violation = float(ev.vel_on) if abs(ev.finger) in {4, 5} and ev.vel_on > velocity_threshold else 0.0
        phys.append(
            [
                stretch,
                crossing,
                hand_pos / 127.0,
                natural_violation,
                strength_violation / 127.0,
                chord_flags[i],
            ]
        )
    return np.asarray(phys, dtype=np.float32)


def basic_features(events: List[NoteEvent], chord_flags: List[int], chord_sizes: List[int]) -> np.ndarray:
    feats = []
    prev_onset = events[0].onset if events else 0.0
    for i, ev in enumerate(events):
        duration = ev.offset - ev.onset
        delta_onset = ev.onset - prev_onset if i > 0 else 0.0
        # context: previous/next same-hand midi deltas
        prev_same = next_same = ev.midi
        next_onset_gap = 0.0
        # previous same hand
        for j in range(i - 1, -1, -1):
            if events[j].channel == ev.channel:
                prev_same = events[j].midi
                break
        # next same hand
        for j in range(i + 1, len(events)):
            if events[j].channel == ev.channel:
                next_same = events[j].midi
                next_onset_gap = events[j].onset - ev.onset
                break
        beat_frac = ev.onset - math.floor(ev.onset)
        chord_span = 0
        same_hand_midis = [other.midi for other in events if other.channel == ev.channel and other.onset < ev.offset and ev.onset < other.offset]
        if same_hand_midis:
            chord_span = max(same_hand_midis) - min(same_hand_midis)
        feats.append(
            [
                ev.midi / 127.0,
                duration,
                delta_onset,
                ev.vel_on / 127.0,
                ev.vel_off / 127.0,
                ev.channel,
                is_black_key(ev.midi),
                chord_flags[i],
                chord_sizes[i],
                chord_span / 48.0,  # normalize by 4 octaves
                (ev.midi - prev_same) / 12.0,
                (next_same - ev.midi) / 12.0,
                next_onset_gap,
                beat_frac,
            ]
        )
        prev_onset = ev.onset
    return np.asarray(feats, dtype=np.float32)


class FeatureBuilder:
    def __init__(self, feature_type: str = "physical", word2vec_dim: int = 16, velocity_threshold: int = 80):
        self.feature_type = feature_type
        self.word2vec_dim = word2vec_dim
        self.velocity_threshold = velocity_threshold
        self.word2vec: Optional[Word2Vec] = None

    def fit_word2vec(self, pieces: Dict[str, List[NoteEvent]]):
        sentences = [[ev.pitch_str for ev in events] for events in pieces.values()]
        self.word2vec = Word2Vec(
            sentences,
            vector_size=self.word2vec_dim,
            window=5,
            min_count=1,
            workers=2,
            sg=1,
            epochs=30,
        )

    def _pitch_embedding(self, events: List[NoteEvent]) -> np.ndarray:
        if self.word2vec is None:
            return np.zeros((len(events), self.word2vec_dim), dtype=np.float32)
        embeds = []
        for ev in events:
            embeds.append(self.word2vec.wv.get_vector(ev.pitch_str))
        return np.asarray(embeds, dtype=np.float32)

    def build(
        self,
        events: List[NoteEvent],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        chord_flags, chord_sizes = detect_chords(events)
        base = basic_features(events, chord_flags, chord_sizes)
        phys = compute_physical_features(events, chord_flags, self.velocity_threshold)
        labels = np.array([FINGER_TO_IDX.get(ev.finger, FINGER_TO_IDX[0]) for ev in events], dtype=np.int64)

        if self.feature_type == "base":
            main_feats = base
            phys_feats = np.zeros_like(phys)
        elif self.feature_type == "word2vec":
            emb = self._pitch_embedding(events)
            main_feats = np.concatenate([base, emb], axis=1)
            phys_feats = np.zeros_like(phys)
        else:
            main_feats = np.concatenate([base, phys], axis=1)
            phys_feats = phys
        return main_feats, phys_feats, labels

