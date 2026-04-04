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


def estimate_seconds_per_beat(events: List[NoteEvent]) -> float:
    """Estimate seconds-per-beat from note onset times.

    Uses the median inter-onset interval (IOI) across all unique onset times,
    filtering out intervals that are too short (chord voicing, < 0.05 s) or
    too long (rests, > 3.0 s).  Falls back to 0.5 s (120 BPM quarter note).
    """
    onsets = sorted({round(ev.onset, 4) for ev in events})
    iois = [b - a for a, b in zip(onsets, onsets[1:]) if 0.05 < b - a < 3.0]
    if not iois:
        return 0.5
    return float(np.median(iois))


def detect_chords(events: List[NoteEvent]) -> Tuple[List[int], List[int]]:
    """Count same-hand notes active at each note's onset (not duration overlap)."""
    flags = [0] * len(events)
    sizes = [1] * len(events)
    for i, ev in enumerate(events):
        active = sum(
            1 for other in events
            if other.channel == ev.channel and other.onset <= ev.onset < other.offset
        )
        sizes[i] = active          # includes self
        flags[i] = 1 if active > 1 else 0
    return flags, sizes


def compute_physical_features(
    events: List[NoteEvent],
    chord_flags: List[int],
) -> np.ndarray:
    """Physical features with genuine information beyond the base feature set.

    All features are computable at inference time (no ground-truth finger labels).

    Columns:
      hand_pos          : mean MIDI of same-hand notes active at onset / 127
                          (≠ ev.midi/127 for chords; reflects where the hand sits)
      hand_pos_delta    : change in hand_pos from previous same-hand onset, in octaves,
                          clipped to [-2, 2]  (captures position jumps / shifts)
      within_chord_rank : normalized rank of this note among simultaneous same-hand notes,
                          0.0 = lowest pitch, 1.0 = highest pitch  (0.0 when alone)
                          directly signals which finger tier is likely needed
      note_is_top       : 1 if this is the highest simultaneous same-hand note
      note_is_bottom    : 1 if this is the lowest simultaneous same-hand note
    """
    prev_hand_pos: dict = {}   # channel -> hand_pos at previous onset
    phys = []

    for i, ev in enumerate(events):
        active_pitches = sorted(
            {ev.midi} | {
                other.midi for other in events
                if other.channel == ev.channel and other.onset <= ev.onset < other.offset
            }
        )
        # active_pitches always includes ev itself (ev.onset <= ev.onset < ev.offset)
        n = len(active_pitches)
        hand_pos = float(np.mean(active_pitches)) if active_pitches else float(ev.midi)

        # Rank within chord: 0 = lowest, 1 = highest
        if n > 1:
            rank = active_pitches.index(ev.midi)
            within_chord_rank = rank / (n - 1)
        else:
            within_chord_rank = 0.0

        note_is_top    = 1.0 if ev.midi == active_pitches[-1] else 0.0
        note_is_bottom = 1.0 if ev.midi == active_pitches[0]  else 0.0

        last_pos = prev_hand_pos.get(ev.channel, hand_pos)
        hand_pos_delta = float(np.clip((hand_pos - last_pos) / 12.0, -2.0, 2.0))
        prev_hand_pos[ev.channel] = hand_pos

        phys.append([
            hand_pos / 127.0,
            hand_pos_delta,
            within_chord_rank,
            note_is_top,
            note_is_bottom,
        ])
    return np.asarray(phys, dtype=np.float32)


def basic_features(
    events: List[NoteEvent],
    chord_flags: List[int],
    chord_sizes: List[int],
    spb: float = 0.5,
) -> np.ndarray:
    """Build basic per-note feature matrix.

    Timing features (duration, delta_onset, next_onset_gap) are normalized by
    `spb` (seconds-per-beat) so values are in beats rather than raw seconds.
    This makes the model tempo-invariant across different MIDI files.
    beat_frac is the position within the beat [0, 1), computed from onset/spb.
    """
    feats = []
    prev_onset = events[0].onset if events else 0.0
    for i, ev in enumerate(events):
        duration    = np.clip((ev.offset - ev.onset) / spb, 0.0, 8.0)
        delta_onset = np.clip((ev.onset - prev_onset) / spb if i > 0 else 0.0, 0.0, 8.0)
        prev_same = next_same = ev.midi
        next_onset_gap = 0.0
        for j in range(i - 1, -1, -1):
            if events[j].channel == ev.channel:
                prev_same = events[j].midi
                break
        for j in range(i + 1, len(events)):
            if events[j].channel == ev.channel:
                next_same = events[j].midi
                next_onset_gap = np.clip((events[j].onset - ev.onset) / spb, 0.0, 8.0)
                break
        beat_frac = (ev.onset / spb) % 1.0  # position within beat [0, 1)
        # Notes active at this note's onset (consistent with detect_chords fix)
        same_hand_midis = [
            other.midi for other in events
            if other.channel == ev.channel and other.onset <= ev.onset < other.offset
        ]
        chord_span = max(same_hand_midis) - min(same_hand_midis) if same_hand_midis else 0
        prev_interval = ev.midi - prev_same   # signed semitones to prev same-hand note
        next_interval = next_same - ev.midi   # signed semitones to next same-hand note
        feats.append(
            [
                ev.midi / 127.0,
                duration,
                delta_onset,
                ev.channel,
                is_black_key(ev.midi),
                chord_flags[i],
                chord_sizes[i],
                chord_span / 48.0,
                prev_interval / 12.0,                          # signed delta pitch (prev)
                float(np.sign(prev_interval)),                 # direction: -1/0/+1
                1.0 if 0 < abs(prev_interval) <= 2 else 0.0,  # is_step (semitone or tone)
                next_interval / 12.0,                          # signed delta pitch (next)
                float(np.sign(next_interval)),                 # direction: -1/0/+1
                1.0 if 0 < abs(next_interval) <= 2 else 0.0,  # is_step (semitone or tone)
                next_onset_gap,
                beat_frac,
            ]
        )
        prev_onset = ev.onset
    return np.asarray(feats, dtype=np.float32)


class FeatureBuilder:
    def __init__(
        self,
        feature_type: str = "physical",
        word2vec_dim: int = 16,
        velocity_threshold: int = 80,
        use_spatial: bool = True,
        use_temporal: bool = True,
        use_hand: bool = True,
        use_fingering: bool = True,
    ):
        self.feature_type = feature_type
        self.word2vec_dim = word2vec_dim
        self.velocity_threshold = velocity_threshold  # kept for config compat
        self.word2vec: Optional[Word2Vec] = None
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal
        self.use_hand = use_hand
        self.use_fingering = use_fingering

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
        spb = estimate_seconds_per_beat(events)
        base_full = basic_features(events, chord_flags, chord_sizes, spb=spb)
        phys_full = compute_physical_features(events, chord_flags)
        # phys_full columns: [hand_pos, hand_pos_delta, within_chord_rank, note_is_top, note_is_bottom]
        phys_selected = phys_full

        labels = np.array([FINGER_TO_IDX.get(ev.finger, FINGER_TO_IDX[0]) for ev in events], dtype=np.int64)

        if self.feature_type == "base":
            main_feats = base_full
            phys_feats = np.zeros_like(phys_selected)
        elif self.feature_type == "word2vec":
            emb = self._pitch_embedding(events)
            main_feats = np.concatenate([base_full, emb], axis=1)
            phys_feats = np.zeros_like(phys_selected)
        else:  # physical variants
            main_feats = np.concatenate([base_full, phys_selected], axis=1)
            phys_feats = phys_selected
        return main_feats, phys_feats, labels

