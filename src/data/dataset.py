import copy
from dataclasses import replace
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .parser import load_pig_dataset, split_pieces, NoteEvent
from .features import FeatureBuilder

# Piano range: A0 (21) to C8 (108)
PIANO_MIN = 21
PIANO_MAX = 108

# Match C++ PitchToSitch() convention: Eb (not D#), Bb (not A#)
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


def _midi_to_pitch_str(midi_num: int) -> str:
    octave = (midi_num // 12) - 1
    return f"{NOTE_NAMES[midi_num % 12]}{octave}"


def transpose_events(events: List[NoteEvent], semitones: int) -> List[NoteEvent]:
    """Transpose all events by *semitones*, dropping notes outside piano range."""
    out: List[NoteEvent] = []
    for ev in events:
        new_midi = ev.midi + semitones
        if new_midi < PIANO_MIN or new_midi > PIANO_MAX:
            continue
        out.append(replace(
            ev,
            midi=new_midi,
            pitch_str=_midi_to_pitch_str(new_midi),
        ))
    for i, ev in enumerate(out):
        ev.idx = i
    return out


def tempo_scale_events(events: List[NoteEvent], factor: float) -> List[NoteEvent]:
    """Scale all timing by *factor* (>1 = slower, <1 = faster).

    Pitch, channel, and finger labels stay the same.
    Only onset/offset are multiplied — this naturally changes
    duration, delta_onset, next_onset_gap, and beat_frac features.
    """
    out: List[NoteEvent] = []
    for ev in events:
        out.append(replace(
            ev,
            onset=ev.onset * factor,
            offset=ev.offset * factor,
        ))
    for i, ev in enumerate(out):
        ev.idx = i
    return out


class FingeringDataset(Dataset):
    def __init__(
        self,
        pieces: Dict[str, List[NoteEvent]],
        piece_ids: List[str],
        feature_builder: FeatureBuilder,
        augment_shifts: List[int] = (),
        augment_speeds: List[float] = (),
    ):
        """
        Parameters
        ----------
        augment_shifts : list of int
            Semitone shifts for pitch augmentation (e.g. [-12, 12]).
        augment_speeds : list of float
            Tempo scale factors (e.g. [0.8, 0.9, 1.1, 1.2]).
            1.0 is the original speed and is always included implicitly.
        Only use augmentation for the *training* set.
        """
        self.items = []
        self.builder = feature_builder

        for pid in piece_ids:
            events = pieces[pid]
            # original + pitch-shifted variants
            variants = [(events, pid)]
            for shift in augment_shifts:
                shifted = transpose_events(events, shift)
                if len(shifted) >= 4:
                    variants.append((shifted, f"{pid}_t{shift:+d}"))

            # for each pitch variant, create tempo-scaled copies
            for evs, tag in variants:
                self._add_piece(evs, tag)
                for speed in augment_speeds:
                    scaled = tempo_scale_events(evs, speed)
                    self._add_piece(scaled, f"{tag}_s{speed:.2f}")

    def _add_piece(self, events: List[NoteEvent], pid: str):
        main, phys, labels = self.builder.build(events)
        self.items.append(
            {
                "piece_id": pid,
                "main": torch.tensor(main, dtype=torch.float32),
                "phys": torch.tensor(phys, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
                "midi": torch.tensor([ev.midi for ev in events], dtype=torch.long),
                "channel": torch.tensor([ev.channel for ev in events], dtype=torch.long),
            }
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_batch(batch: List[Dict]):
    main_seqs = [b["main"] for b in batch]
    phys_seqs = [b["phys"] for b in batch]
    label_seqs = [b["labels"] for b in batch]
    midi_seqs = [b["midi"] for b in batch]
    channel_seqs = [b["channel"] for b in batch]
    lengths = torch.tensor([len(x) for x in main_seqs], dtype=torch.long)
    main_pad = pad_sequence(main_seqs, batch_first=True)
    phys_pad = pad_sequence(phys_seqs, batch_first=True)
    labels_pad = pad_sequence(label_seqs, batch_first=True, padding_value=-100)
    midi_pad = pad_sequence(midi_seqs, batch_first=True, padding_value=-1)
    channel_pad = pad_sequence(channel_seqs, batch_first=True, padding_value=-1)
    mask = torch.arange(main_pad.size(1))[None, :] < lengths[:, None]
    return {
        "main": main_pad,
        "phys": phys_pad,
        "labels": labels_pad,
        "midi": midi_pad,
        "channel": channel_pad,
        "mask": mask,
        "lengths": lengths,
        "piece_ids": [b["piece_id"] for b in batch],
    }


def create_dataloaders(
    root: str,
    feature_type: str,
    word2vec_dim: int,
    velocity_threshold: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
    augment_shifts: List[int] = None,
    augment_speeds: List[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, FeatureBuilder, Dict[str, List[NoteEvent]]]:
    pieces = load_pig_dataset(root)
    builder = FeatureBuilder(
        feature_type=feature_type,
        word2vec_dim=word2vec_dim,
        velocity_threshold=velocity_threshold,
        use_spatial=feature_type != "base",
        use_temporal=feature_type != "base",
        use_hand=feature_type != "base",
        use_fingering=feature_type != "base",
    )
    if feature_type == "word2vec":
        builder.fit_word2vec(pieces)
    train_ids, val_ids, test_ids = split_pieces(pieces, train_ratio, val_ratio, test_ratio, seed)

    shifts = augment_shifts if augment_shifts else []
    speeds = augment_speeds if augment_speeds else []
    train_ds = FingeringDataset(pieces, train_ids, builder, augment_shifts=shifts, augment_speeds=speeds)
    val_ds = FingeringDataset(pieces, val_ids, builder)
    test_ds = FingeringDataset(pieces, test_ids, builder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    return train_loader, val_loader, test_loader, builder, pieces
