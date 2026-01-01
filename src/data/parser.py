import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class NoteEvent:
    idx: int
    onset: float
    offset: float
    pitch_str: str
    midi: int
    vel_on: int
    vel_off: int
    channel: int
    finger: int


PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
    "B#": 0,
}


def pitch_to_midi(pitch: str) -> int:
    """
    Convert spelled pitch like C4, F#3, Bb5 to MIDI number.
    """
    if not pitch:
        return -1
    name = pitch[:-1]
    octave_str = pitch[-1]
    if octave_str not in "0123456789":
        # fallback: treat last char as octave even if non-digit
        octave_str = pitch[-1]
        name = pitch[:-1]
    try:
        octave = int(octave_str)
    except ValueError:
        return -1
    semitone = PITCH_CLASS.get(name)
    if semitone is None:
        # allow odd cases like double-sharp by stripping trailing symbols
        base = name[0]
        accidental = name[1:]
        semitone = PITCH_CLASS.get(base, 0)
        semitone = (semitone + accidental.count("#") - accidental.count("b")) % 12
    return semitone + 12 * (octave + 1)


def parse_finger(value: str) -> int:
    """
    Convert finger string like '4_1' or '-5' to integer label.
    """
    if value is None:
        return 0
    cleaned = str(value).split("_")[0]
    try:
        return int(cleaned)
    except ValueError:
        return 0


def parse_fingering_file(path: str) -> Tuple[str, List[NoteEvent]]:
    """
    Parse a single PIG fingering text file.
    Returns (piece_id, list of NoteEvent).
    """
    piece_id = os.path.basename(path).replace("_fingering.txt", "").replace(".txt", "")
    events: List[NoteEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("//"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 8:
                parts = line.strip().split()
            if len(parts) < 8:
                continue
            idx = int(parts[0])
            onset = float(parts[1])
            offset = float(parts[2])
            pitch = parts[3]
            vel_on = int(float(parts[4]))
            vel_off = int(float(parts[5]))
            channel = int(parts[6])
            finger = parse_finger(parts[7])
            events.append(
                NoteEvent(
                    idx=idx,
                    onset=onset,
                    offset=offset,
                    pitch_str=pitch,
                    midi=pitch_to_midi(pitch),
                    vel_on=vel_on,
                    vel_off=vel_off,
                    channel=channel,
                    finger=finger,
                )
            )
    events.sort(key=lambda x: x.onset)
    return piece_id, events


def load_pig_dataset(root: str) -> Dict[str, List[NoteEvent]]:
    """
    Load all fingering files into a dict mapping piece_id -> events.
    """
    dataset: Dict[str, List[NoteEvent]] = {}
    for fname in os.listdir(root):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(root, fname)
        piece_id, events = parse_fingering_file(path)
        dataset[piece_id] = events
    return dataset


def split_pieces(
    pieces: Dict[str, List[NoteEvent]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split piece ids into train/val/test lists.
    """
    piece_ids = list(pieces.keys())
    random.Random(seed).shuffle(piece_ids)
    n = len(piece_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = piece_ids[:n_train]
    val_ids = piece_ids[n_train : n_train + n_val]
    test_ids = piece_ids[n_train + n_val :]
    return train_ids, val_ids, test_ids




