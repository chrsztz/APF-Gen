from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .parser import load_pig_dataset, split_pieces, NoteEvent
from .features import FeatureBuilder


class FingeringDataset(Dataset):
    def __init__(
        self,
        pieces: Dict[str, List[NoteEvent]],
        piece_ids: List[str],
        feature_builder: FeatureBuilder,
    ):
        self.items = []
        self.builder = feature_builder
        for pid in piece_ids:
            events = pieces[pid]
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
    train_ds = FingeringDataset(pieces, train_ids, builder)
    val_ds = FingeringDataset(pieces, val_ids, builder)
    test_ds = FingeringDataset(pieces, test_ids, builder)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    return train_loader, val_loader, test_loader, builder, pieces

