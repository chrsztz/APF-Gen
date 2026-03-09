# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

APF-Gen is an **Automatic Piano Fingering** system using a CNN-BiLSTM model with physical constraints. It predicts which finger (1–5 for each hand, encoded as -5 to +5) should play each note. Positive fingers = right hand, negative = left hand. The 11 classes are: `[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]`.

Data: **PIG dataset** in `./PIGdata/FingeringFiles/` — tab-delimited `.txt` files where each row is a note event (idx, onset, offset, pitch, vel_on, vel_off, channel, finger).

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train
python -m src.train --config configs/default.yaml

# Evaluate (val + test metrics, optional confusion matrix + error CSV)
python -m src.eval --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt

# Inference (PIG txt, MusicXML, or MIDI input)
python -m src.infer --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --input PIGdata/FingeringFiles/001-1_fingering.txt \
  --xml-out outputs/musicxml/001-1.musicxml \
  [--xml-template mxl/001.musicxml] \
  [--beam] [--top-k 3] [--midi-split 60]

# Web demo (FastAPI + OpenSheetMusicDisplay)
uvicorn web.app:app --reload
# Open http://127.0.0.1:8000, upload a MIDI file
```

TensorBoard logs go to `runs/<arch>_<feature_type>_<timestamp>/`. Best checkpoint saved to `outputs/checkpoints/best.pt`.

## Architecture

### Data pipeline (`src/data/`)
- **`parser.py`**: Parses PIG `.txt` files into `NoteEvent` dataclasses. `channel=0` → right hand, `channel=1` → left hand.
- **`features.py`**: `FeatureBuilder` produces two feature arrays per piece:
  - `main_feats`: 12-dim basic features (midi, duration, delta_onset, chord info, interval context, beat_frac) + optional physical features + optional Word2Vec pitch embeddings
  - `phys_feats`: 5-dim physical constraint features (stretch, crossing, hand_pos, natural_violation, chord_flag) — subset controlled by `use_spatial/temporal/hand/fingering` flags
- **`dataset.py`**: `FingeringDataset` with pitch transposition (+/- semitones) and tempo scaling augmentation (train only). `collate_batch` pads variable-length sequences; padded labels use `-100` (ignored by CrossEntropyLoss).

### Models (`src/models/`)
All models share the same `forward(main_feats, phys_feats, mask) → (main_logits, phys_logits, attn_weights)` interface:
- **`FingeringModel`** (`model.py`): Primary model — CNN (1D conv + BN + LeakyReLU) → BiLSTM → optional attention → `main_head`. Separate `phys_head` operates directly on physical features.
- **`TransformerFingering`** (`transformer_model.py`): Input projection → sinusoidal PE → TransformerEncoder → `main_head` + `phys_head`.
- **`ArLSTM`** / **`ArGNN`** (`ar_models.py`): Autoregressive LSTM variant; ArGNN is Conv+BiLSTM with a distinct name for experiment tagging.

### Loss & training (`src/train.py`)
Combined loss: `(1 - phys_lambda) * CE(main_logits) + phys_lambda * CE(phys_logits)`. Both heads trained against the same ground-truth labels. `phys_lambda=0.3` by default.

### Decoding (`src/utils/decoder.py`)
Beam search with physical transition costs: `alpha * stretch + beta * crossing`. Hand-channel constraints filter candidate fingers (right hand: positive fingers only; left hand: negative only).

### Metrics (`src/utils/metrics.py`)
- `M_gen`: exact matching rate (%)
- `M_high`: best match across multiple annotators (same as M_gen for single annotator)
- `M_soft`: soft matching allowing ±tol finger difference
- `M_cp`: change-position rate (ratio of predicted finger changes to ground-truth changes)

### Export & Web (`src/export/`, `web/`)
- `src/export/musicxml.py`: Converts note events + finger predictions to MusicXML with `<fingering>` tags.
- `src/utils/musicxml_io.py`: Reads/writes fingerings to existing MusicXML files.
- `src/utils/midi_io.py`: Parses MIDI to `NoteEvent` list, splitting hands by pitch threshold (`--midi-split`, default 60).
- `web/app.py`: FastAPI app. Config/checkpoint paths can be overridden via env vars `PIG_CONFIG` and `PIG_CHECKPOINT`.

## Configuration (`configs/default.yaml`)

Key settings:
- `data.feature_type`: `"base"` | `"word2vec"` | `"physical"` — controls which features are used
- `model.arch`: `"cnn_bilstm"` (default) | `"transformer"` | `"arlstm"` | `"argnn"`
- `model.phys_lambda`: weight of physical constraint loss (0 = ignore physics)
- `train.device`: `"mps"` for Apple Silicon; falls back to CPU if CUDA unavailable
- `decoder.use_beam_eval`: whether to use beam search during evaluation
- `data.augment_shifts` / `augment_speeds`: pitch/tempo augmentation applied to training set only
