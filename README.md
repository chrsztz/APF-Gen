# APF-Gen — Automatic Piano Fingering

A PyTorch system for **automatic piano fingering** prediction using a CNN-BiLSTM model with physical-constraint losses and beam-search decoding. Given a sequence of note events, the model predicts which finger (1–5 per hand, encoded as **-5…+5**; positive = right hand, negative = left hand) should play each note.

The pipeline supports multiple input formats — PIG `.txt`, MusicXML, or MIDI — and exports MusicXML with `<fingering>` tags for rendering in any score viewer (a FastAPI + OpenSheetMusicDisplay web demo is included).

---

## Dataset & External Baselines

This repository does **not** redistribute the PIG dataset or the original baseline source code. Please download them from the official project page:

**https://beam.kisarazu.ac.jp/research/PianoFingeringDataset/**

- Extract the dataset into `./PIGdata/` (so that `PIGdata/FingeringFiles/*.txt` exists). See [`PIGdata/README.md`](PIGdata/README.md).
- The original CHMM/FHMM baseline sources, if you need them for comparison, go into `./SourceCode/`. See [`SourceCode/README.md`](SourceCode/README.md).

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Train

```bash
python -m src.train --config configs/default.yaml
```

TensorBoard logs are written to `runs/<arch>_<feature_type>_<timestamp>/`. The best checkpoint is saved to `outputs/checkpoints/best.pt`.

Switch the architecture or feature set via `configs/default.yaml`:

- `model.arch`: `cnn_bilstm` (default) · `transformer` · `arlstm` · `argnn`
- `data.feature_type`: `base` · `word2vec` · `physical`
- `model.phys_lambda`: weight of the physical-constraint loss head (0 disables it)
- `data.augment_shifts` / `data.augment_speeds`: pitch / tempo augmentation (training only)

A batch experiment runner is provided under `scripts/run_experiments.py`, and a collection of pre-trained ablation checkpoints lives in `outputs/checkpoints/` so you can reproduce results directly.

---

## Evaluate

```bash
python -m src.eval --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best.pt
```

Reports the following metrics on the val/test splits:

- **M_gen** — exact matching rate (%)
- **M_high** — best match across annotators (equals `M_gen` for single-annotator data)
- **M_soft** — soft matching with ±tol finger tolerance
- **M_cp** — change-position rate (ratio of predicted finger changes to ground-truth changes)

Beam-search decoding during evaluation can be toggled via `decoder.use_beam_eval` in the config.

---

## Inference

Accepts PIG `.txt`, MusicXML, or MIDI input:

```bash
python -m src.infer --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --input PIGdata/FingeringFiles/001-1_fingering.txt \
  --xml-out outputs/musicxml/001-1.musicxml \
  --xml-template mxl/001.musicxml \   # optional: reuse an existing score as template
  --beam --top-k 3 \                  # optional: beam search with physical transition costs
  --midi-split 60                     # optional: pitch split for MIDI hand-separation
```

- The console prints predicted finger numbers.
- The exported MusicXML embeds `<fingering>` tags on every note.
- Notes with tied durations in MusicXML templates are merged into a single event so they line up with PIG-style note rows.

---

## Web Demo (MIDI → Fingering → MusicXML)

```bash
uvicorn web.app:app --reload
```

Open `http://127.0.0.1:8000`, upload a MIDI file, and the score will be rendered with fingerings via OpenSheetMusicDisplay.

> **Note.** The web demo's bundled example presets reference files under `./PIGdata/FingeringFiles/`. Make sure you have downloaded the PIG dataset (see the [Dataset](#dataset--external-baselines) section above) before running the demo, otherwise the presets will 404.
>
> The checkpoint and config paths can be overridden via the env vars `PIG_CHECKPOINT` and `PIG_CONFIG`.

---

## Repository Layout

```
src/
├── data/         # PIG parser, feature builder, dataset / augmentation
├── models/       # cnn_bilstm, transformer, arlstm, argnn
├── utils/        # beam decoder, metrics, midi_io, musicxml_io
├── export/       # MusicXML writer with <fingering> tags
├── train.py      # training entry point (combined main + physical loss)
├── eval.py       # M_gen / M_high / M_soft / M_cp on val + test
└── infer.py      # CLI inference for PIG-txt / MusicXML / MIDI
web/              # FastAPI demo + OpenSheetMusicDisplay frontend
configs/          # YAML configs (see default.yaml)
scripts/          # batch experiment runner
mxl/              # MusicXML templates used by --xml-template
outputs/          # checkpoints (and inference artefacts)
```
