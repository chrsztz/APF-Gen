# Automatic Piano Fingering (PIG) – CNN-BiLSTM with Physical Constraints

PyTorch implementation following the attached plan. Uses PIG dataset at `./PIGdata` and exports MusicXML for web demo with OpenSheetMusicDisplay.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python -m src.train --config configs/default.yaml
```
Checkpoints: `outputs/checkpoints/best.pt`.

## Evaluate
```bash
python -m src.eval --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```

## Inference
```bash
python -m src.infer --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --input /home/ztz/Desktop/finger/PIGdata/FingeringFiles/001-1_fingering.txt \
  --xml-out outputs/musicxml/001-1.musicxml
```
The console prints predicted finger numbers. The MusicXML includes `<fingering>` tags for every note.

## Web demo (OSMD)
Open `web/osmd_demo.html` in a browser and load a MusicXML exported by `infer.py`. Finger numbers render above the staff.

## Notes
- Config at `configs/default.yaml` controls feature type (`base`, `word2vec`, `physical`), model sizes, and training hyperparameters.
- Word2Vec embeddings are trained on the PIG pitch sequences automatically when `feature_type=word2vec`.
- The provided evaluation metrics implement the plan’s matching and change-position rates; extend as needed for multi-label pieces.


