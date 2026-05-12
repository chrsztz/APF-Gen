# PIG Dataset

This directory is intentionally left empty. The PIG (Piano Fingering) dataset is **not** redistributed in this repository.

To use this project, please download the dataset from the official source:

**https://beam.kisarazu.ac.jp/research/PianoFingeringDataset/**

After downloading, extract the contents so the directory structure looks like:

```
PIGdata/
├── FingeringFiles/      # *.txt — tab-delimited note-event files
├── ScorePDF/            # PDF scores (optional)
├── List.csv
└── README.pdf
```

The training/evaluation pipeline expects `PIGdata/FingeringFiles/*.txt` (this path is configurable via `data.root` in `configs/default.yaml`).
