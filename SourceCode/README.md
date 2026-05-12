# PIG Source Code (CHMM / FHMM baselines)

This directory is intentionally left empty. The original baseline source code (CHMM, FHMM1/2/3) shipped with the PIG dataset is **not** redistributed here.

To obtain the original baseline implementations, please download them from the official PIG project page:

**https://beam.kisarazu.ac.jp/research/PianoFingeringDataset/**

The original release includes:

```
SourceCode/
├── Code/                # C++ sources for CHMM / FHMM baselines
├── compile.sh
├── run_CHMM.sh
├── run_FHMM1.sh
├── run_FHMM2.sh
├── run_FHMM3.sh
└── README.txt
```

These baselines are independent of APF-Gen and are only needed if you wish to reproduce the original PIG paper's HMM results for comparison.
