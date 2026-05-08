# Data

Pre-extracted features are **not** committed to this repository due to their size (~2.5 GB total). Place them here before running any experiments.

## Expected layout

```
IEMOCAP_features_2021/
├── A/
│   ├── comparE.h5              # 130-d OpenSMILE ComParE acoustic features
│   └── comparE_mean_std.h5     # Per-feature mean and std for normalisation
├── V/
│   └── denseface.h5            # 342-d DenseNet face embedding features
├── L/
│   └── bert_large.h5           # 1024-d BERT-large text features
└── target/
    └── {1..10}/                # Fold-level label files (10-fold CV)

MSP-IMPROV_features_2021/
├── A/
│   └── comparE_raw.h5
├── V/
│   └── denseface.h5
├── L/
│   └── bert_large.h5
└── target/
    └── {1..12}/
```

## Obtaining the features

Features are provided by the IF-MMIN authors alongside the IEMOCAP and MSP-IMPROV datasets:

- **IF-MMIN repo**: https://github.com/ZhuoYulang/IF-MMIN
- **IEMOCAP**: access request at https://sail.usc.edu/iemocap/
- **MSP-IMPROV**: access request at https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html

After obtaining access, download the pre-extracted `.h5` feature files from the authors and place them in the directories above.

## Config paths

Dataset configs are in `IF-MMIN/data/config/`. Update `feature_root` and `target_root` to match your local paths if they differ from the repository root layout.
