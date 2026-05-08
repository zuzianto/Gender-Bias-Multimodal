# Gender Bias in Multimodal Emotion Recognition

Investigating gender bias in **IF-MMIN** (Zuo et al., 2022), a multimodal emotion recognition model trained on IEMOCAP. The project compares model behaviour under (1) the natural data distribution and (2) a gender-balanced resampling, and measures gender fairness across all 7 modality-availability combinations the model supports.

---

## Research Questions

- Does training on a gender-balanced dataset reduce gender-correlated disparities in emotion recognition performance?
- Do fairness gaps persist even when only a subset of modalities is available at inference time?
- Are gender-correlated patterns encoded in the model's learned representations?

---

## Approach

### Model

[IF-MMIN](https://github.com/ZhuoYulang/IF-MMIN) integrates three modalities through a two-stage training pipeline:

| Modality | Feature | Dimensionality |
|---|---|---|
| Acoustic (A) | OpenSMILE ComParE | 130-d |
| Visual (V) | DenseNet face embeddings | 342-d |
| Textual (T) | BERT-large | 1024-d |

**Stage 1** — CMD-based invariant feature learning (cross-modal disentanglement).  
**Stage 2** — Full IF-MMIN training with missing-modality imagination modules.

### Training Conditions

| Condition | Description |
|---|---|
| **Natural** | Original IEMOCAP label distribution |
| **Balanced** | Resampled so each emotion label is equally distributed across male and female speakers |

Both conditions use 5-fold cross-validation, 40 epochs per fold, batch size 128.

### Evaluation

Because IF-MMIN handles missing modalities via feature imagination, a single trained model covers all 7 modality subsets without retraining:

`{A}`, `{V}`, `{T}`, `{A,V}`, `{A,T}`, `{V,T}`, `{A,V,T}`

**2 training conditions × 7 modality subsets = 14 evaluation configurations.**

### Fairness Metrics

For each configuration:

- **F1 gap** — F1-score difference between male and female speakers
- **Statistical Parity Difference (SPD)** — difference in the probability of predicting each emotion across genders
- **Equality of Opportunity Difference (EoOD)** — whether the model is equally accurate for both genders conditioned on the true label
- **Gender probing AUROC** — logistic regression probe on learned representations measuring how linearly decodable gender is

---

## Repository Structure

```
Gender-Bias-Multimodal/
├── IF-MMIN/                    # Model codebase (Zuo et al., 2022)
│   ├── data/                   # Dataset loaders and configs
│   ├── models/                 # Model definitions (IFMMIN, MMIN, MISA, …)
│   ├── scripts/                # Original training shell scripts
│   ├── opts/                   # Argument parsing
│   ├── utils/                  # Logging and helpers
│   ├── train_baseline.py       # Stage 1 training entry point
│   └── train_miss.py           # Stage 2 training entry point
├── scripts/                    # Project-specific training and analysis scripts
├── data/
│   ├── IEMOCAP_features_2021/  # Pre-extracted features (not in git — see below)
│   └── MSP-IMPROV_features_2021/
├── results/                    # Experiment outputs (not in git)
└── README.md
```

> **Note:** Large feature files (`.h5`) are excluded from version control. See [`data/README.md`](data/README.md) for instructions on obtaining them.

---

## Data

Pre-extracted features are provided by the IF-MMIN authors and expected at:

```
IEMOCAP_features_2021/
├── A/comparE.h5               # 130-d OpenSMILE acoustic features
├── A/comparE_mean_std.h5      # Normalisation statistics
├── V/denseface.h5             # 342-d DenseNet visual features
├── L/bert_large.h5            # 1024-d BERT-large text features
└── target/                    # Fold-level label files (1–10)

MSP-IMPROV_features_2021/
├── A/comparE_raw.h5
├── V/denseface.h5
├── L/bert_large.h5
└── target/
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/zuzianto/Gender-Bias-Multimodal.git
cd Gender-Bias-Multimodal

# Install dependencies (Python 3.8+)
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn h5py

# Place pre-extracted features in IEMOCAP_features_2021/ and MSP-IMPROV_features_2021/
# Update data/config paths in IF-MMIN/data/config/ if needed
```

---

## Running Experiments

Training scripts are in `IF-MMIN/scripts/`. The main IF-MMIN pipeline for IEMOCAP:

```bash
# Stage 1: CMD baseline (invariant feature learning)
bash IF-MMIN/scripts/CAP_data_aug.sh

# Stage 2: Full IF-MMIN training
bash IF-MMIN/scripts/CAP_IFMMIN.sh
```

Project-specific scripts for balanced training and fairness evaluation will live in `scripts/`.

---

## References

- Zuo, H., Li, R., Liu, Z., Wu, Z., Meng, H., & Cai, L. (2022). *Exploiting modality-invariant feature for robust multimodal emotion recognition with missing modalities*. ICASSP 2022. [[Paper]](https://arxiv.org/abs/2210.15359) [[Code]](https://github.com/ZhuoYulang/IF-MMIN)
- Busso, C. et al. (2008). *IEMOCAP: Interactive emotional dyadic motion capture database*. Language Resources and Evaluation.

---

## License

The IF-MMIN codebase retains its original license (see [`IF-MMIN/LICENSE`](IF-MMIN/LICENSE)). Project-specific code in `scripts/` is MIT licensed.
