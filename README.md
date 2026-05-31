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

---

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

# Install PyTorch with the right CUDA version for your machine — check with nvidia-smi
# Example for CUDA 13.2:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# Install all other dependencies
pip install -r IF-MMIN/requirements.txt
```

Then update `IF-MMIN/data/config/IEMOCAP_config.json` with the paths to your local feature files:

```json
{
  "feature_root": "/path/to/IEMOCAP_features_2021",
  "target_root":  "/path/to/IEMOCAP_features_2021/target"
}
```

See [`data/README.md`](data/README.md) for how to obtain the pre-extracted features.

---

## Running Experiments

All training is run from inside the `IF-MMIN/` directory.  
Arguments: `[num_of_expr]` = experiment index (e.g. `1`), `[GPU_index]` = GPU ID (e.g. `0`; use `-1` for CPU).


### Gender-bias evaluation pipeline (extended)

`train_miss_bias.py` is a drop-in replacement for `train_miss.py` that adds:

| Metric | Per condition | Per emotion |
|---|---|---|
| F1 difference by gender | ✓ | ✓ |
| Statistical Parity Difference | ✓ | ✓ |
| Equality of Opportunity Difference | ✓ | ✓ |
| Gender Predictability AUROC | ✓ (100-trial logistic regression probe) | — |

Evaluated across all **7 modality conditions**: `{A}`, `{V}`, `{T}`, `{A,V}`, `{A,T}`, `{V,T}`, `{A,V,T}`.

Stage 1 is the same — reuse the checkpoint from IF-MMIN. 

```bash
bash scripts/CAP_IFMMIN.sh [num_of_expr] [GPU_index]
```

**Outputs per fold** (in `logs/{name}/results/`):

| File | Contents |
|---|---|
| `result_total.tsv` … `result_zvl.tsv` | Standard WA / UAR / F1 (unchanged) |
| `result_avl.tsv` | WA / UAR / F1 for full-modality condition |
| `bias_{cond}.tsv` | F1 diff / SP / EO per emotion, per condition |
| `gender_auroc_cv{N}.csv` | AUROC mean / std per modality × condition |
| `gender_auroc_cv{N}.png` | Violin plot of AUROC distributions |

### Smoke test (single fold)

To verify the pipeline works before a full 10-fold run, edit both shell scripts to replace `seq 1 1 10` with `seq 1 1 1` and set `--niter=1 --niter_decay=1`, then run both stages. Training is complete when the terminal returns to a prompt and you see:

```
INFO - Tst result acc 0.XXXX uar 0.XXXX f1 0.XXXX
INFO - === Gender Predictability AUROC (mean ± std, 100 trials) ===
```

### Dataset exploration notebook

```bash
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

Covers label distribution, gender balance, emotion × gender cross-tabulation, and feature statistics across all 10 folds.

---

---

## License

The IF-MMIN codebase retains its original license (see [`IF-MMIN/LICENSE`](IF-MMIN/LICENSE)). Project-specific code in `scripts/` is MIT licensed.
