# IF-MMIN

Implementation of the Invariant Feature aware Missing Modality Imagination Network (IF-MMIN) from:

> Zuo et al. (2022) — *"Exploiting Modality-Invariant Feature for Robust Multimodal Emotion Recognition with Missing Modalities"*  
> ICASSP 2022 · [arXiv:2210.15359](https://arxiv.org/abs/2210.15359)

This fork extends the original codebase with gender-bias evaluation metrics.

---

## Environment

Python 3.8+ · PyTorch ≥ 1.0

Install all dependencies:

```bash
# 1. Check your CUDA version (on a GPU machine)
nvidia-smi

# 2. Install PyTorch with the matching CUDA version — examples:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8

# 3. Install everything else
pip install -r requirements.txt
```

---

## Data setup

1. Update the paths in `data/config/IEMOCAP_config.json` to point to your local feature files:

```json
{
  "feature_root": "/path/to/IEMOCAP_features_2021",
  "target_root":  "/path/to/IEMOCAP_features_2021/target"
}
```

2. Download pre-extracted features from the IF-MMIN authors:  
   https://drive.google.com/drive/folders/18nTA5LpTGqE_pRwhnDWN0o4bo-b0lkGt

   Or, if running on a machine with internet access, download directly:

   ```bash
   pip install gdown
   gdown --folder "https://drive.google.com/drive/folders/18nTA5LpTGqE_pRwhnDWN0o4bo-b0lkGt"
   ```

---

## Training — original pipeline (`train_miss.py`)

Training is a two-stage process. All scripts loop over 10 folds by default.  
Pass `[num_of_expr]` as an experiment index (e.g. `1`) and `[GPU_index]` as the GPU to use (e.g. `0`; use `-1` for CPU).

**Stage 1 — invariant feature pretraining**

```bash
bash scripts/CAP_utt_shared.sh AVL [num_of_expr] [GPU_index]
```

This trains the CMD-based specificity + invariance encoders on all three modalities (A, V, L) and saves checkpoints to `checkpoints/CAP_utt_shared_AVL_run{num_of_expr}/`.

**Stage 2 — full IF-MMIN training**

```bash
bash scripts/CAP_IFMMIN.sh [num_of_expr] [GPU_index]
```

> **Known bug in the original script:** `CAP_IFMMIN.sh` passes `--consistent_weight=100` which is not a recognised argument. Change this to `--invariant_weight=100` before running.

Stage 2 loads the Stage 1 checkpoint as a frozen pretrained encoder and trains the Invariant Feature aware Imagination Module (IF-IM).

**Outputs** (written to `logs/{name}/results/` after all 10 folds):

| File | Contents |
|---|---|
| `result_total.tsv` | Overall WA / UAR / F1 across all missing conditions |
| `result_azz.tsv` … `result_zvl.tsv` | Per-condition WA / UAR / F1 (6 conditions) |

---

## Evaluation — gender-bias pipeline (`train_miss_bias.py`)

`train_miss_bias.py` is a drop-in replacement for `train_miss.py` that adds three gender-fairness metrics and a gender predictability AUROC evaluation.

### What it adds

| Metric | Description |
|---|---|
| **F1 difference by gender** | \|F1_male − F1_female\| per emotion class |
| **Statistical Parity (SP)** | P(ŷ=c \| male) − P(ŷ=c \| female) per emotion |
| **Equality of Opportunity (EO)** | TPR_male − TPR_female per emotion |
| **Gender predictability AUROC** | How well a logistic regression predicts speaker gender from model embeddings (100 stratified trials per modality × condition) |

All metrics are computed for **7 modality conditions**: the 6 missing-modality conditions (`azz`, `zvz`, `zzl`, `avz`, `azl`, `zvl`) plus the full-modality condition (`avl`).

Gender is inferred from the IEMOCAP utterance ID trailing token (`_M…` = male, `_F…` = female).

### Running the bias evaluation

Replace `train_miss.py` with `train_miss_bias.py` in the shell scripts, or run directly:

**Stage 1** is identical — run it once and reuse the checkpoint:

```bash
bash scripts/CAP_utt_shared.sh AVL [num_of_expr] [GPU_index]
```

**Stage 2 with bias metrics:**

```bash
# Edit CAP_IFMMIN.sh to call train_miss_bias.py instead of train_miss.py,
# and fix --consistent_weight=100  →  --invariant_weight=100
# Then:
bash scripts/CAP_IFMMIN.sh [num_of_expr] [GPU_index]
```

Or call the script directly for a single fold (e.g. fold 1, GPU 0):

```bash
python train_miss_bias.py \
  --dataset_mode=multimodal_miss --model=IFMMIN \
  --log_dir=./logs --checkpoints_dir=./checkpoints --gpu_ids=0 --image_dir=./shared_image \
  --A_type=comparE  --input_dim_a=130  --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool \
  --V_type=denseface --input_dim_v=342 --embd_size_v=128 --embd_method_v=maxpool \
  --L_type=bert_large --input_dim_l=1024 --embd_size_l=128 \
  --AE_layers=256,128,64 --n_blocks=5 --num_threads=0 --corpus_name=IEMOCAP \
  --ce_weight=1 --mse_weight=1 --invariant_weight=100 \
  --output_dim=4 --cls_layers=128,128 --dropout_rate=0.5 \
  --niter=20 --niter_decay=20 --verbose --print_freq=10 \
  --batch_size=128 --lr=2e-4 --run_idx=1 --weight_decay=1e-5 \
  --name=our_IEMOCAP_bias --suffix=block_5_run_0_1 --has_test \
  --pretrained_path=checkpoints/CAP_utt_shared_AVL_run1 \
  --cvNo=1 --num_classes=4
```

### Outputs

In addition to the standard result files, each fold writes:

| File | Contents |
|---|---|
| `logs/{name}/results/bias_{cond}.tsv` | F1 diff, SP, EO per emotion for each condition |
| `logs/{name}/results/result_avl.tsv` | WA / UAR / F1 for full-modality condition |
| `logs/{name}/results/gender_auroc_cv{N}.csv` | Mean / std AUROC per modality × condition |
| `checkpoints/{name}/{fold}/gender_auroc_cv{N}.png` | Violin plot of AUROC distributions |

---

## Smoke test (single fold, quick check)

To verify the pipeline runs end to end before committing to a full 10-fold run, edit both shell scripts to loop over fold 1 only (`seq 1 1 1` instead of `seq 1 1 10`) and reduce epochs:

```bash
# Stage 1 — fold 1 only
bash scripts/CAP_utt_shared.sh AVL 1 0

# Stage 2 with bias eval — fold 1 only
bash scripts/CAP_IFMMIN.sh 1 0   # after editing to use train_miss_bias.py
```

Training is complete when the terminal returns to a prompt and you see:

```
INFO - Tst result acc 0.XXXX uar 0.XXXX f1 0.XXXX
INFO - === Gender Predictability AUROC (mean ± std, 100 trials) ===
```

---

## Hyperparameters

All arguments are defined in `opts/get_opts.py` and the `modify_commandline_options` method of each model. Key parameters for IF-MMIN:

| Argument | Default | Description |
|---|---|---|
| `--niter` | 20 | Epochs at initial learning rate |
| `--niter_decay` | 20 | Epochs to linearly decay lr to 0 |
| `--batch_size` | 128 | Batch size |
| `--lr` | 2e-4 | Initial Adam learning rate |
| `--ce_weight` | 1 | Weight of classification loss |
| `--mse_weight` | 1 | Weight of imagination loss (Limg) |
| `--invariant_weight` | 100 | Weight of invariance loss (Linv) |
| `--n_blocks` | 5 | Number of autoencoder blocks in IF-IM |
| `--AE_layers` | 256,128,64 | Hidden layer sizes of each AE block |
