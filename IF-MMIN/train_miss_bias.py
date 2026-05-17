"""
train_miss_bias.py — copy of train_miss.py extended with gender-bias evaluation.

Three new metrics computed per missing-modality condition (6 missing + 1 full = 7):
  1. F1 Score Difference by Gender      — |F1_male - F1_female| per emotion
  2. Statistical Parity Difference (SP) — P(ŷ=c|male) - P(ŷ=c|female) per emotion
  3. Equality of Opportunity (EO)       — TPR_male - TPR_female per emotion

Results are written to bias_{condition}.tsv files alongside the existing result_*.tsv files.
Gender is inferred from the trailing token of each IEMOCAP utterance ID
(e.g. 'Ses02F_script03_1_M001' → male, 'Ses05F_impro04_F041' → female).
"""
import os
import time
import fcntl
import numpy as np
from opts.get_opts import Options
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder, LossRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import random

from sklearn.manifold import TSNE
import matplotlib as mpl
import pickle

mpl.use('Agg')
import matplotlib.pyplot as plt

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# ─── constants ────────────────────────────────────────────────────────────────

EMOTIONS   = ['happy', 'angry', 'sad', 'neutral']
# 6 missing conditions that already exist in the dataset + 1 full-modality condition
ALL_CONDITIONS = ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl', 'avl']


# ─── gender helper ────────────────────────────────────────────────────────────

def get_gender(utt_id: str) -> str:
    """Return 'M' or 'F' from an IEMOCAP utterance ID.

    The speaker gender is encoded in the LAST token of the utterance ID:
        'Ses02F_script03_1_M001' → 'M'   (male speaker)
        'Ses05F_impro04_F041'    → 'F'   (female speaker)

    Note: the session letter (e.g. 'F' in 'Ses02F') encodes the *scripted
    character* gender, not the actual speaker — always use the trailing token.
    """
    return utt_id.split('_')[-1][0]


# ─── bias metrics ─────────────────────────────────────────────────────────────

def compute_bias_metrics(preds, labels, genders):
    """Compute all three gender-bias metrics for one set of predictions.

    Parameters
    ----------
    preds   : array-like (N,) — model class predictions (int 0–3)
    labels  : array-like (N,) — ground-truth labels      (int 0–3)
    genders : array-like (N,) — 'M' or 'F' per sample

    Returns
    -------
    dict with keys:
        f1_m        (4,) per-emotion F1 for male speakers
        f1_f        (4,) per-emotion F1 for female speakers
        f1_diff     (4,) |f1_m - f1_f|
        sp          (4,) P(ŷ=c|male) - P(ŷ=c|female) per emotion
        sp_abs_mean     mean |SP| across emotions
        eo          (4,) TPR_male - TPR_female per emotion
        eo_abs_mean     mean |EO| across emotions
    """
    preds   = np.array(preds)
    labels  = np.array(labels)
    genders = np.array(genders)

    n_classes    = len(EMOTIONS)
    labels_range = list(range(n_classes))

    male_mask   = genders == 'M'
    female_mask = genders == 'F'
    preds_m,  labels_m = preds[male_mask],  labels[male_mask]
    preds_f,  labels_f = preds[female_mask], labels[female_mask]

    # 1. F1 Score Difference by Gender ----------------------------------------
    f1_m    = f1_score(labels_m, preds_m, average=None,
                       labels=labels_range, zero_division=0)
    f1_f    = f1_score(labels_f, preds_f, average=None,
                       labels=labels_range, zero_division=0)
    f1_diff = np.abs(f1_m - f1_f)

    # 2. Statistical Parity: SP_c = P(ŷ=c | male) − P(ŷ=c | female) ----------
    sp = np.zeros(n_classes)
    for c in range(n_classes):
        rate_m = np.mean(preds_m == c) if len(preds_m) > 0 else 0.0
        rate_f = np.mean(preds_f == c) if len(preds_f) > 0 else 0.0
        sp[c]  = rate_m - rate_f

    # 3. Equality of Opportunity: EO_c = TPR_male − TPR_female ----------------
    eo = np.zeros(n_classes)
    for c in range(n_classes):
        m_positive = labels_m == c
        f_positive = labels_f == c
        tpr_m = np.mean(preds_m[m_positive] == c) if m_positive.sum() > 0 else 0.0
        tpr_f = np.mean(preds_f[f_positive] == c) if f_positive.sum() > 0 else 0.0
        eo[c] = tpr_m - tpr_f

    return {
        'f1_m':        f1_m,
        'f1_f':        f1_f,
        'f1_diff':     f1_diff,
        'sp':          sp,
        'sp_abs_mean': float(np.mean(np.abs(sp))),
        'eo':          eo,
        'eo_abs_mean': float(np.mean(np.abs(eo))),
    }


# ─── bias result recorder ─────────────────────────────────────────────────────

class BiasResultRecorder:
    """Writes per-fold gender-bias metrics to a TSV file.

    Columns (22 total):
        For each emotion: f1_m, f1_f, f1_diff, sp, eo  (5 × 4 = 20)
        Summary:          sp_abs_mean, eo_abs_mean        (2)

    Appends a mean row once all folds have been written (same behaviour
    as ResultRecorder in utils/logger.py).
    """

    _COLS = []
    for _e in EMOTIONS:
        _COLS += [f'f1_m_{_e}', f'f1_f_{_e}', f'f1_diff_{_e}', f'sp_{_e}', f'eo_{_e}']
    _COLS += ['sp_abs_mean', 'eo_abs_mean']

    def __init__(self, path, total_cv=10):
        self.path     = path
        self.total_cv = total_cv
        if not os.path.exists(path):
            with open(path, 'w') as fh:
                fh.write('\t'.join(self._COLS) + '\n')

    def write_result_to_tsv(self, results, cvNo):
        f_in = open(self.path)
        fcntl.flock(f_in.fileno(), fcntl.LOCK_EX)
        content = f_in.readlines()
        if len(content) < self.total_cv + 1:
            content += ['\n'] * (self.total_cv - len(content) + 1)

        vals = []
        for i in range(len(EMOTIONS)):
            vals += [
                results['f1_m'][i],
                results['f1_f'][i],
                results['f1_diff'][i],
                results['sp'][i],
                results['eo'][i],
            ]
        vals += [results['sp_abs_mean'], results['eo_abs_mean']]
        content[cvNo] = '\t'.join(f'{v:.4f}' for v in vals) + '\n'

        if len(content) >= self.total_cv + 1:
            try:
                rows  = [list(map(float, content[i].split('\t')))
                         for i in range(1, self.total_cv + 1)]
                means = np.mean(rows, axis=0)
                mean_line = '\t'.join(f'{v:.4f}' for v in means) + '\n'
                # replace or append mean row
                if len(content) > self.total_cv + 1:
                    content[self.total_cv + 1] = mean_line
                else:
                    content.append(mean_line)
            except ValueError:
                pass  # some folds not yet written; skip mean for now

        f_out = open(self.path, 'w')
        f_out.writelines(content)
        f_out.close()
        f_in.close()


# ─── helpers (unchanged from train_miss.py) ───────────────────────────────────

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join('checkpoints', expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch) + '_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))


# ─── evaluation functions ─────────────────────────────────────────────────────

def eval(model, val_iter, is_save=False, phase='test', epoch=-1):
    """Evaluate on the missing-modality dataset (6 conditions × N utterances).

    Changes vs. train_miss.py:
      • also collects int2name to derive speaker gender
      • when is_save=True and phase='test', computes bias metrics per condition
        and writes them to bias_recorder_lookup
    """
    model.eval()

    total_pred      = []
    total_label     = []
    total_miss_type = []
    total_int2name  = []          # NEW: needed for gender extraction
    total_data      = 0

    for i, data in enumerate(val_iter):
        total_data += 1
        model.set_input(data)
        model.test()
        pred      = model.pred.argmax(dim=1).detach().cpu().numpy()
        label     = data['label']
        miss_type = np.array(data['miss_type'])
        int2name  = np.array(data['int2name'])    # NEW

        total_pred.append(pred)
        total_label.append(label)
        total_miss_type.append(miss_type)
        total_int2name.append(int2name)           # NEW

    total_pred      = np.concatenate(total_pred)
    total_label     = np.concatenate(total_label)
    total_miss_type = np.concatenate(total_miss_type)
    total_int2name  = np.concatenate(total_int2name)  # NEW

    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1  = f1_score(total_label, total_pred, average='macro')
    cm  = confusion_matrix(total_label, total_pred)

    if is_save:
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, f'{phase}_pred.npy'),     total_pred)
        np.save(os.path.join(save_dir, f'{phase}_label.npy'),    total_label)
        np.save(os.path.join(save_dir, f'{phase}_int2name.npy'), total_int2name)  # NEW

        for part_name in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
            part_idx   = np.where(total_miss_type == part_name)
            part_pred  = total_pred[part_idx]
            part_label = total_label[part_idx]
            acc_part   = accuracy_score(part_label, part_pred)
            uar_part   = recall_score(part_label, part_pred, average='macro')
            f1_part    = f1_score(part_label, part_pred, average='macro')
            np.save(os.path.join(save_dir, f'{phase}_{part_name}_pred.npy'),  part_pred)
            np.save(os.path.join(save_dir, f'{phase}_{part_name}_label.npy'), part_label)
            if phase == 'test':
                recorder_lookup[part_name].write_result_to_tsv(
                    {'acc': acc_part, 'uar': uar_part, 'f1': f1_part},
                    cvNo=opt.cvNo)

                # NEW: gender-bias metrics for this condition ──────────────────
                part_int2name  = total_int2name[part_idx]
                part_genders   = np.array([get_gender(u) for u in part_int2name])
                bias           = compute_bias_metrics(part_pred, part_label, part_genders)
                bias_recorder_lookup[part_name].write_result_to_tsv(bias, cvNo=opt.cvNo)
                _log_bias(part_name, bias)

    model.train()
    return acc, uar, f1, cm


def eval_full_modality(model, full_val_iter, is_save=False, phase='test'):
    """Evaluate the 7th condition: all three modalities present (avl).

    Uses the regular MultimodalDataset (no zeroing) so the model receives
    complete A, V, L features.  In test mode IFMMIN's set_input does not
    apply any missing-index mask, so the model simply uses all features.
    """
    model.eval()

    total_pred     = []
    total_label    = []
    total_int2name = []

    for i, data in enumerate(full_val_iter):
        model.set_input(data)
        model.test()
        pred     = model.pred.argmax(dim=1).detach().cpu().numpy()
        label    = data['label']
        int2name = np.array(data['int2name'])

        total_pred.append(pred)
        total_label.append(label)
        total_int2name.append(int2name)

    total_pred     = np.concatenate(total_pred)
    total_label    = np.concatenate(total_label)
    total_int2name = np.concatenate(total_int2name)

    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1  = f1_score(total_label, total_pred, average='macro')
    cm  = confusion_matrix(total_label, total_pred)

    if is_save and phase == 'test':
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, 'test_avl_pred.npy'),     total_pred)
        np.save(os.path.join(save_dir, 'test_avl_label.npy'),    total_label)
        np.save(os.path.join(save_dir, 'test_avl_int2name.npy'), total_int2name)

        recorder_lookup['avl'].write_result_to_tsv(
            {'acc': acc, 'uar': uar, 'f1': f1}, cvNo=opt.cvNo)

        genders = np.array([get_gender(u) for u in total_int2name])
        bias    = compute_bias_metrics(total_pred, total_label, genders)
        bias_recorder_lookup['avl'].write_result_to_tsv(bias, cvNo=opt.cvNo)
        _log_bias('avl', bias)

    model.train()
    return acc, uar, f1, cm


def _log_bias(condition, bias):
    """Log a one-line bias summary for a condition."""
    per_emo = '  '.join(
        f'{e}:[SP={bias["sp"][i]:+.3f} EO={bias["eo"][i]:+.3f} F1d={bias["f1_diff"][i]:.3f}]'
        for i, e in enumerate(EMOTIONS)
    )
    logger.info(
        f'[Bias {condition}] mean|SP|={bias["sp_abs_mean"]:.4f}  '
        f'mean|EO|={bias["eo_abs_mean"]:.4f}  '
        f'mean_F1diff={bias["f1_diff"].mean():.4f}\n    {per_emo}'
    )


# ─── gender predictability AUROC ──────────────────────────────────────────────

def _run_auroc_trials(embeddings, genders, n_trials=100, test_size=0.2):
    """Run n_trials stratified logistic-regression → AUROC for gender prediction.

    Each trial uses a fresh StratifiedShuffleSplit so class balance is
    preserved (roughly 50/50 M/F as in IEMOCAP) and a freshly-fitted
    LogisticRegression classifier.

    Returns an array of shape (n_trials,) with one AUROC per trial.
    """
    if len(np.unique(genders)) < 2:
        return np.full(n_trials, 0.5)  # degenerate: only one gender present

    sss    = StratifiedShuffleSplit(n_splits=n_trials, test_size=test_size, random_state=0)
    scores = []
    for trial_idx, (trn_idx, tst_idx) in enumerate(sss.split(embeddings, genders)):
        X_trn, X_tst = embeddings[trn_idx], embeddings[tst_idx]
        y_trn, y_tst = genders[trn_idx],    genders[tst_idx]
        clf = LogisticRegression(max_iter=1000, random_state=trial_idx, C=1.0)
        clf.fit(X_trn, y_trn)
        proba  = clf.predict_proba(X_tst)[:, 1]
        scores.append(roc_auc_score(y_tst, proba))
    return np.array(scores)


def _extract_embeddings(model, data_iter, miss_dataset=True):
    """Extract post-encoder embeddings for all samples in data_iter.

    Accesses the intermediate tensors that IFMMIN stores as attributes
    during forward():
        feat_A_miss  — audio LSTM output      (128-d)
        feat_V_miss  — visual LSTM output     (128-d)
        feat_L_miss  — text TextCNN output    (128-d)
        feat_fusion_miss — concatenation A+L+V before AE  (384-d)

    When a modality is missing (zeroed input) its corresponding embedding
    is near-zero, reflecting the information available under that condition.

    Parameters
    ----------
    miss_dataset : bool
        True  → batch has 'miss_type' key (MultimodalMissDataset)
        False → batch is full-modality (MultimodalDataset), assign 'avl'
    """
    model.eval()
    audio_list, visual_list, text_list, fused_list = [], [], [], []
    gender_list, miss_type_list = [], []

    with torch.no_grad():
        for data in data_iter:
            model.set_input(data)
            model.test()   # runs forward(), populates feat_* attributes

            audio_list.append(model.feat_A_miss.cpu().numpy())
            visual_list.append(model.feat_V_miss.cpu().numpy())
            text_list.append(model.feat_L_miss.cpu().numpy())
            fused_list.append(model.feat_fusion_miss.cpu().numpy())

            int2names = np.array(data['int2name'])
            # encode gender as binary: F=1, M=0
            gender_list.append(
                (np.array([get_gender(u) for u in int2names]) == 'F').astype(int))

            if miss_dataset:
                miss_type_list.append(np.array(data['miss_type']))
            else:
                miss_type_list.append(np.full(len(int2names), 'avl'))

    return {
        'audio':  np.concatenate(audio_list),
        'visual': np.concatenate(visual_list),
        'text':   np.concatenate(text_list),
        'fused':  np.concatenate(fused_list),
    }, np.concatenate(gender_list), np.concatenate(miss_type_list)


def eval_gender_auroc(model, tst_miss_iter, full_tst_iter,
                      save_dir, result_dir, cvNo, n_trials=100):
    """Measure gender predictability AUROC for every modality × condition.

    For each of the 7 conditions (6 missing + full), trains a logistic
    regression classifier to predict speaker gender from the model's
    intermediate embeddings.  Repeats n_trials times with different
    StratifiedShuffleSplit seeds; reports mean ± std AUROC.

    AUROC ≈ 0.5 → gender-blind features (ideal for fairness).
    AUROC →  1  → strong gender leakage → higher bias risk.

    Called automatically after the standard test evaluation.
    Returns the nested dict of AUROC arrays for downstream logging.
    """
    logger.info('=== Gender Predictability AUROC — extracting embeddings ===')

    # ── extract from the 6-condition missing-modality test set ────────────────
    miss_embs, miss_genders, miss_types = _extract_embeddings(
        model, tst_miss_iter, miss_dataset=True)

    # ── extract from the full-modality test set (7th condition) ───────────────
    full_embs, full_genders, _ = _extract_embeddings(
        model, full_tst_iter, miss_dataset=False)

    # ── compute AUROC trials per condition × modality ─────────────────────────
    auroc_results = {}
    modality_names = ['audio', 'visual', 'text', 'fused']

    for cond in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
        idx = np.where(miss_types == cond)
        auroc_results[cond] = {
            mod: _run_auroc_trials(miss_embs[mod][idx], miss_genders[idx], n_trials)
            for mod in modality_names
        }

    auroc_results['avl'] = {
        mod: _run_auroc_trials(full_embs[mod], full_genders, n_trials)
        for mod in modality_names
    }

    # ── log summary table ─────────────────────────────────────────────────────
    header = f"{'Cond':<6}" + ''.join(f'{m:>22}' for m in modality_names)
    logger.info('\n=== Gender Predictability AUROC (mean ± std, %d trials) ===' % n_trials)
    logger.info(header)
    for cond in ALL_CONDITIONS:
        row = f'{cond:<6}'
        for mod in modality_names:
            s = auroc_results[cond][mod]
            row += f'   {np.mean(s):.3f} ± {np.std(s):.3f}       '
        logger.info(row)

    # ── save per-fold CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(result_dir, f'gender_auroc_cv{cvNo}.csv')
    with open(csv_path, 'w') as fh:
        fh.write('condition,modality,auroc_mean,auroc_std,auroc_min,auroc_max\n')
        for cond in ALL_CONDITIONS:
            for mod in modality_names:
                s = auroc_results[cond][mod]
                fh.write(f'{cond},{mod},{np.mean(s):.4f},{np.std(s):.4f},'
                         f'{np.min(s):.4f},{np.max(s):.4f}\n')
    logger.info(f'AUROC results saved to {csv_path}')

    # ── violin plot ───────────────────────────────────────────────────────────
    plot_path = os.path.join(save_dir, f'gender_auroc_cv{cvNo}.png')
    _plot_auroc_distributions(auroc_results, modality_names, plot_path)

    model.train()
    return auroc_results


def _plot_auroc_distributions(auroc_results, modality_names, save_path):
    """4-panel violin plot: one panel per modality, x-axis = conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax_idx, mod in enumerate(modality_names):
        ax   = axes[ax_idx]
        data = [auroc_results[cond][mod] for cond in ALL_CONDITIONS]

        parts = ax.violinplot(data, positions=range(len(ALL_CONDITIONS)),
                              showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#4C9BE8')
            pc.set_alpha(0.7)

        # ideal gender-blind reference
        ax.axhline(y=0.5, color='green', linestyle='--', linewidth=1.5,
                   label='AUROC = 0.5  (gender-blind)')

        ax.set_xticks(range(len(ALL_CONDITIONS)))
        ax.set_xticklabels(ALL_CONDITIONS, fontsize=10)
        ax.set_ylim(0.3, 1.0)
        ax.set_ylabel('AUROC', fontsize=11)
        ax.set_title(f'{mod.capitalize()} embeddings', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Gender Predictability AUROC — per Modality & Missing Condition',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'AUROC violin plot saved to {save_path}')


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    opt = Options().parse()
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo))
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)

    result_dir = os.path.join(opt.log_dir, opt.name, 'results')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    total_cv = 10 if opt.corpus_name == 'IEMOCAP' else 12

    # original WA / UAR / F1 recorders (identical to train_miss.py) ──────────
    recorder_lookup = {
        "total": ResultRecorder(os.path.join(result_dir, 'result_total.tsv'), total_cv=total_cv),
        "azz":   ResultRecorder(os.path.join(result_dir, 'result_azz.tsv'),   total_cv=total_cv),
        "zvz":   ResultRecorder(os.path.join(result_dir, 'result_zvz.tsv'),   total_cv=total_cv),
        "zzl":   ResultRecorder(os.path.join(result_dir, 'result_zzl.tsv'),   total_cv=total_cv),
        "avz":   ResultRecorder(os.path.join(result_dir, 'result_avz.tsv'),   total_cv=total_cv),
        "azl":   ResultRecorder(os.path.join(result_dir, 'result_azl.tsv'),   total_cv=total_cv),
        "zvl":   ResultRecorder(os.path.join(result_dir, 'result_zvl.tsv'),   total_cv=total_cv),
        "avl":   ResultRecorder(os.path.join(result_dir, 'result_avl.tsv'),   total_cv=total_cv),  # NEW: full-modality
    }

    # NEW: bias recorders — one TSV per condition ─────────────────────────────
    bias_recorder_lookup = {
        cond: BiasResultRecorder(
            os.path.join(result_dir, f'bias_{cond}.tsv'), total_cv=total_cv)
        for cond in ALL_CONDITIONS
    }

    loss_dir = os.path.join(opt.image_dir, opt.name, 'loss')
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    recorder_loss = LossRecorder(
        os.path.join(loss_dir, 'result_loss.tsv'),
        total_cv=total_cv, total_epoch=opt.niter + opt.niter_decay)

    suffix = '_'.join([opt.model, opt.dataset_mode])
    logger = get_logger(logger_path, suffix)

    # datasets ─────────────────────────────────────────────────────────────────
    if opt.has_test:
        dataset, val_dataset, tst_dataset = create_dataset_with_args(
            opt, set_name=['trn', 'val', 'tst'])
    else:
        dataset, val_dataset = create_dataset_with_args(opt, set_name=['trn', 'val'])

    # NEW: full-modality test set (7th condition) ──────────────────────────────
    # Temporarily switch to the non-missing dataset so features are unmasked.
    # IFMMIN's set_input in eval mode uses features directly without zeroing,
    # so passing unmasked features gives us the all-modalities-present result.
    _saved_mode = opt.dataset_mode
    opt.dataset_mode = 'multimodal'
    if opt.has_test:
        full_tst_dataset = create_dataset_with_args(opt, set_name='tst')
    opt.dataset_mode = _saved_mode  # restore so the model is created correctly

    dataset_size     = len(dataset)
    tst_dataset_size = len(tst_dataset) if opt.has_test else 0
    logger.info('The number of training samples = %d' % dataset_size)
    logger.info('The number of testing samples  = %d' % tst_dataset_size)

    model = create_model(opt)
    model.setup(opt)
    total_iters     = 0
    best_eval_epoch = -1
    best_eval_acc, best_eval_uar, best_eval_f1 = 0, 0, 0

    # training loop (identical to train_miss.py) ───────────────────────────────
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time   = time.time()
        epoch_iter       = 0
        loss_add         = True

        TSNE_model = TSNE()

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iters  += 1
            epoch_iter   += opt.batch_size
            model.set_input(data)
            model.optimize_parameters(epoch)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                logger.info(
                    'Cur epoch {}'.format(epoch) + ' loss ' +
                    ' '.join(
                        map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)
                    ).format(**losses)
                )
            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate(logger)

        acc, uar, f1, cm = eval(model, val_dataset)
        logger.info('Val result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (
            epoch, opt.niter + opt.niter_decay, acc, uar, f1))
        logger.info('\n{}'.format(cm))

        if opt.has_test and opt.verbose:
            acc, uar, f1, cm = eval(model, tst_dataset)
            logger.info('Tst result of epoch %d acc %.4f uar %.4f f1 %.4f' % (epoch, acc, uar, f1))
            logger.info('\n{}'.format(cm))

        if opt.corpus_name == 'IEMOCAP':
            if uar > best_eval_uar:
                best_eval_epoch = epoch
                best_eval_uar   = uar
                best_eval_acc   = acc
                best_eval_f1    = f1
            select_metric = 'uar'
            best_metric   = best_eval_uar
        elif opt.corpus_name == 'MSP':
            if f1 > best_eval_f1:
                best_eval_epoch = epoch
                best_eval_uar   = uar
                best_eval_acc   = acc
                best_eval_f1    = f1
            select_metric = 'f1'
            best_metric   = best_eval_f1
        else:
            raise ValueError(f'corpus name must be IEMOCAP or MSP, but got {opt.corpus_name}')

    logger.info('Best eval epoch %d found with %s %f' % (best_eval_epoch, select_metric, best_metric))

    # final test on best checkpoint ────────────────────────────────────────────
    if opt.has_test:
        logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
        model.load_networks(best_eval_epoch)

        _ = eval(model, val_dataset, is_save=True, phase='val', epoch=best_eval_epoch)
        acc, uar, f1, cm = eval(model, tst_dataset, is_save=True, phase='test', epoch=best_eval_epoch)
        logger.info('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
        logger.info('\n{}'.format(cm))
        recorder_lookup['total'].write_result_to_tsv(
            {'acc': acc, 'uar': uar, 'f1': f1}, cvNo=opt.cvNo)

        # NEW: 7th condition — full modality (avl) ────────────────────────────
        logger.info('Evaluating full-modality condition (avl) ...')
        acc_avl, uar_avl, f1_avl, cm_avl = eval_full_modality(
            model, full_tst_dataset, is_save=True, phase='test')
        logger.info('Full-modality (avl) acc %.4f uar %.4f f1 %.4f' % (acc_avl, uar_avl, f1_avl))
        logger.info('\n{}'.format(cm_avl))

        # Gender Predictability AUROC — called after all standard eval is done
        logger.info('Running Gender Predictability AUROC evaluation ...')
        eval_gender_auroc(
            model, tst_dataset, full_tst_dataset,
            save_dir=model.save_dir, result_dir=result_dir, cvNo=opt.cvNo)

    else:
        recorder_lookup['total'].write_result_to_tsv(
            {'acc': best_eval_acc, 'uar': best_eval_uar, 'f1': best_eval_f1},
            cvNo=opt.cvNo)

    clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)
