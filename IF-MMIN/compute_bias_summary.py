"""
compute_bias_summary.py — aggregate gender-bias metrics across all 10 folds.

IEMOCAP's 10-fold CV is speaker-based: each test fold contains utterances
from exactly one speaker (thus one gender). Bias metrics require both genders,
so they must be computed on the combined predictions from all 10 folds.

This script:
  1. Loads per-fold test predictions saved by train_miss_bias.py
  2. Combines predictions from all folds (covering all speakers / both genders)
  3. Computes F1-diff, Statistical Parity, and Equality of Opportunity
     for each of the 7 modality conditions
  4. Writes a summary TSV and prints a report

Usage (run from IF-MMIN/):
    python compute_bias_summary.py --checkpoint_dir checkpoints/our_IEMOCAP_block_5_run_0_1
"""

import os
import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score

EMOTIONS       = ['happy', 'angry', 'sad', 'neutral']
MISS_CONDS     = ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']
ALL_CONDITIONS = MISS_CONDS + ['avl']


# ─── gender helper ─────────────────────────────────────────────────────────────

def get_gender(utt_id: str) -> str:
    """Return 'M' or 'F' from an IEMOCAP utterance ID.

    The speaker gender is in the LAST token of the utterance ID:
        'Ses02F_script03_1_M001' → 'M'
        'Ses05F_impro04_F041'    → 'F'
    """
    token = utt_id.split('_')[-1]
    if token[0] in ('M', 'F'):
        return token[0]
    # fallback: check the session letter (less reliable, only if trailing token fails)
    session_token = utt_id.split('_')[0]   # e.g. 'Ses01M' or 'Ses01F'
    if len(session_token) >= 6 and session_token[5] in ('M', 'F'):
        return session_token[5]
    return 'U'   # unknown


# ─── bias computation ─────────────────────────────────────────────────────────

def compute_bias_metrics(preds, labels, genders):
    """Compute F1-diff, SP, and EO for one set of predictions."""
    preds   = np.array(preds)
    labels  = np.array(labels)
    genders = np.array(genders)

    n_classes    = len(EMOTIONS)
    labels_range = list(range(n_classes))
    zeros        = np.zeros(n_classes)

    male_mask   = genders == 'M'
    female_mask = genders == 'F'
    preds_m,  labels_m = preds[male_mask],  labels[male_mask]
    preds_f,  labels_f = preds[female_mask], labels[female_mask]

    n_m, n_f = len(preds_m), len(preds_f)
    print(f'    male samples: {n_m}   female samples: {n_f}   unknown: {(~male_mask & ~female_mask).sum()}')

    if n_m == 0 or n_f == 0:
        print('    WARNING: one gender group still empty — skipping this condition.')
        return None

    f1_m    = f1_score(labels_m, preds_m, average=None, labels=labels_range, zero_division=0)
    f1_f    = f1_score(labels_f, preds_f, average=None, labels=labels_range, zero_division=0)
    f1_diff = np.abs(f1_m - f1_f)

    sp = np.array([np.mean(preds_m == c) - np.mean(preds_f == c) for c in range(n_classes)])

    eo = np.zeros(n_classes)
    for c in range(n_classes):
        m_pos = labels_m == c
        f_pos = labels_f == c
        tpr_m = np.mean(preds_m[m_pos] == c) if m_pos.sum() > 0 else 0.0
        tpr_f = np.mean(preds_f[f_pos] == c) if f_pos.sum() > 0 else 0.0
        eo[c] = tpr_m - tpr_f

    return {
        'f1_m': f1_m, 'f1_f': f1_f, 'f1_diff': f1_diff,
        'sp': sp, 'sp_abs_mean': float(np.mean(np.abs(sp))),
        'eo': eo, 'eo_abs_mean': float(np.mean(np.abs(eo))),
        'acc_m': accuracy_score(labels_m, preds_m),
        'acc_f': accuracy_score(labels_f, preds_f),
        'uar_m': recall_score(labels_m, preds_m, average='macro', zero_division=0),
        'uar_f': recall_score(labels_f, preds_f, average='macro', zero_division=0),
    }


# ─── loading helpers ───────────────────────────────────────────────────────────

def load_fold(fold_dir):
    """Load test predictions, labels, int2names, and miss_types for one fold."""
    def npy(name):
        path = os.path.join(fold_dir, name)
        if not os.path.exists(path):
            return None
        return np.load(path, allow_pickle=True)

    pred      = npy('test_pred.npy')
    label     = npy('test_label.npy')
    int2name  = npy('test_int2name.npy')
    miss_type = npy('test_miss_type.npy')

    # avl (full modality) stored separately
    avl_pred     = npy('test_avl_pred.npy')
    avl_label    = npy('test_avl_label.npy')
    avl_int2name = npy('test_avl_int2name.npy')

    if pred is None:
        return None

    return {
        'pred': pred, 'label': label,
        'int2name': int2name, 'miss_type': miss_type,
        'avl_pred': avl_pred, 'avl_label': avl_label,
        'avl_int2name': avl_int2name,
    }


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to checkpoint directory, e.g. checkpoints/our_IEMOCAP_block_5_run_0_1')
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Where to write the summary TSV (defaults to checkpoint_dir)')
    args = parser.parse_args()

    out_dir = args.out_dir or args.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── collect predictions from all folds ────────────────────────────────────
    all_pred, all_label, all_int2name, all_miss_type = [], [], [], []
    all_avl_pred, all_avl_label, all_avl_int2name   = [], [], []

    loaded_folds = 0
    for fold in range(1, args.n_folds + 1):
        fold_dir = os.path.join(args.checkpoint_dir, str(fold))
        data = load_fold(fold_dir)
        if data is None:
            print(f'Fold {fold}: no predictions found, skipping.')
            continue
        loaded_folds += 1
        all_pred.append(data['pred'])
        all_label.append(data['label'])
        all_int2name.append(data['int2name'])
        all_miss_type.append(data['miss_type'])
        if data['avl_pred'] is not None:
            all_avl_pred.append(data['avl_pred'])
            all_avl_label.append(data['avl_label'])
            all_avl_int2name.append(data['avl_int2name'])

    print(f'\nLoaded predictions from {loaded_folds} / {args.n_folds} folds.')

    if loaded_folds == 0:
        print('No fold predictions found. Make sure train_miss_bias.py has completed at least one fold.')
        return

    all_pred      = np.concatenate(all_pred)
    all_label     = np.concatenate(all_label)
    all_int2name  = np.concatenate(all_int2name)
    all_miss_type = np.concatenate(all_miss_type)

    all_genders = np.array([get_gender(str(u)) for u in all_int2name])

    # sanity-check gender extraction
    unique, counts = np.unique(all_genders, return_counts=True)
    print(f'Gender distribution across all folds: {dict(zip(unique, counts))}')

    # ── compute bias per condition ─────────────────────────────────────────────
    results = {}

    for cond in MISS_CONDS:
        print(f'\n--- Condition: {cond} ---')
        idx = all_miss_type == cond
        bias = compute_bias_metrics(all_pred[idx], all_label[idx], all_genders[idx])
        results[cond] = bias

    # full modality (avl)
    if all_avl_pred:
        avl_pred     = np.concatenate(all_avl_pred)
        avl_label    = np.concatenate(all_avl_label)
        avl_int2name = np.concatenate(all_avl_int2name)
        avl_genders  = np.array([get_gender(str(u)) for u in avl_int2name])
        print('\n--- Condition: avl (full modality) ---')
        results['avl'] = compute_bias_metrics(avl_pred, avl_label, avl_genders)
    else:
        print('\nNo avl (full-modality) predictions found.')
        results['avl'] = None

    # ── print summary table ────────────────────────────────────────────────────
    print('\n' + '='*80)
    print('GENDER BIAS SUMMARY (combined across all folds)')
    print('='*80)
    header = f"{'Cond':<6}  {'UAR_M':>7} {'UAR_F':>7} {'UAR_diff':>9}  " + \
             '  '.join(f'SP_{e[:3]}' for e in EMOTIONS) + '  ' + \
             '  '.join(f'EO_{e[:3]}' for e in EMOTIONS)
    print(header)
    print('-'*80)

    for cond in ALL_CONDITIONS:
        b = results.get(cond)
        if b is None:
            print(f'{cond:<6}  (no data)')
            continue
        uar_diff = b['uar_m'] - b['uar_f']
        sp_str   = '  '.join(f'{v:+.3f}' for v in b['sp'])
        eo_str   = '  '.join(f'{v:+.3f}' for v in b['eo'])
        print(f"{cond:<6}  {b['uar_m']:>7.4f} {b['uar_f']:>7.4f} {uar_diff:>+9.4f}  {sp_str}  {eo_str}")

    print()
    print('Per-emotion F1 difference (|F1_male - F1_female|):')
    print(f"{'Cond':<6}  " + '  '.join(f'{e:>10}' for e in EMOTIONS))
    for cond in ALL_CONDITIONS:
        b = results.get(cond)
        if b is None:
            continue
        vals = '  '.join(f'{v:>10.4f}' for v in b['f1_diff'])
        print(f'{cond:<6}  {vals}')

    # ── write TSV ─────────────────────────────────────────────────────────────
    tsv_path = os.path.join(out_dir, f'bias_summary_{loaded_folds}folds.tsv')
    with open(tsv_path, 'w') as fh:
        cols = (['condition', 'n_folds', 'uar_m', 'uar_f', 'uar_diff', 'sp_abs_mean', 'eo_abs_mean'] +
                [f'f1_m_{e}' for e in EMOTIONS] + [f'f1_f_{e}' for e in EMOTIONS] +
                [f'f1_diff_{e}' for e in EMOTIONS] +
                [f'sp_{e}' for e in EMOTIONS] + [f'eo_{e}' for e in EMOTIONS])
        fh.write('\t'.join(cols) + '\n')
        for cond in ALL_CONDITIONS:
            b = results.get(cond)
            if b is None:
                continue
            vals = ([cond, loaded_folds, b['uar_m'], b['uar_f'],
                     b['uar_m'] - b['uar_f'], b['sp_abs_mean'], b['eo_abs_mean']] +
                    list(b['f1_m']) + list(b['f1_f']) + list(b['f1_diff']) +
                    list(b['sp'])  + list(b['eo']))
            fh.write('\t'.join(f'{v:.4f}' if isinstance(v, float) else str(v)
                               for v in vals) + '\n')

    print(f'\nSummary saved to {tsv_path}')


if __name__ == '__main__':
    main()
