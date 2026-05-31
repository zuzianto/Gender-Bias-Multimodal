import argparse
import os
import re
import numpy as np

EMOTION_ORDER = ["angry", "happy", "sad", "neutral"]
GENDER_PATTERN = re.compile(r"Ses\d+(F|M)")


def decode_name(name):
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="replace")
    if isinstance(name, np.ndarray) and name.dtype.kind == "S":
        return name.astype(str)
    return str(name)


def parse_gender(name):
    if isinstance(name, np.ndarray):
        name = name.item() if name.size == 1 else name[0]
    name = decode_name(name)
    match = GENDER_PATTERN.search(name)
    if match:
        return "female" if match.group(1) == "F" else "male"
    if "_F" in name or "F_" in name:
        return "female"
    if "_M" in name or "M_" in name:
        return "male"
    return "unknown"


def load_split_data(split_dir, split_name):
    names_path = os.path.join(split_dir, f"{split_name}_int2name.npy")
    labels_path = os.path.join(split_dir, f"{split_name}_label.npy")

    if not os.path.isfile(names_path) or not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Missing files for {split_name}: {names_path} or {labels_path}")

    names = np.load(names_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    if len(names) != len(labels):
        raise ValueError(f"Length mismatch for {split_name}: {len(names)} names vs {len(labels)} labels")

    return names, labels


def summarize_split(names, labels):
    stats = {
        "overall": {"male": 0, "female": 0, "unknown": 0},
        "by_emotion": {emotion: {"male": 0, "female": 0, "unknown": 0} for emotion in EMOTION_ORDER}
    }

    for name, onehot in zip(names, labels):
        gender = parse_gender(name)
        stats["overall"][gender] = stats["overall"].get(gender, 0) + 1

        if onehot.ndim == 0:
            onehot = np.asarray([onehot])
        if onehot.sum() != 1:
            raise ValueError(f"Expected one-hot label for example {decode_name(name)}; got {onehot}")

        emotion_idx = int(np.argmax(onehot))
        emotion = EMOTION_ORDER[emotion_idx]
        stats["by_emotion"][emotion][gender] = stats["by_emotion"][emotion].get(gender, 0) + 1

    return stats


def format_stats(summary):
    lines = []
    total = sum(summary["overall"].values())
    lines.append(f"  Total examples: {total}")
    for gender, count in summary["overall"].items():
        pct = (count / total * 100) if total else 0.0
        lines.append(f"    {gender.title()}: {count} ({pct:.1f}%)")

    lines.append("  Emotion / gender breakdown:")
    for emotion in EMOTION_ORDER:
        em_stats = summary["by_emotion"][emotion]
        emo_total = sum(em_stats.values())
        if emo_total == 0:
            lines.append(f"    {emotion.title()}: no examples")
            continue
        parts = []
        for gender in ["male", "female", "unknown"]:
            count = em_stats.get(gender, 0)
            pct = (count / emo_total * 100) if emo_total else 0.0
            parts.append(f"{gender.title()} {count} ({pct:.1f}%)")
        lines.append(f"    {emotion.title()}: {emo_total} -> {', '.join(parts)}")
    return "\n".join(lines)


def summarize_fold(fold_dir, fold_name):
    fold_summary = {}
    for split_name in ["trn", "tst", "val"]:
        names, labels = load_split_data(fold_dir, split_name)
        fold_summary[split_name] = summarize_split(names, labels)
    return fold_summary


def collect_overall(base_dir):
    overall_summary = {"trn": {"names": [], "labels": []}, "tst": {"names": [], "labels": []}, "val": {"names": [], "labels": []}}
    for entry in sorted(os.listdir(base_dir), key=lambda x: int(x) if x.isdigit() else x):
        fold_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(fold_dir):
            continue
        for split_name in ["trn", "tst", "val"]:
            names, labels = load_split_data(fold_dir, split_name)
            overall_summary[split_name]["names"].append(names)
            overall_summary[split_name]["labels"].append(labels)

    combined = {}
    all_names = []
    all_labels = []
    for split_name, data in overall_summary.items():
        if not data["names"]:
            continue
        names = np.concatenate(data["names"], axis=0)
        labels = np.concatenate(data["labels"], axis=0)
        combined[split_name] = summarize_split(names, labels)
        all_names.append(names)
        all_labels.append(labels)

    if all_names and all_labels:
        all_names = np.concatenate(all_names, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_data_summary = summarize_split(all_names, all_labels)
    else:
        all_data_summary = None

    return combined, all_data_summary


def main():
    parser = argparse.ArgumentParser(description="Summarize gender/emotion ratios in IEMOCAP target folds.")
    parser.add_argument("base_dir", nargs="?", default=r"IEMOCAP_features/IEMOCAP_features_2021/target",
                        help="Base folder containing fold subfolders 1..10")
    parser.add_argument("--output-csv", default=None,
                        help="Optional CSV file to save the per-fold summary table")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    print(f"Base directory: {base_dir}")
    print("\nPer-fold summaries:")

    for entry in sorted(os.listdir(base_dir), key=lambda x: int(x) if x.isdigit() else x):
        fold_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(fold_dir):
            continue
        print(f"\nFold {entry}")
        fold_summary = summarize_fold(fold_dir, entry)
        for split_name in ["trn", "tst", "val"]:
            print(f"  {split_name.upper()}:")
            print(format_stats(fold_summary[split_name]))
            print()

    print("\nOverall combined dataset summary:")
    overall_summary, all_data_summary = collect_overall(base_dir)
    for split_name in ["trn", "tst", "val"]:
        if split_name not in overall_summary:
            continue
        print(f"  Combined {split_name.upper()}:" )
        print(format_stats(overall_summary[split_name]))
        print()

    if all_data_summary is not None:
        print("  Combined ALL SPLITS (all folds):")
        print(format_stats(all_data_summary))
        print()

    # Optionally save a simple CSV summary if requested.
    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["fold", "split", "emotion", "gender", "count", "percent_of_emotion"])
            for entry in sorted(os.listdir(base_dir), key=lambda x: int(x) if x.isdigit() else x):
                fold_dir = os.path.join(base_dir, entry)
                if not os.path.isdir(fold_dir):
                    continue
                fold_summary = summarize_fold(fold_dir, entry)
                for split_name in ["trn", "tst", "val"]:
                    stats = fold_summary[split_name]
                    for emotion in EMOTION_ORDER:
                        em_stats = stats["by_emotion"][emotion]
                        emo_total = sum(em_stats.values())
                        if emo_total == 0:
                            continue
                        for gender in ["male", "female", "unknown"]:
                            count = em_stats.get(gender, 0)
                            pct = (count / emo_total * 100) if emo_total else 0.0
                            writer.writerow([entry, split_name, emotion, gender, count, f"{pct:.2f}"])
        print(f"Saved summary CSV to {args.output_csv}")


if __name__ == "__main__":
    main()