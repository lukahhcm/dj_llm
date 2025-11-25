#!/usr/bin/env python3
"""
使用示例:
    python evaluation/evaluate_llm_filter.py \
        --pred-file ../dj_llm_data/results/llm/test_cc_100_qwen_32b_batch_1.jsonl \
        --ground-truth-root ../dj_llm_data/results/ground_truth
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM filter predictions against ground truth."
    )
    parser.add_argument(
        "--pred-file",
        required=True,
        help="Path to the LLM prediction JSONL file.",
    )
    parser.add_argument(
        "--ground-truth-root",
        required=True,
        help="Root directory that contains ground-truth JSONL files grouped by operator/level.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=6.0,
        help="Threshold used when measuring score-label consistency (default: 6.0).",
    )
    return parser.parse_args()


def normalize_key(value: Optional[str]) -> Optional[str]:
    return value.lower() if isinstance(value, str) else None


class GroundTruthStore:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._cache: Dict[Tuple[str, str], Dict[int, int]] = {}

    def _load_file(self, operator: str, level: str) -> Dict[int, int]:
        op_key = normalize_key(operator)
        lvl_key = normalize_key(level)
        if op_key is None or lvl_key is None:
            raise ValueError("operator_name and level must be present in metadata")
        cache_key = (op_key, lvl_key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        gt_path = os.path.join(self.root_dir, operator, lvl_key, "results.jsonl")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing ground truth file: {gt_path}")

        sample_map: Dict[int, int] = {}
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_idx = row.get("sample_index")
                decision = row.get("operator_output", {}).get("decision")
                if decision is None:
                    decision = "KEEP" if row.get("filter_decision") else "REMOVE"
                sample_map[sample_idx] = 1 if str(decision).upper() == "KEEP" else 0

        self._cache[cache_key] = sample_map
        return sample_map

    def get_label(self, operator: str, level: str, sample_index: int) -> Optional[int]:
        sample_map = self._load_file(operator, level)
        return sample_map.get(sample_index)


def parse_llm_output(raw: str) -> Tuple[Optional[float], Optional[int]]:
    raw = (raw or "").strip()
    if not raw:
        return None, None
    try:
        elem = ET.fromstring(raw)
        score_text = elem.findtext("score")
        decision_text = elem.findtext("decision")
        score = float(score_text) if score_text is not None else None
        label = 1 if str(decision_text).strip().upper() == "KEEP" else 0
        return score, label
    except ET.ParseError:
        return None, None


def compute_precision(tp: int, fp: int) -> Optional[float]:
    denom = tp + fp
    if denom == 0:
        return None
    return tp / denom


def compute_recall(tp: int, fn: int) -> Optional[float]:
    denom = tp + fn
    if denom == 0:
        return None
    return tp / denom


def compute_f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def compute_auc(labels: List[int], scores: List[float]) -> Optional[float]:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    tp = fp = 0
    prev_fpr = prev_tpr = 0.0
    auc = 0.0
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / pos
        fpr = fp / neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr, prev_tpr = fpr, tpr
    return auc


def summarize_group(
    records: Iterable[Tuple[int, int, Optional[float]]],
    score_threshold: float,
) -> Dict[str, Optional[float]]:
    tp = tn = fp = fn = 0
    labels: List[int] = []
    scores: List[float] = []
    consistency_matches = total_records = 0

    for y_true, y_pred, score in records:
        total_records += 1
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1

        if score is not None:
            labels.append(y_true)
            scores.append(score)
            inferred_keep = 1 if score >= score_threshold else 0
            if inferred_keep == y_pred:
                consistency_matches += 1
        else:
            # If score missing, exclude from AUC + consistency
            pass

    accuracy = (tp + tn) / total_records if total_records else None
    precision = compute_precision(tp, fp)
    recall = compute_recall(tp, fn)
    f1 = compute_f1(precision, recall)
    auc = compute_auc(labels, scores) if scores else None
    consistency = (
        consistency_matches / len(scores) if scores else None
    )

    return {
        "samples": total_records,
        "accuracy": accuracy,
        "precision_keep": precision,
        "recall_keep": recall,
        "f1_keep": f1,
        "auc_roc": auc,
        "score_label_consistency": consistency,
    }


def main() -> None:
    args = parse_args()
    gt_store = GroundTruthStore(args.ground_truth_root)

    groups: Dict[Tuple[str, str], List[Tuple[int, int, Optional[float]]]] = defaultdict(list)
    skipped = 0

    with open(args.pred_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata", {})
            operator = metadata.get("operator_name")
            level = metadata.get("level")

            sample_idx = row.get("sample_index")
            llm_output = row.get("llm_output", "")
            score, pred_label = parse_llm_output(llm_output)
            if pred_label is None:
                skipped += 1
                continue
            try:
                gt_label = gt_store.get_label(operator, level, sample_idx)
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}", file=sys.stderr)
                skipped += 1
                continue

            if gt_label is None:
                print(
                    f"[WARN] Missing ground truth for sample_index={sample_idx} "
                    f"operator={operator} level={level}",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            key = (operator, level)
            groups[key].append((gt_label, pred_label, score))

    if not groups:
        print("No matching samples found. Please check filters and inputs.")
        if skipped:
            print(f"Skipped samples: {skipped}")
        return

    overall_records: List[Tuple[int, int, Optional[float]]] = []
    per_group_outputs = []
    for key, records in groups.items():
        overall_records.extend(records)
        metrics = summarize_group(records, args.score_threshold)
        operator, level = key
        print(f"\n=== operator={operator} | level={level} ===")
        per_group_outputs.append(
            {"operator": operator, "level": level, **metrics}
        )
        for metric_name, value in metrics.items():
            if value is None:
                print(f"{metric_name:>26}: N/A")
            else:
                if metric_name == "samples":
                    print(f"{metric_name:>26}: {int(value)}")
                else:
                    print(f"{metric_name:>26}: {value:.4f}")

    overall_metrics = summarize_group(overall_records, args.score_threshold)
    print("\n=== OVERALL ===")
    for metric_name, value in overall_metrics.items():
        if value is None:
            print(f"{metric_name:>26}: N/A")
        else:
            if metric_name == "samples":
                print(f"{metric_name:>26}: {int(value)}")
            else:
                print(f"{metric_name:>26}: {value:.4f}")

    if skipped:
        print(f"\nSkipped samples (parse/lookup issues): {skipped}")

    pred_path = Path(args.pred_file).resolve()
    eval_dir = pred_path.parent / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / f"{pred_path.stem}_evaluation.json"
    summary = {
        "pred_file": str(pred_path),
        "score_threshold": args.score_threshold,
        "per_group": per_group_outputs,
        "overall": overall_metrics,
        "skipped_samples": skipped,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()

