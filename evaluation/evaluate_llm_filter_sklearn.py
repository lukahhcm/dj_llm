#!/usr/bin/env python3
"""
使用示例:
    python evaluation/evaluate_llm_filter_sklearn.py \
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

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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


def summarize_group(
    records: Iterable[Tuple[int, int, Optional[float]]],
    score_threshold: float,
) -> Dict[str, Optional[float]]:
    y_true_list: List[int] = []
    y_pred_list: List[int] = []
    labels: List[int] = []
    scores: List[float] = []
    consistency_matches = 0

    for y_true, y_pred, score in records:
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

        if score is not None:
            labels.append(y_true)
            scores.append(score)
            inferred_keep = 1 if score >= score_threshold else 0
            if inferred_keep == y_pred:
                consistency_matches += 1

    total_records = len(y_true_list)
    if total_records == 0:
        return {
            "samples": 0,
            "accuracy": None,
            "precision_keep": None,
            "recall_keep": None,
            "f1_keep": None,
            "auc_roc": None,
            "score_label_consistency": None,
        }

    # 使用 sklearn.metrics 计算指标
    accuracy = accuracy_score(y_true_list, y_pred_list)

    # precision: 当 tp+fp == 0 (即没有预测为正的样本) 时返回 None
    # recall: 当 tp+fn == 0 (即没有真实为正的样本) 时返回 None
    pred_positive_count = sum(y_pred_list)  # tp + fp
    true_positive_count = sum(y_true_list)  # tp + fn

    if pred_positive_count > 0:
        precision = precision_score(y_true_list, y_pred_list, zero_division=0)
    else:
        precision = None

    if true_positive_count > 0:
        recall = recall_score(y_true_list, y_pred_list, zero_division=0)
    else:
        recall = None

    # f1: 当 precision 或 recall 为 None 时返回 None
    # 当 precision 和 recall 都为 0 时，sklearn 的 f1_score 会返回 0（zero_division=0）
    if precision is not None and recall is not None:
        f1 = f1_score(y_true_list, y_pred_list, zero_division=0)
    else:
        f1 = None

    # AUC 需要正负样本都存在
    auc = None
    if scores and len(set(labels)) == 2:  # 确保有正负两类
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = None

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

