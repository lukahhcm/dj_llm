#!/usr/bin/env python3
"""
Aggregate evaluation JSON files into operator-centric tables.

Usage example:
    # Process 32b model evaluations
    python evaluation/aggregate_operator_evaluations.py \
        --eval-dir ../dj_llm_data/results/llm/evaluation \
        --model-size 32b \
        --output ../dj_llm_data/results/llm/evaluation/operator_tables_qwen_32b.txt
    
    # Process 8b model evaluations
    python evaluation/aggregate_operator_evaluations.py \
        --eval-dir ../dj_llm_data/results/llm/evaluation \
        --model-size 8b \
        --output ../dj_llm_data/results/llm/evaluation/operator_tables_qwen_8b.txt
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


MetricEntry = Dict[str, Optional[float]]
OperatorData = Dict[str, Dict[int, MetricEntry]]  # level -> batch -> metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LLM filter evaluation JSON files into operator tables."
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Directory containing *_evaluation.json files.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the generated tables. Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--model-size",
        choices=["8b", "32b"],
        help="Filter evaluation files by model size (8b or 32b). If omitted, processes all files.",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        default=["LOOSE", "MEDIUM", "STRICT"],
        help="Display order for levels (default: LOOSE MEDIUM STRICT).",
    )
    return parser.parse_args()


def read_eval_file(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_batch_id(stem: str) -> Optional[int]:
    match = re.search(r"batch_(\d+)", stem)
    if match:
        return int(match.group(1))
    return None


def collect_metrics(eval_dir: Path, model_size: Optional[str] = None) -> Dict[str, OperatorData]:
    operator_metrics: Dict[str, OperatorData] = defaultdict(
        lambda: defaultdict(dict)
    )
    for path in sorted(eval_dir.glob("*_evaluation.json")):
        # Filter by model size if specified
        if model_size:
            if model_size not in path.stem:
                continue
        
        batch_id = extract_batch_id(path.stem)
        if batch_id is None:
            continue

        payload = read_eval_file(path)
        for entry in payload.get("per_group", []):
            operator = entry["operator"]
            level = entry["level"]
            operator_metrics[operator][level][batch_id] = entry

    return operator_metrics


def format_value(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_table(
    operator: str,
    level_map: OperatorData,
    levels_order: List[str],
    metric_columns: List[tuple],
) -> str:
    rows: List[List[str]] = []
    all_batches = sorted(
        {batch for metrics in level_map.values() for batch in metrics.keys()}
    )

    headers = ["Level", "Batch"] + [label for _, label in metric_columns]
    col_widths = [len(h) for h in headers]

    for level in levels_order:
        batch_dict = level_map.get(level, {})
        for batch in all_batches:
            metrics = batch_dict.get(batch)
            if not metrics:
                continue
            row = [level, str(batch)]
            for key, _ in metric_columns:
                row.append(format_value(metrics.get(key)))
            rows.append(row)

    if not rows:
        return ""

    for row in rows:
        col_widths = [
            max(width, len(str(cell))) for width, cell in zip(col_widths, row)
        ]

    lines = [f"=== Operator: {operator} ==="]
    header_line = " | ".join(
        h.ljust(w) for h, w in zip(headers, col_widths)
    )
    separator = "-+-".join("-" * w for w in col_widths)
    lines.append(header_line)
    lines.append(separator)
    for row in rows:
        lines.append(
            " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).resolve()
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    operator_metrics = collect_metrics(eval_dir, args.model_size)
    metric_columns = [
        ("accuracy", "Accuracy"),
        ("precision_keep", "Precision"),
        ("recall_keep", "Recall"),
        ("f1_keep", "F1"),
        ("auc_roc", "AUC"),
        ("score_label_consistency", "Score/Label"),
    ]

    tables: List[str] = []
    for operator in sorted(operator_metrics.keys()):
        table = render_table(
            operator, operator_metrics[operator], args.levels, metric_columns
        )
        if table:
            tables.append(table)

    output_text = "\n".join(tables).strip()
    if not output_text:
        output_text = "No tables generated. Verify evaluation files."

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text + "\n")
        print(f"Saved tables to {output_path}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()

