#!/usr/bin/env python3
"""
Compare operator evaluation metrics between two model sizes using the existing
aggregate evaluation outputs.

Usage example:
    python evaluation/compare_model_differences.py \
        --eval-dir ../dj_llm_data/results/llm/evaluation \
        --model-a 32b \
        --model-b 8b \
        --output ../dj_llm_data/results/llm/evaluation/operator_diff_32b_vs_8b.txt
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from aggregate_operator_evaluations import (
    MetricEntry,
    OperatorData,
    collect_metrics,
    format_value,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-operator metric differences between two model sizes."
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Directory containing *_evaluation.json files.",
    )
    parser.add_argument(
        "--model-a",
        required=True,
        choices=["8b", "32b"],
        help="First model size (the minuend).",
    )
    parser.add_argument(
        "--model-b",
        required=True,
        choices=["8b", "32b"],
        help="Second model size (the subtrahend).",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        default=["LOOSE", "MEDIUM", "STRICT"],
        help="Display order for levels (default: LOOSE MEDIUM STRICT).",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Prints to stdout if omitted.",
    )
    return parser.parse_args()


def format_delta(a: Optional[float], b: Optional[float]) -> str:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "N/A"
    diff = a - b
    formatted = f"{diff:.4f}".rstrip("0").rstrip(".")
    if formatted == "":
        formatted = "0"
    if diff > 0:
        return f"+{formatted}"
    return formatted


def render_comparison_table(
    operator: str,
    model_a_map: OperatorData,
    model_b_map: OperatorData,
    model_a_label: str,
    model_b_label: str,
    levels_order: List[str],
    metric_columns: List[tuple],
) -> str:
    rows: List[List[str]] = []
    all_batches = sorted(
        {
            batch
            for metrics in list(model_a_map.values()) + list(model_b_map.values())
            for batch in metrics.keys()
        }
    )

    headers = ["Level", "Batch"] + [f"Δ {label}" for _, label in metric_columns]
    col_widths = [len(h) for h in headers]

    for level in levels_order:
        batches_a = model_a_map.get(level, {})
        batches_b = model_b_map.get(level, {})
        for batch in all_batches:
            metrics_a = batches_a.get(batch)
            metrics_b = batches_b.get(batch)
            if not metrics_a and not metrics_b:
                continue
            row = [level, str(batch)]
            for key, _ in metric_columns:
                value_a = metrics_a.get(key) if metrics_a else None
                value_b = metrics_b.get(key) if metrics_b else None
                row.append(format_delta(value_a, value_b))
            rows.append(row)

    if not rows:
        return ""

    for row in rows:
        col_widths = [
            max(width, len(str(cell))) for width, cell in zip(col_widths, row)
        ]

    lines = [
        f"=== Operator: {operator} ({model_a_label} - {model_b_label}) ==="
    ]
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

    if args.model_a == args.model_b:
        raise ValueError("Model sizes must be different to compute differences.")

    metric_columns = [
        ("accuracy", "Accuracy"),
        ("precision_keep", "Precision"),
        ("recall_keep", "Recall"),
        ("f1_keep", "F1"),
        ("auc_roc", "AUC"),
    ]

    model_a_metrics = collect_metrics(eval_dir, args.model_a)
    model_b_metrics = collect_metrics(eval_dir, args.model_b)
    all_operators = sorted(
        set(model_a_metrics.keys()) | set(model_b_metrics.keys())
    )

    tables = []
    for operator in all_operators:
        table = render_comparison_table(
            operator,
            model_a_metrics.get(operator, defaultdict(dict)),
            model_b_metrics.get(operator, defaultdict(dict)),
            args.model_a,
            args.model_b,
            args.levels,
            metric_columns,
        )
        if table:
            tables.append(table)

    output_text = "\n".join(tables).strip()
    if not output_text:
        output_text = "No comparison tables generated. Verify evaluation files."

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text + "\n")
        print(f"Saved comparison tables to {output_path}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()

