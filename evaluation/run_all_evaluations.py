#!/usr/bin/env python3
"""
自动运行所有评估脚本

1. 对 ../dj_llm_data/results/llm/ 目录下的所有 jsonl 文件运行 evaluate_llm_filter_sklearn.py
2. 运行 aggregate_operator_evaluations.py 为不同大小的模型生成表格

使用示例:
    # 从 dj_llm 目录运行
    python evaluation/run_all_evaluations.py --evaluate-script evaluation/evaluate_llm_filter_sklearn.py
    python evaluation/run_all_evaluations.py --evaluate-script evaluation/evaluate_llm_filter.py
    
    # 或从 evaluation 目录运行
    cd evaluation && python run_all_evaluations.py
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def find_jsonl_files(results_dir: Path) -> List[Path]:
    """查找所有 jsonl 文件，排除 evaluation 目录"""
    jsonl_files = []
    for path in results_dir.glob("*.jsonl"):
        if path.is_file():
            jsonl_files.append(path)
    return sorted(jsonl_files)


def run_evaluation(pred_file: Path, ground_truth_root: Path, script_path: Path) -> bool:
    """运行单个评估脚本"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {pred_file.name}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        str(script_path),
        "--pred-file",
        str(pred_file),
        "--ground-truth-root",
        str(ground_truth_root),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
        )
        print(f"✓ Successfully evaluated {pred_file.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to evaluate {pred_file.name}: {e}", file=sys.stderr)
        return False


def run_aggregate(
    aggregate_script: Path,
    eval_dir: Path,
    model_size: str,
    output_path: Path,
) -> bool:
    """运行聚合脚本生成表格"""
    print(f"\n{'='*80}")
    print(f"Generating tables for {model_size} model")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        str(aggregate_script),
        "--eval-dir",
        str(eval_dir),
        "--model-size",
        model_size,
        "--output",
        str(output_path),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
        )
        print(f"✓ Successfully generated table for {model_size} model")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate table for {model_size} model: {e}", file=sys.stderr)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically run evaluations for all JSONL files and generate tables."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../dj_llm_data/results/llm",
        help="Directory containing prediction JSONL files (default: ../dj_llm_data/results/llm)",
    )
    parser.add_argument(
        "--ground-truth-root",
        type=str,
        default="../dj_llm_data/results/ground_truth",
        help="Root directory containing ground truth files (default: ../dj_llm_data/results/ground_truth)",
    )
    parser.add_argument(
        "--evaluate-script",
        type=str,
        default="evaluation/evaluate_llm_filter_sklearn.py",
        help="Path to evaluation script (default: evaluation/evaluate_llm_filter_sklearn.py)",
    )
    parser.add_argument(
        "--aggregate-script",
        type=str,
        default="evaluation/aggregate_operator_evaluations.py",
        help="Path to aggregate script (default: evaluation/aggregate_operator_evaluations.py)",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip running evaluations, only generate tables",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip generating tables, only run evaluations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # 解析路径
    # script_dir 是 evaluation/ 目录
    script_dir = Path(__file__).parent.resolve()
    # 如果从 dj_llm 目录运行，需要找到 dj_llm 目录（script_dir 的父目录）
    # 如果从 evaluation 目录运行，也需要找到 dj_llm 目录
    dj_llm_dir = script_dir.parent.resolve()
    
    # 数据路径相对于 dj_llm 目录
    results_dir = (dj_llm_dir / args.results_dir).resolve()
    ground_truth_root = (dj_llm_dir / args.ground_truth_root).resolve()
    
    # 脚本路径：如果包含 "evaluation/"，说明是从 dj_llm 目录运行的；否则是相对于 script_dir
    evaluate_script_path = Path(args.evaluate_script)
    if "evaluation/" in str(evaluate_script_path) or evaluate_script_path.is_absolute():
        evaluate_script = evaluate_script_path.resolve()
    else:
        evaluate_script = (script_dir / args.evaluate_script).resolve()
    
    # aggregate_script 同样处理
    aggregate_script_path = Path(args.aggregate_script)
    if aggregate_script_path.is_absolute():
        aggregate_script = aggregate_script_path.resolve()
    elif "evaluation/" in str(aggregate_script_path):
        aggregate_script = aggregate_script_path.resolve()
    else:
        aggregate_script = (script_dir / args.aggregate_script).resolve()
    
    eval_dir = results_dir / "evaluation"
    
    # 验证路径
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not ground_truth_root.exists():
        print(f"Error: Ground truth root not found: {ground_truth_root}", file=sys.stderr)
        sys.exit(1)
    
    if not evaluate_script.exists() and not args.skip_evaluation:
        print(f"Error: Evaluation script not found: {evaluate_script}", file=sys.stderr)
        sys.exit(1)
    
    if not aggregate_script.exists() and not args.skip_aggregate:
        print(f"Error: Aggregate script not found: {aggregate_script}", file=sys.stderr)
        sys.exit(1)
    
    # 步骤1: 运行所有评估
    if not args.skip_evaluation:
        print("\n" + "="*80)
        print("STEP 1: Running evaluations for all JSONL files")
        print("="*80)
        
        jsonl_files = find_jsonl_files(results_dir)
        if not jsonl_files:
            print(f"No JSONL files found in {results_dir}")
        else:
            print(f"Found {len(jsonl_files)} JSONL files to evaluate")
            
            success_count = 0
            for jsonl_file in jsonl_files:
                if run_evaluation(jsonl_file, ground_truth_root, evaluate_script):
                    success_count += 1
            
            print(f"\n{'='*80}")
            print(f"Evaluation summary: {success_count}/{len(jsonl_files)} files processed successfully")
            print(f"{'='*80}")
    
    # 步骤2: 生成表格
    if not args.skip_aggregate:
        print("\n" + "="*80)
        print("STEP 2: Generating operator tables")
        print("="*80)
        
        if not eval_dir.exists():
            print(f"Error: Evaluation directory not found: {eval_dir}", file=sys.stderr)
            sys.exit(1)
        
        # 为 8b 和 32b 模型分别生成表格
        model_sizes = ["8b", "32b"]
        all_success = True
        
        for model_size in model_sizes:
            output_path = eval_dir / f"operator_tables_qwen_{model_size}.txt"
            if run_aggregate(aggregate_script, eval_dir, model_size, output_path):
                print(f"Table saved to: {output_path}")
            else:
                all_success = False
        
        if all_success:
            print(f"\n{'='*80}")
            print("All tables generated successfully")
            print(f"{'='*80}")
        else:
            print(f"\n{'='*80}")
            print("Some tables failed to generate")
            print(f"{'='*80}")
            sys.exit(1)
    
    print("\n✓ All tasks completed!")


if __name__ == "__main__":
    main()

