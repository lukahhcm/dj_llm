#!/usr/bin/env python3
"""
生成标准答案脚本

使用data-juicer的算子对数据集进行筛选，生成标准答案，用于与LLM结果进行对比。

使用示例:
    # 运行所有算子（从user_request文件中自动提取）
    python generate_ground_truth.py \
        --dataset ../dj_llm_data/test_data/test_cc_100.jsonl \
        --base-output-dir ../dj_llm_data/results/ground_truth \
        --user-request-file ./user_request_length.jsonl

    # 只运行指定算子
    python generate_ground_truth.py \
        --operator-name text_length_filter \
        --dataset ../dj_llm_data/test_data/test_cc_100.jsonl \
        --base-output-dir ../dj_llm_data/results/ground_truth \
        --user-request-file ./user_request_length.jsonl
"""
import os
import sys
import json
import argparse
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional

# 设置 HuggingFace 镜像源（如果没有设置的话）
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"Set HF_ENDPOINT={os.environ['HF_ENDPOINT']}")

# Add data-juicer to path
sys.path.insert(0, str(Path(__file__).parent.parent / "data-juicer"))

from loguru import logger

# 导入算子模块以触发注册（算子通过装饰器自动注册）
import data_juicer.ops.filter  # noqa: F401
import data_juicer.ops.mapper

from data_juicer.ops.base_op import OPERATORS
from data_juicer.utils.constant import Fields


def load_test_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load test dataset from file."""
    logger.info(f"Loading test dataset from: {dataset_path}")
    
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def load_user_requests(file_path: str, operator_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load all request objects from a JSONL file."""
    logger.info(f"Loading user requests from: {file_path}")
    requests: List[Dict[str, Any]] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                
                # 如果指定了算子名称，只加载匹配的
                if operator_name and obj.get("operator_name") != operator_name:
                    continue
                
                val = obj.get("user_request")
                if isinstance(val, str) and val.strip():
                    requests.append(obj)
    except Exception as e:
        logger.error(f"Failed to read user_request file: {e}")
        return []

    if not requests:
        logger.warning("No valid user_request found; proceeding with empty list")
    return requests


def extract_all_operator_names(file_path: str) -> List[str]:
    """从文件中提取所有唯一的算子名称"""
    operator_names = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                
                op_name = obj.get("operator_name")
                if op_name and isinstance(op_name, str):
                    operator_names.add(op_name)
    except Exception as e:
        logger.error(f"Failed to read user_request file: {e}")
        return []
    
    return sorted(list(operator_names))


def create_filter_operator(operator_name: str, parameters: Dict[str, Any]) -> Any:
    """
    根据算子名称和参数创建算子实例
    
    Args:
        operator_name: 算子名称
        parameters: 算子参数
        
    Returns:
        算子实例
    """
    if operator_name not in OPERATORS.modules:
        raise ValueError(f"Unknown operator: {operator_name}")
    
    op_class = OPERATORS.modules[operator_name]
    
    # 根据不同的算子名称映射参数
    op_kwargs = {}
    
    if operator_name == "text_length_filter":
        op_kwargs["min_len"] = parameters.get("min_len", 10)
        op_kwargs["max_len"] = parameters.get("max_len", sys.maxsize)
    elif operator_name == "average_line_length_filter":
        op_kwargs["min_len"] = parameters.get("min_len", 10)
        op_kwargs["max_len"] = parameters.get("max_len", sys.maxsize)
    elif operator_name == "maximum_line_length_filter":
        op_kwargs["min_len"] = parameters.get("min_len", 10)
        op_kwargs["max_len"] = parameters.get("max_len", sys.maxsize)
    elif operator_name == "token_num_filter":
        op_kwargs["min_num"] = parameters.get("min_num", 1)
        op_kwargs["max_num"] = parameters.get("max_num", sys.maxsize)
        # 如果提供了 hf_tokenizer 参数，使用它；否则尝试使用本地 Qwen tokenizer
        if "hf_tokenizer" in parameters:
            op_kwargs["hf_tokenizer"] = parameters["hf_tokenizer"]
        else:
            # 尝试使用本地 Qwen tokenizer（测试显示可以正常工作）
            possible_tokenizers = [
                "/mnt/data/shared/qwen/Qwen3-32B",
                "/mnt/data/shared/qwen/Qwen3-8B",
                "/mnt/data/shared/qwen/Qwen3-4B-Base",
                "/mnt/data/shared/qwen/Qwen3-1.7B-Base",
            ]
            # 选择一个存在的模型路径作为 tokenizer
            tokenizer_found = False
            for tokenizer_path in possible_tokenizers:
                if os.path.exists(tokenizer_path):
                    op_kwargs["hf_tokenizer"] = tokenizer_path
                    logger.info(f"Using local Qwen tokenizer for {operator_name}: {tokenizer_path}")
                    tokenizer_found = True
                    break
            if not tokenizer_found:
                # 如果都没有找到，使用默认值（可能需要网络，可能会失败）
                logger.warning(
                    f"No local tokenizer found for {operator_name}. "
                    f"Will use default tokenizer 'EleutherAI/pythia-6.9b-deduped' (may require network connection)."
                )
    elif operator_name == "words_num_filter":
        op_kwargs["min_num"] = parameters.get("min_num", 1)
        op_kwargs["max_num"] = parameters.get("max_num", sys.maxsize)
    else:
        # 通用情况：直接传递参数
        op_kwargs = parameters
    
    logger.debug(f"Creating operator {operator_name} with kwargs: {op_kwargs}")
    return op_class(**op_kwargs)


def generate_output_path(
    base_output_dir: str,
    operator_name: Optional[str] = None,
    level: Optional[str] = None,
    default_filename: str = "results.jsonl"
) -> str:
    """根据算子名称和级别生成输出路径"""
    path_parts = [base_output_dir]
    
    if operator_name:
        path_parts.append(operator_name)
    
    if level:
        level_lower = level.lower()
        path_parts.append(level_lower)
    
    output_dir = Path(*path_parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return str(output_dir / default_filename)


def process_dataset_with_operator(
    samples: List[Dict[str, Any]],
    operator_name: str,
    parameters: Dict[str, Any]
) -> List[bool]:
    """
    使用算子对数据集进行筛选，返回筛选决策列表
    
    Args:
        samples: 样本列表
        operator_name: 算子名称
        parameters: 算子参数
        
    Returns:
        List[bool]: 每个样本的筛选决策（True=KEEP, False=REMOVE）
    """
    # 创建算子实例
    op = create_filter_operator(operator_name, parameters)
    
    # 准备数据集格式（需要包含text和stats字段）
    dataset_dict = {
        "text": [sample.get("text", "") for sample in samples],
        Fields.stats: [{} for _ in samples],
        Fields.context: [{} for _ in samples],
    }
    
    # 计算统计信息
    logger.debug(f"Computing stats for {len(samples)} samples with {operator_name}...")
    
    # 批量处理
    if op.is_batched_op():
        # 检查 compute_stats_batched 方法是否支持 context 参数
        sig = inspect.signature(op.compute_stats_batched)
        has_context_param = "context" in sig.parameters
        
        # 使用批量接口
        if has_context_param:
            batch_dict = op.compute_stats_batched(dataset_dict, context=True)
        else:
            batch_dict = op.compute_stats_batched(dataset_dict)
        
        # 进行筛选决策
        decision_iter = op.process_batched(batch_dict)
        decisions = list(decision_iter)
    else:
        # 单样本处理
        # 检查 compute_stats_single 方法是否支持 context 参数
        sig = inspect.signature(op.compute_stats_single)
        has_context_param = "context" in sig.parameters
        
        decisions = []
        for i, sample in enumerate(samples):
            sample_dict = {
                "text": sample.get("text", ""),
                Fields.stats: {},
                Fields.context: {},
            }
            if has_context_param:
                result = op.compute_stats_single(sample_dict, context=True)
            else:
                result = op.compute_stats_single(sample_dict)
            decision = op.process_single({Fields.stats: result[Fields.stats]})
            decisions.append(decision)
    
    return decisions


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save processed results to file."""
    logger.info(f"Saving results to: {output_path}")
    
    # Create output directory if not exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(results)} samples to {output_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate Ground Truth using Data-Juicer Operators")
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to input test dataset")
    parser.add_argument("--base-output-dir", type=str, required=True,
                        help="Base output directory for organizing results")
    parser.add_argument("--user-request-file", type=str, required=True,
                        help="Path to JSONL file containing user requests")
    parser.add_argument("--operator-name", type=str, default=None,
                        help="Operator name (optional, if not specified, runs all)")
    
    return parser.parse_args()


def main():
    import sys
    args = parse_args()
    
    # Load test dataset
    samples = load_test_dataset(args.dataset)
    
    # 确定要运行的算子
    if args.operator_name:
        operator_names = [args.operator_name]
        logger.info(f"Using specified operator: {args.operator_name}")
    else:
        operator_names = extract_all_operator_names(args.user_request_file)
        if not operator_names:
            logger.error(f"No operators found in {args.user_request_file}")
            sys.exit(1)
        logger.info(f"Found {len(operator_names)} operator(s): {operator_names}")
    
    # 按算子和级别分组处理
    LEVEL_MAPPING = {
        "LOOSE": "loose",
        "MEDIUM": "medium",
        "STRICT": "strict",
    }
    
    total_kept = 0
    total_samples = 0
    
    for operator_name in operator_names:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Processing operator: {operator_name}")
        logger.info("=" * 80)
        
        # 加载该算子的所有用户请求（按级别分组）
        all_requests = load_user_requests(args.user_request_file, operator_name)
        
        # 按级别分组
        requests_by_level = {}
        for req in all_requests:
            level = req.get("level", "").upper()
            if level not in requests_by_level:
                requests_by_level[level] = []
            requests_by_level[level].append(req)
        
        for level, requests in requests_by_level.items():
            logger.info(f"Processing level: {level}")
            
            # 使用第一个请求的参数（同一级别应该有相同的参数）
            if not requests:
                continue
            
            request_obj = requests[0]
            parameters = request_obj.get("parameters", {})
            user_request = request_obj.get("user_request", "")
            
            logger.info(f"  Parameters: {parameters}")
            logger.info(f"  User request: {user_request}")
            
            # 使用算子处理数据集
            try:
                decisions = process_dataset_with_operator(samples, operator_name, parameters)
            except Exception as e:
                logger.error(f"Failed to process dataset with {operator_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 生成输出路径
            level_lower = LEVEL_MAPPING.get(level, level.lower())
            output_path = generate_output_path(
                base_output_dir=args.base_output_dir,
                operator_name=operator_name,
                level=level_lower,
                default_filename="results.jsonl"
            )
            
            # 准备结果
            results = []
            kept_count = 0
            
            for i, (sample, decision) in enumerate(zip(samples, decisions)):
                result = {
                    "sample_index": i,
                    "operator_name": operator_name,
                    "level": level_lower,
                    "user_request": user_request,
                    "parameters": parameters,
                    "metadata": request_obj,
                    "original": sample,
                    "filter_decision": bool(decision),  # true=KEEP, false=REMOVE
                    "operator_output": {
                        "decision": "KEEP" if decision else "REMOVE",
                        "parameters": parameters,
                    }
                }
                
                if decision:
                    kept_count += 1
                
                results.append(result)
            
            # 保存结果
            save_results(results, output_path)
            
            removed_count = len(samples) - kept_count
            keep_rate = kept_count / len(samples) * 100 if samples else 0
            
            logger.info(f"  Filter Results: {kept_count}/{len(samples)} samples kept ({keep_rate:.1f}%), {removed_count} removed")
            
            total_kept += kept_count
            total_samples += len(samples)
    
    # 打印最终统计
    logger.info("")
    logger.info("=" * 80)
    logger.info("Ground Truth Generation Summary")
    logger.info("=" * 80)
    logger.info(f"Total samples processed: {total_samples}")
    logger.info(f"Total kept: {total_kept}")
    logger.info("=" * 80)
    logger.info("Ground truth generation completed successfully")


if __name__ == "__main__":
    main()

