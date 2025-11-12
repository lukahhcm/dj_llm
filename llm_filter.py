#!/usr/bin/env python3
"""
使用示例:
对文本进行打分与筛选:
   python llm_filter.py \
     --dataset ../data/test_data/test.jsonl \
     --output ../data/results/llm/test_results.jsonl \
     --model /mnt/data/shared/qwen/Qwen3-32B \
     --model-type vllm \
     --user-request-file ../data/user_request.jsonl \
     --gpu 0-7

    python llm_filter.py \
     --dataset ../data/test_data/test_cc_100.jsonl \
     --output ../data/results/llm/test_cc_100_results.jsonl \
     --model /mnt/data/shared/qwen/Qwen3-32B \
     --model-type vllm \
     --user-request-file ../data/user_request.jsonl \
     --gpu 0-7

输出格式:
- 每条样本包含 LLM 原始输出字符串 (scores 与 KEEP/REMOVE)，以及 filter_decision 字段 (true=KEEP, false=REMOVE)
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Parse GPU argument early before importing torch/vllm
def get_gpu_arg():
    """Extract --gpu argument from sys.argv before importing torch"""
    try:
        if '--gpu' in sys.argv:
            idx = sys.argv.index('--gpu')
            if idx + 1 < len(sys.argv):
                return sys.argv[idx + 1]
    except (ValueError, IndexError):
        pass
    return None

# Set CUDA_VISIBLE_DEVICES before importing torch
gpu_str = get_gpu_arg()
if gpu_str:
    # Parse GPU IDs (e.g., '0-7' or '0,1,2,3')
    gpu_ids = []
    for part in gpu_str.split(','):
        if '-' in part:
            start, end = part.split('-')
            gpu_ids.extend(range(int(start), int(end) + 1))
        else:
            gpu_ids.append(int(part))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    print(f"Setting CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

from loguru import logger

# Add data-juicer to path
sys.path.insert(0, str(Path(__file__).parent.parent / "data-juicer"))

try:
    from data_juicer.utils.model_utils import (
        get_model,
        prepare_model,
        update_sampling_params,
    )
    from data_juicer.utils.lazy_loader import LazyLoader
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please make sure data-juicer is installed and all dependencies are available.")
    sys.exit(1)

torch = LazyLoader("torch")
vllm = LazyLoader("vllm")


# ================================= 统一LLM处理器类 =================================

class UnifiedLLMProcessor:
    """统一的LLM处理器，基于llm_analysis_filter优化逻辑"""
    
    def __init__(
        self,
        model_name: str,
        model_type: str = "vllm",  # "vllm", "huggingface", "api"
        *,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        try_num: int = 3,
        model_params: Dict = {},
        sampling_params: Dict = {},
        **kwargs,
    ):
        """
        初始化统一LLM处理器
        
        Args:
            model_name: 模型名称或路径
            model_type: 模型类型 ("vllm", "huggingface", "api")
            api_endpoint: API端点（仅API模式）
            response_path: API响应路径（仅API模式）
            try_num: 重试次数
            model_params: 模型参数
            sampling_params: 采样参数
        """
        self.model_name = model_name
        self.model_type = model_type
        self.try_num = try_num
        
        # 确定模型类型
        self.enable_vllm = (model_type == "vllm")
        self.is_hf_model = (model_type == "huggingface")
        self.is_api_model = (model_type == "api")
        
        # 更新采样参数（复用llm_analysis_filter的优化）
        sampling_params = update_sampling_params(sampling_params, model_name, self.enable_vllm)
        
        # 初始化模型（复用llm_analysis_filter的逻辑）
        if self.enable_vllm:
            assert torch.cuda.device_count() >= 1, "must be executed in CUDA"
            # cannot initialize vllm replicas on different GPUs  
            # self.num_proc = 1  # 如果需要多进程处理可以取消注释
            if model_params.get("tensor_parallel_size") is None:
                tensor_parallel_size = torch.cuda.device_count()
                logger.info(f"Set tensor_parallel_size to {tensor_parallel_size} for vllm.")
                model_params["tensor_parallel_size"] = tensor_parallel_size
            
            self.model_key = prepare_model(
                model_type="vllm", 
                pretrained_model_name_or_path=model_name, 
                **model_params
            )
            self.sampling_params = vllm.SamplingParams(**sampling_params)
            
        elif self.is_hf_model:
            self.model_key = prepare_model(
                model_type="huggingface",
                pretrained_model_name_or_path=model_name,
                return_pipe=True,
                trust_remote_code=True,
                **model_params,
            )
            self.sampling_params = sampling_params
            
        else:  # API模式
            self.sampling_params = sampling_params
            self.model_key = prepare_model(
                model_type="api",
                model=model_name,
                endpoint=api_endpoint,
                response_path=response_path,
                **model_params,
            )
    
    def _get_model(self, rank=None):
        """获取模型实例（复用llm_analysis_filter的逻辑）"""
        if self.enable_vllm or self.is_hf_model:
            model, _ = get_model(self.model_key, rank, True)  # use_cuda=True
            return model
        else:
            return get_model(self.model_key, rank, True)
    
    def generate_response(
        self, 
        system_prompt: str, 
        user_content: str, 
        rank: Optional[int] = None
    ) -> Tuple[str, float]:
        """
        生成响应（复用llm_analysis_filter的高效逻辑）
        
        Args:
            system_prompt: 系统提示词
            user_content: 用户输入内容
            rank: 进程rank
            
        Returns:
            Tuple[str, float]: (生成的响应, 推理时间(秒))
        """
        model = self._get_model(rank)
        
        # 构建消息（与llm_analysis_filter相同的格式）
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        # 重试机制（完全复用llm_analysis_filter的逻辑）
        for attempt in range(self.try_num):
            try:
                # 记录推理开始时间
                start_time = time.time()
                
                if self.enable_vllm:
                    response = model.chat(messages, self.sampling_params)
                    output = response[0].outputs[0].text
                elif self.is_hf_model:
                    response = model(messages, return_full_text=False, **self.sampling_params)
                    output = response[0]["generated_text"]
                else:  # API模式
                    output = model(messages, **self.sampling_params)
                
                # 记录推理结束时间
                inference_time = time.time() - start_time
                
                if output and output.strip():
                    return output.strip(), inference_time
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
        
        logger.error(f"All {self.try_num} attempts failed")
        return "", 0.0


# ================================= 工具函数 =================================

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


def load_user_requests(file_path: str) -> List[Dict[str, Any]]:
    """Load all request objects from a JSONL file.

    Each object is returned in full (metadata preserved), and should contain a
    non-empty 'user_request' string field used for prompting.
    """
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
                val = obj.get("user_request")
                if isinstance(val, str) and val.strip():
                    requests.append(obj)
    except Exception as e:
        logger.error(f"Failed to read user_request file: {e}")
        return []

    if not requests:
        logger.warning("No valid user_request found; proceeding with empty list")
    return requests


def parse_gpu_ids(gpu_str):
    """Parse GPU IDs from string (e.g., '0,1,2,3' or '0-7')"""
    if not gpu_str:
        return None
    
    gpu_ids = []
    for part in gpu_str.split(','):
        if '-' in part:
            start, end = part.split('-')
            gpu_ids.extend(range(int(start), int(end) + 1))
        else:
            gpu_ids.append(int(part))
    
    return sorted(gpu_ids)


def parse_filter_output(raw_output: str) -> int:
    """
    解析filter任务的输出，提取KEEP/REMOVE决策
    学习llm_analysis_filter中的parse_output函数思路
    
    Args:
        raw_output: LLM的原始输出
        
    Returns:
        int: 1代表KEEP，0代表REMOVE
    """
    if not raw_output or not raw_output.strip():
        logger.warning("Empty output from LLM, defaulting to REMOVE")
        return 0
    
    # 转换为大写并去除空格，便于匹配
    output_clean = raw_output.strip().upper()
    
    # 检查是否包含KEEP或REMOVE
    if "KEEP" in output_clean:
        return 1
    elif "REMOVE" in output_clean:
        return 0
    else:
        # 如果都不包含，记录警告并默认为REMOVE（保守策略）
        logger.warning(f"Cannot parse filter decision from output: '{raw_output}', defaulting to REMOVE")
        return 0


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
    parser = argparse.ArgumentParser(description="LLM Scoring & Filtering Script - Unified Version")
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to input test dataset")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save processed results")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="vllm",
                        choices=["vllm", "huggingface", "api"],
                        help="Model type")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Specify GPU IDs to use (e.g., '0,1,2,3' or '0-7')")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--try-num", type=int, default=3,
                        help="Number of retry attempts")
    parser.add_argument("--user-request-file", type=str, default="/mnt/workspace/chenming.hyc/llm_dj_gap/user_request.jsonl",
                        help="Path to JSONL file containing 'user_request' field(s)")
    
    return parser.parse_args()


# ================================= 主程序 =================================

def main():
    args = parse_args()
    
    # Parse GPU IDs if provided
    gpu_ids = parse_gpu_ids(args.gpu) if args.gpu else None
    if gpu_ids:
        logger.info(f"Using GPUs: {gpu_ids}")
    
    # Load all user requests
    user_requests = load_user_requests(args.user_request_file)
    
    # Load test dataset
    samples = load_test_dataset(args.dataset)
    
    # 准备模型参数（基于llm_analysis_filter的优化）
    model_params = {}
    if args.tensor_parallel_size:
        model_params["tensor_parallel_size"] = args.tensor_parallel_size
    
    sampling_params = {}
    if args.temperature:
        sampling_params["temperature"] = args.temperature
    if args.max_tokens:
        # 先设置用户指定的值，update_sampling_params会保留已有参数
        if args.model_type == "vllm":
            sampling_params["max_tokens"] = args.max_tokens
        else:
            sampling_params["max_new_tokens"] = args.max_tokens

    enable_vllm = (args.model_type == "vllm")
    sampling_params = update_sampling_params(sampling_params, args.model, enable_vllm)
    
    # Initialize unified processor
    processor = UnifiedLLMProcessor(
        model_name=args.model,
        model_type=args.model_type,
        try_num=args.try_num,
        model_params=model_params,
        sampling_params=sampling_params
    )
    
    # Process each sample（每条数据遍历所有request）
    results = []
    kept_count = 0
    inference_times = []  # 记录所有推理时间
    total_start_time = time.time()  # 记录总开始时间
    
    for i, sample in enumerate(samples):
        logger.info(f"Processing sample {i+1}/{len(samples)}")
        for r_idx, request_obj in enumerate(user_requests or []):
            logger.info(f"  Using request {r_idx+1}/{len(user_requests) if user_requests else 1}")
            
            input_text = sample.get("text", "")
            user_request = request_obj.get("user_request", "")

            # 构建系统提示，包含当前 user_request 与样本文本
            system_prompt = (
                "You are an experienced text processing expert. Carefully analyze the input text sample according to the given user request. "
                "Provide a single overall score (a floating-point number from 0 to 10, where higher values indicate better quality) that reflects how well the text aligns with the user’s specific need."
                "Then, determine whether the sample should be retained in light of the score and the request. "
                "Output only the scores and the retention decision, separated by a space. Use \"KEEP\" to retain the sample and \"REMOVE\" to filter it out.\n\n"
                "The user request on text processing is:\n"
                f"{user_request}\n"
                "The input text is:\n"
                f"{input_text}"
                "/no_think"
            )

            # 生成输出（仅使用系统提示，用户内容为空）
            output, inference_time = processor.generate_response(system_prompt, "")
            inference_times.append(inference_time)
            filter_decision = parse_filter_output(output)
            
            # Create result
            result = {
                "sample_index": i,
                "request_index": r_idx,
                "user_request": user_request,
                "metadata": request_obj,  # 保留原始request的整条所有字段
                "original": sample,
                "llm_output": output,
                "filter_decision": bool(filter_decision),  # true=KEEP, false=REMOVE
                "inference_time": round(inference_time, 4),  # 推理时间（秒），保留4位小数
            }

            if filter_decision:
                kept_count += 1
                logger.debug(f"  Sample {i+1} (req {r_idx}): KEEP (inference_time: {inference_time:.4f}s)")
            else:
                logger.debug(f"  Sample {i+1} (req {r_idx}): REMOVE (inference_time: {inference_time:.4f}s)")
            
            results.append(result)
    
    # 计算总时间和统计信息
    total_time = time.time() - total_start_time
    total_inference_time = sum(inference_times)
    avg_inference_time = total_inference_time / len(inference_times) if inference_times else 0
    min_inference_time = min(inference_times) if inference_times else 0
    max_inference_time = max(inference_times) if inference_times else 0
    
    # 打印统计信息
    total_samples = len(samples) * len(user_requests) if user_requests else len(samples)
    removed_count = total_samples - kept_count
    keep_rate = kept_count / total_samples * 100 if total_samples > 0 else 0
    logger.info(f"Filter Results: {kept_count}/{total_samples} samples kept ({keep_rate:.1f}%), {removed_count} removed")
    
    # 打印时间统计信息
    logger.info("=" * 60)
    logger.info("Inference Time Statistics:")
    logger.info(f"  Total samples processed: {len(inference_times)}")
    logger.info(f"  Total inference time: {total_inference_time:.4f}s ({total_inference_time/60:.2f} minutes)")
    logger.info(f"  Average inference time per sample: {avg_inference_time:.4f}s")
    logger.info(f"  Min inference time: {min_inference_time:.4f}s")
    logger.info(f"  Max inference time: {max_inference_time:.4f}s")
    logger.info(f"  Total wall-clock time: {total_time:.4f}s ({total_time/60:.2f} minutes)")
    logger.info(f"  Overhead time: {total_time - total_inference_time:.4f}s ({(total_time - total_inference_time)/total_time*100:.2f}%)")
    if avg_inference_time > 0:
        logger.info(f"  Throughput: {1/avg_inference_time:.2f} samples/second")
    logger.info("=" * 60)
    
    # Save results
    save_results(results, args.output)
    
    logger.info("LLM filtering completed successfully")


if __name__ == "__main__":
    main()
