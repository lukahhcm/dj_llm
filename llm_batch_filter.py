#!/usr/bin/env python3
"""
使用示例:
对文本进行打分与筛选:
    python llm_batch_filter.py \
        --dataset ../dj_llm_data/test_data/test_cc_100.jsonl \
        --output ../dj_llm_data/results/test_cc_100_qwen_32b_batch_5.jsonl \
        --model /mnt/data/shared/qwen/Qwen3-32B \
        --model-type vllm \
        --user-request-file ./user_request_length.jsonl \
        --gpu 0-7 \
        --batch-size 5


输出格式:
- 每条样本包含 LLM 原始输出字符串 (scores 与 KEEP/REMOVE)，以及 filter_decision 字段 (true=KEEP, false=REMOVE)
"""
import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

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
gpu_env_message = None
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
    gpu_env_message = f"Setting CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"

from loguru import logger

# LOG_DIR = Path(__file__).parent / "logs"
# LOG_DIR.mkdir(parents=True, exist_ok=True)
# log_file_path = LOG_DIR / f"llm_batch_filter_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
# logger.add(log_file_path, encoding="utf-8", enqueue=True)
# logger.info(f"Logging to {log_file_path}")

# if gpu_env_message:
#     logger.info(gpu_env_message)

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
            gpu_count = torch.cuda.device_count()
            assert gpu_count >= 1, "must be executed in CUDA"
            logger.info(f"Detected {gpu_count} GPU(s) (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
            # cannot initialize vllm replicas on different GPUs  
            # self.num_proc = 1  # 如果需要多进程处理可以取消注释
            if model_params.get("tensor_parallel_size") is None:
                tensor_parallel_size = gpu_count
                logger.info(f"Auto-detected tensor_parallel_size: {tensor_parallel_size} (using all {gpu_count} visible GPU(s))")
                model_params["tensor_parallel_size"] = tensor_parallel_size
            else:
                tensor_parallel_size = model_params["tensor_parallel_size"]
                logger.info(f"Using user-specified tensor_parallel_size: {tensor_parallel_size}")
            
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

    def generate_response_batch(
        self, 
        system_prompt: str, 
        user_content: str,  # 包含多条文本的用户内容
        rank: Optional[int] = None
    ) -> Tuple[str, float]:
        """
        生成响应（批量处理多条文本在一个prompt中）
        
        Args:
            system_prompt: 系统提示词
            user_content: 包含多条文本的用户内容
            rank: 进程rank
            
        Returns:
            Tuple[str, float]: (生成的响应, 推理时间(秒))
        """
        model = self._get_model(rank)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        for attempt in range(self.try_num):
            try:
                start_time = time.time()
                
                if self.enable_vllm:
                    response = model.chat(messages, self.sampling_params)
                    output = response[0].outputs[0].text
                elif self.is_hf_model:
                    response = model(messages, return_full_text=False, **self.sampling_params)
                    output = response[0]["generated_text"]
                else:  # API模式
                    output = model(messages, **self.sampling_params)
                
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


def parse_batch_filter_output(raw_output: str, batch_size: int) -> List[Tuple[str, int]]:
    """
    解析批量filter任务的输出，提取每条文本的分数和KEEP/REMOVE决策
    
    支持格式: <result id="1"><score>8.5</score><decision>KEEP</decision></result>
    
    Args:
        raw_output: LLM的原始输出
        batch_size: 批处理大小
        
    Returns:
        List[Tuple[str, int]]: 列表，每项为 (原始输出行, 决策)
                               决策: 1代表KEEP，0代表REMOVE
    """
    if not raw_output or not raw_output.strip():
        logger.warning(f"Empty output from LLM, returning default decisions for {batch_size} items")
        return [("", 0) for _ in range(batch_size)]
    
    results = [("", 0) for _ in range(batch_size)]  # 初始化结果列表
    
    # 记录原始输出用于调试
    logger.debug(f"Raw LLM output:\n{raw_output}")
    
    # 改进的正则表达式：更严格地匹配格式
    # 匹配: <result id="1"><score>8.5</score><decision>KEEP</decision></result>
    pattern = r'<result\s+id=["\']?(\d+)["\']?[^>]*>\s*<score>([^<]+)</score>\s*<decision>([^<]+)</decision>\s*</result>'
    matches = list(re.finditer(pattern, raw_output, re.IGNORECASE | re.DOTALL))
    
    if not matches:
        logger.warning(f"No matches found with primary pattern. Trying fallback patterns...")
        # 备选方案1：更灵活的匹配
        pattern_fallback = r'id=["\']?(\d+)["\']?.*?decision["\']?\s*[:=]\s*["\']?([A-Za-z]+)["\']?'
        matches = list(re.finditer(pattern_fallback, raw_output, re.IGNORECASE | re.DOTALL))
    
    matched_ids = set()
    for match in matches:
        try:
            result_id = int(match.group(1))
            
            # 处理第二组：score（可能在更灵活的模式中为决策）
            if len(match.groups()) >= 3:
                decision_text = match.group(3).strip().upper()
            else:
                decision_text = match.group(2).strip().upper()
            
            full_match = match.group(0)  # 完整匹配的原始文本
            
            if 1 <= result_id <= batch_size:
                decision = 1 if "KEEP" in decision_text else 0
                results[result_id - 1] = (full_match, decision)
                matched_ids.add(result_id)
                logger.debug(f"Parsed result id={result_id}: {decision_text} -> {decision}")
            else:
                logger.warning(f"Result ID {result_id} out of range [1, {batch_size}]")
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing match: {e}")
    
    # 检查是否所有ID都被匹配
    missing_ids = []
    for i in range(1, batch_size + 1):
        if i not in matched_ids:
            missing_ids.append(i)
            results[i - 1] = ("", 0)  # 默认为REMOVE
    
    if missing_ids:
        logger.warning(f"Missing outputs for text ids: {missing_ids}")
    
    return results



def generate_output_path(
    base_output_dir: str,
    operator_name: Optional[str] = None,
    level: Optional[str] = None,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    default_filename: str = "results.jsonl"
) -> str:
    """
    根据算子名称、级别、模型名称和batch-size生成输出路径
    
    Args:
        base_output_dir: 基础输出目录
        operator_name: 算子名称（例如: text_length_filter）
        level: 级别（例如: loose, medium, strict）
        model_name: 模型名称（例如: Qwen3-32B）
        batch_size: batch大小（例如: 5）
        default_filename: 默认文件名
        
    Returns:
        str: 生成的输出路径
    """
    path_parts = [base_output_dir]
    
    if operator_name:
        path_parts.append(operator_name)
    
    if level:
        # 将级别转换为小写，以便与文件夹名称匹配
        level_lower = level.lower()
        path_parts.append(level_lower)
    
    if model_name:
        # 提取模型名称（去掉路径前缀）
        model_basename = Path(model_name).name
        path_parts.append(model_basename)
    
    if batch_size is not None:
        path_parts.append(f"batch_{batch_size}")
    
    output_dir = Path(*path_parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return str(output_dir / default_filename)


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
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save processed results (required unless --auto-organize-output or --base-output-dir is provided)")
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
    parser.add_argument("--batch-size", type=int, default=1,
                    help="Number of text samples to process in one LLM inference call")
    parser.add_argument("--base-output-dir", type=str, default=None,
                    help="Base output directory for organizing results. If set, will auto-organize by operator_name/level/model_name/batch_size")
    parser.add_argument("--auto-organize-output", action="store_true",
                    help="Auto-organize output by operator_name/level/model_name/batch_size from user_request file")
    
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
    
    # Determine output path
    output_path = None
    if args.auto_organize_output or args.base_output_dir:
        # Extract operator name and level from first user request if available
        operator_name = None
        level = None
        if user_requests:
            first_request = user_requests[0]
            operator_name = first_request.get("operator_name")
            level_str = first_request.get("level", "").upper()
            # Convert LOOSE/MEDIUM/STRICT to loose/medium/strict
            if level_str == "LOOSE":
                level = "loose"
            elif level_str == "MEDIUM":
                level = "medium"
            elif level_str == "STRICT":
                level = "strict"
        
        # Extract model name from model path
        model_name = Path(args.model).name
        
        if args.base_output_dir:
            base_dir = Path(args.base_output_dir)
        elif args.output:
            base_dir = Path(args.output).parent
        else:
            base_dir = Path.cwd()
        
        output_path = generate_output_path(
            base_output_dir=str(base_dir),
            operator_name=operator_name,
            level=level,
            model_name=model_name,
            batch_size=args.batch_size,
            default_filename="results.jsonl"
        )
        logger.info(f"Auto-organized output path: {output_path}")
    elif args.output:
        output_path = args.output
    else:
        logger.error("Output path not specified. Provide --output or enable --auto-organize-output/--base-output-dir.")
        sys.exit(1)
    
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
    
    # Process samples with batching
    results = []
    kept_count = 0
    inference_times = []
    total_start_time = time.time()

    for r_idx, request_obj in enumerate(user_requests or []):
        logger.info(f"Processing with request {r_idx+1}/{len(user_requests) if user_requests else 1}")
        user_request = request_obj.get("user_request", "")
        
        # 按 batch_size 分组处理样本
        for batch_start in range(0, len(samples), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]
            batch_size = len(batch_samples)
            
            logger.info(f"  Processing batch {batch_start//args.batch_size + 1} "
                    f"(samples {batch_start+1}-{batch_end}, batch_size={batch_size})")
          
            # 构建批处理的用户内容：把多条文本整合到一个context里
            user_content_parts = []
            for idx, sample in enumerate(batch_samples, 1):
                input_text = sample.get("text", "")
                user_content_parts.append(f"<text id=\"{idx+1}\">\n{input_text}\n</text>")

            combined_user_content = "\n".join(user_content_parts)

            system_prompt = (
                "You are an experienced text processing expert. Your task is to evaluate each input text sample according to the user's specific processing request\n\n"

                "You will be given:\n"
                "1. A user request that describes what kind of texts are acceptable. This may be a concrete instruction or a more general goal.\n"
                "2. A list of input texts, numbered sequentially starting from 1.\n\n"
                
                "For each text, you should provide:\n"
                "1. A single overall score (a float from 0.0 to 10.0) reflecting how well the text satisfies the user's request—higher scores indicate better alignment.\n"
                "2. A retention decision: \"KEEP\" if the text meets the user's criteria, or \"REMOVE\" otherwise.\n\n"

                "IMPORTANT OUTPUT FORMAT:\n"
                "Output EXACTLY ONE line per input text in this format:\n"
                "<result id=\"ID\"><score>SCORE</score><decision>DECISION</decision></result>\n\n"
                "Where:\n"
                "- ID: the text ID (1, 2, 3, etc., matching the input order)\n"
                "- SCORE: a floating-point number from 0 to 10\n"
                "- DECISION: either KEEP or REMOVE\n\n"

                "Example output format:\n"
                "<result id=\"1\"><score>8.5</score><decision>KEEP</decision></result>\n"
                "<result id=\"2\"><score>3.2</score><decision>REMOVE</decision></result>\n\n"
                "Keep the output order same as input texts. Do not output anything else.\n\n"
            )

            user_content = (
                "The user request on text processing is:\n"
                f"{user_request}\n\n"
                "The input texts are:\n\n"
                f"{combined_user_content}\n\n"
                "Now output your analysis (one result line per text):\n"
                "/no_think"
            )


            # 一次LLM调用处理整个batch
            raw_output, inference_time = processor.generate_response_batch(system_prompt, user_content)
            inference_times.append(inference_time)
            logger.info(
                f"    Inference time: {inference_time:.4f}s "
                f"(batch_size={batch_size})"
            )
            
            # 解析批量输出
            batch_results = parse_batch_filter_output(raw_output, batch_size)
            
            # 逐条保存结果
            for i, (sample, (output_line, filter_decision)) in enumerate(zip(batch_samples, batch_results)):
                sample_idx = batch_start + i
                
                result = {
                    "sample_index": sample_idx,
                    "request_index": r_idx,
                    "user_request": user_request,
                    "metadata": request_obj,
                    "original": sample,
                    "llm_output": output_line,
                    "filter_decision": bool(filter_decision),  # true=KEEP, false=REMOVE
                    "inference_time": round(inference_time / batch_size, 4),  # 平均分配推理时间
                }
                
                if filter_decision:
                    kept_count += 1
                    logger.debug(f"  Sample {sample_idx+1}: KEEP")
                else:
                    logger.debug(f"  Sample {sample_idx+1}: REMOVE")
                
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
    logger.info(f"  Total inference calls: {len(inference_times)}")  # 改为调用次数
    logger.info(f"  Total samples processed: {total_samples}")
    logger.info(f"  Total inference time: {sum(inference_times):.4f}s ({sum(inference_times)/60:.2f} minutes)")
    logger.info(f"  Average inference time per call: {sum(inference_times)/len(inference_times) if inference_times else 0:.4f}s")
    logger.info(f"  Average inference time per sample: {sum(inference_times)/total_samples if total_samples else 0:.4f}s")
    logger.info(f"  Min inference time: {min_inference_time:.4f}s")
    logger.info(f"  Max inference time: {max_inference_time:.4f}s")
    logger.info(f"  Total wall-clock time: {total_time:.4f}s ({total_time/60:.2f} minutes)")
    logger.info(f"  Overhead time: {total_time - total_inference_time:.4f}s ({(total_time - total_inference_time)/total_time*100:.2f}%)")
    if avg_inference_time > 0:
        logger.info(f"  Throughput: {1/avg_inference_time:.2f} samples/second")
    logger.info("=" * 60)
    
    # Save results
    save_results(results, output_path)
    
    logger.info("LLM filtering completed successfully")


if __name__ == "__main__":
    main()
