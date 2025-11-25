#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
request_gen.py
--------------

Generate STRICT/LOOSE user-facing requests for Data-Juicer operator chains by
combining operator documentation, code excerpts, and an application scenario,
then prompting a local vLLM model (e.g., Qwen).

Input format (JSON/JSONL, one entry per line):
{
  "id": "avg_line_len_demo",
  "scenario": "General Data Clean",
  "type": "op",   # or "recipe" for multi-operator pipelines
  "operators": [
    {"name": "average_line_length_filter", "parameters": {"min_len": 50, "max_len": 250}}
  ]
}

Example:
    python request_gen.py \
        --model /mnt/data/shared/qwen/Qwen3-32B \
        --chain-file ./user_request/op_list.jsonl \
        --loose-output ./user_request/op/op_loose.jsonl \
        --strict-output ./user_request/op/op_strict.jsonl \
        --gpu 0-7
"""

import argparse
import ast
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# -----------------------------------------------------------------------------
# Early GPU handling
# -----------------------------------------------------------------------------


def _extract_gpu_cli_arg() -> Optional[str]:
    try:
        idx = sys.argv.index("--gpu")
        return sys.argv[idx + 1]
    except (ValueError, IndexError):
        return None


_gpu = _extract_gpu_cli_arg()
if _gpu:
    ids: List[int] = []
    for seg in _gpu.split(","):
        if "-" in seg:
            start, end = seg.split("-")
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(seg))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in ids)
    logger.info(f"Setting CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")


# -----------------------------------------------------------------------------
# Data-Juicer imports
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_JUICER_ROOT = PROJECT_ROOT / "data-juicer"
OPS_DIR = DATA_JUICER_ROOT / "data_juicer" / "ops"
DOCS_DIR = DATA_JUICER_ROOT / "docs"
sys.path.insert(0, str(DATA_JUICER_ROOT))

try:
    from data_juicer.utils.model_utils import (
        get_model,
        prepare_model,
        update_sampling_params,
    )
    from data_juicer.utils.lazy_loader import LazyLoader
except ImportError as exc:
    logger.error(f"Failed to import data-juicer modules: {exc}")
    sys.exit(1)

torch = LazyLoader("torch")
vllm = LazyLoader("vllm")


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class OperatorInfo:
    op_name: str
    class_name: str
    file_path: str
    docstring: Optional[str]
    class_source: str
    doc_markdown: Optional[str]


@dataclass
class OperatorSpec:
    name: str
    parameters: Dict[str, Any]


@dataclass
class OperatorChain:
    chain_id: str
    scenario: str
    type: str  # "op" (single operator) or "recipe" (multi-operator chain)
    operators: List[OperatorSpec]
    notes: Optional[str] = None


@dataclass
class RequestResult:
    chain: OperatorChain
    strict_request: str
    loose_request: str
    raw_output: str
    inference_time: float
    success: bool
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# Operator extraction helpers
# -----------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as fh:
        return fh.read()


def _get_op_name_from_assigns(module: ast.Module) -> Optional[str]:
    for node in module.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "OP_NAME":
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
                if hasattr(ast, "Str") and isinstance(value, ast.Str):
                    return value.s
    return None


def _resolve_registered_name(dec: ast.AST, fallback: Optional[str]) -> Optional[str]:
    if not isinstance(dec, ast.Call):
        return None
    func = dec.func
    if not isinstance(func, ast.Attribute) or func.attr != "register_module":
        return None
    if not dec.args:
        return None
    arg0 = dec.args[0]
    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
        return arg0.value
    if hasattr(ast, "Str") and isinstance(arg0, ast.Str):
        return arg0.s
    if isinstance(arg0, ast.Name) and arg0.id == "OP_NAME":
        return fallback
    return None


def _get_docstring_from_class(node: ast.ClassDef) -> Optional[str]:
    if not node.body:
        return None
    doc = ast.get_docstring(node, clean=False)
    return doc


def _slice_source(source: str, node: ast.AST) -> str:
    if hasattr(ast, "get_source_segment"):
        seg = ast.get_source_segment(source, node)
        if seg is not None:
            return seg
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)
    return "".join(lines[start:end])


def _discover_operator_files() -> List[Path]:
    candidates: List[Path] = []
    for sub_dir in ("filter", "mapper"):
        base = OPS_DIR / sub_dir
        if base.exists():
            candidates.extend(base.rglob("*.py"))
    excluded_tokens = {"image", "audio", "video", "annotation"}
    filtered = [
        path
        for path in candidates
        if path.name not in {"__init__.py", "base.py"}
        and not any(token in str(path).lower() for token in excluded_tokens)
    ]
    logger.info(f"Discovered {len(filtered)} operator files under OPS.")
    return sorted(filtered)


def _guess_doc_path(op_name: str) -> Optional[Path]:
    # docs/operators/{filter|mapper} directories mirror operator categories
    for category in ("filter", "mapper"):
        md_path = DOCS_DIR / "operators" / category / f"{op_name}.md"
        if md_path.exists():
            return md_path
    return None


def _load_markdown(path: Optional[Path]) -> Optional[str]:
    if path and path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to read markdown {path}: {exc}")
    return None


def extract_operator_metadata() -> Dict[str, OperatorInfo]:
    operator_index: Dict[str, OperatorInfo] = {}
    for file_path in _discover_operator_files():
        try:
            src = _read_text(file_path)
            module = ast.parse(src)
        except Exception as exc:
            logger.warning(f"Failed to parse {file_path}: {exc}")
            continue
        op_name_const = _get_op_name_from_assigns(module)
        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            resolved_name = None
            for dec in node.decorator_list:
                resolved_name = _resolve_registered_name(dec, op_name_const)
                if resolved_name:
                    break
            if not resolved_name:
                continue
            doc_markdown = _load_markdown(_guess_doc_path(resolved_name))
            info = OperatorInfo(
                op_name=resolved_name,
                class_name=node.name,
                file_path=str(file_path),
                docstring=_get_docstring_from_class(node),
                class_source=_slice_source(src, node),
                doc_markdown=doc_markdown,
            )
            if resolved_name not in operator_index:
                operator_index[resolved_name] = info
    logger.info(f"Indexed {len(operator_index)} operators with metadata.")
    return operator_index


# -----------------------------------------------------------------------------
# Chain loading
# -----------------------------------------------------------------------------


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_operator_chains(path: str) -> List[OperatorChain]:
    chain_path = Path(path)
    if not chain_path.exists():
        raise FileNotFoundError(f"Chain file not found: {chain_path}")
    text = chain_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Chain file is empty.")
    try:
        data = json.loads(text) if text[0] == "[" else _load_jsonl(chain_path)
    except Exception as exc:
        raise ValueError(f"Failed to parse chain file {chain_path}: {exc}") from exc

    chains: List[OperatorChain] = []
    for idx, entry in enumerate(data, 1):
        chain_id = entry.get("chain_id") or entry.get("id") or f"chain_{idx}"
        scenario = entry.get("scenario") or entry.get("application") or entry.get("scene") or ""
        operators = entry.get("operators") or entry.get("chain") or entry.get("ops") or []
        notes = entry.get("notes")
        chain_type = entry.get("type") or entry.get("chain_type") or entry.get("kind")
        if isinstance(chain_type, str):
            chain_type = chain_type.strip().lower()
        if not chain_type:
            chain_type = "op" if len(operators) <= 1 else "recipe"
        if not scenario:
            raise ValueError(f"Entry {chain_id} missing 'scenario'.")
        if not operators:
            raise ValueError(f"Entry {chain_id} missing 'operators'.")
        specs: List[OperatorSpec] = []
        for op in operators:
            name = op.get("name") or op.get("operator") or op.get("op_name")
            params = op.get("parameters") or {}
            if not name:
                raise ValueError(f"Chain {chain_id} contains operator without 'name'.")
            specs.append(OperatorSpec(name=name, parameters=params))
        chains.append(
            OperatorChain(
                chain_id=chain_id,
                scenario=scenario,
                type=chain_type,
                operators=specs,
                notes=notes,
            )
        )
    logger.info(f"Loaded {len(chains)} operator chains.")
    return chains


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------

REQUEST_SYSTEM_PROMPT = """You are an expert at turning technical data processing pipelines into natural language requests that real users might make.

I will give you:
- A sequence of one or more operators, possibly with parameters
- Documentation or code explaining what each operator does
- The application scenario motivating the pipeline

Your task: produce TWO user-facing requests:

STRICT
- Start with one high-level instruction describing the overall goal
- Then list EACH operator in execution order step by step in natural language
- In every step, express the key thresholds/parameters in natural language, NEVER use mathematical symbols.

LOOSE
- A short plain-language sentence summarizing the desired data from user's perspective without mentioning any specific operator.
- Do NOT include ANY specific step or operator, keep it high-level
- Do NOT include ANY numeric thresholds or percentages related to operators

To lock in tone, here are two reference IO pairs. Follow the same structure (STRICT section with short imperative lines, LOOSE section as one sentence). Operator chains are comma-separated for readability; your actual input will still be JSON with operators/parameters.

Example 1
Input chain (for illustration only):
clean_html, clean_email, clean_ip, language_id_filter, text_chunk, minhash_dedup, llm_quality_filter
Target output:
<answer><strict>Prepare these web pages for RAG:
- Remove HTML tags, emails, IPs.
- Split into semantic chunks (300–800 chars).
- Deduplicate similar chunks.
- Keep only English with quality score over 7.</strict><loose>Please turn these web pages into clean, non-repeating English text chunks ready for RAG.</loose></answer>

Example 2
Input chain (for illustration only):
dialog_split, intent_detection, sentiment_detection, generate_qa, optimize_qa, selector(topk=1000)
Target output:
<answer><strict>From this customer service log:
- Extract dialog turns.
- Detect user intent and sentiment.
- Generate optimized QA pairs.
- Select top 1,000 pairs by diversity and quality.</strict><loose>Please turn this customer service log into 1,000 diverse, high-quality QA pairs that capture user intent and sentiment.</loose></answer>

CRITICAL:
- Wrap the final reply in <answer> ... </answer>
- Inside <answer>, include <strict> ... </strict> and <loose> ... </loose>
- Nothing else should appear outside those tags

/no_think
"""


def build_chain_prompt(
    chain: OperatorChain,
    operator_index: Dict[str, OperatorInfo],
    max_doc_chars: int,
    max_code_chars: int,
    max_markdown_chars: int,
) -> str:
    lines: List[str] = []
    lines.append(f"Application scenario:\n{chain.scenario.strip()}\n")
    if chain.notes:
        lines.append(f"Additional notes:\n{chain.notes.strip()}\n")
    lines.append(f"Chain type: {chain.type}")
    lines.append("Operator chain:")
    for idx, spec in enumerate(chain.operators, 1):
        info = operator_index.get(spec.name)
        lines.append(f"{idx}. Operator: {spec.name}")
        if spec.parameters:
            lines.append(f"   Parameters: {json.dumps(spec.parameters, ensure_ascii=False)}")
        if info:
            if info.docstring:
                doc = info.docstring.strip()
                if len(doc) > max_doc_chars:
                    doc = doc[: max_doc_chars - 3] + "..."
                lines.append(f"   Docstring:\n{doc}")
            if info.doc_markdown:
                md = info.doc_markdown.strip()
                if len(md) > max_markdown_chars:
                    md = md[: max_markdown_chars - 3] + "..."
                lines.append(f"   Documentation excerpt:\n{md}")
            snippet = info.class_source.strip()
            if len(snippet) > max_code_chars:
                snippet = snippet[: max_code_chars - 3] + "..."
            lines.append("   Code excerpt:\n----BEGIN CODE----")
            lines.append(snippet)
            lines.append("----END CODE----")
        else:
            lines.append("   (Operator metadata not found)")
    lines.append("")
    lines.append("Generate STRICT and LOOSE user requests now.")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# vLLM wrapper
# -----------------------------------------------------------------------------


class UnifiedLLMProcessor:
    def __init__(
        self,
        model_name: str,
        try_num: int = 3,
        tensor_parallel_size: Optional[int] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.try_num = try_num
        gpu_count = torch.cuda.device_count()
        if gpu_count < 1:
            raise RuntimeError("CUDA GPUs are required for vLLM inference.")
        model_params: Dict[str, Any] = {}
        if tensor_parallel_size:
            model_params["tensor_parallel_size"] = tensor_parallel_size
        else:
            model_params["tensor_parallel_size"] = gpu_count
            logger.info(f"Auto tensor_parallel_size={gpu_count}")
        sampling_params = sampling_params or {}
        sampling_params = update_sampling_params(sampling_params, model_name, True)
        self.model_key = prepare_model(
            model_type="vllm",
            pretrained_model_name_or_path=model_name,
            **model_params,
        )
        self.sampling_params = vllm.SamplingParams(**sampling_params)

    def _get_model(self):
        model, _ = get_model(self.model_key, None, True)
        return model

    def chat(self, system_prompt: str, user_prompt: str) -> Tuple[str, float]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        model = self._get_model()
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.try_num + 1):
            try:
                start = time.time()
                response = model.chat(messages, self.sampling_params)
                text = response[0].outputs[0].text.strip()
                duration = time.time() - start
                if text:
                    return text, duration
            except Exception as exc:
                logger.warning(f"Attempt {attempt} failed: {exc}")
                last_exc = exc
        raise RuntimeError(f"Generation failed after {self.try_num} attempts: {last_exc}")

    def shutdown(self):
        logger.debug("UnifiedLLMProcessor shutdown requested (vLLM handles lifecycle).")


# -----------------------------------------------------------------------------
# Output parsing
# -----------------------------------------------------------------------------


def parse_response(raw_output: str) -> Tuple[Optional[str], Optional[str]]:
    if not raw_output:
        return None, None
    answer_match = re.search(r"<answer>(.*?)</answer>", raw_output, re.DOTALL | re.IGNORECASE)
    payload = answer_match.group(1).strip() if answer_match else raw_output.strip()
    strict_match = re.search(r"<strict>(.*?)</strict>", payload, re.DOTALL | re.IGNORECASE)
    loose_match = re.search(r"<loose>(.*?)</loose>", payload, re.DOTALL | re.IGNORECASE)
    strict = strict_match.group(1).strip() if strict_match else None
    loose = loose_match.group(1).strip() if loose_match else None
    return strict, loose


# -----------------------------------------------------------------------------
# Main generation loop
# -----------------------------------------------------------------------------


def generate_requests(
    chains: List[OperatorChain],
    operator_index: Dict[str, OperatorInfo],
    processor: UnifiedLLMProcessor,
    max_doc_chars: int,
    max_code_chars: int,
    max_markdown_chars: int,
) -> List[RequestResult]:
    results: List[RequestResult] = []
    for chain in chains:
        prompt = build_chain_prompt(
            chain,
            operator_index=operator_index,
            max_doc_chars=max_doc_chars,
            max_code_chars=max_code_chars,
            max_markdown_chars=max_markdown_chars,
        )
        logger.info(f"Generating requests for chain {chain.chain_id} ...")
        try:
            raw_output, duration = processor.chat(REQUEST_SYSTEM_PROMPT, prompt)
            strict_req, loose_req = parse_response(raw_output)
            if not strict_req or not loose_req:
                raise ValueError("STRICT/LOOSE fields missing in model output.")
            result = RequestResult(
                chain=chain,
                strict_request=strict_req,
                loose_request=loose_req,
                raw_output=raw_output,
                inference_time=duration,
                success=True,
            )
        except Exception as exc:
            logger.error(f"Chain {chain.chain_id} failed: {exc}")
            result = RequestResult(
                chain=chain,
                strict_request="",
                loose_request="",
                raw_output="",
                inference_time=0.0,
                success=False,
                error=str(exc),
            )
        results.append(result)
    return results


def export_level_requests(
    results: List[RequestResult],
    level: str,
    output_path: Optional[str],
) -> None:
    if not output_path:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    level_key = level.upper()
    records: List[Dict[str, Any]] = []
    for item in results:
        if not item.success or item.chain.type != "op" or not item.chain.operators:
            continue
        request_text = item.loose_request if level == "loose" else item.strict_request
        record = {
            "id": item.chain.chain_id,
            "scenario": item.chain.scenario,
            "type": item.chain.type,
            "operators": [asdict(spec) for spec in item.chain.operators],
            "notes": item.chain.notes,
            "level": level_key,
            "user_request": request_text,
        }
        records.append(record)
    if not records:
        logger.warning(f"No {level_key} operator requests written (check type=='op').")
        return
    with output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} {level_key} operator requests to {output}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate personalized STRICT/LOOSE user requests for operator chains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, type=str, help="Path to local Qwen/Llama checkpoint.")
    parser.add_argument(
        "--chain-file",
        type=str,
        default="./user_request/op_list.jsonl",
        help="JSON/JSONL file describing operator chains.",
    )
    parser.add_argument(
        "--loose-output",
        type=str,
        default="./user_request/op/op_loose.jsonl",
        help="Optional JSONL path containing only loose-level requests (op only).",
    )
    parser.add_argument(
        "--strict-output",
        type=str,
        default="./user_request/op/op_strict.jsonl",
        help="Optional JSONL path containing only strict-level requests (op only).",
    )
    parser.add_argument("--gpu", type=str, default=None, help="GPU ids, e.g. 0-3 or 0,1,2.")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Tensor parallel size override.")
    parser.add_argument("--temperature", type=float, default=0.35, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Generation max tokens.")
    parser.add_argument("--try-num", type=int, default=3, help="Retry attempts when generation fails.")
    parser.add_argument("--max-doc-chars", type=int, default=900, help="Docstring excerpt length.")
    parser.add_argument("--max-code-chars", type=int, default=1400, help="Code excerpt length.")
    parser.add_argument("--max-markdown-chars", type=int, default=900, help="Markdown excerpt length.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper())


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    logger.info("Indexing operators...")
    operator_index = extract_operator_metadata()

    logger.info("Loading operator chains...")
    chains = load_operator_chains(args.chain_file)

    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    processor = UnifiedLLMProcessor(
        model_name=args.model,
        try_num=args.try_num,
        tensor_parallel_size=args.tensor_parallel_size,
        sampling_params=sampling_params,
    )

    try:
        results = generate_requests(
            chains=chains,
            operator_index=operator_index,
            processor=processor,
            max_doc_chars=args.max_doc_chars,
            max_code_chars=args.max_code_chars,
            max_markdown_chars=args.max_markdown_chars,
        )
    finally:
        processor.shutdown()

    export_level_requests(results, "loose", args.loose_output)
    export_level_requests(results, "strict", args.strict_output)

    success = sum(1 for r in results if r.success)
    logger.info(f"Generation finished: {success}/{len(results)} chains succeeded.")


if __name__ == "__main__":
    main()

