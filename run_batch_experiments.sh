#!/bin/bash

set -e

DATASET="${1:-../dj_llm_data/test_data/test_cc_100.jsonl}"
USER_REQUEST_FILE="${2:-./user_request_length.jsonl}"
BASE_OUTPUT_DIR="${3:-../dj_llm_data/results/llm}"
GPU="${4:-0-7}"

MODELS=(
    "Qwen3-1.7B-Base:/mnt/data/shared/qwen/Qwen3-1.7B-Base"
    "Qwen3-4B-Base:/mnt/data/shared/qwen/Qwen3-4B-Base"
    "Qwen3-8B:/mnt/data/shared/qwen/Qwen3-8B"
    "Qwen3-32B:/mnt/data/shared/qwen/Qwen3-32B"
)

BATCH_SIZES=(1 2 5)

if [[ ! -f "$USER_REQUEST_FILE" ]]; then
    echo "Error: User request file not found: $USER_REQUEST_FILE"
    exit 1
fi

if [[ ! -f "$DATASET" ]]; then
    echo "Error: Dataset file not found: $DATASET"
    exit 1
fi

echo "Starting batch experiments..."
echo "Dataset: $DATASET"
echo "User request file: $USER_REQUEST_FILE"
echo "Base output dir: $BASE_OUTPUT_DIR"
echo "GPU: $GPU"
echo ""

total=0
completed=0
failed=0

# 遍历模型
for model_info in "${MODELS[@]}"; do
    model_name="${model_info%%:*}"
    model_path="${model_info##*:}"
    
    if [[ ! -d "$model_path" ]]; then
        echo "⚠ Model path does not exist: $model_path, skipping"
        continue
    fi
    
    echo "================================================================================="
    echo "Loading model: $model_name ($model_path)"
    echo "================================================================================="
    
    # 遍历batch_size
    for batch_size in "${BATCH_SIZES[@]}"; do
        total=$((total + 1))
        
        echo "  [$total] Processing with batch_size=$batch_size"
        
        # 构建输出路径和文件名
        output_file="$BASE_OUTPUT_DIR/${model_name}_batch_${batch_size}.jsonl"
        mkdir -p "$BASE_OUTPUT_DIR"
        
        # 调用 llm_batch_filter，一次性处理所有请求
        if python3 llm_batch_filter.py \
            --dataset "$DATASET" \
            --model "$model_path" \
            --model-type vllm \
            --user-request-file "$USER_REQUEST_FILE" \
            --batch-size "$batch_size" \
            --gpu "$GPU" \
            --output "$output_file"; then
            echo "    ✓ Completed: $output_file"
            completed=$((completed + 1))
        else
            echo "    ✗ Failed"
            failed=$((failed + 1))
        fi
    done
done

echo ""
echo "================================================================================="
echo "Batch Experiments Summary"
echo "================================================================================="
echo "Total experiments: $total"
echo "Completed: $completed"
echo "Failed: $failed"
echo "================================================================================="

if [[ $failed -gt 0 ]]; then
    exit 1
fi
