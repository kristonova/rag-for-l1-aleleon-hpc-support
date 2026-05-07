#!/bin/bash

# Interactive configuration
echo "=========================================="
echo "vLLM Benchmark Configuration"
echo "=========================================="

# Prompt for server host
read -p "Enter server host (default: 0.0.0.0): " SERVER_HOST_INPUT
SERVER_HOST="${SERVER_HOST_INPUT:-0.0.0.0}"

# Prompt for server port
read -p "Enter server port (default: 8005): " SERVER_PORT_INPUT
SERVER_PORT="${SERVER_PORT_INPUT:-8005}"

# Fixed configuration
MODEL_PATH="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
MODEL_BASENAME=$(basename $MODEL_PATH)
INPUT_LENGTHS=(1000 5000 10000 50000 100000)
OUTPUT_LEN=128
NUM_PROMPTS=10  # Number of requests per concurrency level
RESULT_DIR="./results-$SERVER_HOST-$(date +%Y%m%d-%H%M%S)"

# Create result directory
mkdir -p "$RESULT_DIR"

# Concurrency levels to test
CONCURRENCY_LEVELS=(1 2 4 8 16)

# Outer loop: iterate through each input length
for INPUT_LEN in "${INPUT_LENGTHS[@]}"; do
    echo "=========================================="
    echo "Running benchmark with input length: $INPUT_LEN"
    echo "=========================================="
    
    # Create subdirectory for this input length
    INPUT_LEN_DIR="$RESULT_DIR/input_${INPUT_LEN}"
    mkdir -p "$INPUT_LEN_DIR"
    
    # Inner loop: iterate through each concurrency level
    for CONCURRENCY in "${CONCURRENCY_LEVELS[@]}"; do
        echo "----------------------------------------"
        echo "Running benchmark with $CONCURRENCY concurrent requests (input_len=$INPUT_LEN)"
        echo "----------------------------------------"
        
        OUTPUT_FILE="$INPUT_LEN_DIR/benchmark_concurrency_${CONCURRENCY}_input${INPUT_LEN}_output${OUTPUT_LEN}.json"
        
        vllm bench serve \
            --backend vllm \
            --model "$MODEL_PATH" \
            --host "$SERVER_HOST" \
            --port "$SERVER_PORT" \
            --endpoint /v1/completions \
            --dataset-name random \
            --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$CONCURRENCY" \
            --num-warmups 1 \
            --request-rate inf \
            --ignore-eos \
            --trust-remote-code \
            --percentile-metrics "ttft,itl,tpot" \
            --metric-percentiles "50,99" \
            --save-result \
            --save-detailed \
            --result-dir "$INPUT_LEN_DIR" \
            --result-filename "${MODEL_BASENAME}_concurrency${CONCURRENCY}_input${INPUT_LEN}_output${OUTPUT_LEN}" \
            --metadata "concurrency=$CONCURRENCY input_len=$INPUT_LEN output_len=$OUTPUT_LEN"
        
        echo "Results saved to: $OUTPUT_FILE"
        echo ""
    done
done

echo "All benchmarks completed. Results are in: $RESULT_DIR"

