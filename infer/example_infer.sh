#!/bin/bash
echo "O-Researcher Inference Examples"
echo "================================"

# Get script directory (works from any location)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

# Define paths
INFER_SCRIPT="$DIR/infer.py"
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$PROJECT_ROOT/results"

# Create results directory if not exists
mkdir -p "$RESULTS_DIR"

[[ -f "$ENV_FILE" ]] || { echo "Error: Config file not found: $ENV_FILE"; exit 1; }
source "$ENV_FILE"

echo "Environment variables:"
echo "  MODEL_NAME: $MODEL_NAME"
echo "  MODEL_URL: $MODEL_URL"
echo "  WEBSEARCH_URL: $WEBSEARCH_URL"
echo "  CRAWL_PAGE_URL: $CRAWL_PAGE_URL"
echo ""
echo "Paths:"
echo "  Script: $INFER_SCRIPT"
echo "  Data: $DATA_DIR"
echo "  Results: $RESULTS_DIR"
echo ""

# =============================================================================
# Example 1: Basic inference with default parameters
# =============================================================================
echo "Example 1: Basic inference"
python "$INFER_SCRIPT" \
    --input_file "$DATA_DIR/example.jsonl" \
    --output_file "$RESULTS_DIR/output.jsonl"

# # =============================================================================
# # Example 2: Custom q_key and a_key
# # =============================================================================
# echo ""
# echo "Example 2: Custom input/output keys"
# python "$INFER_SCRIPT" \
#     --input_file "$DATA_DIR/example.jsonl" \
#     --output_file "$RESULTS_DIR/output.jsonl" \
#     --q_key "prompt" \
#     --a_key "answer"

# # =============================================================================
# # Example 3: High parallel processing
# # =============================================================================
# echo ""
# echo "Example 3: Parallel processing (30 workers)"
# python "$INFER_SCRIPT" \
#     --input_file "$DATA_DIR/example.jsonl" \
#     --output_file "$RESULTS_DIR/parallel_output.jsonl" \
#     --parallel 30

# # =============================================================================
# # Example 4: Multiple rounds
# # =============================================================================
# echo ""
# echo "Example 4: Multiple rounds (3 rounds)"
# python "$INFER_SCRIPT" \
#     --input_file "$DATA_DIR/example.jsonl" \
#     --output_file "$RESULTS_DIR/multi_round.jsonl" \
#     --round 3

# # =============================================================================
# # Example 5: Full parameters
# # =============================================================================
# echo ""
# echo "Example 5: Full parameters"
# python "$INFER_SCRIPT" \
#     --input_file "$DATA_DIR/example.jsonl" \
#     --output_file "$RESULTS_DIR/full_output.jsonl" \
#     --q_key "question" \
#     --a_key "answer" \
#     --temperature 1.0 \
#     --top_p 0.9 \
#     --max_tokens 4096 \
#     --total_tokens 81920 \
#     --max_steps 100 \
#     --parallel 30 \
#     --round 1

echo ""
echo "================================"
echo "All examples completed!"
echo "Output files are in: $RESULTS_DIR"
echo ""
echo "Available parameters:"
echo "  --input_file      : Input JSON/JSONL file (required)"
echo "  --output_file     : Output JSONL file (required)"
echo "  --q_key           : Question field key (default: question)"
echo "  --a_key           : Answer field key (default: answer)"
echo "  --temperature     : Generation temperature (default: 1.0)"
echo "  --top_p           : Top-p sampling (default: 0.9)"
echo "  --max_tokens      : Max tokens per generation (default: 4096)"
echo "  --total_tokens    : Max total tokens (default: 81920)"
echo "  --max_steps       : Max inference steps per question (default: 100)"
echo "  --parallel        : Parallel workers (default: 1)"
echo "  --round           : Number of inference rounds (default: 1)"
