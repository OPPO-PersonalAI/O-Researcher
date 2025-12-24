echo "O-Researcher Inference Example"
echo "=============================="

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

[[ -f "$ENV_FILE" ]] || { echo "Error: Config file not found: $ENV_FILE"; exit 1; }
source "$ENV_FILE"

echo "Environment variables set:"
echo "- MODEL_NAME: $MODEL_NAME"
echo "- MODEL_URL: $MODEL_URL"
echo "- OPENAI_API_URL: $OPENAI_API_URL"
echo "- WEBSEARCH_URL: $WEBSEARCH_URL"
echo "- CRAWL_PAGE_URL: $CRAWL_PAGE_URL"
echo ""

# Example 1: Auto mode with default parameters
echo "Example 1: Auto mode with default parameters"
python infer_main.py \
    --input_file ../data/example.json \
    --output_file ../results/auto_output.jsonl \
    --temperature 1.0 \
    --max_steps_agent 20 \ 
    --parallel 1

echo ""

# Example 2: Force agentic mode
echo "Example 2: Force agentic mode"
python infer_main.py \
    --input_file ../data/example.json \
    --output_file ../results/agentic_output.jsonl \
    --adaptive toolcalling_agent \
    --max_steps_agent 100 \
    --temperature 0.8

echo ""

# Example 3: Force reasoning mode
echo "Example 3: Force reasoning mode"
python infer_main.py \
    --input_file ../data/example.json \
    --output_file ../results/reasoning_output.jsonl \
    --adaptive reasoning_agent \
    --temperature 0.5

echo ""

# Example 4: Force instant mode 
echo "Example 4: Force instant mode"
python infer_main.py \
    --input_file ../data/example.json \
    --output_file ../results/instant_output.jsonl \
    --adaptive instant \
    --temperature 0.3

echo ""

# Example 5: High-performance parallel processing
echo "Example 5: High-performance parallel processing"
python infer_main.py \
    --input_file ../data/example.json \
    --output_file ../results/parallel_output.jsonl \
    --adaptive auto \
    --parallel_per_dataset 10 \
    --max_steps_agent 80

# Example 6: Basic inference with custom q_key and a_key
echo "Example 6: Basic inference with custom keys"
python infer.py \
    --input_file /home/notebook/code/group/eason/AFM_module/deep_research_bench/data/prompt_data/query_five.jsonl \
    --output_file ../results/output.jsonl \
    --q_key "prompt" \
    --a_key "answer" \
    --temperature 1.0 \
    --top_p 0.9 \
    --total_tokens 81920 \
    --retry_attempt 100 \
    --parallel 30

echo ""
echo "All examples completed!"
echo "Check the results directory for output files."
echo ""
echo "Note: Make sure to:"
echo "1. Replace all placeholder values with your actual API keys and URLs"
echo "2. Ensure all tool servers are running before executing inference"
echo "3. Create the necessary input data files in the ./data/ directory"