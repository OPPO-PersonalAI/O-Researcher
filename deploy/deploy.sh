#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

[[ -f "$ENV_FILE" ]] || { echo "Error: Config file not found: $ENV_FILE"; exit 1; }
source "$ENV_FILE"

# System optimization
export TORCHDYNAMO_VERBOSE=${TORCHDYNAMO_VERBOSE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export VLLM_USE_V1=${VLLM_USE_V1:-1}

# Model configuration
model_path=${MODEL_PATH:?Please set MODEL_PATH in .env}
base_modelname=${MODEL_NAME:?Please set MODEL_NAME in .env}
base_port=${MODEL_BASE_PORT:-8000}
deploy_host=${DEPLOY_HOST:?Please set DEPLOY_HOST in .env}

# Deployment parameters
INSTANCES=${DEPLOY_INSTANCES:-1}
GPUS_PER_INSTANCE=${DEPLOY_GPUS_PER_INSTANCE:-2}
max_model_len=${DEPLOY_MAX_MODEL_LEN:-131072}
LOG_DIR=${DEPLOY_LOG_DIR:-"$DIR/logs"}
WAIT_TIMEOUT=${DEPLOY_WAIT_TIMEOUT:-120}
PID_DIR="$DIR/pids"

mkdir -p "$LOG_DIR" "$PID_DIR"
ip_sanitized=$(echo "$deploy_host" | tr '.' '_')
log_prefix="${base_modelname}_${ip_sanitized}"

cmd=$1
[[ "$cmd" =~ ^(start|stop|status)$ ]] || { echo "Usage: $0 [start|stop|status]"; exit 1; }

# Port check function - uses multiple methods for compatibility
check_port_listening() {
    local port=$1
    # Method 1: ss (most common on modern Linux)
    ss -tln 2>/dev/null | grep -q ":${port} " && return 0
    # Method 2: netstat
    netstat -tln 2>/dev/null | grep -q ":${port} " && return 0
    # Method 3: /proc/net/tcp (always available on Linux)
    local hex_port=$(printf '%04X' "$port")
    grep -qi ":${hex_port} " /proc/net/tcp 2>/dev/null && return 0
    # Method 4: lsof (if available)
    lsof -i:"$port" &>/dev/null && return 0
    return 1
}

# Get PIDs on port - uses multiple methods
get_port_pids() {
    local port=$1
    # Try lsof first
    local pids=$(lsof -t -i:"$port" 2>/dev/null)
    if [[ -z "$pids" ]]; then
        # Try ss + /proc
        pids=$(ss -tlnp 2>/dev/null | grep ":${port} " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u)
    fi
    if [[ -z "$pids" ]]; then
        # Try netstat
        pids=$(netstat -tlnp 2>/dev/null | grep ":${port} " | awk '{print $7}' | cut -d'/' -f1 | sort -u)
    fi
    echo "$pids"
}

# Stop process by port
stop_by_port() {
    local port=$1
    echo "Stopping process on port $port..."
    
    # Method 1: Find vLLM processes by command line
    PIDS=$(ps -ef | grep "vllm" | grep -E "\-\-port[= ]$port( |$)" | grep -v grep | awk '{print $2}')
    if [ -n "$PIDS" ]; then
        echo "  Found vLLM PIDs: $PIDS"
        kill $PIDS 2>/dev/null
        sleep 2
        kill -9 $PIDS 2>/dev/null
    fi
    
    # Method 2: Kill by port directly (fuser)
    if command -v fuser &>/dev/null; then
        fuser -k ${port}/tcp 2>/dev/null && echo "  Killed via fuser"
        sleep 1
    fi
    
    # Method 3: Get PIDs from port and kill
    PORT_PIDS=$(get_port_pids "$port")
    if [ -n "$PORT_PIDS" ]; then
        echo "  Found port PIDs: $PORT_PIDS"
        echo "$PORT_PIDS" | xargs kill -9 2>/dev/null
        sleep 1
    fi
    
    # Verify
    if check_port_listening "$port"; then
        echo "  Warning: Port $port still in use"
    else
        echo "  Port $port cleaned ✓"
    fi
}

check_instance_status() {
    local port=$1 instance_name=$2
    if check_port_listening "$port"; then
        local pid=$(get_port_pids "$port" | head -1)
        # Health check
        local http_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://${deploy_host}:${port}/health" 2>/dev/null)
        if [ "$http_code" == "200" ]; then
            echo "${instance_name}: RUNNING (port: $port, PID: ${pid:-?}) - healthy ✓"
        else
            echo "${instance_name}: RUNNING (port: $port, PID: ${pid:-?}) - loading/unhealthy (HTTP: $http_code)"
        fi
    else
        echo "${instance_name}: STOPPED (port: $port)"
    fi
}

if [[ "$cmd" == "start" ]]; then
    echo "Starting deployment: ${INSTANCES} instance(s), model: ${base_modelname}"
    echo "Note: Model deployment may take several minutes depending on model size."
    echo "      Progress will be shown below. Logs are saved to: $LOG_DIR"
    echo ""

    for ((i=0; i<INSTANCES; i++)); do
        start_gpu=$((i * GPUS_PER_INSTANCE))
        end_gpu=$((start_gpu + GPUS_PER_INSTANCE - 1))
        gpu_list=$(seq $start_gpu $end_gpu | tr '\n' ',' | sed 's/,$//')
        port=$((base_port + i))
        instance_name="${base_modelname}_inst${i}"
        log_file="${LOG_DIR}/${log_prefix}_inst${i}.log"
        pidf="$PID_DIR/${instance_name}.pid"
        
        echo "Starting instance ${instance_name}: port ${port}, GPU ${gpu_list}"
        
        # Check if already running
        if check_port_listening "$port"; then
            echo "Warning: ${instance_name} port $port already in use, run: $0 stop"
            continue
        fi
        if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
            echo "Warning: ${instance_name} already running"
            continue
        fi
        
        nohup bash -c "
            export CUDA_VISIBLE_DEVICES=${gpu_list}
            vllm serve ${model_path} \
                --served-model-name ${base_modelname} \
                --max-model-len ${max_model_len} \
                --max-seq-len ${max_model_len} \
                --tensor-parallel-size ${GPUS_PER_INSTANCE} \
                --gpu-memory-utilization 0.8 \
                --trust-remote-code \
                --uvicorn-log-level debug \
                --host 0.0.0.0 \
                --port ${port}
        " > "$log_file" 2>&1 &
        
        echo $! > "$pidf"
        
        # Wait for service to start
        echo "Waiting for instance ${instance_name} to start..."
        echo "  This may take several minutes (timeout: ${WAIT_TIMEOUT}s)..."
        start_time=$(date +%s)
        progress_shown=0
        last_progress_time=0
        
        while [ $(( $(date +%s) - start_time )) -lt $WAIT_TIMEOUT ]; do
            elapsed=$(( $(date +%s) - start_time ))
            if [[ $elapsed -gt $((last_progress_time + 30)) ]]; then
                echo "  Still initializing... (${elapsed}s elapsed)"
                last_progress_time=$elapsed
            fi
            if check_port_listening "$port"; then
                echo "  Port $port is listening, service starting..."
                sleep 3
                echo "  Service is ready ✓"
                break
            else
                if [[ $progress_shown -eq 0 ]] && kill -0 "$(cat "$pidf" 2>/dev/null)" 2>/dev/null; then
                    echo "  Process is running, initializing..."
                    progress_shown=1
                fi
            fi
            sleep 3
        done
        
        sleep 2
        actual_pid=$(get_port_pids "$port" | head -1)
        if [[ -n "$actual_pid" ]]; then
            echo "$actual_pid" > "$pidf"
            echo "${instance_name} started successfully (PID: $actual_pid, port: $port)"
            echo ""
        else
            echo "Warning: ${instance_name} not ready yet (timeout ${WAIT_TIMEOUT}s)"
            echo "         Process may still be loading. Check log: $log_file"
            echo "         Use '$0 stop' to stop if needed"
        fi
    done

    echo ""
    echo "Deployment complete"
    echo "Server address: $deploy_host"
    for ((i=0; i<INSTANCES; i++)); do
        echo "  - http://$deploy_host:$((base_port + i))"
    done
    echo "Log directory: $LOG_DIR"

elif [[ "$cmd" == "stop" ]]; then
    echo "Stopping vLLM services (${INSTANCES} instance(s))..."
    echo ""
    
    # Stop each instance by port
    for ((i=0; i<INSTANCES; i++)); do
        port=$((base_port + i))
        stop_by_port "$port"
        rm -f "$PID_DIR/${base_modelname}_inst${i}.pid"
    done
    
    echo ""
    echo "Waiting for processes to terminate..."
    sleep 30
    
    # Verify
    echo ""
    echo "Verification:"
    has_remaining=0
    for ((i=0; i<INSTANCES; i++)); do
        port=$((base_port + i))
        if check_port_listening "$port"; then
            echo "  Warning: Port $port still in use, retrying..."
            fuser -k -9 ${port}/tcp 2>/dev/null
            sleep 2
            if check_port_listening "$port"; then
                echo "  Port $port still in use after retry"
                has_remaining=1
            else
                echo "  Port $port released ✓"
            fi
        else
            echo "  Port $port released ✓"
        fi
    done
    
    if [ $has_remaining -eq 1 ]; then
        echo ""
        echo "Some ports still in use. Try: lsof -i:<port>"
    fi
    
    # Show GPU status
    echo ""
    echo "GPU status:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "  (no GPU processes or nvidia-smi not available)"

else
    echo "Deployment service status:"
    for ((i=0; i<INSTANCES; i++)); do
        check_instance_status $((base_port + i)) "${base_modelname}_inst${i}"
    done
    echo ""
    echo "Access URLs:"
    for ((i=0; i<INSTANCES; i++)); do
        echo "  - http://$deploy_host:$((base_port + i))"
    done
fi