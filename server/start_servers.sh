#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DIR")"
source "$PROJECT_ROOT/.env"
echo $PROJECT_ROOT
: "${SERVER_HOST:?Please set SERVER_HOST in .env}"
: "${CRAWL_PAGE_PORT:?Please set CRAWL_PAGE_PORT in .env}"
: "${WEBSEARCH_PORT:?Please set WEBSEARCH_PORT in .env}"

# Configure worker count
CRAWL_PAGE_WORKERS=${CRAWL_PAGE_WORKERS:-10}
WEBSEARCH_WORKERS=${WEBSEARCH_WORKERS:-10} 

LOG_DIR="$DIR/logs/$SERVER_HOST";   mkdir -p "$LOG_DIR"
PID_DIR="$DIR/pids/$SERVER_HOST";   mkdir -p "$PID_DIR"

cmd=$1
if [[ ! "$cmd" =~ ^(start|stop|status|test)$ ]]; then
  echo "Usage: $0 [start|stop|status|test]"
  exit 1
fi

# =============================================================================
# Helper function: Stop service and all child processes
# =============================================================================
stop_service_with_children() {
  local name=$1 pidf=$2 port=$3
  echo "Stopping ${name}..."
  
  # 1. Stop process group via PID file
  if [[ -f "$pidf" ]]; then
    local pid=$(cat "$pidf" 2>/dev/null)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      # Get actual process group ID
      local pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
      if [[ -n "$pgid" ]] && [[ "$pgid" != "0" ]]; then
        kill -TERM -"$pgid" 2>/dev/null
      else
        kill -TERM "$pid" 2>/dev/null
      fi
      # Wait for process to end
      for i in {1..5}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      # Force kill on timeout
      if kill -0 "$pid" 2>/dev/null; then
        [[ -n "$pgid" ]] && kill -9 -"$pgid" 2>/dev/null
        kill -9 "$pid" 2>/dev/null
      fi
      echo "${name} stopped (PID $pid)"
    fi
    rm -f "$pidf"
  fi
  
  # 2. Force cleanup all processes on port
  sleep 1
  local pids=$(lsof -t -i:"$port" 2>/dev/null)
  if [[ -n "$pids" ]]; then
    echo "  Cleaning remaining processes on port $port: $pids"
    echo "$pids" | xargs kill -9 2>/dev/null
    sleep 1
  fi
  
  # Check final status
  if ! lsof -i:"$port" &>/dev/null; then
    echo "${name} port $port released"
  else
    echo "Port $port still in use, check manually: lsof -i:$port"
  fi
}

# =============================================================================
# Helper function: Check service status
# =============================================================================
check_service_status() {
  local service_name=$1
  local pidf=$2
  local port=$3
  local workers=$4
  
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    main_pid=$(cat "$pidf")
    echo "${service_name} running (main PID: $main_pid)"
    
    # Find all child processes
    child_pids=($(pgrep -P "$main_pid" 2>/dev/null))
    if [[ ${#child_pids[@]} -gt 0 ]]; then
      echo "   └─ Worker count: ${#child_pids[@]} (expected: $workers)"
      echo "   └─ Worker PIDs: ${child_pids[*]}"
    else
      echo "   └─ Single process mode (no worker child processes detected)"
    fi
    
    # Show port listening status
    local port_count=$(lsof -t -i:"$port" 2>/dev/null | wc -l)
    echo "   └─ Port $port listening processes: $port_count"
    
    # Show memory usage
    local mem_usage=$(ps -o rss= -p "$main_pid" 2>/dev/null)
    if [[ -n "$mem_usage" ]]; then
      mem_mb=$((mem_usage / 1024))
      echo "   └─ Memory usage: ${mem_mb} MB"
    fi
    
  elif lsof -i:"$port" &>/dev/null; then
    echo "${service_name} port $port in use, but PID file invalid or process abnormal"
    echo "   Processes using port:"
    lsof -i:"$port" 2>/dev/null | grep LISTEN
  else
    echo "${service_name} not running, port $port not in use"
  fi
}


# ---------------------------------------------
#                start
# ---------------------------------------------
if [[ "$cmd" == "start" ]]; then
  # CrawlPage
  pidf="$PID_DIR/${SERVER_HOST}_CrawlPage_$CRAWL_PAGE_PORT.pid"
  logf="$LOG_DIR/CrawlPage_$CRAWL_PAGE_PORT.log"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "CrawlPage already running (PID $(cat "$pidf"))"
  else
    echo "Starting CrawlPage on $SERVER_HOST:$CRAWL_PAGE_PORT (Workers: $CRAWL_PAGE_WORKERS)..."
    nohup uvicorn crawl_page_server:app \
      --host "$SERVER_HOST" \
      --port "$CRAWL_PAGE_PORT" \
      --workers "$CRAWL_PAGE_WORKERS" \
      --log-level info \
      --timeout-keep-alive 120 \
      --app-dir "$DIR" \
      > "$logf" 2>&1 &
    echo $! > "$pidf"
    # Use health check to verify startup (max 15 seconds)
    started=0
    for i in {1..15}; do
      if curl -s --connect-timeout 1 "http://$SERVER_HOST:$CRAWL_PAGE_PORT/health" &>/dev/null; then
        echo "CrawlPage started successfully (PID: $(cat "$pidf"), address: $SERVER_HOST:$CRAWL_PAGE_PORT)"
        started=1
        break
      fi
      sleep 1
    done
    [[ $started -eq 0 ]] && echo "CrawlPage startup failed, check log: $logf" && tail -3 "$logf"
  fi

  # WebSearch
  pidf="$PID_DIR/${SERVER_HOST}_WebSearch_$WEBSEARCH_PORT.pid"
  logf="$LOG_DIR/WebSearch_$WEBSEARCH_PORT.log"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "WebSearch already running (PID $(cat "$pidf"))"
  else
    echo "Starting WebSearch on $SERVER_HOST:$WEBSEARCH_PORT (Workers: $WEBSEARCH_WORKERS)..."
    nohup uvicorn cache_serper_server:app \
      --host "$SERVER_HOST" \
      --port "$WEBSEARCH_PORT" \
      --workers "$WEBSEARCH_WORKERS" \
      --log-level info \
      --timeout-keep-alive 120 \
      --app-dir "$DIR" \
      > "$logf" 2>&1 &
    echo $! > "$pidf"
    # Use health check to verify startup (max 15 seconds)
    started=0
    for i in {1..15}; do
      if curl -s --connect-timeout 1 "http://$SERVER_HOST:$WEBSEARCH_PORT/health" &>/dev/null; then
        echo "WebSearch started successfully (PID: $(cat "$pidf"), address: $SERVER_HOST:$WEBSEARCH_PORT)"
        started=1
        break
      fi
      sleep 1
    done
    [[ $started -eq 0 ]] && echo "WebSearch startup failed, check log: $logf" && tail -3 "$logf"
  fi

  # Wait for services to start
  echo ""
  echo "Waiting for services to start..."
  sleep 3
  echo "All services started!"
  echo ""
  echo "Check status: $0 status"
  echo "View logs: tail -f $LOG_DIR/*.log"

# ---------------------------------------------
#                test
# ---------------------------------------------
elif [[ "$cmd" == "test" ]]; then
  echo "-------------------- Testing web search ------------------"
  python -u "$DIR/test_cache_serper_server.py" \
          "http://$SERVER_HOST:$WEBSEARCH_PORT/search"
  echo "------------------------- Test done --------------------------"
  
  echo "-------------------- Testing crawl page -------------------"
  python -u "$DIR/test_crawl_page_simple.py" \
          "http://$SERVER_HOST:$CRAWL_PAGE_PORT/crawl_page"
  echo "------------------------- Test done --------------------------"


# ---------------------------------------------
#                stop
# ---------------------------------------------
elif [[ "$cmd" == "stop" ]]; then
  echo "Stopping all services and child processes..."
  echo ""

  # CrawlPage
  stop_service_with_children "CrawlPage" \
    "$PID_DIR/${SERVER_HOST}_CrawlPage_$CRAWL_PAGE_PORT.pid" \
    "$CRAWL_PAGE_PORT"

  # WebSearch
  stop_service_with_children "WebSearch" \
    "$PID_DIR/${SERVER_HOST}_WebSearch_$WEBSEARCH_PORT.pid" \
    "$WEBSEARCH_PORT"
  
  echo ""
  echo "All services stopped"

# =============================================================================
#                               STATUS
# =============================================================================
else
  echo "=========================================="
  echo "          Service Status Check"
  echo "=========================================="
  
  check_service_status "CrawlPage" \
    "$PID_DIR/${SERVER_HOST}_CrawlPage_$CRAWL_PAGE_PORT.pid" \
    "$CRAWL_PAGE_PORT"
  
  check_service_status "WebSearch" \
    "$PID_DIR/${SERVER_HOST}_WebSearch_$WEBSEARCH_PORT.pid" \
    "$WEBSEARCH_PORT"
  
  echo "=========================================="
fi