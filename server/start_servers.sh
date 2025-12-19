#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DIR")"
source "$PROJECT_ROOT/.env"
echo $PROJECT_ROOT
: "${SERVER_HOST:?请在 .env 中设置 SERVER_HOST}"
: "${CRAWL_PAGE_PORT:?请在 .env 中设置 CRAWL_PAGE_PORT}"
: "${WEBSEARCH_PORT:?请在 .env 中设置 WEBSEARCH_PORT}"

# 配置 workers 数量
CRAWL_PAGE_WORKERS=${CRAWL_PAGE_WORKERS:-10}
WEBSEARCH_WORKERS=${WEBSEARCH_WORKERS:-10} 

LOG_DIR="$DIR/logs/$SERVER_HOST";   mkdir -p "$LOG_DIR"
PID_DIR="$DIR/pids/$SERVER_HOST";   mkdir -p "$PID_DIR"

cmd=$1
if [[ ! "$cmd" =~ ^(start|stop|status|test)$ ]]; then
  echo "用法: $0 [start|stop|status|test]"
  exit 1
fi

# =============================================================================
# 辅助函数：停止服务及其所有子进程
# =============================================================================
stop_service_with_children() {
  local name=$1 pidf=$2 port=$3
  echo "正在停止 ${name}..."
  
  # 1. 通过 PID 文件停止进程组
  if [[ -f "$pidf" ]]; then
    local pid=$(cat "$pidf" 2>/dev/null)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      # 获取真正的进程组 ID
      local pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
      if [[ -n "$pgid" ]] && [[ "$pgid" != "0" ]]; then
        kill -TERM -"$pgid" 2>/dev/null
      else
        kill -TERM "$pid" 2>/dev/null
      fi
      # 等待进程结束
      for i in {1..5}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      # 超时强杀
      if kill -0 "$pid" 2>/dev/null; then
        [[ -n "$pgid" ]] && kill -9 -"$pgid" 2>/dev/null
        kill -9 "$pid" 2>/dev/null
      fi
      echo "${name} 已停止 (PID $pid)"
    fi
    rm -f "$pidf"
  fi
  
  # 2. 强制清理端口上的所有进程
  sleep 1
  local pids=$(lsof -t -i:"$port" 2>/dev/null)
  if [[ -n "$pids" ]]; then
    echo "  清理端口 $port 上的残留进程: $pids"
    echo "$pids" | xargs kill -9 2>/dev/null
    sleep 1
  fi
  
  # 检查最终状态
  if ! lsof -i:"$port" &>/dev/null; then
    echo "${name} 端口 $port 已释放"
  else
    echo "端口 $port 仍被占用，请手动检查: lsof -i:$port"
  fi
}

# =============================================================================
# 辅助函数：检查服务状态
# =============================================================================
check_service_status() {
  local service_name=$1
  local pidf=$2
  local port=$3
  local workers=$4
  
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    main_pid=$(cat "$pidf")
    echo "${service_name} 运行中 (主进程 PID: $main_pid)"
    
    # 查找所有子进程
    child_pids=($(pgrep -P "$main_pid" 2>/dev/null))
    if [[ ${#child_pids[@]} -gt 0 ]]; then
      echo "   └─ Worker 进程数: ${#child_pids[@]} (预期: $workers)"
      echo "   └─ Worker PIDs: ${child_pids[*]}"
    else
      echo "   └─ 单进程模式 (未检测到 worker 子进程)"
    fi
    
    # 显示端口监听情况
    local port_count=$(lsof -t -i:"$port" 2>/dev/null | wc -l)
    echo "   └─ 端口 $port 监听进程数: $port_count"
    
    # 显示内存使用
    local mem_usage=$(ps -o rss= -p "$main_pid" 2>/dev/null)
    if [[ -n "$mem_usage" ]]; then
      mem_mb=$((mem_usage / 1024))
      echo "   └─ 内存使用: ${mem_mb} MB"
    fi
    
  elif lsof -i:"$port" &>/dev/null; then
    echo "${service_name} 端口 $port 被占用，但 PID 文件无效或进程异常"
    echo "   占用端口的进程:"
    lsof -i:"$port" 2>/dev/null | grep LISTEN
  else
    echo "${service_name} 未运行, 且端口 $port 未被占用"
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
    echo "CrawlPage 已在运行 (PID $(cat "$pidf"))"
  else
    echo "启动 CrawlPage 端口 $SERVER_HOST:$CRAWL_PAGE_PORT (Workers: $CRAWL_PAGE_WORKERS)..."
    nohup uvicorn crawl_page_server:app \
      --host "$SERVER_HOST" \
      --port "$CRAWL_PAGE_PORT" \
      --workers "$CRAWL_PAGE_WORKERS" \
      --log-level info \
      --timeout-keep-alive 120 \
      --app-dir "$DIR" \
      > "$logf" 2>&1 &
    echo $! > "$pidf"
    # 使用健康检查验证启动（最多 15 秒）
    started=0
    for i in {1..15}; do
      if curl -s --connect-timeout 1 "http://$SERVER_HOST:$CRAWL_PAGE_PORT/health" &>/dev/null; then
        echo "CrawlPage 启动成功 (PID: $(cat "$pidf"), 地址: $SERVER_HOST:$CRAWL_PAGE_PORT)"
        started=1
        break
      fi
      sleep 1
    done
    [[ $started -eq 0 ]] && echo "CrawlPage 启动失败，请查看日志: $logf" && tail -3 "$logf"
  fi

  # WebSearch
  pidf="$PID_DIR/${SERVER_HOST}_WebSearch_$WEBSEARCH_PORT.pid"
  logf="$LOG_DIR/WebSearch_$WEBSEARCH_PORT.log"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "WebSearch 已在运行 (PID $(cat "$pidf"))"
  else
    echo "启动 WebSearch 端口 $SERVER_HOST:$WEBSEARCH_PORT (Workers: $WEBSEARCH_WORKERS)..."
    nohup uvicorn cache_serper_server:app \
      --host "$SERVER_HOST" \
      --port "$WEBSEARCH_PORT" \
      --workers "$WEBSEARCH_WORKERS" \
      --log-level info \
      --timeout-keep-alive 120 \
      --app-dir "$DIR" \
      > "$logf" 2>&1 &
    echo $! > "$pidf"
    # 使用健康检查验证启动（最多 15 秒）
    started=0
    for i in {1..15}; do
      if curl -s --connect-timeout 1 "http://$SERVER_HOST:$WEBSEARCH_PORT/health" &>/dev/null; then
        echo "WebSearch 启动成功 (PID: $(cat "$pidf"), 地址: $SERVER_HOST:$WEBSEARCH_PORT)"
        started=1
        break
      fi
      sleep 1
    done
    [[ $started -eq 0 ]] && echo "WebSearch 启动失败，请查看日志: $logf" && tail -3 "$logf"
  fi

  # 等待服务启动
  echo ""
  echo "等待服务启动..."
  sleep 3
  echo "所有服务启动完成！"
  echo ""
  echo "查看状态: $0 status"
  echo "查看日志: tail -f $LOG_DIR/*.log"

# ---------------------------------------------
#                test
# ---------------------------------------------
elif [[ "$cmd" == "test" ]]; then
  echo "--------------------开始测试 web search ------------------"
  python -u "$DIR/test_cache_serper_server.py" \
          "http://$SERVER_HOST:$WEBSEARCH_PORT/search"
  echo "-------------------------测试结束--------------------------"
  
  echo "--------------------开始测试 crawl page -------------------"
  python -u "$DIR/test_crawl_page_simple.py" \
          "http://$SERVER_HOST:$CRAWL_PAGE_PORT/crawl_page"
  echo "-------------------------测试结束--------------------------"


# ---------------------------------------------
#                stop
# ---------------------------------------------
elif [[ "$cmd" == "stop" ]]; then
  echo "正在停止所有服务及其子进程..."
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
  echo "所有服务已停止"

# =============================================================================
#                               STATUS
# =============================================================================
else
  echo "=========================================="
  echo "          服务状态检查"
  echo "=========================================="
  
  check_service_status "CrawlPage" \
    "$PID_DIR/${SERVER_HOST}_CrawlPage_$CRAWL_PAGE_PORT.pid" \
    "$CRAWL_PAGE_PORT"
  
  check_service_status "WebSearch" \
    "$PID_DIR/${SERVER_HOST}_WebSearch_$WEBSEARCH_PORT.pid" \
    "$WEBSEARCH_PORT"
  
  echo "=========================================="
fi