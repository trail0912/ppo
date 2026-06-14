#!/bin/bash
# ==============================================================================
# 训练资源监控脚本
# 用法:
#   ./scripts/monitor_resources.sh [输出文件]
#   默认每 30 秒检查一次，Ctrl+C 退出
#
#   watch 模式（每 30 秒刷新一次终端）:
#   watch -n 30 ./scripts/monitor_resources.sh
# ==============================================================================

OUTPUT_FILE="${1:-/dev/stdout}"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$OUTPUT_FILE"
}

echo "=========================================="
echo " Resource Monitor (interval: 30s)"
echo " Start: $(date)"
echo "=========================================="

trap 'echo ""; echo "Monitoring stopped."; exit 0' INT

while true; do
    echo ""
    echo "=== $(date) ==="

    # ---- Gazebo 模型数（关键：>1 说明有残留） ----
    MODEL_COUNT=$(gz model --list 2>/dev/null | wc -l)
    if [ "$MODEL_COUNT" -gt 1 ]; then
        log "WARN: Gazebo model count = $MODEL_COUNT (>1 means stale models!)"
        gz model --list 2>/dev/null
    else
        log "Gazebo model count: $MODEL_COUNT"
    fi

    # ---- ROS 节点数 ----
    NODE_COUNT=$(ros2 node list 2>/dev/null | wc -l)
    log "ROS 2 node count: $NODE_COUNT"

    # ---- 内存占用 ----
    log "Memory usage:"
    free -h | head -2 | tee -a "$OUTPUT_FILE"

    # ---- CPU/内存 Top 进程 ----
    log "Top processes (by memory):"
    ps aux --sort=-%mem 2>/dev/null | head -6 | tee -a "$OUTPUT_FILE"

    # ---- 文件描述符（检查 ZeroMQ socket 泄漏） ----
    GZ_FD_COUNT=$(lsof -p $(pgrep -f "gz sim" | head -1) 2>/dev/null | wc -l)
    if [ -n "$GZ_FD_COUNT" ] && [ "$GZ_FD_COUNT" -gt 0 ]; then
        log "Gazebo open file descriptors: $GZ_FD_COUNT (high => possible socket leak)"
    fi

    BRIDGE_FD_COUNT=$(lsof -p $(pgrep -f "ros_gz_bridge" | head -1) 2>/dev/null | wc -l)
    if [ -n "$BRIDGE_FD_COUNT" ] && [ "$BRIDGE_FD_COUNT" -gt 0 ]; then
        log "ros_gz_bridge open FDs: $BRIDGE_FD_COUNT"
    fi

    # ---- TCP 连接状态统计 ----
    TIME_WAIT=$(ss -tan state time-wait 2>/dev/null | wc -l)
    if [ "$TIME_WAIT" -gt 100 ]; then
        log "WARN: TCP TIME_WAIT count: $TIME_WAIT (high => socket cleanup issue)"
    fi

    # ---- 僵尸进程检查 ----
    ZOMBIES=$(ps aux 2>/dev/null | grep -c 'defunct')
    if [ "$ZOMBIES" -gt 5 ]; then
        log "WARN: Zombie processes: $ZOMBIES"
    fi

    sleep 30
done
