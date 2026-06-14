#!/bin/bash
# ==============================================================================
# 训练环境一键设置脚本
# 使用前: source config/setup_training_env.sh
# ==============================================================================

# ---- 系统资源限制 ----
# 提高文件描述符上限，防止 ZeroMQ/TCP 连接数耗尽
ulimit -n 65536 2>/dev/null && echo "[setup] ulimit -n = 65536" || echo "[setup] WARN: cannot raise ulimit"
# 提高进程数上限
ulimit -u 16384 2>/dev/null && echo "[setup] ulimit -u = 16384" || echo "[setup] WARN: cannot raise ulimit -u"

# ---- ROS 2 DDS ----
# 切换到轻量级 CycloneDDS（需要先安装: sudo apt install ros-humble-rmw-cyclonedds-cpp）
if [ -f /opt/ros/humble/setup.bash ]; then
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    echo "[setup] RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
else
    echo "[setup] WARN: CycloneDDS may not be installed"
fi

# 加载 CycloneDDS 配置（单机优化：禁用组播、限制参与者数）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CYCLONE_CONFIG="$SCRIPT_DIR/cyclonedds_config.xml"
if [ -f "$CYCLONE_CONFIG" ]; then
    export CYCLONEDDS_URI="file://$CYCLONE_CONFIG"
    echo "[setup] CYCLONEDDS_URI=$CYCLONEDDS_URI"
else
    echo "[setup] WARN: cyclonedds_config.xml not found at $CYCLONE_CONFIG"
fi

# ---- Gazebo Ignition Transport ----
# 单机训练：限制通信为本地回环
export GZ_PARTITION=training
export GZ_IP=127.0.0.1

# ---- ROS 2 日志 ----
# 减少日志 I/O：WARN 级别及以上
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=30

# ---- 确认设置 ----
echo "=========================================="
echo " Training Environment Variables"
echo "=========================================="
echo "RMW_IMPLEMENTATION = $RMW_IMPLEMENTATION"
echo "GZ_PARTITION       = $GZ_PARTITION"
echo "GZ_IP              = $GZ_IP"
echo "LOG_THRESHOLD      = WARN (30)"
echo "CYCLONEDDS_URI     = $CYCLONEDDS_URI"
echo "=========================================="
