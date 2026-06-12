#!/bin/bash
# ==============================================================================
# 训练环境一键设置脚本
# 使用前: source config/setup_training_env.sh
# ==============================================================================

# ---- ROS 2 DDS ----
# 切换到轻量级 CycloneDDS（需要先安装: sudo apt install ros-humble-rmw-cyclonedds-cpp）
if [ -f /opt/ros/humble/setup.bash ]; then
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    echo "[setup] RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
else
    echo "[setup] WARN: CycloneDDS may not be installed"
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
echo "=========================================="
