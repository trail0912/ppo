# ROS 2 DDS 与 Gazebo Ignition Transport 优化指南

## 问题背景

长时间训练中，ROS 2 DDS 和 Gazebo Ignition Transport 的组播通信可能
因资源耗尽而阻塞，导致 `ros_gz_sim` 卡死在 "Requesting list of world names"。

## 方案一：切换到 CycloneDDS（推荐）

CycloneDDS 比 Fast-DDS 更轻量，适合单机仿真训练：

```bash
# 安装 CycloneDDS
sudo apt install ros-humble-rmw-cyclonedds-cpp

# 训练前设置环境变量
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

## 方案二：限制 Fast-DDS 资源使用

如果继续使用 Fast-DDS，创建以下配置文件限制 participant 资源：

### 步骤 1：创建 `fastdds_profile.xml`

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<dds>
  <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
      <transport_descriptor>
        <transport_id>UDPv4Transport</transport_id>
        <type>UDPv4</type>
        <!-- 限制接收缓冲区大小 -->
        <maxInitialPeersRange>10</maxInitialPeersRange>
        <maxMessageSize>65000</maxMessageSize>
        <sendBufferSize>65536</sendBufferSize>
        <receiveBufferSize>65536</receiveBufferSize>
      </transport_descriptor>
    </transport_descriptors>

    <participant profile_name="training_participant">
      <rtps>
        <name>training_node</name>
        <builtin>
          <!-- 减少发现协议流量 -->
          <discovery_config>
            <leaseDuration>
              <sec>60</sec>
            </leaseDuration>
            <leaseAnnouncement>
              <sec>10</sec>
            </leaseAnnouncement>
          </discovery_config>
        </builtin>
        <!-- 限制 RTPS 分配 -->
        <allocation>
          <participants>
            <initial>1</initial>
            <maximum>4</maximum>
            <increment>1</increment>
          </participants>
          <readers>
            <initial>4</initial>
            <maximum>32</maximum>
            <increment>1</increment>
          </readers>
          <writers>
            <initial>4</initial>
            <maximum>32</maximum>
            <increment>1</increment>
          </writers>
          <total_readers>
            <initial>4</initial>
            <maximum>32</maximum>
            <increment>1</increment>
          </total_readers>
          <total_writers>
            <initial>4</initial>
            <maximum>32</maximum>
            <increment>1</increment>
          </total_writers>
        </allocation>
      </rtps>
    </participant>
  </profiles>
</dds>
```

### 步骤 2：训练前加载配置

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/fastdds_profile.xml
```

## 方案三：使用 CycloneDDS + 配置（最佳）

创建 `cyclonedds_config.xml`：

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="https://cdds.io/config
            https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
  <Domain id="any">
    <General>
      <NetworkInterfaceAddress>lo</NetworkInterfaceAddress>
      <!-- 单机训练使用 loopback，避免组播网络开销 -->
      <AllowMulticast>false</AllowMulticast>
      <DontRoute>true</DontRoute>
    </General>
    <Discovery>
      <Peers>
        <Peer address="localhost"/>
      </Peers>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>9</MaxAutoParticipantIndex>
    </Discovery>
    <Internal>
      <Watermarks>
        <WhcHigh>500kB</WhcHigh>
      </Watermarks>
      <MaxParticipants>4</MaxParticipants>
      <MaxQueuedSamples>64</MaxQueuedSamples>
    </Internal>
  </Domain>
</CycloneDDS>
```

然后：
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$(pwd)/config/cyclonedds_config.xml
```

## Gazebo Ignition Transport 优化

### 限制 ZeroMQ 资源

Gazebo Harmonic 使用 gz-transport (基于 ZeroMQ)。可以通过环境变量限制：

```bash
# 限制 ZeroMQ 接收超时
export GZ_TRANSPORT_TOPIC_STATISTICS_PERIOD=0

# 如果是 Gazebo Fortress 或更早版本，限制通信范围
export GZ_PARTITION=training
export GZ_IP=127.0.0.1
export GZ_RELAY=127.0.0.1
```

## 训练前一键设置脚本

将此脚本保存为 `setup_training_env.sh`，训练前执行：

```bash
#!/bin/bash

# 使用 CycloneDDS + 本地回环
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Gazebo 限制通信范围
export GZ_PARTITION=training
export GZ_IP=127.0.0.1

# ROS 2 日志级别（减少日志 I/O）
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=30  # WARN 及以上

# 确认设置
echo "=== Training Environment Settings ==="
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "GZ_PARTITION=$GZ_PARTITION"
echo "GZ_IP=$GZ_IP"
echo "====================================="
```

## 资源监控建议

训练过程中定期检查：

```bash
# 每 30 秒检查一次
watch -n 30 'echo "Gazebo models:"; gz model --list 2>/dev/null | wc -l;
echo "ROS nodes:"; ros2 node list 2>/dev/null | wc -l;
echo "Memory:"; free -h | head -2;
echo "Top processes:"; ps aux --sort=-%mem | head -6'
```

如果发现 Gazebo 模型数量持续增长（> 1），说明 reset 逻辑仍有泄漏。
