from launch import LaunchDescription
import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    pkg = get_package_share_directory('fishbot_description')

    xacro_file = os.path.join(pkg, 'urdf', 'mini_diff', 'fishbot.urdf.xacro')
    world_file = os.path.join(pkg, 'world2', 'robot_room.world')

    resource_path = os.path.dirname(pkg)
    existing = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    if existing:
        resource_path += ":" + existing

    # ================= model =================
    declare_model = DeclareLaunchArgument(
        'model',
        default_value=xacro_file
    )

    robot_description = ParameterValue(
        Command(['xacro ', LaunchConfiguration('model')]),
        value_type=str
    )

    # ================= robot_state_publisher =================
    # 发布到 /tf_uncorrected，由 odom_tf_broadcaster 统一加 epoch offset 后转发到 /tf
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True
        }],
        remappings=[
            ('/tf', '/tf_uncorrected'),
        ],
        output='screen'
    )

    # ================= Gazebo =================
    gz = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_file, '--render-engine', 'ogre2'],
        additional_env={'GZ_SIM_RESOURCE_PATH': resource_path},
        output='screen'
    )

    # ================= spawn =================
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/robot_description',
            '-name', 'mini_diff_robot'
        ],
        output='screen'
    )

    spawn_delay = TimerAction(period=5.0, actions=[spawn])

    # ================= FIX 1: bridge =================
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/scan_raw@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/world/default/model/mini_diff_robot/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/world/default/control@ros_gz_interfaces/srv/ControlWorld',
            '/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose',
        ],
    )

    # ================= FIX 2: merge Gazebo real joints + passive joints -> /joint_states =================
    joint_state_merger = Node(
        package='fishbot_application',
        executable='passive_joint_pub',
        parameters=[{
            'use_sim_time': True
        }],
        output='screen'
    )

    # ================= FIX 3: odom->base_footprint dynamic TF from /odom =================
    odom_tf_broadcaster = Node(
        package='fishbot_application',
        executable='odom_tf_broadcaster',
        parameters=[{
            'use_sim_time': True
        }],
        output='screen'
    )

    return LaunchDescription([
        declare_model,
        rsp,
        gz,
        spawn_delay,
        bridge,
        joint_state_merger,
        odom_tf_broadcaster,
    ])