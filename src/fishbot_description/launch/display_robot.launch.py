from launch import LaunchDescription
import launch
import os
import launch_ros

from ament_index_python.packages import get_package_share_directory
import launch_ros.parameter_descriptions


def generate_launch_description():

    urdf_package_path = get_package_share_directory('fishbot_description')

    default_urdf_path = os.path.join(
        urdf_package_path,
        'urdf',
        'mini_diff_robot.urdf'
    )

    default_rviz2_config_path = os.path.join(
        urdf_package_path,
        'config',
        'display_robot_model.rviz'
    )

    # ---------------- model arg ----------------
    action_declare_arg_mode_path = launch.actions.DeclareLaunchArgument(
        name='model',
        default_value=str(default_urdf_path),
        description='加载的模型文件路径'
    )

    # ---------------- robot description ----------------
    robot_description_value = launch_ros.parameter_descriptions.ParameterValue(
        launch.substitutions.Command([
            'xacro ', launch.substitutions.LaunchConfiguration('model')
        ]),
        value_type=str
    )

    # ---------------- robot_state_publisher ----------------
    action_robot_state_publisher = launch_ros.actions.Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "robot_description": robot_description_value,
            "use_sim_time": True
        }]
    )

    # ---------------- joint_state_publisher ----------------
    action_joint_state_publisher = launch_ros.actions.Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{
            "use_sim_time": True
        }]
    )

    # ---------------- RViz ----------------
    action_rviz_node = launch_ros.actions.Node(
        package="rviz2",
        executable="rviz2",
        arguments=['-d', default_rviz2_config_path],
        parameters=[{
            "use_sim_time": True
        }]
    )

    return launch.LaunchDescription([
        action_declare_arg_mode_path,
        action_joint_state_publisher,
        action_robot_state_publisher,
        action_rviz_node
    ])