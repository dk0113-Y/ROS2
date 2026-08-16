"""Start only the Wheeltec base, lidar, odometry filtering, and TF chain."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    wheeltec_share = FindPackageShare('turn_on_wheeltec_robot')
    wheeltec_launch = PathJoinSubstitution([wheeltec_share, 'launch'])
    default_param_file = PathJoinSubstitution(
        [wheeltec_share, 'config', 'wheeltec_param.yaml']
    )
    default_model_file = PathJoinSubstitution(
        [wheeltec_share, 'config', 'robot_model.yaml']
    )

    car_mode = LaunchConfiguration('car_mode')
    lidar_type = LaunchConfiguration('lidar_type')
    param_file = LaunchConfiguration('wheeltec_param_yaml')
    model_file = LaunchConfiguration('robot_model_yaml')

    base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([wheeltec_launch, 'base_serial.launch.py'])
        ),
        launch_arguments={
            'car_mode': car_mode,
            'imu_mode_yaml': param_file,
            'wheeltec_param_yaml': param_file,
        }.items(),
    )
    ekf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([wheeltec_launch, 'wheeltec_ekf.launch.py'])
        ),
        launch_arguments={
            'carto_slam': 'false',
            'robot_nav': 'false',
        }.items(),
    )
    robot_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [wheeltec_launch, 'robot_mode_description.launch.py']
            )
        ),
        launch_arguments={
            'car_mode': car_mode,
            'robot_model_yaml': model_file,
            'wheeltec_param_yaml': param_file,
        }.items(),
    )
    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([wheeltec_launch, 'wheeltec_lidar.launch.py'])
        ),
        launch_arguments={
            'lidar_type': lidar_type,
            'lidar_type_yaml': param_file,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'car_mode',
            default_value='',
            description='Wheeltec car mode; empty uses wheeltec_param.yaml',
        ),
        DeclareLaunchArgument(
            'lidar_type',
            default_value='',
            description='Wheeltec lidar type; empty uses wheeltec_param.yaml',
        ),
        DeclareLaunchArgument(
            'wheeltec_param_yaml',
            default_value=default_param_file,
            description='Wheeltec robot and lidar parameter file',
        ),
        DeclareLaunchArgument(
            'robot_model_yaml',
            default_value=default_model_file,
            description='Wheeltec robot model transform file',
        ),
        LogInfo(
            msg=(
                'Starting minimal realcar bringup only: base, EKF, robot TF, '
                'and lidar. No DRL, Nav2, exploration, or cmd_vel publisher.'
            )
        ),
        base,
        ekf,
        robot_description,
        lidar,
    ])
