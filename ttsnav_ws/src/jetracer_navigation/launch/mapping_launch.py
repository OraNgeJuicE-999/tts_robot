"""Bringup for re-running SLAM mapping (slam_toolbox), at the LIDAR's current
mounted height/position -- so the resulting map matches what the sensor
actually sees now, instead of amcl trying to scan-match live data against a
map captured at a different LIDAR geometry.

Deliberately does NOT start map_server/amcl/lifecycle_manager (unlike
bringup_launch.py) -- slam_toolbox owns the map->odom transform and builds
the map itself here, and would fight amcl over that if both ran at once.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use the /clock topic (Isaac Sim) instead of the system clock',
    )

    slam_params_path = os.path.expanduser('~/tts_robot/slam_config/mapper_params_online_async.yaml')

    pointcloud_to_laserscan_node = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        output='screen',
        remappings=[
            ('cloud_in', '/point_cloud'),
            ('scan', '/scan'),
        ],
        parameters=[{
            'target_frame': 'base_link',
            'transform_tolerance': 0.1,
            'min_height': -0.5,
            'max_height': 1.0,
            'angle_min': -3.14159,
            'angle_max': 3.14159,
            'range_min': 0.1,
            'range_max': 12.0,
            'use_sim_time': use_sim_time,
        }],
    )

    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params_path, {'use_sim_time': use_sim_time}],
    )

    # slam_toolbox is a managed lifecycle node here (same as map_server/amcl
    # in bringup_launch.py) -- without something to configure+activate it,
    # it just sits unconfigured forever: no /scan subscription, no /map
    # publisher, nothing. bringup_launch.py already needs this same pattern
    # for map_server/amcl; this was missed here on the first pass.
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_mapping',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['slam_toolbox'],
            'bond_timeout': 0.0,
        }],
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,
        pointcloud_to_laserscan_node,
        slam_toolbox_node,
        lifecycle_manager_node,
    ])
