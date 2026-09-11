"""Bringup for the JetRacer navigation stack (localization mode).

Starts the "general infrastructure" nodes together: robot_state_publisher,
map_server + amcl (with their lifecycle transitions handled automatically),
and pointcloud_to_laserscan. a_star_planner and rviz2 are intentionally left
out — run those manually so you can rebuild/restart them independently
without tearing down everything else.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml_file = LaunchConfiguration('map_yaml_file')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use the /clock topic (Isaac Sim) instead of the system clock',
    )

    declare_map_yaml_file_cmd = DeclareLaunchArgument(
        'map_yaml_file',
        default_value=os.path.expanduser('~/tts_robot/maps/office1_map.yaml'),
        description=(
            'Map yaml to load. Defaults to the slam_toolbox-built map; pass '
            'map_yaml_file:=<path>/office1_ground_truth.yaml to localize '
            'against the Occupancy Map Generator ground-truth map instead.'
        ),
    )

    urdf_path = os.path.expanduser('~/tts_robot/jetracer.urdf')
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
    )

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': map_yaml_file,
            'use_sim_time': use_sim_time,
        }],
    )

    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[{
            'base_frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'global_frame_id': 'map',
            'scan_topic': '/scan',
            'use_sim_time': use_sim_time,
            # Defaults are light for a repetitive/symmetric office layout
            # (many near-identical cubicle rows), which is exactly the kind
            # of environment where the particle filter can suddenly favor a
            # wrong-but-similar-looking hypothesis instead of gradually
            # losing confidence. More particles keep a richer set of
            # competing hypotheses alive so a plausible-but-wrong one is
            # less likely to dominate via resampling noise alone; tighter
            # update thresholds mean corrections happen more often instead
            # of letting motion (e.g. a sharp turn at a waypoint transition)
            # accumulate before AMCL gets a chance to catch a divergence.
            'min_particles': 500,
            'max_particles': 5000,
            'update_min_d': 0.1,
            'update_min_a': 0.1,
        }],
    )

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

    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server', 'amcl'],
        }],
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,
        declare_map_yaml_file_cmd,
        robot_state_publisher_node,
        map_server_node,
        amcl_node,
        pointcloud_to_laserscan_node,
        lifecycle_manager_node,
    ])
