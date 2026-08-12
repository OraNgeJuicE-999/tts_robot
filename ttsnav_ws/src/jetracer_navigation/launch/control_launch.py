"""Launches the app-layer nodes together: a_star_planner, dwa_controller,
goal_bridge, and vlm_bridge. Kept separate from bringup_launch.py (which
covers the rarely-restarted infrastructure: robot_state_publisher,
map_server, amcl, pointcloud_to_laserscan) so this whole group can be
killed/relaunched on its own while iterating, without tearing down
localization.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='jetracer_navigation',
            executable='a_star_planner',
            name='a_star_planner',
            output='screen',
        ),
        Node(
            package='jetracer_navigation',
            executable='dwa_controller',
            name='dwa_controller',
            output='screen',
        ),
        Node(
            package='jetracer_navigation',
            executable='goal_bridge',
            name='goal_bridge',
            output='screen',
        ),
        Node(
            package='jetracer_navigation',
            executable='vlm_bridge',
            name='vlm_bridge',
            output='screen',
        ),
    ])