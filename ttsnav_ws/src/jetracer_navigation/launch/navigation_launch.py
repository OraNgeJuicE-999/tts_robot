"""Launch the JetRacer path-planning stack: a_star_planner (global planner,
publishes /plan) and dwa_controller (local controller, follows /plan and
publishes ackermann_cmd).

Run bringup_launch.py first (robot_state_publisher, map_server, amcl,
pointcloud_to_laserscan) -- this is kept separate so the planner/controller
can be rebuilt and restarted independently while iterating on them.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    a_star_planner_node = Node(
        package='jetracer_navigation',
        executable='a_star_planner',
        name='a_star_planner',
        output='screen',
    )

    dwa_controller_node = Node(
        package='jetracer_navigation',
        executable='dwa_controller',
        name='dwa_controller',
        output='screen',
    )

    return LaunchDescription([
        a_star_planner_node,
        dwa_controller_node,
    ])