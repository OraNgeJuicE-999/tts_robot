"""ROS2 node that bridges named-location commands into nav goals."""

import json
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler
from ament_index_python import get_package_share_directory

from jetracer_navigation.locations import load_location_map, get_location, LocationNotFoundError


class GoalBridge(Node):
    def __init__(self):
        super().__init__('goal_bridge')

        location_path = os.path.join(get_package_share_directory('jetracer_navigation'), 'config', 'locations.json')
        self.declare_parameter('location_file', location_path)
        self.location_file = self.get_parameter('location_file').value

        self.location_map = load_location_map(self.location_file)

        # Fixed patrol route for a full-survey command. Order matters -- this
        # is walked start to finish, one waypoint per /goal_reached signal.
        self.course = [
            "TOP_LEFT", "TOP_RIGHT",
            "HALLWAY1_RIGHT", "HALLWAY1_LEFT",
            "HALLWAY2_LEFT", "HALLWAY2_RIGHT",
            "HALLWAY3_RIGHT", "HALLWAY3_LEFT",
            "BOTTOM_RIGHT", "BOTTOM_LEFT",
        ]
        self.course_index = None  # None = not currently running a course
        self.course_started = False

        self.command_sub = self.create_subscription(String, '/robot_command', self.command_callback, 10)
        self.goal_reached_sub = self.create_subscription(Bool, '/goal_reached', self.goal_reached_callback, 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

    def command_callback(self, msg):
        try:
            command = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Command JSON Error: {exc}")
            return

        command_name = command.get("command_name")

        if command_name == "navigate":
            loc = command.get("parameters", {}).get("destination")
            if loc is None:
                self.get_logger().warning(f"navigate command missing destination: {command}")
                return
            self.publish_goal_for_location(loc)
            return

        elif command_name == "survey":
            self.start_course()
            return

    def start_course(self):
        """Begin walking self.course from the first waypoint, and (since
        goal_reached_callback calls back into this same method) advance to
        the next one on every later call, stopping once the last waypoint
        has been reached rather than published."""
        if self.course_index is None:
            self.course_index = 0
            self.course_started = True
        elif self.course_index >= len(self.course) - 1:
            self.course_started = False
            self.course_index = None
            return
        else:
            self.course_index += 1

        self.publish_goal_for_location(self.course[self.course_index])



    def goal_reached_callback(self, msg):
        """Advance to the next course waypoint when dwa_controller signals
        /goal_reached, if a course is currently running."""
        
        if self.course_started:
            self.start_course()
        return

    def publish_goal_for_location(self, name):
        """Look up name via locations.py and publish it as a /goal_pose."""
        try:
            x, y, yaw = get_location(self.location_map, name)
        except LocationNotFoundError as e:
            self.get_logger().warning(f'Location Library Error: {e}')
            return

        pub = PoseStamped()
        pub.header.frame_id = 'map'
        pub.header.stamp = self.get_clock().now().to_msg()
        pub.pose.position.x = x
        pub.pose.position.y = y
        pub.pose.position.z = 0.0
        qx, qy, qz, qw = quaternion_from_euler(0, 0, yaw)
        pub.pose.orientation.x = qx
        pub.pose.orientation.y = qy
        pub.pose.orientation.z = qz
        pub.pose.orientation.w = qw

        self.goal_pub.publish(pub)


def main(args=None):
    rclpy.init(args=args)
    node = GoalBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()