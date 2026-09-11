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
            # TODO: once validate_course() below is implemented, decide
            # whether "survey" should gate on its result before calling
            # start_course() at all.
            self.start_course()
            return

    def validate_course(self) -> bool:
        """TODO (course dry-run validation): before committing to driving
        the whole survey route, check that every consecutive waypoint pair
        in self.course actually has a feasible A* path -- so an infeasible
        leg (e.g. one blocked by a_star_planner's inflation_radius) is
        caught before the robot ever leaves the first waypoint, instead of
        discovered mid-course when it's stuck partway through with nowhere
        to go. Worth working out:
          - Where should this run: once at node startup (as soon as a map
            is available), or every time "survey" is triggered?
          - goal_bridge doesn't currently subscribe to /map at all, and has
            no A* planning logic of its own. a_star.py's AStarImplementation
            is kept separate from a_star_planner.py specifically so it's
            reusable/testable standalone (see that module's own docstring)
            -- does that mean goal_bridge should import it directly and
            keep its own copy of the map, or should this ask a_star_planner
            to do the check instead (and if so, how -- a service call?),
            so the planning logic and its inflation_radius aren't
            duplicated in two places that could drift out of sync?
          - If a leg turns out infeasible, what should happen: refuse to
            start the whole course, skip just that leg, or log a warning
            and let it fail the way it does today (silently, discovered
            only when the robot gets there)?
        """
        pass

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