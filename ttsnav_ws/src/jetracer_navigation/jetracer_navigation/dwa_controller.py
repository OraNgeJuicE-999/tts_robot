"""ROS2 node that runs the DWA local controller: follows the /plan path published
by a_star_planner while reacting to live obstacles from /scan, publishing
AckermannDriveStamped commands (same message type/topic convention as
teleop_keyboard.py) at a fixed control-loop rate.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import tf2_ros
from tf_transformations import euler_from_quaternion 

from nav_msgs.msg import Path, OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

from jetracer_navigation.dwa import DWA, DWAConfig
from jetracer_navigation.primitives import RobotState, PathNode, PixelCoords
from jetracer_navigation.utils import euclidean_distance


class DWAController(Node):
    def __init__(self):
        super().__init__('dwa_controller')

        self.dwa = DWA(config=DWAConfig())

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.path = None           # latest nav_msgs/Path from /plan
        self.world_map = None      # local obstacle representation built from /scan
        self.map_info = None
        self.lookahead_distance = 1.2  # meters; tune above the vehicle's min turning radius (~0.78m)
        self.last_v = 0.0
        self.last_steer = 0.0

        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, scan_qos)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, 'ackermann_cmd', 10)
        self.timer = self.create_timer(1.0 / 15.0, self.control_loop)

    def path_callback(self, msg):

        self.path = msg.poses
        self.get_logger().info(f"path_callback: received {len(self.path)} waypoints")


    def scan_callback(self, msg):
        """Turn the latest LaserScan into a small local occupancy grid (same
        1=occupied/0=free convention as self.world_map elsewhere), centered on
        the robot, for clearance_cost() to check trajectories against."""
        try:
            t = self.tf_buffer.lookup_transform('map', msg.header.frame_id, rclpy.time.Time())
        except tf2_ros.TransformException:
            return
        origin_x = t.transform.translation.x
        origin_y = t.transform.translation.y
        quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
        _, _, origin_yaw = euler_from_quaternion(quat)

        resolution = 0.05
        grid_size = int(2 * msg.range_max / resolution)
        grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

        grid_origin_x = origin_x - msg.range_max
        grid_origin_y = origin_y - msg.range_max

        for i, r in enumerate(msg.ranges):
            if not np.isfinite(r) or r < msg.range_min or r > msg.range_max:
                continue

            angle = msg.angle_min + i * msg.angle_increment
            x_local = r * np.cos(angle)
            y_local = r * np.sin(angle)

            x_map = origin_x + x_local * np.cos(origin_yaw) - y_local * np.sin(origin_yaw)
            y_map = origin_y + x_local * np.sin(origin_yaw) + y_local * np.cos(origin_yaw)

            px = int((x_map - grid_origin_x) / resolution)
            py = int((y_map - grid_origin_y) / resolution)

            if 0 <= px < grid_size and 0 <= py < grid_size:
                grid[py][px] = 1

        self.world_map = grid

        map_info = MapMetaData()
        map_info.resolution = resolution
        map_info.width = grid_size
        map_info.height = grid_size
        map_info.origin.position.x = grid_origin_x
        map_info.origin.position.y = grid_origin_y
        self.map_info = map_info


    def get_current_state(self) -> RobotState:
        
        t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        xx = t.transform.translation.x
        yy = t.transform.translation.y
        quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
        _, _, yaw = euler_from_quaternion(quat)

        return RobotState(xx, yy, yaw, v=self.last_v, steer=self.last_steer)
        

    def closest_point_on_segment(self, p, a, b):
        ab_x = b[0] - a[0]
        ab_y = b[1] - a[1]
        ab_len_sq = ab_x ** 2 + ab_y ** 2
        if ab_len_sq == 0.0:
            return a, float(np.hypot(p[0] - a[0], p[1] - a[1]))

        t = ((p[0] - a[0]) * ab_x + (p[1] - a[1]) * ab_y) / ab_len_sq
        t = max(0.0, min(1.0, t))
        closest = (a[0] + t * ab_x, a[1] + t * ab_y)
        dist = float(np.hypot(p[0] - closest[0], p[1] - closest[1]))
        return closest, dist

    def get_local_goal(self) -> tuple:
        state = self.get_current_state()
        p = (state.x, state.y)

        best_dist = np.inf
        best_i = None
        best_point = None
        for i in range(len(self.path) - 1):
            a = (self.path[i].pose.position.x, self.path[i].pose.position.y)
            b = (self.path[i + 1].pose.position.x, self.path[i + 1].pose.position.y)
            point, dist = self.closest_point_on_segment(p, a, b)
            if dist < best_dist:
                best_dist = dist
                best_i = i
                best_point = point

        if best_i is None:
            last = self.path[-1].pose.position
            return (last.x, last.y)

        remaining = self.lookahead_distance
        next_point = (self.path[best_i + 1].pose.position.x, self.path[best_i + 1].pose.position.y)
        first_leg = float(np.hypot(next_point[0] - best_point[0], next_point[1] - best_point[1]))
        if remaining <= first_leg:
            frac = remaining / first_leg if first_leg > 0 else 0.0
            return (
                best_point[0] + frac * (next_point[0] - best_point[0]),
                best_point[1] + frac * (next_point[1] - best_point[1]),
            )
        remaining -= first_leg

        for i in range(best_i + 1, len(self.path) - 1):
            a = (self.path[i].pose.position.x, self.path[i].pose.position.y)
            b = (self.path[i + 1].pose.position.x, self.path[i + 1].pose.position.y)
            seg_len = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            if remaining <= seg_len:
                frac = remaining / seg_len if seg_len > 0 else 0.0
                return (a[0] + frac * (b[0] - a[0]), a[1] + frac * (b[1] - a[1]))
            remaining -= seg_len

        last = self.path[-1].pose.position
        return (last.x, last.y)

    def control_loop(self):
        """Timer callback: run one DWA cycle and publish a command."""
        t0 = self.get_clock().now()

        if self.path is None or self.world_map is None:
            self.get_logger().info(
                f"control_loop: waiting (path={'yes' if self.path is not None else 'no'}, "
                f"world_map={'yes' if self.world_map is not None else 'no'})"
            )
            self.Ackermann_cmd_publisher(0.0, 0.0)
            return

        state = self.get_current_state()
        goal = self.get_local_goal()
        v, steer, traj = self.dwa.plan(state, goal, self.world_map, self.map_info)

        dt = (self.get_clock().now() - t0).nanoseconds / 1e9
        self.get_logger().info(
            f"control_loop: state=({state.x:.2f},{state.y:.2f},{state.theta:.2f}) "
            f"last=({self.last_v:.2f},{self.last_steer:.2f}) goal={goal} "
            f"v={v} steer={steer} took={dt:.3f}s"
        )

        if v is None:
            self.get_logger().warning("control_loop: no valid trajectory found, stopping")
            self.Ackermann_cmd_publisher(0.0, 0.0)
            return

        self.Ackermann_cmd_publisher(v, steer)
            

    def Ackermann_cmd_publisher(self, speed:float, steer:float):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        # DWA's internal bicycle model uses the standard convention (positive speed
        # = forward), but this robot's real Ackermann interface is inverted (see
        # teleop_keyboard.py: 'w' = forward drives self.speed negative) — so the
        # sign is flipped only here, at the final publish boundary, leaving DWA's
        # own math and self.last_v in its normal convention.
        msg.drive.speed = -float(speed)
        msg.drive.steering_angle = float(steer)
        self.cmd_pub.publish(msg)
        self.last_v = float(speed)
        self.last_steer = float(steer)


def main(args=None):
    rclpy.init(args=args)
    node = DWAController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()