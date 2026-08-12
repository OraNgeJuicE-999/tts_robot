"""ROS2 node that runs the DWA local controller: follows the /plan path published
by a_star_planner while reacting to live obstacles from /scan, publishing
AckermannDriveStamped commands (same message type/topic convention as
teleop_keyboard.py) at a fixed control-loop rate.
"""

import cv2
import numpy as np
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import tf2_ros
from tf_transformations import euler_from_quaternion 

from nav_msgs.msg import Path, OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool

from jetracer_navigation.dwa import DWA, DWAConfig
from jetracer_navigation.primitives import RobotState, PathNode, PixelCoords
from jetracer_navigation.utils import euclidean_distance, inflated_obstacles


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
        self.goal_tolerance = 0.3      # meters; how close to the final path point counts as "reached"
        self.goal_reached_published = False  # edge-trigger: only publish once per path, not every cycle
        self.check_on_path_dist = 1.0
        # Local-map obstacle inflation, in pixels at the local grid's 0.05m/px
        # resolution. Covers the robot's own half-width (~0.0625m, from the
        # 0.125m-wide wheel boxes in jetracer.xacro) plus one control step's
        # worth of travel at max_speed*dt (1.0 * 0.1 = 0.1m) -- without that
        # second term, a trajectory's sampled states can land on either side
        # of a wall that's only ~1px wide without ever landing on the wall
        # pixel itself, since collision is only checked at each sampled state,
        # not swept continuously between them.
        self.local_inflation_radius = 5
        self.on_path = False

        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, scan_qos)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, 'ackermann_cmd', 10)
        self.debug_map_pub = self.create_publisher(OccupancyGrid, '/dwa_local_map', 10)  # add a Map display on this topic in RViz to see what clearance_cost sees
        self.goal_reached_pub = self.create_publisher(Bool, '/goal_reached', 10)  # goal_bridge listens for this to advance a survey course
        self.timer = self.create_timer(1.0 / 15.0, self.control_loop)

    def path_callback(self, msg):

        self.path = msg.poses
        self.goal_reached_published = False  # new goal in play -- allow /goal_reached to fire again
        self.get_logger().info(f"path_callback: received {len(self.path)} waypoints")


    def scan_callback(self, msg):
        """Turn the latest LaserScan into a small local occupancy grid (same
        1=occupied/0=free convention as self.world_map elsewhere), centered on
        the robot, for clearance_cost() to check trajectories against.

        Visibility-fan filled, not just point-marked: the grid starts entirely
        occupied (1) -- same "unknown = unsafe" convention a_star_planner
        already uses for the global map (raw_map == -1 -> 1). Each ray fills
        its own thin triangular wedge free (0), spanning exactly its own
        angular slice (angle +/- angle_increment/2) out to its own measured
        distance -- never connected to any other ray's endpoint. Adjacent
        rays' slices exactly abut, so coverage is still seamless (no 1px-line
        gaps), but two adjacent rays looking at different surfaces (a corner)
        just produce two wedges of very different length side by side --
        never a chord cutting through whatever's behind the corner. (An
        earlier version built one polygon across a whole run of consecutive
        valid rays, which was fast but chorded straight through corners; a
        range-jump threshold to break the run was tried and abandoned -- the
        jump at a real corner isn't reliably larger than the jump along an
        ordinary wall at a grazing angle, so no constant threshold separates
        them without misfiring on one or the other.)

        A reading above range_max (typically +inf, meaning "no obstacle
        within sensor range" per REP 117) is treated as confirmed clear to
        range_max, not unknown -- its wedge is filled (capped at range_max),
        just without marking a hit point. NaN or below range_min is a
        genuine invalid reading and contributes nothing.

        Known tradeoff, chosen deliberately: this sensor's simulated minRange
        (jetracer.usda) not matching this message's range_min means a target
        closer than the sensor's true minimum range comes back exactly as
        non-finite as a target beyond range_max -- the two are
        indistinguishable from /scan data alone, so treating overrange as
        confirmed-free can also fill in a "dome" over a wall that's actually
        just inside the sensor's blind zone. The minRange fix narrows that
        blind zone a lot (0.1m instead of 0.4m), but doesn't eliminate the
        ambiguity below the new minRange. Chosen anyway, over the more
        conservative "treat all non-finite as unknown" alternative, to avoid
        losing legitimate long-sightline free space DWA can otherwise plan
        through.
        """
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
        grid_origin_x = origin_x - msg.range_max
        grid_origin_y = origin_y - msg.range_max

        grid = np.ones((grid_size, grid_size), dtype=np.uint8)

        origin_px = int((origin_x - grid_origin_x) / resolution)
        origin_py = int((origin_y - grid_origin_y) / resolution)

        def to_grid_cell(angle, r):
            x_local = r * np.cos(angle)
            y_local = r * np.sin(angle)
            x_map = origin_x + x_local * np.cos(origin_yaw) - y_local * np.sin(origin_yaw)
            y_map = origin_y + x_local * np.sin(origin_yaw) + y_local * np.cos(origin_yaw)
            px = int((x_map - grid_origin_x) / resolution)
            py = int((y_map - grid_origin_y) / resolution)
            return px, py

        half_increment = msg.angle_increment / 2.0
        hit_points = []

        for i, r in enumerate(msg.ranges):
            # NaN / below range_min is a genuine invalid reading -- unknown,
            # contributes nothing.
            if np.isnan(r) or r < msg.range_min:
                continue

            is_hit = np.isfinite(r) and r <= msg.range_max
            r_eff = r if is_hit else msg.range_max

            angle = msg.angle_min + i * msg.angle_increment

            # This ray's own thin wedge -- origin plus the two edges of its
            # angular slice, both at r_eff. cv2.fillPoly clips to the image
            # automatically, so no bounds check is needed for the fill itself.
            p_left = to_grid_cell(angle - half_increment, r_eff)
            p_right = to_grid_cell(angle + half_increment, r_eff)
            polygon = np.array([[origin_px, origin_py], list(p_left), list(p_right)], dtype=np.int32)
            cv2.fillPoly(grid, [polygon.reshape((-1, 1, 2))], 0)

            if is_hit:
                px, py = to_grid_cell(angle, r_eff)
                if 0 <= px < grid_size and 0 <= py < grid_size:
                    hit_points.append((px, py))

        for px, py in hit_points:
            grid[py][px] = 1

        # Pad obstacles out so is_standable()/clearance_cost() see a wall with
        # real margin instead of a razor-thin, exactly-where-the-ray-landed
        # boundary. Exempt the robot's own current cell (like start/goal in
        # the global planner) so a robot already close to a wall doesn't read
        # its own position as a collision.
        origin_px_coords = PixelCoords(origin_px, origin_py)
        grid = inflated_obstacles(grid, self.local_inflation_radius, start=origin_px_coords)

        self.get_logger().info(f"origin cell ({origin_px},{origin_py}) = {grid[origin_py][origin_px]}")

        self.world_map = grid

        map_info = MapMetaData()
        map_info.resolution = resolution
        map_info.width = grid_size
        map_info.height = grid_size
        map_info.origin.position.x = grid_origin_x
        map_info.origin.position.y = grid_origin_y
        map_info.origin.orientation.w = 1.0
        self.map_info = map_info

        self.publish_debug_map(grid, map_info, msg.header.stamp)

    def publish_debug_map(self, grid, map_info, stamp):
        debug_msg = OccupancyGrid()
        debug_msg.header.frame_id = 'map'
        debug_msg.header.stamp = stamp
        debug_msg.info = map_info
        debug_msg.data = (grid * 100).astype(np.int8).flatten().tolist()
        self.debug_map_pub.publish(debug_msg)


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

        if best_dist <= self.check_on_path_dist:
            self.on_path = True
        else:
            self.on_path = False

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

    def check_goal_reached(self, state):
        """Publish True on /goal_reached, once, when the robot gets within
        self.goal_tolerance of the final point in self.path.

        TODO:
          1. last = self.path[-1].pose.position
          2. dist = math.hypot(state.x - last.x, state.y - last.y) -- plain
             float math, NOT PathNode/PixelCoords (those round to whole
             pixels/meters, see get_local_goal's own history of that bug).
          3. if dist <= self.goal_tolerance and not self.goal_reached_published:
                 self.goal_reached_pub.publish(Bool(data=True))
                 self.goal_reached_published = True
             (the self.goal_reached_published flag is what stops this from
             publishing every single cycle once the robot is parked at the
             goal -- it only resets to False in path_callback, when a new
             /plan arrives.)
        """
        last = self.path[-1].pose.position
        dist = math.hypot(state.x - last.x, state.y - last.y)
        if dist <= self.goal_tolerance and not self.goal_reached_published:
            self.goal_reached_pub.publish(Bool(data=True))
            self.goal_reached_published = True

    def control_loop(self):
        """Timer callback: run one DWA cycle and publish a command."""
        t0 = self.get_clock().now()

        if not self.path or self.world_map is None:
            self.get_logger().info(
                f"control_loop: waiting (path={'yes' if self.path else 'no'}, "
                f"world_map={'yes' if self.world_map is not None else 'no'})"
            )
            self.Ackermann_cmd_publisher(0.0, 0.0)
            return

        state = self.get_current_state()
        self.check_goal_reached(state)
        goal = self.get_local_goal()

        # TODO (lane-check): dwa.plan() needs some minimal signal for "is the
        # robot back on the lane" to know when a reverse maneuver has solved
        # the problem it started for. get_local_goal() above already walks
        # self.path with closest_point_on_segment() to find the nearest point
        # on the path to the robot -- what's the smallest additional piece of
        # data you could compute alongside/from that (without duplicating all
        # of get_local_goal's lookahead-walking logic) that plan() would
        # actually need? Where does that get computed, and how does it get
        # threaded into the plan() call below once dwa.py's signature grows
        # to accept it?
        v, steer, traj = self.dwa.plan(state, goal, self.on_path, self.world_map, self.map_info)

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