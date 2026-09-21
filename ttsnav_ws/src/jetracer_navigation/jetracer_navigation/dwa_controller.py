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
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
import tf2_ros
from tf_transformations import euler_from_quaternion 

from nav_msgs.msg import Path, OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool, ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from jetracer_navigation.dwa import DWA, DWAConfig
from jetracer_navigation.primitives import RobotState, PathNode, PixelCoords
from jetracer_navigation.utils import (
    euclidean_distance,
    inflated_obstacles,
    rollout_trajectory,
    build_distance_map,
    clearance_cost,
    closest_point_on_segment,
    bresenham,
    world_to_pixel,
)


class DWAController(Node):
    def __init__(self):
        super().__init__('dwa_controller')

        self.dwa = DWA(config=DWAConfig())

        # Diagnostic bypass: when True, control_loop skips dwa.plan() entirely
        # and just steers geometrically toward get_local_goal() (pure pursuit,
        # fixed speed, no obstacle awareness at all). Use this to check
        # whether /plan itself is a sane, drivable path -- independent of
        # whether DWA's search/cost logic is doing anything reasonable --
        # before spending more time debugging DWA specifically. Flip back to
        # False to restore normal obstacle-aware planning.
        self.dwa_enabled = True
        self.pure_pursuit_speed = 0.6  # m/s, fixed forward speed while bypassing DWA

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.path = None           # latest nav_msgs/Path from /plan
        self.world_map = None      # local obstacle representation built from /scan
        self.map_info = None
        self.lookahead_distance = 1.2  # meters; tune above the vehicle's min turning radius (~0.78m)
        self.last_v = 0.0
        self.last_steer = 0.0
        # meters; how close to the final path point counts as "reached".
        # Widened 0.3 -> 0.5 -> 1.0 -> 1.5: at 1.0 it was barely larger than
        # the car's own minimum turning radius (wheelbase / tan(max_steer)
        # = 0.32 / tan(0.322) ~= 0.96m), so depending on approach angle it
        # often couldn't curve tightly enough to actually enter the circle
        # in one pass -- it had to loop around and retry repeatedly. 1.5m
        # gives real margin over that radius.
        self.goal_tolerance = 0.5
        self.goal_reached_published = False  # edge-trigger: only publish once per path, not every cycle
        self.min_dist_to_goal = math.inf  # closest approach so far this path -- see check_goal_reached
        self.last_logged_mode = None  # edge-trigger for _log_mode: only print control_loop's mode on change
        self.was_on_obstacle_cell = False  # edge-trigger for the [ON_OBSTACLE] diagnostic, kept separate from last_logged_mode
        self.world_map_pose_x = None  # pose scan_callback was at when it last built self.world_map -- see control_loop's staleness diagnostic
        self.world_map_pose_y = None
        # Local-map obstacle inflation, in pixels at the local grid's 0.05m/px
        # resolution. Covers the robot's own half-width (~0.12m, half of the
        # Leatherback's real 0.24m trackWidth) plus one control step's worth
        # of travel -- without that second term, a trajectory's sampled
        # states can land on either side of a wall that's only ~1px wide
        # without ever landing on the wall pixel itself, since collision is
        # only checked at each sampled state, not swept continuously between
        # them. Deliberately sized off the ACHIEVABLE travel on an early,
        # accel-limited tick (~0.1m, not max_speed*dt's 0.3m) -- a cold-start
        # candidate never actually reaches max_speed*dt in one step, so
        # sizing this off max_speed swallowed the only cells reachable from
        # a stop and made the car unable to ever start moving.
        # (0.12 + 0.1 = 0.22m ~= 4.4px)
        self.local_inflation_radius = 5

        # Stuck-recovery: dwa.plan() is forward-only, so getting unstuck is
        # handled here instead, as a bounded recovery behavior (Nav2's
        # BackUp pattern) rather than folded into the planner's per-cycle
        # cost search -- see start_recovery()/run_recovery() below.
        self.failed_count = 0                # consecutive cycles with no valid forward command
        self.stuck_cycle_threshold = 5       # ~1/3s at 15Hz before treating it as genuinely stuck, not a one-frame blip
        self.recovery_state = False
        self.recovery_distance = 0.3         # meters to back away before declaring success
        self.recovery_time_cap = 5.0         # seconds; safety net if recovery_distance is unreachable (e.g. truly wedged)
        self.recovery_speed = 0.3            # m/s, fixed reverse speed used during recovery
        self.recovery_start_pos = None
        self.recovery_start_time = None
        self.recovery_v = 0.0                # chosen once in start_recovery() (or escalated in run_recovery()), held for the whole maneuver
        self.recovery_steer = 0.0            # chosen once in start_recovery(), held for the whole maneuver
        self.recovery_attempts = 0           # consecutive backup timeouts -- escalates to find_any_escape() after a couple
        self.min_scan_dist = None
        self.previous_world_map = None
        # grid_origin_x/y from the cycle that produced previous_world_map --
        # needed to shift the persisted grid by however many cells the
        # robot moved since then, since the grid always recenters on the
        # robot's CURRENT position each cycle. Without this, array index
        # (240,240) silently means "wherever the robot was last cycle"
        # instead of "wherever it is now," and persisted marks drift
        # relative to the robot every cycle they aren't freshly reobserved.
        self.previous_grid_origin_x = None
        self.previous_grid_origin_y = None

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
        self.candidates_pub = self.create_publisher(MarkerArray, '/dwa_candidates', 10)  # add a MarkerArray display on this topic to see every (v, steer) rollout DWA evaluated this tick
        self.goal_reached_pub = self.create_publisher(Bool, '/goal_reached', 10)  # goal_bridge listens for this to advance a survey course
        self.timer = self.create_timer(1.0 / 15.0, self.control_loop)

    def path_callback(self, msg):

        self.path = msg.poses
        self.goal_reached_published = False  # new goal in play -- allow /goal_reached to fire again
        self.min_dist_to_goal = math.inf  # reset closest-approach tracking for the new path
        self.get_logger().info(f"path_callback: received {len(self.path)} waypoints")

        # A fresh path's very first control_loop tick sees world_map built
        # from just one scan frame at the robot's fixed starting pose -- a
        # single frame's blind spots/near-range artifacts can make every
        # forward candidate look blocked even in open space, and nothing
        # fixes that until the robot actually moves and collects a couple
        # of fresh, different-angle scans. Force a short clearing backup
        # immediately rather than waiting stuck_cycle_threshold cycles for
        # normal stuck-detection to notice and trigger the same thing.
        if self.world_map is not None:
            try:
                self.start_recovery(self.get_current_state(), reason="fresh path, clearing map before planning")
            except Exception as exc:
                self.get_logger().warning(f"path_callback: couldn't start clearing backup, skipping: {exc}")


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
            # Look up the transform AT THE SCAN'S OWN TIMESTAMP, not "whatever's
            # freshest right now" (rclpy.time.Time()) -- using "latest" here means
            # every scan gets projected into the map frame using the pose the
            # robot happens to be at when this callback executes, not the pose it
            # was actually at when the LIDAR data was captured. That gap grows
            # with turning speed and executor/transport latency, and reprojects
            # every single scan at a slightly different (wrong) origin --
            # producing exactly a flickering, unstable local grid even in a
            # perfectly static, obstacle-fixed environment.
            # A short bounded wait is safe here now that main() runs a
            # MultiThreadedExecutor: the TF listener's own subscription
            # callback (the one that appends new data to tf_buffer) runs on
            # a different thread than this callback, so blocking here
            # doesn't starve it the way it would have on the old
            # single-threaded executor. Observed lag between a scan's own
            # timestamp and the newest available transform was a *constant*
            # ~17ms (one /tf publish tick), not occasional jitter -- 0.05s
            # comfortably covers that without risking a real multi-cycle
            # stall turning into a long block.
            t = self.tf_buffer.lookup_transform(
                'map', msg.header.frame_id, rclpy.time.Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=0.05),
            )
        except tf2_ros.TransformException as e:
            # Previously silent -- made this loud on purpose while chasing
            # why self.world_map stays empty even with /scan and TF both
            # confirmed healthy from the outside. If this fires every
            # cycle, the exception text below is the actual reason, not
            # speculation about it.
            self.get_logger().warning(
                f"scan_callback: TF lookup failed, skipping this scan: {e}",
                throttle_duration_sec=2.0,
            )
            return
        origin_x = t.transform.translation.x
        origin_y = t.transform.translation.y
        quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
        _, _, origin_yaw = euler_from_quaternion(quat)

        resolution = 0.05
        grid_size = int(2 * msg.range_max / resolution)
        grid_origin_x = origin_x - msg.range_max
        grid_origin_y = origin_y - msg.range_max
        # The exact pose this world_map build is anchored to -- control_loop
        # diffs this against its own, separately-TF-looked-up state each
        # tick, to check whether world_map is actually stale relative to
        # where the robot currently believes it is (scan_callback only
        # updates world_map when its own TF lookup succeeds, which TF
        # extrapolation failures skip roughly half the time).
        self.world_map_pose_x = origin_x
        self.world_map_pose_y = origin_y

        # TODO (costmap persistence + confidence smoothing): this grid is
        # rebuilt from all-occupied every single call, with zero memory of
        # the previous cycle -- so single-scan sensor noise (dropped/noisy
        # rays, the minRange/overrange ambiguity already noted above) and
        # any TF/localization jitter both show up directly as flicker,
        # which propagates into build_distance_map()/clearance_cost() in
        # dwa.py on every plan() call: a trajectory that scored fine last
        # cycle can suddenly read as a collision for no real reason.
        # Priority order to work through:
        #   1. Persistence: instead of starting from all-occupied each
        #      call, incrementally mark newly-observed occupied cells and
        #      clear cells each ray actually passed through, reusing
        #      self.world_map from the previous cycle as the starting
        #      point. bresenham() in utils.py returns every cell on a line
        #      and is already used for line-of-sight checks elsewhere
        #      (check_collision_free) -- would it fit the raytrace-clear
        #      step here, in place of (or alongside) the current
        #      cv2.fillPoly wedge fill? Worth benchmarking first: it's a
        #      plain Python per-cell loop, and a /scan can be 700+ rays up
        #      to range_max/resolution = 240 cells each -- cv2.fillPoly is
        #      a fast batch C call, bresenham() called per-ray per-scan is
        #      a very different cost profile. Does it fit the 15Hz control
        #      budget, or does the existing wedge-fill approach need to
        #      stay for the fill step with bresenham only handling
        #      something narrower?
        #   2. Confidence layer: a per-cell counter or log-odds float array
        #      (separate from the binary grid clearance_cost/is_standable
        #      actually consume), incrementing on a hit and decaying on a
        #      clear ray-pass, only promoting a cell to "occupied" in the
        #      binary grid once it crosses a threshold -- so one noisy
        #      frame can't flip a cell by itself. What threshold (N
        #      consecutive hits? a log-odds value?), and what's the right
        #      increment/decrement magnitude relative to how fast a real
        #      obstacle should be trusted?
        #   3. Decay: now that cells persist instead of resetting to
        #      all-occupied every call, a cell the robot moved away from
        #      (or an obstacle that moved) needs to actively clear over
        #      time rather than staying occupied forever. What's the right
        #      decay rate, and does it differ from the confidence-layer
        #      decrement in (2), or is it the same mechanism?
        #   4. Hard safety floor: keep a separate, always-instantaneous
        #      minimum-distance check straight from the raw current /scan
        #      (not smoothed, not confidence-gated) so improving map
        #      stability never trades away collision safety. Where does
        #      this hook in -- scan_callback publishing a
        #      self.min_scan_distance that control_loop checks directly
        #      alongside dwa.plan(), bypassing the smoothed grid entirely?
        #   5. Frame anchoring (only if 1-3 don't resolve the instability):
        #      this grid is currently rebuilt in the map frame every call,
        #      using a fresh tf_buffer lookup each time -- once it
        #      persists across cycles instead, any localization jitter (or
        #      a real AMCL correction) no longer just shows up as one
        #      frame's flicker, it *smears* previously-marked cells to the
        #      wrong place, since old cells don't get retroactively
        #      corrected when the pose estimate shifts. This is the same
        #      concern already flagged as "wait until AMCL/amcl_pose
        #      covariance is verified before doing this" -- anchoring the
        #      rolling grid in the odom frame instead (only using map
        #      frame for the DWA lookahead goal) is Nav2's own answer to
        #      exactly this problem. Don't reach for this until (1)-(3)
        #      are in and still don't resolve the instability.
        #
        # Acceptance check: with the robot and obstacles stationary,
        # consecutive /dwa_local_map frames (debug_map_pub below) shouldn't
        # flip individual cells between occupied/free frame to frame.
        # clearance_cost()/build_distance_map() in dwa.py/utils.py
        # shouldn't need to change at all -- they keep consuming a binary
        # grid, just a more stable one. Worth a small standalone
        # script/test subscribing to /dwa_local_map for N consecutive
        # frames while stationary and diffing them cell-by-cell before
        # trusting this by eye in RViz.
        
        if self.previous_world_map is None:
            grid = np.ones((grid_size, grid_size), dtype=np.uint8)
        else:
            # The grid recenters on the robot's CURRENT position every
            # cycle, so array index (240,240) always means "the robot,
            # right now" -- but self.previous_world_map's cells were
            # written relative to LAST cycle's origin. Shift the persisted
            # array by however many cells the origin moved, so a
            # stationary world point lands at the same array index it
            # would if it had just been freshly observed, instead of
            # silently dragging along with wherever the robot used to be.
            dx_px = int(round((grid_origin_x - self.previous_grid_origin_x) / resolution))
            dy_px = int(round((grid_origin_y - self.previous_grid_origin_y) / resolution))

            grid = np.ones((grid_size, grid_size), dtype=np.uint8)

            # new_grid[:, dst] = previous_world_map[:, dst + dx_px] (same
            # idea for rows/dy_px) -- derived from: a world point at old
            # column px_old is now at new column px_old - dx_px, since the
            # origin itself moved dx_px cells in the same direction.
            # Whatever doesn't overlap (newly-revealed edge, or the robot
            # having moved more than a full grid-width since last cycle)
            # is left at the blank/unknown default above.
            dst_x_start, dst_x_end = max(0, -dx_px), min(grid_size, grid_size - dx_px)
            dst_y_start, dst_y_end = max(0, -dy_px), min(grid_size, grid_size - dy_px)
            src_x_start, src_x_end = dst_x_start + dx_px, dst_x_end + dx_px
            src_y_start, src_y_end = dst_y_start + dy_px, dst_y_end + dy_px

            if src_x_end > src_x_start and src_y_end > src_y_start:
                grid[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                    self.previous_world_map[src_y_start:src_y_end, src_x_start:src_x_end]

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
        min_dist = math.inf

        for i, r in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment

            # NaN / below range_min is not a trustworthy obstacle reading,
            # but it's also not "occupied" -- the grid starts entirely
            # occupied (np.ones above) and only gets cleared along rays that
            # reach here, so simply skipping these left a permanent ring of
            # phantom obstacles at exactly this range around the robot on
            # every cycle, since a real lidar reliably returns sub-range_min
            # readings for empty space immediately around its own mount.
            # Clear this ray out to range_min as free instead -- the actual
            # close-range safety net is min_scan_dist's hard stop in
            # control_loop, not this costmap.
            if np.isnan(r) or r < msg.range_min:
                hitx, hity = to_grid_cell(angle, msg.range_min)
                for x, y in bresenham(origin_px, hitx, origin_py, hity):
                    if 0 <= x < grid_size and 0 <= y < grid_size:
                        grid[y][x] = 0
                continue

            is_hit = np.isfinite(r) and r <= msg.range_max
            r_eff = r if is_hit else msg.range_max

            if np.isnan(r) or r < min_dist:
                min_dist = r

            # This ray's own thin wedge -- origin plus the two edges of its
            # angular slice, both at r_eff. cv2.fillPoly clips to the image
            # automatically, so no bounds check is needed for the fill itself.

            # Original Version
            # p_left = to_grid_cell(angle - half_increment, r_eff)
            # p_right = to_grid_cell(angle + half_increment, r_eff)
            # polygon = np.array([[origin_px, origin_py], list(p_left), list(p_right)], dtype=np.int32)
            # cv2.fillPoly(grid, [polygon.reshape((-1, 1, 2))], 0)

            # Revised
            hitx, hity = to_grid_cell(angle, r_eff)
            path = bresenham(origin_px, hitx, origin_py, hity)
            for x, y in path:
                # to_grid_cell() can land exactly on grid_size at max range
                # along a cardinal direction (e.g. r_eff=range_max puts
                # px/py at 480 on a 480-wide grid) -- cv2.fillPoly used to
                # clip this for free, a raw numpy index does not, so this
                # needs its own bounds check the way the hit_points
                # collection below already does. Also note grid[y][x], not
                # grid[x][y] -- same row/column convention as everywhere
                # else in this file (see line ~332's grid[py][px] = 1).
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    grid[y][x] = 0

            if is_hit:
                px, py = to_grid_cell(angle, r_eff)
                if 0 <= px < grid_size and 0 <= py < grid_size:
                    hit_points.append((px, py))

        for px, py in hit_points:
            grid[py][px] = 1

        self.min_scan_dist = min_dist

        # Save the RAW (pre-inflation) grid as next cycle's persistence
        # starting point, before inflated_obstacles() pads it below. If the
        # already-inflated grid were persisted instead, next cycle would
        # inflate it *again* on top of the existing padding, and the padding
        # around every obstacle would keep growing every single cycle
        # instead of staying constant -- inflation has to be re-derived
        # fresh from the raw observations each time, not accumulated.
        self.previous_world_map = grid.copy()
        # Origin this grid was built at -- next cycle's shift math needs
        # both halves of the delta (its own new origin, and this one).
        self.previous_grid_origin_x = grid_origin_x
        self.previous_grid_origin_y = grid_origin_y

        # Pad obstacles out so is_standable()/clearance_cost() see a wall with
        # real margin instead of a razor-thin, exactly-where-the-ray-landed
        # boundary. Exempt the robot's own current cell (like start/goal in
        # the global planner) so a robot already close to a wall doesn't read
        # its own position as a collision.
        origin_px_coords = PixelCoords(origin_px, origin_py)
        grid = inflated_obstacles(grid, self.local_inflation_radius, start=origin_px_coords)

        self.get_logger().debug(f"origin cell ({origin_px},{origin_py}) = {grid[origin_py][origin_px]}")

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
        

    def get_local_goal(self) -> tuple:
        state = self.get_current_state()
        p = (state.x, state.y)

        best_dist = np.inf
        best_i = None
        best_point = None
        for i in range(len(self.path) - 1):
            a = (self.path[i].pose.position.x, self.path[i].pose.position.y)
            b = (self.path[i + 1].pose.position.x, self.path[i + 1].pose.position.y)
            point, dist = closest_point_on_segment(p, a, b)
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

    def check_goal_reached(self, state):
        """Publish True on /goal_reached, once, when the robot is
        (a) currently within self.goal_tolerance of the final point, or
        (b) has already gotten that close at some point and is now moving
        away from its closest approach -- a non-holonomic car isn't
        guaranteed to be able to curve tightly enough to sit inside
        goal_tolerance from every approach angle (its minimum turning
        radius can exceed goal_tolerance), so requiring a literal re-entry
        can make it loop past the goal repeatedly instead of accepting
        "that was as close as it's going to get." self.min_dist_to_goal is
        reset to inf in path_callback whenever a fresh /plan arrives.
        """
        if self.goal_reached_published:
            return
        last = self.path[-1].pose.position
        dist = math.hypot(state.x - last.x, state.y - last.y)
        self.min_dist_to_goal = min(self.min_dist_to_goal, dist)

        directly_close = dist <= self.goal_tolerance
        passed_closest_approach = (
            self.min_dist_to_goal <= self.goal_tolerance and dist > self.min_dist_to_goal + 0.05
        )
        if directly_close or passed_closest_approach:
            self.goal_reached_pub.publish(Bool(data=True))
            self.goal_reached_published = True

    def _log_mode(self, mode, message, level='info'):
        """Print one line only when `mode` differs from the last call --
        control_loop runs every tick, so logging on every call regardless of
        outcome buries the moments that actually matter (a mode changing)
        under a constant stream of "still doing the same thing" lines."""
        if mode != self.last_logged_mode:
            # Deliberately two separate calls, not one aliased through a
            # variable -- rclpy's logger tracks per-call-site state by
            # source line, and refuses (raises ValueError) if the same line
            # is ever used at two different severities across calls.
            if level == 'warning':
                self.get_logger().warning(f"[{mode}] {message}")
            else:
                self.get_logger().info(f"[{mode}] {message}")
            self.last_logged_mode = mode

    def control_loop(self):
        """Timer callback: run one DWA cycle and publish a command."""
        if not self.path or self.world_map is None:
            self._log_mode(
                "WAIT",
                f"no path/map yet (path={'yes' if self.path else 'no'}, "
                f"world_map={'yes' if self.world_map is not None else 'no'})"
            )
            self.Ackermann_cmd_publisher(0.0, 0.0)
            return

        state = self.get_current_state()
        self.check_goal_reached(state)

        if self.recovery_state:
            self.run_recovery(state)
            return

        goal = self.get_local_goal()

        if not self.dwa_enabled:
            final = self.path[-1].pose.position
            dist_to_final = math.hypot(state.x - final.x, state.y - final.y)
            if dist_to_final <= self.goal_tolerance:
                self._log_mode("STOPPED", f"reached goal (dist={dist_to_final:.2f}m)")
                self.failed_count = 0
                self.Ackermann_cmd_publisher(0.0, 0.0)
                return

            # Pure pursuit has zero obstacle awareness of its own -- it just
            # geometrically chases get_local_goal() along /plan, so nothing
            # above this stops it from driving straight into a wall that
            # isn't (accurately) reflected in the global map. Piggyback on
            # the same live /scan-derived min_scan_dist safety floor the DWA
            # branch below already gates on, so bypassing DWA only removes
            # local trajectory *planning*, not this hard stop.
            if self.min_scan_dist is not None and self.min_scan_dist < 0.3:
                self._log_mode("STOPPED", f"obstacle at {self.min_scan_dist:.2f}m", level='warning')
                self.Ackermann_cmd_publisher(0.0, 0.0)
                return

            v, steer = self.pure_pursuit_speed, self.pure_pursuit_steer(state, goal)
            self._log_mode(
                "FOLLOW",
                f"pure pursuit -- state=({state.x:.2f},{state.y:.2f},{state.theta:.2f}) goal={goal}"
            )
            self.failed_count = 0
            self.Ackermann_cmd_publisher(v, steer)
            return

        # Diagnostic: is the robot's own current cell reading as occupied in
        # world_map? inflated_obstacles() is supposed to exempt this exact
        # cell (see scan_callback's "start=origin_px_coords" comment) so a
        # robot near a wall never reads its own position as a collision --
        # this checks live whether that exemption is actually holding, since
        # if it isn't, every forward candidate collides on its very first
        # sampled state and only v=0 ever looks valid.
        origin_cell = world_to_pixel(state.x, state.y, self.map_info)
        opx, opy = origin_cell.x, origin_cell.y
        on_obstacle = (
            0 <= opy < self.world_map.shape[0] and 0 <= opx < self.world_map.shape[1]
            and self.world_map[opy][opx] == 1
        )
        if on_obstacle != self.was_on_obstacle_cell:
            if on_obstacle:
                self.get_logger().warning(
                    f"[ON_OBSTACLE] robot's own cell ({opx},{opy}) reads OCCUPIED in "
                    f"world_map -- every forward DWA candidate will collide immediately "
                    f"until this clears, which is why the car won't move"
                )
            else:
                self.get_logger().info(f"[ON_OBSTACLE] cleared -- robot's own cell ({opx},{opy}) is free again")
            self.was_on_obstacle_cell = on_obstacle

        # Taper velocity_cost's "ideal" speed proportionally to euclidean
        # distance from the FINAL goal (not the local lookahead goal, which
        # stays a fixed lookahead_distance away right up until the very
        # end), once within slowdown_distance -- a single-step drop (e.g.
        # straight to 1.0 m/s) still holds that speed all the way to
        # goal_tolerance and overshoots; a proportional (P) taper keeps
        # slowing continuously as distance shrinks, like the P term of a
        # PID loop with dist_to_final as the error. Floored at
        # min_approach_speed so it keeps creeping in rather than
        # stalling before actually reaching goal_tolerance.
        final_point = self.path[-1].pose.position
        dist_to_final = math.hypot(state.x - final_point.x, state.y - final_point.y)

        # check_goal_reached() only publishes a notification for
        # goal_bridge (course advancement) -- it never commands a stop.
        # The pure-pursuit branch above has its own explicit "close enough,
        # publish zero speed" check; DWA never got the equivalent, so
        # regardless of goal_tolerance the car just kept crawling forward
        # forever at at least min_approach_speed once near the goal. Stop
        # outright once actually reached, before ever calling dwa.plan().
        if self.goal_reached_published:
            self._log_mode("STOPPED", f"goal reached (dist={dist_to_final:.2f}m)")
            self.failed_count = 0
            self.Ackermann_cmd_publisher(0.0, 0.0)
            return

        slowdown_distance = 1.5  # must stay > goal_tolerance -- otherwise "reached" can fire before deceleration even starts
        min_approach_speed = 0.2
        if dist_to_final < slowdown_distance:
            target_speed = max(min_approach_speed, self.dwa.config.max_speed * (dist_to_final / slowdown_distance))
        else:
            target_speed = self.dwa.config.max_speed

        path_points = [(p.pose.position.x, p.pose.position.y) for p in self.path]
        v, steer, traj, candidates = self.dwa.plan(
            state, goal, self.world_map, self.map_info, path_points, target_speed=target_speed
        )
        self.publish_candidates(candidates, v, steer)

        # Direct breakdown instead of guessing: are v>0 candidates even being
        # generated (window too narrow), colliding (map/inflation issue), or
        # valid-but-losing on cost (a scoring bug, not a collision one)?
        num_total = len(candidates)
        num_positive_v = sum(1 for c in candidates if c[0] > 0.0)
        num_valid_positive_v = sum(1 for c in candidates if c[0] > 0.0 and c[3])
        self.get_logger().info(
            f"[candidates] total={num_total} with_v>0={num_positive_v} valid_and_v>0={num_valid_positive_v}",
            throttle_duration_sec=1.0,
        )

        # v=0 is *always* a valid candidate whenever the robot's own cell is
        # free (standing still can't collide with anything it isn't already
        # overlapping), so "v is None" alone can't detect the common real
        # stuck case: every v>0 candidate this cycle was a genuine collision
        # (num_valid_positive_v == 0, not merely outscored on cost -- that
        # would still leave some v>0 candidate marked valid), and v=0 wins
        # by default. Without this, the car can sit pinned at (v=0,
        # steer=min_steer) indefinitely -- start_recovery() never fires,
        # since it's never technically "zero candidates at all". Gated on
        # num_positive_v > 0 (at least one v>0 candidate was actually
        # sampled and rejected this cycle) so a cold-start/near-goal window
        # that simply hasn't opened up to include any v>0 candidates yet --
        # a legitimate, non-stuck situation -- doesn't trip this.
        blocked = v is not None and num_positive_v > 0 and num_valid_positive_v == 0

        if v is None or blocked:
            # "Stuck" here means either zero candidates survived at all
            # (v is None), or v=0 only "won" because every v>0 candidate
            # collided (blocked) -- not "v stayed near 0 for a while," which
            # is often completely legitimate (slowing near a goal) and was
            # exactly the ambiguity that caused most of the trouble with the
            # old reverse-hysteresis approach.
            self.failed_count += 1
            reason = (
                "no valid trajectory" if v is None
                else f"every v>0 candidate collided ({num_positive_v} sampled)"
            )
            self._log_mode(
                "STUCK",
                f"{reason} ({self.failed_count}/{self.stuck_cycle_threshold} consecutive)",
                level='warning',
            )
            self.Ackermann_cmd_publisher(0.0, 0.0)
            if self.failed_count > self.stuck_cycle_threshold:
                self.start_recovery(state)
            return

        self._log_mode(
            "FOLLOW",
            f"state=({state.x:.2f},{state.y:.2f},{state.theta:.2f}) goal={goal} v={v} steer={steer}"
        )
        # Throttled, not edge-triggered -- unlike _log_mode above, this prints
        # on a fixed cadence regardless of whether the mode name changed, so
        # v/steer staying flat (or ramping) across many ticks in the same
        # FOLLOW mode is actually visible instead of hidden after the first.
        map_drift_x = state.x - self.world_map_pose_x if self.world_map_pose_x is not None else None
        map_drift_y = state.y - self.world_map_pose_y if self.world_map_pose_y is not None else None
        self.get_logger().info(
            f"[v] state=({state.x:.2f},{state.y:.2f},{state.theta:.2f}) v={v} steer={steer} last_v={self.last_v} "
            f"map_drift=({map_drift_x},{map_drift_y})",
            throttle_duration_sec=1.0,
        )

        if self.min_scan_dist is not None and self.min_scan_dist >= 0.3:
            self.failed_count = 0
            self.Ackermann_cmd_publisher(v, steer)
        else:
            self._log_mode(
                "STOPPED",
                f"obstacle at {self.min_scan_dist}m -- DWA picked v={v} steer={steer} but obstacle gate zeroed it",
                level='warning',
            )
            self.Ackermann_cmd_publisher(0.0, 0.0)
            self.failed_count += 1
            # Without this, the obstacle gate can zero a perfectly valid
            # DWA pick forever -- v isn't None and no v>0 candidate
            # collided, so the "blocked" stuck-detection above never sees
            # this as stuck at all, and it can oscillate FOLLOW/STOPPED
            # indefinitely without ever triggering recovery.
            if self.failed_count > self.stuck_cycle_threshold:
                self.start_recovery(state)

    def publish_candidates(self, candidates, chosen_v, chosen_s):
        """Visualize every (v, steer) rollout DWA evaluated this tick as a
        MarkerArray -- one LINE_STRIP per candidate, colored red (collided),
        gray (valid but not picked), or green (the one actually chosen).
        Add a Marker display on /dwa_candidates in RViz to see this."""
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        clear = Marker()
        clear.header.frame_id = 'map'
        clear.header.stamp = now
        clear.ns = 'dwa_candidates'
        clear.id = -1  # distinct from every candidate's id (0..N-1) so RViz doesn't see a duplicate (ns, id) pair
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for i, (v, s, traj, valid) in enumerate(candidates):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = now
            marker.ns = 'dwa_candidates'
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.points = [Point(x=st.x, y=st.y, z=0.0) for st in traj.states]

            if valid and v == chosen_v and s == chosen_s:
                marker.scale.x = 0.05
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            elif valid:
                marker.scale.x = 0.02
                marker.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.4)
            else:
                marker.scale.x = 0.02
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.3)

            marker_array.markers.append(marker)

        self.candidates_pub.publish(marker_array)

    def pure_pursuit_steer(self, state, goal) -> float:
        """Standard pure-pursuit steering angle toward `goal` (already at
        ~lookahead_distance away, from get_local_goal()), ignoring obstacles
        entirely -- only used while self.dwa_enabled is False, to sanity-check
        that /plan is a drivable path on its own."""
        dx = goal[0] - state.x
        dy = goal[1] - state.y

        # goal in the robot's own frame (x forward, y left)
        lx = dx * math.cos(state.theta) + dy * math.sin(state.theta)
        ly = -dx * math.sin(state.theta) + dy * math.cos(state.theta)

        L = math.hypot(lx, ly)
        if L < 1e-3:
            return 0.0

        curvature = 2.0 * ly / (L ** 2)
        steer = math.atan(self.dwa.config.wheelbase * curvature)
        return max(self.dwa.config.min_steer, min(self.dwa.config.max_steer, steer))

    def start_recovery(self, state, reason=None):
        """Begin a bounded backup recovery: pick a steering direction once
        (whichever of left/straight/right has the most clearance right now),
        then back away in that fixed direction until either
        recovery_distance is covered or recovery_time_cap is hit."""
        self.recovery_v = -self.recovery_speed
        self.recovery_steer = self.choose_recovery_steer(state)
        self.recovery_start_pos = (state.x, state.y)
        self.recovery_start_time = self.get_clock().now()
        self.recovery_state = True
        self.failed_count = 0
        reason = reason or f"stuck for {self.stuck_cycle_threshold}+ cycles"
        self.get_logger().warning(
            f"[RECOVER] {reason}, starting recovery backup (steer={self.recovery_steer:.3f})"
        )
        self.last_logged_mode = "RECOVER"

    def choose_recovery_steer(self, state):
        """Pick the steering angle for the recovery backup, once, by rolling
        out a short reverse trajectory at min/0/max steer and comparing
        clearance -- the same rollout_trajectory/clearance_cost machinery
        dwa.py's normal planning already uses, just a one-shot decision here
        instead of a per-cycle search."""
        # TODO (coarse candidate set): only three discrete angles get
        # considered here -- full-left, straight, full-right -- unlike
        # normal forward planning in dwa.py, which sweeps the whole steer
        # range at steer_resolution. bicycle_step already handles negative
        # v correctly (the turn direction physically flips vs. forward at
        # the same steer angle, which is correct Ackermann behavior, not a
        # bug), so the math this runs on is sound -- the open question is
        # just whether 3 buckets is coarse enough to sometimes pick a worse
        # escape angle than a finer sweep would. Worth it for a one-shot
        # decision, given the extra rollout/cost-eval calls are cheap
        # relative to plan()'s full per-cycle search? Or does the coarse
        # choice not actually matter much in practice, since this only
        # needs to clear the immediate obstacle, not find an optimal path?
        distance_map = build_distance_map(self.world_map)
        candidates = [self.dwa.config.min_steer, 0.0, self.dwa.config.max_steer]

        best_steer = 0.0
        best_clearance = -math.inf
        for steer in candidates:
            traj = rollout_trajectory(
                state, -self.recovery_speed, steer,
                self.dwa.config.predict_time, self.dwa.config.dt, self.dwa.config.wheelbase,
            )
            cost = clearance_cost(traj, self.world_map, distance_map, self.map_info)
            if cost == math.inf:
                continue
            clearance = -cost  # clearance_cost = 10 - dist_m, so a lower cost means more clearance
            if clearance > best_clearance:
                best_clearance = clearance
                best_steer = steer
        return best_steer

    def run_recovery(self, state):
        """Execute one cycle of an in-progress recovery backup: keep
        reversing at the fixed speed/steering chosen in start_recovery()
        until the distance target is reached (success) or the time cap is
        hit (give up this attempt)."""
        traveled = math.hypot(state.x - self.recovery_start_pos[0], state.y - self.recovery_start_pos[1])
        elapsed = (self.get_clock().now() - self.recovery_start_time).nanoseconds / 1e9

        if traveled >= self.recovery_distance:
            self.get_logger().info(f"[RECOVER] complete, travelled {traveled:.2f}m")
            self.recovery_state = False
            self.recovery_attempts = 0
            return

        if elapsed >= self.recovery_time_cap:
            # Straight backup didn't clear it -- rather than give up and let
            # normal planning immediately re-detect "stuck" and retry the
            # exact same backup direction again, escalate after a couple of
            # failed attempts: search every direction (forward and reverse,
            # full steer range, no path/heading preference at all -- just
            # "is this collision-free") and commit to whichever has the most
            # clearance. This is deliberately not a good path back onto
            # /plan -- it's a last resort to stop being fully stuck, on the
            # assumption that once it's moved at all, normal DWA planning
            # gets another real shot at finding its way back.
            self.recovery_attempts += 1
            if self.recovery_attempts >= 2:
                escape_v, escape_s = self.find_any_escape(state)
                if escape_v is None:
                    self.get_logger().error(
                        f"[RECOVER] no escape direction found at all after "
                        f"{self.recovery_attempts} failed attempts -- genuinely boxed in, giving up"
                    )
                    self.recovery_state = False
                    self.recovery_attempts = 0
                    return
                self.get_logger().warning(
                    f"[RECOVER] backup failed {self.recovery_attempts} times -- "
                    f"escalating to best-available direction (v={escape_v:.2f}, steer={escape_s:.3f})"
                )
                self.recovery_v = escape_v
                self.recovery_steer = escape_s
                self.recovery_start_pos = (state.x, state.y)
                self.recovery_start_time = self.get_clock().now()
                self.recovery_attempts = 0
                return
            self.get_logger().warning(
                f"[RECOVER] timed out after {elapsed:.1f}s, only "
                f"travelled {traveled:.2f}m of {self.recovery_distance}m -- retrying backup"
            )
            self.recovery_state = False
            return

        self.Ackermann_cmd_publisher(self.recovery_v, self.recovery_steer)

    def find_any_escape(self, state):
        """Last resort after bounded backup recovery has failed repeatedly:
        search forward AND reverse across the full steer range for any
        collision-free short rollout, ignoring path/heading cost entirely,
        and return whichever has the most clearance. Returns (None, None)
        if truly nothing around the robot is collision-free at all."""
        distance_map = build_distance_map(self.world_map)
        best_v = None
        best_s = None
        best_clearance = -math.inf
        for v in (self.recovery_speed, -self.recovery_speed):
            for s in np.arange(self.dwa.config.min_steer, self.dwa.config.max_steer, self.dwa.config.steer_resolution):
                traj = rollout_trajectory(
                    state, v, s, self.dwa.config.predict_time, self.dwa.config.dt, self.dwa.config.wheelbase,
                )
                cost = clearance_cost(traj, self.world_map, distance_map, self.map_info)
                if cost == math.inf:
                    continue
                clearance = -cost
                if clearance > best_clearance:
                    best_clearance = clearance
                    best_v = v
                    best_s = s
        return best_v, best_s

    def Ackermann_cmd_publisher(self, speed:float, steer:float):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        # DWA's internal bicycle model and this robot's real Ackermann interface
        # both use the standard convention (positive speed = forward) — see
        # teleop_keyboard.py: 'w' calls increase_speed(), which makes self.speed
        # more positive and publishes it unmodified. No sign flip needed here.
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.cmd_pub.publish(msg)
        self.last_v = float(speed)
        self.last_steer = float(steer)


def main(args=None):
    rclpy.init(args=args)
    node = DWAController()
    # MultiThreadedExecutor, not the default single-threaded one -- lets
    # scan_callback's bounded lookup_transform() wait (above) actually work:
    # the TF listener's own subscription callback needs to run concurrently
    # to deliver the transform being waited on, which a single-threaded
    # executor can't do while stuck inside that same wait.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()