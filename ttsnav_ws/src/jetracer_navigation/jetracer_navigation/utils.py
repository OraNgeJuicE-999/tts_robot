"""Helper functions: map checking, coordinate conversion, distances, path
post-processing (shortcutting/smoothing), and visualization."""

from typing import List, Optional

import cv2
import numpy as np
import math

from numpy.typing import NDArray
from jetracer_navigation.primitives import PixelCoords, PathNode, RobotState, Trajectory

# Map checking functions ======================================================


def bresenham(x0: int, x1: int, y0: int, y1: int) -> list[tuple[int, int]]:
    """Return every grid cell on the line from (x0, y0) to (x1, y1).

    TODO: implement Bresenham's line algorithm (integer-only line rasterization).
    Used by check_collision_free to test line-of-sight between two cells.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    m = dx if dx > dy else dy
    sx = 1 if x0 <= x1 else -1
    sy = 1 if y0 <= y1 else -1
    result = []

    err = m / 2
    x = x0
    y = y0
    for n in range(m):
        if m == dx:
            err -= dy
            x += sx
            if err < 0:
                y += sy
                err += dx
        else:
            err -= dx
            y += sy
            if err < 0:
                x += sx
                err += dy
            
        result.append((x, y))
        
    return result


def is_in_bounds(coords: PixelCoords, occupancy_grid: NDArray) -> bool:

    h = occupancy_grid.shape[0]
    w = occupancy_grid.shape[1]

    if coords.x_ >= w or coords.y_ >= h:
        return False
    elif coords.x_ < 0 or coords.y_ < 0:
        return False
    return True


def is_standable(coords: PixelCoords, occupancy_grid) -> bool:
    if occupancy_grid[coords.y_][coords.x_] == 1:
        return False
    return True


def check_collision_free(
    occupancy_map: NDArray,
    source_node: PathNode,
    target_node: PathNode,
) -> bool:
    
    if is_in_bounds(source_node.coords, occupancy_map) and is_in_bounds(target_node.coords, occupancy_map):
        path = bresenham(source_node.coords.x_, target_node.coords.x_, source_node.coords.y_, target_node.coords.y_)

        for coord in path:
            if not is_standable(PixelCoords(coord[0], coord[1]), occupancy_map):
                return False

        return True
    else:
        raise RuntimeError()


def get_8_neighbors(
    coords: PixelCoords,
    occupancy_grid,
    node_registry: dict[tuple[int, int], PathNode],
) -> List[PathNode]:
    
    DIRECTION = {
        (0, 1), (1, 0), (1, 1), (-1, 0), (0, -1), (-1, -1), (1, -1), (-1, 1)
    }
    result = []
    for idx in DIRECTION:
        xx = coords.x_ + idx[0]
        yy = coords.y_ + idx[1]
        if is_in_bounds(PixelCoords(xx, yy), occupancy_grid) and is_standable(PixelCoords(xx, yy), occupancy_grid) and check_collision_free(occupancy_grid, PathNode(coords), PathNode(PixelCoords(xx, yy))):
            if node_registry.get((xx, yy)) is None:
                node_registry[(xx, yy)] = PathNode(coords=PixelCoords(xx, yy))
            result.append(node_registry[(xx, yy)])

    return result
# ======================================================

# Coordinate conversion ======================================================


def world_to_pixel(wx: float, wy: float, map_info) -> PixelCoords:
    
    dx = wx - map_info.origin.position.x
    dy = wy - map_info.origin.position.y

    xx = dx / map_info.resolution
    yy = dy / map_info.resolution

    return PixelCoords(xx, yy)


def pixel_to_world(px: int, py: int, map_info) -> tuple[float, float]:

    xx = map_info.origin.position.x + px * map_info.resolution
    yy = map_info.origin.position.y + py * map_info.resolution

    return (xx, yy)

# ======================================================

# Distance calculation ======================================================


def manhattan_distance(start: PathNode, goal: PathNode) -> float:
    dx = abs(start.coords.x - goal.coords.x)
    dy = abs(start.coords.y - goal.coords.y)
    return dx + dy


def euclidean_distance(start: PathNode, goal: PathNode) -> float:
    dx = start.coords.x - goal.coords.x
    dy = start.coords.y - goal.coords.y
    return (dx**2 + dy**2) ** 0.5


# ======================================================

# World transform ======================================================


def inflated_obstacles(
    occupancy_map: NDArray,
    inflation_radius: int,
    start: Optional[PixelCoords] = None,
    goal: Optional[PixelCoords] = None,
) -> NDArray:
    """Expand obstacles outward so the planner keeps the (physically-sized) robot
    a safe distance from walls/obstacles.

    TODO:
      - build a square structuring element sized from inflation_radius
      - cv2.dilate the occupancy_map with it
      - make sure start/goal cells stay clear (0) even if inflation covered them
    """
    size = 2 * inflation_radius - 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    new_map = cv2.dilate(occupancy_map.astype(np.uint8), kernel)

    if start is not None:
        if is_in_bounds(start, new_map): 
            if not is_standable(start, new_map):
                new_map[start.y_][start.x_] = 0

    if goal is not None:
        if is_in_bounds(goal, new_map):
            if not is_standable(goal, new_map):
                new_map[goal.y_][goal.x_] = 0

    return new_map

def world_2_occupancy_map(
    world_map: NDArray,
    start: PixelCoords,
    goal: PixelCoords,
    inflation_radius: int = 5,
):
    inflated_map = inflated_obstacles(world_map, inflation_radius, start, goal)
    return world_map, inflated_map

# ======================================================

# Path generation ======================================================


def reconstruct_path(goal_node: PathNode) -> List[PathNode]:
    """Walk parent pointers from goal_node back to the start, then reverse."""
    path_list = []
    current_node = goal_node
    while current_node:
        path_list.append(current_node)
        current_node = current_node.parent
    return path_list[::-1]


def shortcut_path(path: List[PathNode], occupancy_map: NDArray) -> List[PathNode]:
    """Greedy line-of-sight shortcutting: remove nodes bypassed by a clear straight line."""
    if len(path) < 3:
        return path
    result = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if check_collision_free(occupancy_map, path[i], path[j]):
                break
            j -= 1
        result.append(path[j])
        i = j
    return result


def is_path_collision_free(path: List[PathNode], occupancy_map: NDArray) -> bool:
    """Check that every consecutive pair of points in path has a clear line of sight."""
    for i in range(len(path) - 1):
        if not check_collision_free(occupancy_map, path[i], path[i + 1]):
            return False
    return True


def natural_cubic_spline(t, value):
    """Fit a natural cubic spline through (t, value) control points.

    Returns (a, b, c, d): per-segment coefficients such that on segment i,
    S(x) = a[i] + b[i]*(x - t[i]) + c[i]*(x - t[i])**2 + d[i]*(x - t[i])**3.
    """
    if len(t) != len(value):
        raise ValueError("t and value must have the same length")
    if len(t) < 2:
        raise ValueError("Need at least two points")
    if any(t[i + 1] <= t[i] for i in range(len(t) - 1)):
        raise ValueError("t must be strictly increasing")

    n = len(t) - 1

    h = [0.0] * n
    alpha = [0.0] * (n + 1)
    l = [0.0] * (n + 1)
    mu = [0.0] * (n + 1)
    z = [0.0] * (n + 1)
    a = value[:]              # a[i] = value[i]
    b = [0.0] * n
    c = [0.0] * (n + 1)
    d = [0.0] * n

    # step 1: compute h[i]
    for i in range(n):
        h[i] = t[i + 1] - t[i]

    # step 2: compute alpha[i]
    for i in range(1, n):
        alpha[i] = 3 * ((a[i + 1] - a[i]) / h[i]) - 3 * ((a[i] - a[i - 1]) / h[i - 1])

    # step 3: forward pass, l, mu, z arrays
    l[0] = 1.0
    mu[0] = 0.0
    z[0] = 0.0

    for i in range(1, n):
        l[i] = 2 * (t[i + 1] - t[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    # ending
    l[n] = 1.0
    z[n] = 0.0
    c[n] = 0.0

    # step 4: backward pass — compute b, c, d
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    return a, b, c, d


def compute_turning_angle(prev: PixelCoords, curr: PixelCoords, next: PixelCoords):
    """Return the angle (degrees) between the incoming and outgoing path segments at curr."""
    v1 = np.array([(curr.x - prev.x), (curr.y - prev.y)])
    v2 = np.array([(next.x - curr.x), (next.y - curr.y)])

    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)

    if l1 == 0 or l2 == 0:
        return 0.0

    dot = np.dot(v1, v2)
    angle = np.degrees(np.arccos(np.clip(dot / (l1 * l2), -1.0, 1.0)))

    return angle


def turn_region(path_coords: List[PathNode], threshold):
    """Return the set of path indices where the turning angle exceeds threshold."""
    turn_indices = set()

    for i, _ in enumerate(path_coords):
        if i == 0 or i == len(path_coords) - 1:
            continue

        angle = compute_turning_angle(
            path_coords[i - 1].coords,
            path_coords[i].coords,
            path_coords[i + 1].coords,
        )

        # Straight ~0, Turn 1
        if angle > threshold:
            turn_indices.add(i)

    return turn_indices


def expand_turn_regions(turn_indices, path_length, radius):
    """Pad each detected turn index outward by radius on both sides."""
    expand = set()
    for idx in turn_indices:
        for j in range(max(0, idx - radius), min(path_length, idx + radius)):
            expand.add(j)

    return expand


def reduce_straight_regions(turn_indices, path_length, interval):
    """Pick a sparse set of indices along straight sections (every `interval` points),
    always keeping the first and last index."""
    straight = set()
    straight.add(0)
    straight.add(path_length - 1)
    # Everything not a turn is considered straight
    for i in range(path_length):
        if i in turn_indices:
            continue
        if i == 0 or i == path_length - 1:
            continue
        if i % interval == 0:
            straight.add(i)

    return straight


def resample_path_uniform(path: List[PathNode], spacing: float = 3.0) -> List[PathNode]:
    """Resample the path to roughly uniform pixel spacing, so turn detection isn't
    biased by uneven node density."""
    if len(path) < 2:
        return path
    result = [path[0]]
    carry = 0.0
    for i in range(1, len(path)):
        dx = path[i].coords.x - path[i - 1].coords.x
        dy = path[i].coords.y - path[i - 1].coords.y
        seg_len = np.sqrt(dx * dx + dy * dy)
        carry += seg_len
        if carry >= spacing:
            result.append(path[i])
            carry = 0.0
    if result[-1] is not path[-1]:
        result.append(path[-1])
    return result


def generate_adaptive_waypoints(
    path_coords,
    straight_interval=15,
    turn_threshold=8.0,
    turn_density=10,
):
    """Combine turn_region + expand_turn_regions + reduce_straight_regions to pick a
    reduced set of "important" waypoints: dense near turns, sparse on straightaways."""
    length = len(path_coords)
    turn_indices = turn_region(path_coords, turn_threshold)
    turn_indices = expand_turn_regions(turn_indices, length, turn_density)
    straight_indices = reduce_straight_regions(turn_indices, length, straight_interval)
    smooth_indices = turn_indices | straight_indices

    return [path_coords[i] for i in sorted(smooth_indices)]


def smooth_path_generation(
    path_coords: List[PathNode],
    straight_interval=20,
    turn_density=5,
    num_points: int = 300,
    occupancy_map: Optional[NDArray] = None,
) -> List[PathNode]:
    """Full smoothing pipeline: shortcut -> uniform resample -> adaptive waypoint
    selection -> Chaikin corner-cutting subdivision -> uniform arc-length resample."""
    if len(path_coords) < 4:
        return []

    if occupancy_map is not None:
        path_coords = shortcut_path(path_coords, occupancy_map)
        if len(path_coords) < 4:
            return path_coords

    path_coords = resample_path_uniform(path_coords, spacing=3.0)
    if len(path_coords) < 4:
        return path_coords

    control = generate_adaptive_waypoints(path_coords, straight_interval, turn_density)
    if len(control) < 2:
        return []

    # Chaikin corner-cutting subdivision: approximating B-spline.
    # Unlike the cubic spline it does NOT interpolate control points, so it
    # cannot oscillate through A* zigzag pixels (fixes lateral drift) and it
    # rounds every corner into a proper arc instead of tracing the staircase
    # (fixes hexagonal-looking turns).  4 passes gives strong smoothing.
    pts = [(n.coords.x, n.coords.y) for n in control]
    for _ in range(4):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            new_pts.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            new_pts.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        new_pts.append(pts[-1])
        pts = new_pts

    # uniform arc-length resample to num_points
    arc = [0.0]
    for i in range(len(pts) - 1):
        dx, dy = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
        arc.append(arc[-1] + np.sqrt(dx * dx + dy * dy))

    smoothed = []
    j = 0
    for dist in np.linspace(0.0, arc[-1], num_points):
        while j < len(arc) - 2 and arc[j + 1] < dist:
            j += 1
        seg = arc[j + 1] - arc[j]
        t = (dist - arc[j]) / seg if seg > 0 else 0.0
        x = pts[j][0] + t * (pts[j + 1][0] - pts[j][0])
        y = pts[j][1] + t * (pts[j + 1][1] - pts[j][1])
        smoothed.append(PathNode(PixelCoords(x, y)))

    if occupancy_map is not None and not is_path_collision_free(smoothed, occupancy_map):
        return control

    return smoothed


# ======================================================

# DWA (local control) ======================================================


def bicycle_step(state: RobotState, v: float, steer: float, dt: float, wheelbase: float) -> RobotState:
    """Advance state by one timestep under the bicycle (Ackermann) kinematic model,
    given a commanded (v, steer)."""

    nextstate = RobotState(state.x, state.y, state.theta)
    nextstate.theta = state.theta + (v * np.tan(steer) / wheelbase) * dt
    nextstate.x = state.x + v * np.cos(state.theta) * dt
    nextstate.y = state.y + v * np.sin(state.theta) * dt
    nextstate.v = v
    nextstate.steer = steer

    return nextstate

    


def rollout_trajectory(
    state: RobotState,
    v: float,
    steer: float,
    predict_time: float,
    dt: float,
    wheelbase: float,
) -> Trajectory:
    """Simulate a candidate (v, steer) forward for predict_time seconds by repeatedly
    calling bicycle_step, collecting the resulting states into a Trajectory.
    """
    traj = []
    dummy = RobotState(state.x, state.y, state.theta, state.v, state.steer)
    for idx in range(int(predict_time/dt)):
        
        next_state = bicycle_step(dummy, v, steer, dt, wheelbase)
        traj.append(next_state)
        dummy = next_state

    return Trajectory(traj, v, steer)



def dynamic_window(
    state: RobotState,
    min_speed: float,
    max_speed: float,
    min_steer: float,
    max_steer: float,
    max_accel: float,
    max_steer_rate: float,
    dt: float,
) -> tuple:
    
    v_lo = max(min_speed, state.v - max_accel * dt)
    v_hi = min(max_speed, state.v + max_accel * dt)

    steer_lo = max(min_steer, state.steer - max_steer_rate * dt)
    steer_hi = min(max_steer, state.steer + max_steer_rate * dt)

    return (v_lo, v_hi, steer_lo, steer_hi)


def build_distance_map(occupancy_map: NDArray) -> NDArray:
    """Precompute each free cell's distance (in pixels) to the nearest occupied
    cell, once per occupancy_map. clearance_cost() looks this up per trajectory
    point instead of recomputing a full distance transform on every call — it's
    expensive enough that redoing it per-candidate (dozens of times per control
    cycle) makes plan() fall behind its control-loop budget."""
    mask = np.full(occupancy_map.shape, 127, dtype=np.uint8)
    mask[occupancy_map == 0] = 255
    mask[occupancy_map == 1] = 0
    return cv2.distanceTransform(mask, cv2.DIST_L2, 5)


def clearance_cost(
    trajectory: Trajectory,
    occupancy_map: NDArray,
    distance_map: NDArray,
    map_info,
    safe_distance: float = 1.0,
) -> float:
    """safe_distance (meters): once the trajectory is at least this far from the
    nearest obstacle, additional distance stops being rewarded — otherwise this
    term (unbounded raw pixel distance) dwarfs heading/velocity in the weighted
    sum even in wide-open space, far past any real safety margin."""

    traj = []
    for coords in trajectory.states:
        mcoords = world_to_pixel(coords.x, coords.y, map_info)
        traj.append(mcoords)
        if not is_in_bounds(mcoords, occupancy_map) or not is_standable(mcoords, occupancy_map):
            return math.inf

    dist_px = math.inf
    for mm in traj:
        if dist_px > distance_map[mm.y_][mm.x_]:
            dist_px = distance_map[mm.y_][mm.x_]

    dist_m = min(dist_px * map_info.resolution, safe_distance)
    cost = 10 - dist_m

    return cost

def heading_cost(trajectory: Trajectory, goal_point: tuple) -> float:
    
    state = trajectory.states[-1]
    yy = goal_point[1] - state.y
    xx = goal_point[0] - state.x
    # Reverse trajectories lead with the rear, not the nose -- compare against
    # theta+pi so backing straight toward a goal behind the car scores as
    # well-aligned instead of looking like it's facing away from the goal.
    facing = state.theta if trajectory.v >= 0 else state.theta + math.pi
    angle = np.arctan2(yy, xx) - facing
    return abs(np.arctan2(np.sin(angle), np.cos(angle)))


def velocity_cost(trajectory: Trajectory, target_speed: float) -> float:

    return target_speed - trajectory.v


def closest_point_on_segment(p: tuple, a: tuple, b: tuple) -> tuple:
    """Closest point on segment a->b to p, and the distance to it. Shared by
    dwa_controller.py's get_local_goal (walking the path for a lookahead
    point) and path_alignment_cost below (scoring candidates against the
    whole path), so the two don't carry separate copies of the same
    point-to-segment math."""
    ab_x = b[0] - a[0]
    ab_y = b[1] - a[1]
    ab_len_sq = ab_x ** 2 + ab_y ** 2
    if ab_len_sq == 0.0:
        return a, float(math.hypot(p[0] - a[0], p[1] - a[1]))

    t = ((p[0] - a[0]) * ab_x + (p[1] - a[1]) * ab_y) / ab_len_sq
    t = max(0.0, min(1.0, t))
    closest = (a[0] + t * ab_x, a[1] + t * ab_y)
    dist = float(math.hypot(p[0] - closest[0], p[1] - closest[1]))
    return closest, dist


def path_alignment_cost(trajectory: Trajectory, path: list) -> float:
    """Average distance (meters) from every sampled point along `trajectory`
    to the nearest point on `path` (a list of (x, y) tuples, in order).

    Complements heading_cost rather than replacing it: heading_cost only
    checks whether the trajectory's final state points toward a single
    lookahead point, so a candidate that swings away from the actual path
    mid-maneuver and back can still score well there. Averaging over every
    sampled state (not just the last one) penalizes that drift directly,
    for the same reason clearance_cost checks every sampled state instead
    of only the endpoint.
    """
    if len(path) < 2:
        return 0.0

    total = 0.0
    for state in trajectory.states:
        best_dist = math.inf
        for i in range(len(path) - 1):
            _, dist = closest_point_on_segment((state.x, state.y), path[i], path[i + 1])
            if dist < best_dist:
                best_dist = dist
        total += best_dist

    return total / len(trajectory.states)


# ======================================================

# Visualization ======================================================


def draw_path(
    img: NDArray,
    path: List[PathNode],
    show_path: bool = True,
    path_color: tuple = (60, 179, 113),
    outline_color: tuple = (255, 255, 255),
    path_thickness: int = 2,
    outline_thickness: int = 4,
    arrow_color: tuple = (0, 165, 255),
    arrow_interval: int = 30,
    arrow: bool = True,
) -> NDArray:
    """Draw the path onto a copy of img, with an outline for contrast and
    periodic direction arrows.

    TODO: build a polyline from path coords, draw the white outline then the
    colored line on top, then draw arrows every arrow_interval points.
    """
    pass


def draw_target(
    img: NDArray,
    coords: PixelCoords,
    color: tuple = (0, 0, 255),
    radius: int = 3,
) -> NDArray:
    """Draw a filled circle marker at coords onto a copy of img.

    TODO: implement with cv2.circle.
    """
    pass


# ======================================================
