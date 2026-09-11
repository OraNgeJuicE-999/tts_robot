"""Dynamic Window Approach, adapted for Ackermann-steered (bicycle model) robots.

No ROS2 imports here on purpose — this is pure search/scoring logic over
(v, steer) candidates, testable standalone with plain arrays/tuples, the same
way primitives.py/utils.py/a_star.py are kept separate from a_star_planner.py.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from jetracer_navigation.primitives import RobotState, Trajectory
from jetracer_navigation.utils import (
    dynamic_window,
    rollout_trajectory,
    build_distance_map,
    clearance_cost,
    heading_cost,
    velocity_cost,
    path_alignment_cost,
    bresenham,
)


@dataclass
class DWAConfig:
    """Tunable parameters for the DWA search. Fill these in against the JetRacer's
    real limits (steer bounds match the ±0.322 rad revolute joint limits in
    jetracer.xacro; wheelbase should match the physical front/rear axle spacing)."""

    min_speed: float = 0.0  # forward-only -- see dwa_controller.py's stuck-recovery TODOs for how reverse is now handled
    max_speed: float = 1.0
    min_steer: float = -0.322
    max_steer: float = 0.322
    max_accel: float = 1.0
    max_steer_rate: float = 1.0
    wheelbase: float = 0.26

    predict_time: float = 1.0   # seconds simulated per candidate rollout
    dt: float = 0.1             # simulation timestep within a rollout
    v_resolution: float = 0.05  # sampling step across the speed window
    steer_resolution: float = 0.05  # sampling step across the steer window

    heading_weight: float = 1.0
    clearance_weight: float = 1.0
    velocity_weight: float = 10.0
    path_alignment_weight: float = 1.0  # same order of magnitude as heading_weight to start -- retune once both are visible in the same run


class DWA:
    def __init__(self, config: DWAConfig):
        self.config = config

    def plan(
        self,
        state: RobotState,
        goal_point: Tuple[float, float],
        occupancy_map: NDArray,
        map_info,
        path: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[Trajectory]]:
        """Return the best (v, steer, trajectory) for this control cycle, or
        (None, None, None) if every sampled candidate collides.

        Forward-only. Getting unstuck (backing away from an obstacle) is
        deliberately not this function's job anymore -- an earlier version
        tried to handle it here via a reverse search plus a hysteresis/
        debounce/lane-check state machine on top of a cost-margin
        comparison, and every fix to it uncovered a new edge case, because
        margin-based tuning doesn't scale against a cost function whose
        candidate-to-candidate differences vary a lot depending on the
        scene. Modern stacks (Nav2) don't solve this inside the local
        planner's per-cycle cost function either -- they use a bounded
        recovery behavior (Nav2's nav2_behaviors BackUp): the planner just
        reports "no valid command found," and a separate state machine
        reverses a fixed distance/time, then hands control back. See the
        TODOs in dwa_controller.py's __init__/control_loop for that piece.

        `path` (list of (x, y) tuples, in order) is optional and only used
        for path-alignment scoring (see cost_scoring) -- pass None to fall
        back to pure heading-toward-goal_point scoring, e.g. for tests that
        don't have a real path handy.
        """
        window = dynamic_window(
            state, self.config.min_speed, self.config.max_speed,
            self.config.min_steer, self.config.max_steer,
            self.config.max_accel, self.config.max_steer_rate, self.config.dt
            )
        (vl, vh, sl, sh) = window

        v_range = np.arange(vl, vh, self.config.v_resolution)
        s_range = np.arange(sl, sh, self.config.steer_resolution)

        t_start = time.time()

        # Distance transform is expensive (~480x480) — build it once per plan()
        # call, not once per candidate.
        distance_map = build_distance_map(occupancy_map)
        t_after_distance_map = time.time()

        best_v, best_s, best_traj, best_cost, num_candidates, num_valid = self._search(
            v_range, s_range, state, occupancy_map, distance_map, map_info, goal_point, path
        )

        t_end = time.time()
        print(
            f"[DWA.plan] window=(v:{vl:.3f}..{vh:.3f}, s:{sl:.3f}..{sh:.3f}) "
            f"candidates={num_candidates} valid={num_valid} "
            f"best=(v={best_v}, s={best_s}, cost={best_cost:.3f}) "
            f"distance_map={t_after_distance_map - t_start:.4f}s "
            f"total={t_end - t_start:.4f}s"
        )

        return best_v, best_s, best_traj

    def _search(self, v_values, s_range, state, occupancy_map, distance_map, map_info, goal_point, path=None):
        """Score every (v, steer) candidate in v_values x s_range, returning
        the lowest-cost collision-free one (or None/inf if none are valid)."""
        best_traj_cost = np.inf
        best_v = None
        best_s = None
        best_traj = None
        num_candidates = 0
        num_valid = 0
        for v in v_values:
            for s in s_range:
                num_candidates += 1
                traj = rollout_trajectory(state, v, s, self.config.predict_time, self.config.dt, self.config.wheelbase)
                cost, clear, head, vel, align = self.cost_scoring(traj, occupancy_map, distance_map, map_info, goal_point, path)
                print(f"    v={v:.3f} s={s:.3f} clear={clear} head={head} vel={vel} align={align} total={cost}")
                if cost is not None:
                    num_valid += 1
                    if cost < best_traj_cost:
                        best_s = s
                        best_v = v
                        best_traj = traj
                        best_traj_cost = cost
        return best_v, best_s, best_traj, best_traj_cost, num_candidates, num_valid

    def cost_scoring(self, traj:Trajectory, occupancy_map:NDArray, distance_map:NDArray, map_info, goal_point, path=None):
        clear = clearance_cost(traj, occupancy_map, distance_map, map_info)
        if clear == np.inf:
            return None, clear, None, None, None

        head = heading_cost(traj, goal_point)
        vel = velocity_cost(traj, self.config.max_speed)

        # Sits alongside heading_cost rather than replacing it: heading_cost
        # only checks the final state's bearing to goal_point, so it can't
        # tell a candidate that swings away from the path and back apart
        # from one that stayed close the whole time -- path_alignment_cost
        # (averaged over every sampled state) covers that gap. path is
        # optional (e.g. for standalone tests without a real /plan handy),
        # so this term drops to 0 rather than erroring when it's absent.
        align = path_alignment_cost(traj, path) if path else 0.0

        total_cost = (
            clear * self.config.clearance_weight
            + head * self.config.heading_weight
            + vel * self.config.velocity_weight
            + align * self.config.path_alignment_weight
        )
        return total_cost, clear, head, vel, align
        
