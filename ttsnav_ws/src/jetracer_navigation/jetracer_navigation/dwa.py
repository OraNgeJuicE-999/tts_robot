"""Dynamic Window Approach, adapted for Ackermann-steered (bicycle model) robots.

No ROS2 imports here on purpose — this is pure search/scoring logic over
(v, steer) candidates, testable standalone with plain arrays/tuples, the same
way primitives.py/utils.py/a_star.py are kept separate from a_star_planner.py.
"""

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
    """Tunable parameters for the DWA search, against the Leatherback rig's
    real Ackermann Controller limits (from tester.usd's ROS_Ackermann_Drive
    node property panel: wheelBase=0.32, maxWheelVelocity=60.0 rad/s *
    wheelRadius=0.052m ~= 3.12 m/s top speed).

    min_steer/max_steer: the property panel's maxWheelRotation (0.7854 rad,
    45deg) caused a full physics break when actually commanded near a wall
    -- turning back that sharply broke the car and ended the Isaac Sim
    session. Given this rig already shows PhysX "disjointed body transforms"
    warnings on its suspension/steering joints independent of anything DWA
    does, that parameter is not trustworthy as a safe commandable range.
    Dialed back to a deliberately conservative value (matching the old
    jetracer.xacro joint limit that was never a problem) until the real safe
    range is verified incrementally, in small steps, well clear of walls."""

    min_speed: float = 0.0  # forward-only -- see dwa_controller.py's stuck-recovery TODOs for how reverse is now handled
    max_speed: float = 1.5  # halved from the measured ~3.0 physical ceiling -- more margin while retesting after the steering incident
    min_steer: float = -0.322
    max_steer: float = 0.322
    max_accel: float = 1.0
    max_steer_rate: float = 1.0
    wheelbase: float = 0.32

    # predict_time shortened from 1.0s -- a full 1s rollout at speed reaches
    # 1-3m ahead, which in a small/tight environment means one distant
    # obstacle invalidates an entire candidate even though the next half-
    # second of it is completely clear. Shorter horizon = more responsive,
    # less falsely blocked in clutter.
    #
    # dt is NOT purely "rollout resolution" despite the name/comment below
    # -- dynamic_window() (utils.py) reuses this exact same value to compute
    # how far v/steer can move in one control cycle (state.v +/- max_accel*dt
    # etc.). Dropping it to 0.05 alongside v_resolution=0.05 shrank that
    # window to a single width-0.05 slice, meaning np.arange(0, 0.05, 0.05)
    # from a cold stop produces only [0.0] -- no v>0 candidate is ever even
    # sampled, not merely rejected. That's what broke movement after the
    # predict_time change; keep dt at the original 0.1 until dynamic_window
    # is changed to take the real control-loop period instead of reusing
    # this rollout-step value.
    predict_time: float = 0.4   # seconds simulated per candidate rollout
    dt: float = 0.1             # simulation timestep within a rollout AND accel/steer-rate window size (see above)
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
        target_speed: Optional[float] = None,
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

        `target_speed` overrides velocity_cost's "ideal" speed for this call
        only (defaults to config.max_speed) -- dwa_controller.py passes a
        tapered value as the robot nears the final goal, so candidates near
        that (now-lower) target score best instead of the search always
        preferring the fastest reachable candidate regardless of proximity
        to the goal. This does NOT shrink the sampled window itself (still
        drawn from config.max_speed via dynamic_window) -- a faster escape
        is still available and preferred over colliding, this only changes
        which speed is treated as "ideal" when candidates are otherwise
        similar.
        """
        window = dynamic_window(
            state, self.config.min_speed, self.config.max_speed,
            self.config.min_steer, self.config.max_steer,
            self.config.max_accel, self.config.max_steer_rate, self.config.dt
            )
        (vl, vh, sl, sh) = window

        v_range = np.arange(vl, vh, self.config.v_resolution)
        s_range = np.arange(sl, sh, self.config.steer_resolution)

        # Distance transform is expensive (~480x480) — build it once per plan()
        # call, not once per candidate.
        distance_map = build_distance_map(occupancy_map)

        effective_target_speed = self.config.max_speed if target_speed is None else target_speed
        best_v, best_s, best_traj, _, _, _, candidates = self._search(
            v_range, s_range, state, occupancy_map, distance_map, map_info, goal_point, path,
            target_speed=effective_target_speed,
        )

        return best_v, best_s, best_traj, candidates

    def _search(self, v_values, s_range, state, occupancy_map, distance_map, map_info, goal_point, path=None, target_speed=None):
        """Score every (v, steer) candidate in v_values x s_range, returning
        the lowest-cost collision-free one (or None/inf if none are valid),
        plus every candidate evaluated (v, steer, trajectory, valid) --
        dwa_controller.py visualizes this full set as a MarkerArray, since
        the log line this used to print per-candidate quickly becomes
        unreadable at 15Hz with dozens of candidates per tick."""
        best_traj_cost = np.inf
        best_v = None
        best_s = None
        best_traj = None
        num_candidates = 0
        num_valid = 0
        candidates = []
        for v in v_values:
            for s in s_range:
                num_candidates += 1
                traj = rollout_trajectory(state, v, s, self.config.predict_time, self.config.dt, self.config.wheelbase)
                cost, _, _, _, _ = self.cost_scoring(
                    traj, occupancy_map, distance_map, map_info, goal_point, path, target_speed=target_speed
                )
                valid = cost is not None
                candidates.append((v, s, traj, valid))
                if valid:
                    num_valid += 1
                    # Strict "<" alone always keeps whichever tie is found
                    # first. That's invisible for almost every case (real
                    # cost differences break ties in practice), but at v=0
                    # every steer produces the exact same stationary
                    # trajectory (bicycle_step's turn rate is v-scaled, so
                    # v=0 zeroes it regardless of steer) -- every steer in
                    # the sampled range ties exactly, and always keeping the
                    # first one found locks the robot onto whichever steer
                    # happens to be the window's current edge. Since the
                    # window recenters on last_steer each tick, that's a
                    # self-reinforcing drift toward the steer limit, not a
                    # one-off. Break ties toward steer=0 instead so a v=0
                    # result actually means "stopped, wheels centered," not
                    # "stopped, wheels locked hard over."
                    is_new_best = cost < best_traj_cost or (
                        cost == best_traj_cost and best_s is not None and abs(s) < abs(best_s)
                    )
                    if is_new_best:
                        best_s = s
                        best_v = v
                        best_traj = traj
                        best_traj_cost = cost
        return best_v, best_s, best_traj, best_traj_cost, num_candidates, num_valid, candidates

    def cost_scoring(self, traj:Trajectory, occupancy_map:NDArray, distance_map:NDArray, map_info, goal_point, path=None, target_speed=None):
        clear = clearance_cost(traj, occupancy_map, distance_map, map_info)
        if clear == np.inf:
            return None, clear, None, None, None

        head = heading_cost(traj, goal_point)
        vel = velocity_cost(traj, self.config.max_speed if target_speed is None else target_speed)

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
        
