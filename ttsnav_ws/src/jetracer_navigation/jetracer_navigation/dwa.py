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
)


@dataclass
class DWAConfig:
    """Tunable parameters for the DWA search. Fill these in against the JetRacer's
    real limits (steer bounds match the ±0.322 rad revolute joint limits in
    jetracer.xacro; wheelbase should match the physical front/rear axle spacing)."""

    min_speed: float = -0.3  # allow limited reverse so DWA can back away from a wall to clear a turn
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
    reverse_penalty: float = 5.0  # margin forward must lose by before reverse is allowed to win


class DWA:
    def __init__(self, config: DWAConfig):
        self.config = config
        self.heading_state = "forward"
        self.change_state = False
        self.elasped_time = 0
        self.state_change_tolerance = 10

    def plan(
        self,
        state: RobotState,
        goal_point: Tuple[float, float],
        on_path: bool,
        occupancy_map: NDArray,
        map_info,
    ) -> Tuple[Optional[float], Optional[float], Optional[Trajectory]]:
        """Return the best (v, steer, trajectory) for this control cycle, or
        (None, None, None) if every sampled candidate collides.

        Forward-biased, not forward-gated: search both the forward (v >= 0)
        and reverse (v < 0) portions of the window, but only let reverse win
        if its best cost beats forward's by more than reverse_penalty. A hard
        gate ("only look at reverse if zero forward candidates are
        collision-free") is too blunt -- in open space plenty of forward
        candidates stay collision-free even when every one of them has a
        terrible heading_cost (goal roughly behind the robot), so a hard gate
        commits to the least-bad forward option regardless of how badly
        misaligned it is with the path, instead of ever considering a much
        better-aligned reverse option. Comparing the two pools' best costs
        (with reverse required to win by a margin, not just win) fixes that:
        reverse still only wins outright when forward is truly blocked
        (cost = inf, no finite penalty saves it), but a forward option that's
        merely bad and not blocked can still lose to a reverse option that's
        clearly better, rather than winning by default.
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

        forward_v = v_range[v_range >= 0]
        reverse_v = v_range[v_range < 0]

        fwd_v, fwd_s, fwd_traj, fwd_cost, fwd_n, fwd_valid = self._search(forward_v, s_range, state, occupancy_map, distance_map, map_info, goal_point)
        rev_v, rev_s, rev_traj, rev_cost, rev_n, rev_valid = self._search(reverse_v, s_range, state, occupancy_map, distance_map, map_info, goal_point)

        num_candidates = fwd_n + rev_n
        num_valid = fwd_valid + rev_valid

        # A forward candidate at v~=0 makes no real progress -- it's only
        # cheap because standing still is trivially collision-free and
        # velocity_cost rewards being close to max_speed, not because it's a
        # meaningful "forward is fine" signal. Treat it the same as forward
        # being truly blocked (cost=inf) ONLY when deciding whether to enter
        # reverse (the heading_state == "forward" branch below) -- not when
        # deciding whether to exit it. Once already reversing, a v~=0
        # forward candidate can be a completely legitimate "the maneuver
        # worked, forward is genuinely better now" signal (good heading,
        # good clearance, just not yet accelerated), and suppressing it
        # there would mean forward could never win back. best_v/best_s/
        # best_traj/best_cost still use the real fwd_cost whenever forward
        # is actually selected, so v=0 remains available as the genuine
        # fallback if reverse turns out to be just as blocked.
        fwd_cost_cmp = np.inf if fwd_v is not None and abs(fwd_v) < 1e-6 else fwd_cost

        # Flip the committed direction only once a disagreement has persisted
        # for more than state_change_tolerance *consecutive* cycles -- see
        # below for how elasped_time/change_state are kept tracking that
        # instead of "cycles since the last flip".
        # on_path alone isn't enough -- it's just "near the path line", which
        # can stay true throughout being stuck (an obstacle sitting on the
        # path doesn't push the robot off of it). Requiring the plain
        # (non-sticky) comparison to also agree forward is at least
        # acceptable means this only fires once progress is actually viable
        # again, not just whenever the robot happens to be near the line.
        if on_path and fwd_cost <= rev_cost + self.config.reverse_penalty:
            self.heading_state = "forward"
            self.change_state = False
            self.elasped_time = 0

        if self.change_state and self.elasped_time > self.state_change_tolerance:
            self.heading_state = "reverse" if self.heading_state == "forward" else "forward"
            self.change_state = False
            self.elasped_time = 0

        if self.heading_state == "forward":
            if fwd_cost_cmp <= rev_cost + self.config.reverse_penalty:
                best_v, best_s, best_traj, best_cost = fwd_v, fwd_s, fwd_traj, fwd_cost
                used_reversed = False
                # Agrees with the committed direction -- any pending
                # disagreement streak is broken, not just paused.
                self.change_state = False
                self.elasped_time = 0
            else:
                best_v, best_s, best_traj, best_cost = rev_v, rev_s, rev_traj, rev_cost
                used_reversed = True
                # Disagrees -- only reset the counter at the *start* of a
                # new streak, so it actually counts consecutive cycles
                # instead of restarting on every disagreeing cycle (which
                # would never reach the tolerance) or never restarting at
                # all (which let a single stale blip flip the state almost
                # immediately whenever elasped_time was already large from
                # an unrelated long-stable streak).
                if not self.change_state:
                    self.change_state = True
                    self.elasped_time = 0
        else:
            # Real fwd_cost here, not fwd_cost_cmp -- the v~=0 suppression
            # is only meant to make it easy to *enter* reverse when forward
            # is genuinely offering no progress. Applying it here too would
            # mean a forward candidate can never win back from committed
            # reverse whenever its best move happens to be standing still,
            # even if that's because the maneuver actually worked and
            # forward's heading/clearance are now legitimately excellent --
            # exactly the situation that stood still is masking, not this
            # time in the other direction.
            if fwd_cost + self.config.reverse_penalty <= rev_cost:
                best_v, best_s, best_traj, best_cost = fwd_v, fwd_s, fwd_traj, fwd_cost
                used_reversed = False
                if not self.change_state:
                    self.change_state = True
                    self.elasped_time = 0
            else:
                best_v, best_s, best_traj, best_cost = rev_v, rev_s, rev_traj, rev_cost
                used_reversed = True
                self.change_state = False
                self.elasped_time = 0


        self.elasped_time += 1

        t_end = time.time()
        print(
            f"[DWA.plan] window=(v:{vl:.3f}..{vh:.3f}, s:{sl:.3f}..{sh:.3f}) "
            f"reverse_fallback={used_reversed} candidates={num_candidates} valid={num_valid} "
            f"best=(v={best_v}, s={best_s}, cost={best_cost:.3f}) "
            f"distance_map={t_after_distance_map - t_start:.4f}s "
            f"total={t_end - t_start:.4f}s"
        )

        return best_v, best_s, best_traj

    def _search(self, v_values, s_range, state, occupancy_map, distance_map, map_info, goal_point):
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
                cost, clear, head, vel = self.cost_scoring(traj, occupancy_map, distance_map, map_info, goal_point)
                print(f"    v={v:.3f} s={s:.3f} clear={clear} head={head} vel={vel} total={cost}")
                if cost is not None:
                    num_valid += 1
                    if cost < best_traj_cost:
                        best_s = s
                        best_v = v
                        best_traj = traj
                        best_traj_cost = cost
        return best_v, best_s, best_traj, best_traj_cost, num_candidates, num_valid

    def cost_scoring(self, traj:Trajectory, occupancy_map:NDArray, distance_map:NDArray, map_info, goal_point):
        clear = clearance_cost(traj, occupancy_map, distance_map, map_info)
        if clear == np.inf:
            return None, clear, None, None

        head = heading_cost(traj, goal_point)
        vel = velocity_cost(traj, self.config.max_speed)

        total_cost = clear * self.config.clearance_weight + head * self.config.heading_weight + vel * self.config.velocity_weight
        return total_cost, clear, head, vel
        
