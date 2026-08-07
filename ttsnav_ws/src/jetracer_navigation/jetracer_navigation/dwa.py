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

    min_speed: float = 0.0
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


class DWA:
    def __init__(self, config: DWAConfig):
        self.config = config

    def plan(
        self,
        state: RobotState,
        goal_point: Tuple[float, float],
        occupancy_map: NDArray,
        map_info,
    ) -> Tuple[Optional[float], Optional[float], Optional[Trajectory]]:
        """Return the best (v, steer, trajectory) for this control cycle, or
        (None, None, None) if every sampled candidate collides.

        TODO:
          1. cfg = self.config; get (v_lo, v_hi, steer_lo, steer_hi) from
             dynamic_window(state, cfg.min_speed, cfg.max_speed, cfg.min_steer,
             cfg.max_steer, cfg.max_accel, cfg.max_steer_rate, cfg.dt)
          2. loop over v in that v range (step cfg.v_resolution) and steer in
             that steer range (step cfg.steer_resolution) — build the candidate grid
          3. for each (v, steer): rollout_trajectory(state, v, steer,
             cfg.predict_time, cfg.dt, cfg.wheelbase)
          4. score it: clearance = clearance_cost(traj, occupancy_map, map_info) —
             skip/discard the candidate if this indicates a collision
          5. otherwise total_cost = heading_weight*heading_cost(traj, goal_point)
             + clearance_weight*clearance + velocity_weight*velocity_cost(traj, cfg.max_speed)
          6. track the (v, steer, traj) with the lowest total_cost seen so far
          7. after the loop, return the best candidate found (or None, None, None)
        """
        # Dynamic Window
        window = dynamic_window(
            state, self.config.min_speed, self.config.max_speed, 
            self.config.min_steer, self.config.max_steer, 
            self.config.max_accel, self.config.max_steer_rate, self.config.dt
            )

        # Window Range
        (vl, vh, sl, sh) = window

        v_range = np.arange(vl, vh, self.config.v_resolution)
        s_range = np.arange(sl, sh, self.config.steer_resolution)

        t_start = time.time()

        # Distance transform is expensive (~480x480) — build it once per plan()
        # call, not once per candidate.
        distance_map = build_distance_map(occupancy_map)
        t_after_distance_map = time.time()

        # Create Motion Rollout Trajectory
        best_traj_cost = np.inf
        best_v = None
        best_s = None
        best_traj = None
        num_candidates = 0
        num_valid = 0
        for v in v_range:
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

        t_end = time.time()
        print(
            f"[DWA.plan] window=(v:{vl:.3f}..{vh:.3f}, s:{sl:.3f}..{sh:.3f}) "
            f"candidates={num_candidates} valid={num_valid} "
            f"best=(v={best_v}, s={best_s}, cost={best_traj_cost:.3f}) "
            f"distance_map={t_after_distance_map - t_start:.4f}s "
            f"total={t_end - t_start:.4f}s"
        )

        return best_v, best_s, best_traj

    def cost_scoring(self, traj:Trajectory, occupancy_map:NDArray, distance_map:NDArray, map_info, goal_point):
        clear = clearance_cost(traj, occupancy_map, distance_map, map_info)
        if clear == np.inf:
            return None, clear, None, None

        head = heading_cost(traj, goal_point)
        vel = velocity_cost(traj, self.config.max_speed)

        total_cost = clear * self.config.clearance_weight + head * self.config.heading_weight + vel * self.config.velocity_weight
        return total_cost, clear, head, vel
        
