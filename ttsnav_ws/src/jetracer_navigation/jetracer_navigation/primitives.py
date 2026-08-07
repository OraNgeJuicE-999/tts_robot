"""Core data structures shared by the planner: pixel coordinates and search nodes."""

from dataclasses import dataclass, field
from typing import List


class PixelCoords:
    """A single (x, y) grid/pixel coordinate.

    TODO: __init__ should round and store x/y as ints (mirrors ORB_SLAM3_Relocalization).
    """

    def __init__(self, x, y):
        self.x_ = round(x)
        self.y_ = round(y)

    @property
    def x(self):
        return self.x_

    @property
    def y(self):
        return self.y_

    def __eq__(self, other):
        return (self.x_, self.y_) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x_, self.y_))

    def to_tuple(self):
        return (self.x_, self.y_)


@dataclass
class PathNode:
    """A single node in the search graph, tied to a PixelCoords location."""

    coords: PixelCoords
    parent: "PathNode" = None
    g_cost: float = 0.0
    h_cost: float = 0.0
    cost: float = 0.0

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

    def __lt__(self, other: "PathNode") -> bool:
        return self.f_cost < other.f_cost

    def has_parent(self) -> bool:
        if self.parent is None:
            return False
        return True

    def __eq__(self, other):
        if isinstance(other, PathNode):
            return self.coords == other.coords
        return False

    def __hash__(self):
        return hash(self.coords)


@dataclass
class RobotState:
    """Continuous world-frame robot state for DWA (meters/radians), as opposed to
    PixelCoords/PathNode which are grid-based and only used by A*."""

    x: float
    y: float
    theta: float        # heading, radians
    v: float = 0.0       # current linear speed, m/s
    steer: float = 0.0   # current steering angle, radians


@dataclass
class Trajectory:
    """A short rollout produced by simulating one (v, steer) candidate forward in
    time with the bicycle model, plus the command that produced it."""

    states: List[RobotState] = field(default_factory=list)
    v: float = 0.0
    steer: float = 0.0
