"""Generic planner base class (template method pattern).

Subclasses (e.g. AStarImplementation in a_star.py) implement preloop/step/postloop;
this base class handles the shared setup, validation, and main loop so any search
algorithm can plug into the same interface.
"""

from abc import abstractmethod
from typing import Optional

from numpy.typing import NDArray

from jetracer_navigation.primitives import PathNode, PixelCoords
from jetracer_navigation.utils import world_2_occupancy_map, is_standable, is_in_bounds


class Planner:
    def __init__(
        self,
        world_map: NDArray,
        start_coords: PixelCoords,
        goal_coords: PixelCoords,
        goal_threshold: float = 3,
        iter_limit: int = 10000,
        inflation_radius: int = 8,
        inflated_map: Optional[NDArray] = None,
    ):
        self.world_map = world_map.copy()
        self.start_coords = start_coords
        self.goal_coords = goal_coords
        self.goal_threshold = goal_threshold
        self.iter_limit = iter_limit
        self.found_path = False
        self.search_done = False
        self.occupancy_map = None
        self.inflated_obstacle_map = None
        self.inflation_radius = inflation_radius
        self._cached_inflated_map = inflated_map

    def set_map(self):
        if self._cached_inflated_map is None:
            self.occupancy_map, self.inflated_obstacle_map= world_2_occupancy_map(self.world_map, self.start_coords, self.goal_coords, self.inflation_radius)

        else: 
            self.occupancy_map = self.world_map
            self.inflated_obstacle_map = self._cached_inflated_map

    def normalize_start(self):
        if not isinstance(self.start_coords, PixelCoords):
            self.start_coords = PixelCoords(self.start_coords[0], self.start_coords[1]) # Add a function in utils (maybe)

    def normalize_goal(self):
        if not isinstance(self.goal_coords, PixelCoords):
            self.goal_coords = PixelCoords(self.goal_coords[0], self.goal_coords[1])

    def is_ready(self):
        if self.occupancy_map is None or self.start_coords is None or self.goal_coords is None:
            return False
        
        return True
    
    def validate_request(self):

        if self.is_ready():
            if is_in_bounds(self.start_coords, self.occupancy_map) and is_in_bounds(self.goal_coords, self.occupancy_map):
                if is_standable(self.start_coords, self.occupancy_map) and is_standable(self.goal_coords, self.occupancy_map):
                    return True
                else: 
                    raise ValueError("Start or Goal pose not standable")
            else: 
                raise ValueError("Start or Goal pose not in bounds")
        else:
            raise ValueError("Not Ready!")

    def plan(self):

        self.set_map()
        self.normalize_start()
        self.normalize_goal()
        self.found_path = False
        self.search_done = False
        # Verify everything is ready
        self.validate_request()
        # Preloop
        self.preloop()

        while True:

            self.step()
            if self.found_path or self.search_done:
                break

        result = self.postloop()

        return result

    @abstractmethod
    def preloop(self) -> None:
        """One-time setup before the search loop starts (e.g. seed the open set)."""
        ...

    @abstractmethod
    def step(self) -> None:
        """A single search iteration (e.g. pop one node, expand its neighbors)."""
        ...

    @abstractmethod
    def postloop(self) -> tuple[list[PathNode], set[PathNode]]:
        """Called once the loop ends; build and return (path, visited_nodes)."""
        ...
