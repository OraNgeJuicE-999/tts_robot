"""A* search, implemented as a Planner subclass (see planner.py)."""

import heapq

from jetracer_navigation.planner import Planner
from jetracer_navigation.primitives import PathNode, PixelCoords
from jetracer_navigation.utils import (
    get_8_neighbors,
    reconstruct_path,
    euclidean_distance,
    check_collision_free,
)


class AStarImplementation(Planner):
    def preloop(self):
        
        self.heapq = []
        self.open_set = set()
        self.g_cost = {}
        self.h_cost = {}
        self.node_registry = {}
        self.visited_nodes = set()
        self.counter = 0

        self.start_node = PathNode(coords=self.start_coords)
        self.goal_node = PathNode(coords=self.goal_coords)

        self.node_registry[self.start_node.coords.to_tuple()] = self.start_node
        self.node_registry[self.goal_node.coords.to_tuple()] = self.goal_node

        self.g_cost[self.start_node] = 0.0
        self.h_cost[self.start_node] = euclidean_distance(self.start_node, self.goal_node)

        heapq.heappush(self.heapq, (self.h_cost[self.start_node], self.counter, self.start_node))
        self.counter += 1



    def step(self):
      
      if not self.heapq:
          self.search_done = True
          return

      f_cost, counter, node = heapq.heappop(self.heapq)

      if node in self.visited_nodes:
        return

      self.visited_nodes.add(node)

      if check_collision_free(self.inflated_obstacle_map, node, self.goal_node) and euclidean_distance(node, self.goal_node) < self.goal_threshold:
        self.found_path = True
        self.search_done = True
        self.goal_node.parent = node
        return

      for neighbors in get_8_neighbors(node.coords, self.inflated_obstacle_map, self.node_registry):
        if neighbors in self.visited_nodes:
           continue

        tentative_g_cost = self.g_cost[node] + euclidean_distance(node, neighbors)
        if tentative_g_cost < self.g_cost.get(neighbors, float('inf')):
            neighbors.parent = node
            self.g_cost[neighbors] = tentative_g_cost
            self.h_cost[neighbors] = euclidean_distance(neighbors, self.goal_node)
            f_cost = tentative_g_cost + self.h_cost[neighbors]
            heapq.heappush(self.heapq, (f_cost, self.counter, neighbors))
            self.counter += 1

    def postloop(self):
      
      if self.found_path:
          path = reconstruct_path(self.goal_node)
          return path, self.visited_nodes

      else:
         return [], self.visited_nodes
        
