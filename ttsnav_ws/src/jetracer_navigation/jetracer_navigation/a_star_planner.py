"""ROS2 node that plans a global path with A*, triggered by RViz's 2D Goal Pose tool."""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import tf2_ros

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from typing import List

from jetracer_navigation.a_star import AStarImplementation
from jetracer_navigation.utils import world_to_pixel, pixel_to_world, smooth_path_generation


class AStarPlanner(Node):
    def __init__(self):
        super().__init__('a_star_planner')

        self.world_map = None      # 2D numpy occupancy grid, built in map_callback
        self.map_info = None       # last received OccupancyGrid.info (resolution/origin)

        self.occupied_threshold = 65

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.path_pub = self.create_publisher(Path, '/plan', 10)

    def map_callback(self, msg):
        
        raw_map = msg.data
        self.map_info = msg.info

        raw_map = np.array(raw_map, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        
        binary = np.zeros_like(raw_map, dtype=np.uint8)
        binary[raw_map >= self.occupied_threshold] = 1
        binary[raw_map == -1] = 1
        self.world_map = binary

    def get_current_pose(self):
        
        t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())

        return (t.transform.translation.x, t.transform.translation.y)


    def goal_callback(self, msg):
        
        if self.map_info is None or self.world_map is None:
            self.get_logger().warning("No map received yet, ignoring goal")
            return

        start_world = self.get_current_pose()
        start = world_to_pixel(start_world[0], start_world[1], self.map_info)
        goal = world_to_pixel(msg.pose.position.x, msg.pose.position.y, self.map_info)

        planner_instance = AStarImplementation(
            self.world_map, start, goal, goal_threshold=3, inflation_radius=8
        )
        path, visited_node = planner_instance.plan()
        if not path:
            self.get_logger().warning("No path found")
            return

        path = smooth_path_generation(path, occupancy_map=planner_instance.inflated_obstacle_map)
        if not path:
            self.get_logger().warning("Path smoothing produced an empty path, not publishing")
            return

        path_ = Path()
        path_.header.frame_id = 'map'
        path_.header.stamp = self.get_clock().now().to_msg()

        for node in path:
            wx, wy = pixel_to_world(node.coords.x, node.coords.y, self.map_info)

            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = path_.header.stamp
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0

            path_.poses.append(pose)

        self.path_pub.publish(path_)
        self.get_logger().info(
            f"Published path with {len(path_.poses)} waypoints "
            f"from pixel {start} to {goal}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
