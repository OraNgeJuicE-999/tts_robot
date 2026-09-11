#!/usr/bin/env python3
"""One-shot CLI: reads the current ground-truth pose from /odom (Isaac Sim's
physics transform, not a drifting estimate) and publishes it to /initialpose
so AMCL starts from the robot's real position instead of an eyeballed RViz
click. Assumes map and odom currently share an origin -- true for this sim
setup since /odom is ground truth and the map was generated from the same
Isaac Sim world, but not a general-purpose assumption on real hardware.

Usage: ros2 run jetracer_navigation set_initial_pose
"""
import sys

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped


class SetInitialPose(Node):
    def __init__(self):
        super().__init__('set_initial_pose')
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.done = False

    def odom_cb(self, msg: Odometry):
        if self.done:
            return
        self.done = True

        # Publisher needs at least one subscriber (amcl) connected before a
        # single publish is guaranteed to land -- give it a moment.
        for _ in range(20):
            if self.pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.pose.position.x = msg.pose.pose.position.x
        pose.pose.pose.position.y = msg.pose.pose.position.y
        pose.pose.pose.position.z = 0.0
        pose.pose.pose.orientation = msg.pose.pose.orientation
        self.pub.publish(pose)

        p = pose.pose.pose.position
        q = pose.pose.pose.orientation
        self.get_logger().info(
            f'Published initial pose from ground-truth odom: '
            f'x={p.x:.3f} y={p.y:.3f} quat_z={q.z:.4f} quat_w={q.w:.4f}'
        )
        rclpy.shutdown()


def main():
    rclpy.init()
    node = SetInitialPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
