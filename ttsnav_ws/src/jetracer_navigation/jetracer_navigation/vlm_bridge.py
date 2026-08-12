"""ROS2 node that bridges ASR transcripts to VLM-issued robot commands.

Subscribes to raw voice transcripts (std_msgs/String) on /transcript, calls
the VLM backend via VLMClient.get_robot_command(), and republishes its JSON
response as a std_msgs/String on /robot_command -- the topic goal_bridge.py
subscribes to, to turn a "navigate" command into a /goal_pose.

vlm_client.py itself stays ROS-free on purpose (same split as dwa.py vs
dwa_controller.py) -- this node is the only place that imports it.
"""

import asyncio
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from jetracer_navigation.vlm_client import VLMClient, VLMClientError


class VLMBridge(Node):
    def __init__(self):
        super().__init__('vlm_bridge')

        self.declare_parameter('base_url', 'http://140.96.96.16:8080')
        self.base_url = self.get_parameter('base_url').value

        self.session_id = None

        self.transcript_sub = self.create_subscription(String, '/transcript', self.transcript_callback, 10)
        self.command_pub = self.create_publisher(String, '/robot_command', 10)

    def transcript_callback(self, msg):
        try:
            command = asyncio.run(self._fetch_command(msg.data))
        except VLMClientError as exc:
            self.get_logger().warning(f"VLM request failed: {exc}")
            return

        out = String()
        out.data = json.dumps(command)
        self.command_pub.publish(out)

    async def _fetch_command(self, transcript: str) -> dict:
        # A fresh VLMClient per call (rather than one held on self and reused
        # across asyncio.run() calls) -- asyncio.run() tears its event loop
        # down after every call, and httpx's async client gets bound to
        # whichever loop was active on its first request, so a client held
        # open across calls would break on the second one.
        async with VLMClient(base_url=self.base_url, session_id=self.session_id) as client:
            await client.start_session()
            self.session_id = client.session_id
            return await client.get_robot_command(transcript)


def main(args=None):
    rclpy.init(args=args)
    node = VLMBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()