from rclpy .node import Node
from ackermann_msgs.msg import AckermannDriveStamped
import sys
import termios
import tty 
import select
import rclpy 

class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')
        self.pub_cmd = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 10)

        self.speed = 0.0
        self.steering = 0.0
        self.timeout = 0.1

    def getKey(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try: 
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], self.timeout)
            if rlist:
                key = sys.stdin.read(1)
            else:
                key = ''
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        return key
    
    def increase_speed(self):
        self.speed += 0.1
        self.pub_cmds()
    def decrease_speed(self):
        self.speed -= 0.1
        self.pub_cmds()
    def increase_steering(self):
        self.steering = max(-0.322, min(0.322, self.steering + 0.1))
        self.pub_cmds()
    def decrease_steering(self):
        self.steering = max(-0.322, min(0.322, self.steering - 0.1))
        self.pub_cmds()
    def stop(self):
        self.speed = 0.0
        self.steering = 0.0
        self.pub_cmds()

    def pub_cmds(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = self.speed
        msg.drive.steering_angle = self.steering
        self.pub_cmd.publish(msg)

def main():
    rclpy.init()
    node = TeleopKeyboard()
    while True:
        key = node.getKey()
        if key == 'w':
            node.increase_speed()
        elif key == 's':
            node.decrease_speed()
        elif key == 'a':
            node.increase_steering()
        elif key == 'd':
            node.decrease_steering()
        elif key == ' ':
            node.stop()
        elif key == 'q':
            node.stop()
            rclpy.shutdown()
            break


if __name__ == '__main__':
    main()
    