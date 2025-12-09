#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

HAVE_HARDWARE = False
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo as adafruit_servo
    HAVE_HARDWARE = True
except Exception as e:
    print(f"[servo_pca9685] Hardware libs not available, running in dummy mode: {e}")


class ServoPCA9685(Node):
    def __init__(self):
        super().__init__('servo_driver')

        # Parameters
        self.declare_parameter('i2c_addr', 0x40)
        self.declare_parameter('channels', [0, 1])        # [pan, tilt]
        self.declare_parameter('startup_deg', [90.0, 90.0])
        self.declare_parameter('neutral_deg', [90.0, 90.0])
        self.declare_parameter('min_deg', [45.0, 45.0])
        self.declare_parameter('max_deg', [175.0, 175.0])

        i2c_addr     = self.get_parameter('i2c_addr').value
        channels     = self.get_parameter('channels').value
        startup_deg  = self.get_parameter('startup_deg').value
        neutral_deg  = self.get_parameter('neutral_deg').value
        min_deg      = self.get_parameter('min_deg').value
        max_deg      = self.get_parameter('max_deg').value

        if len(channels) != 2:
            raise RuntimeError("servo_pca9685: 'channels' param must have length 2 (pan, tilt)")

        self.pan_ch, self.tilt_ch = channels
        self.pan_neutral_deg, self.tilt_neutral_deg = neutral_deg
        self.pan_min_deg, self.tilt_min_deg = min_deg
        self.pan_max_deg, self.tilt_max_deg = max_deg

        self.get_logger().info(
            f"ServoPCA9685 starting on channels {channels} at 0x{i2c_addr:02X} "
            f"(hardware={'yes' if HAVE_HARDWARE else 'no/dummy'})"
        )

        self.pan_servo = None
        self.tilt_servo = None

        if HAVE_HARDWARE:
            try:
                # Same as in servo_test.py
                i2c = busio.I2C(board.SCL, board.SDA)
                pca = PCA9685(i2c, address=i2c_addr)
                pca.frequency = 50

                self.pca = pca
                self.pan_servo = adafruit_servo.Servo(
                    pca.channels[self.pan_ch], min_pulse=500, max_pulse=2500
                )
                self.tilt_servo = adafruit_servo.Servo(
                    pca.channels[self.tilt_ch], min_pulse=500, max_pulse=2500
                )

                # Center
                self.pan_servo.angle = startup_deg[0]
                self.tilt_servo.angle = startup_deg[1]
                time.sleep(0.3)

                self.get_logger().info("Servo hardware initialized successfully.")
            except Exception as e:
                self.get_logger().error(f"Failed to init PCA9685/servos: {e}")
                self.pan_servo = None
                self.tilt_servo = None

        # Subscribe to head command
        self.sub = self.create_subscription(
            Float64MultiArray,
            '/head_cmd',
            self.head_cmd_callback,
            10
        )

    def head_cmd_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn("Received /head_cmd with fewer than 2 elements, ignoring.")
            return

        yaw_cmd = msg.data[0]    # radians, left/right
        pitch_cmd = msg.data[1]  # radians, up/down

        # radians -> degrees
        pan_deg = yaw_cmd * 180.0 / math.pi + self.pan_neutral_deg
        tilt_deg = pitch_cmd * 180.0 / math.pi + self.tilt_neutral_deg

        # clamp
        pan_deg = max(self.pan_min_deg, min(self.pan_max_deg, pan_deg))
        tilt_deg = max(self.tilt_min_deg, min(self.tilt_max_deg, tilt_deg))

        if self.pan_servo is not None and self.tilt_servo is not None:
            try:
                self.pan_servo.angle = pan_deg
                self.tilt_servo.angle = tilt_deg
            except Exception as e:
                self.get_logger().error(f"Failed to move servos: {e}")
        else:
            self.get_logger().info(
                f"[dummy] /head_cmd -> yaw={yaw_cmd:.3f} rad, pitch={pitch_cmd:.3f} rad "
                f"→ pan_deg={pan_deg:.1f}, tilt_deg={tilt_deg:.1f}"
            )

    def destroy_node(self):
        # Optional: center or deinit PCA on shutdown
        try:
            if self.pan_servo is not None:
                self.pan_servo.angle = self.pan_neutral_deg
            if self.tilt_servo is not None:
                self.tilt_servo.angle = self.tilt_neutral_deg
            if hasattr(self, "pca") and self.pca is not None:
                self.pca.deinit()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ServoPCA9685()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

