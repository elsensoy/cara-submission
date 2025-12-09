#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray


class ModelGazeMapper(Node):
    """
    Model-based gaze mapper for Cara:

    Subscribes:
      - /faces/primary_center : geometry_msgs/Point
          x, y = face center in pixel coordinates

    Publishes:
      - /head_cmd : std_msgs/Float64MultiArray
          data[0] = yaw (rad), data[1] = pitch (rad)

    Features:
      - g-h (α-β) filter to smooth + predict face position
      - exponential (DS) gaze dynamics: dθ/dt = -(θ - θ_target)/τ
      - angular speed limit (max_rate_rad_s)
      - automatic return-to-neutral if no face for timeout
    """

    def __init__(self):
        super().__init__("model_gaze_mapper")

        # --- Parameters ---
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("fov_deg", [62.0, 49.0])  # [horizontal, vertical]
        self.declare_parameter("neutral_yaw_rad", 0.0)   # facing straight ahead
        self.declare_parameter("neutral_pitch_rad", 0.0) # slight down = positive or negative  
        self.declare_parameter("yaw_limits", [-1.22, 1.22])   #  about ±70°
        self.declare_parameter("pitch_limits", [-0.17, 0.61]) # -10°..+35°
        self.declare_parameter("tau_yaw", 0.35)          # time constants (s)
        self.declare_parameter("tau_pitch", 0.35)
        self.declare_parameter("max_rate_rad_s", 3.0)    # about 170°/s
        self.declare_parameter("no_face_timeout", 2.0)   # seconds
        self.declare_parameter("gh_g", 0.5)              # g-h filter gains (0..1)
        self.declare_parameter("gh_h", 0.1)
        self.declare_parameter("min_confidence", 0.0)    # if you encode conf in Point.z
        self.declare_parameter("deadband_norm", 0.02)    # normalized error deadband [-1..1]

        # read params
        self.W = self.get_parameter("image_width").value
        self.H = self.get_parameter("image_height").value
        fov_deg = self.get_parameter("fov_deg").value
        self.fov_h = math.radians(fov_deg[0])
        self.fov_v = math.radians(fov_deg[1])

        self.neutral_yaw = self.get_parameter("neutral_yaw_rad").value
        self.neutral_pitch = self.get_parameter("neutral_pitch_rad").value

        yaw_limits = self.get_parameter("yaw_limits").value
        pitch_limits = self.get_parameter("pitch_limits").value
        self.yaw_min, self.yaw_max = yaw_limits[0], yaw_limits[1]
        self.pitch_min, self.pitch_max = pitch_limits[0], pitch_limits[1]

        self.tau_yaw = self.get_parameter("tau_yaw").value
        self.tau_pitch = self.get_parameter("tau_pitch").value
        self.max_rate = self.get_parameter("max_rate_rad_s").value
        self.no_face_timeout = self.get_parameter("no_face_timeout").value

        self.gh_g = self.get_parameter("gh_g").value
        self.gh_h = self.get_parameter("gh_h").value
        self.min_conf = self.get_parameter("min_confidence").value
        self.db_norm = self.get_parameter("deadband_norm").value

        # --- Internal state ---
        # current "commanded" head orientation (what we believe we’re asking for)
        self.yaw = self.neutral_yaw
        self.pitch = self.neutral_pitch

        # target orientation (from face)
        self.yaw_target = self.neutral_yaw
        self.pitch_target = self.neutral_pitch

        # g-h filter state for face position [x, y, vx, vy]
        self.face_x = None
        self.face_y = None
        self.vx = 0.0
        self.vy = 0.0

        self.last_face_time = 0.0
        self.prev_time = time.time()

        # --- ROS I/O ---
        self.sub_face = self.create_subscription(
            Point,
            "/faces/primary_center",
            self.face_callback,
            10,
        )
        self.pub_cmd = self.create_publisher(
            Float64MultiArray,
            "/head_cmd",
            10,
        )

        # main timer at 60 Hz
        self.timer = self.create_timer(1.0 / 60.0, self.update)

        self.get_logger().info("ModelGazeMapper initialized (model-based gaze tracking).")

    @staticmethod
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    # --- Callbacks ---
    def face_callback(self, msg: Point):
        """
        Face detector callback: msg.x, msg.y = pixel coords.
        Interpret msg.z as optional 'confidence' (0..1).
        """
        conf = msg.z if not math.isnan(msg.z) else 1.0
        if conf < self.min_conf:
            # ignore very low-confidence detections
            return

        t = time.time()
        dt = max(1e-3, t - self.prev_time)

        z_x, z_y = msg.x, msg.y

        if self.face_x is None:
            # first measurement: initialize state
            self.face_x = z_x
            self.face_y = z_y
            self.vx = 0.0
            self.vy = 0.0
        else:
            # ----- g-h (alpha-beta) filter -----
            # predict
            x_pred = self.face_x + self.vx * dt
            y_pred = self.face_y + self.vy * dt

            # innovation
            r_x = z_x - x_pred
            r_y = z_y - y_pred

            g = self.gh_g
            h = self.gh_h

            # update position
            self.face_x = x_pred + g * r_x
            self.face_y = y_pred + g * r_y

            # update velocity
            self.vx = self.vx + (h * r_x / dt)
            self.vy = self.vy + (h * r_y / dt)

        # record last time we saw a face
        self.last_face_time = t

    # --- Main update loop ---
    def update(self):
        t = time.time()
        dt = max(1e-3, t - self.prev_time)
        self.prev_time = t

        has_face = (self.face_x is not None) and ((t - self.last_face_time) < self.no_face_timeout)

        if has_face:
            # use predicted face location (one step ahead) for smoother behavior
            x_pred = self.face_x + self.vx * dt
            y_pred = self.face_y + self.vy * dt

            # clamp within image just in case
            x_pred = self.clamp(x_pred, 0.0, float(self.W - 1))
            y_pred = self.clamp(y_pred, 0.0, float(self.H - 1))

            # compute normalized offsets: ex,ey in [-1,1]
            ex = (x_pred - self.W / 2.0) / (self.W / 2.0)
            ey = (y_pred - self.H / 2.0) / (self.H / 2.0)

            # deadband to avoid noise-driven jitter
            if abs(ex) < self.db_norm:
                ex = 0.0
            if abs(ey) < self.db_norm:
                ey = 0.0

            # map pixel-based error -> angle offsets using camera FOV
            yaw_offset = -ex * (self.fov_h / 2.0)   # right pixel -> negative yaw
            pitch_offset = ey * (self.fov_v / 2.0)  # down pixel -> positive pitch

            # desired absolute orientation
            self.yaw_target = self.neutral_yaw + yaw_offset
            self.pitch_target = self.neutral_pitch + pitch_offset

        else:
            # no face recently: slowly drift back to neutral
            self.yaw_target = self.neutral_yaw
            self.pitch_target = self.neutral_pitch

        # --- Dynamical system gaze: exponential convergence ---
        # dθ/dt = -(θ - θ_target)/τ
        dyaw_dt = -(self.yaw - self.yaw_target) / max(self.tau_yaw, 1e-3)
        dpitch_dt = -(self.pitch - self.pitch_target) / max(self.tau_pitch, 1e-3)

        # limit angular speed
        max_step = self.max_rate * dt
        dyaw = self.clamp(dyaw_dt * dt, -max_step, max_step)
        dpitch = self.clamp(dpitch_dt * dt, -max_step, max_step)

        # integrate
        self.yaw += dyaw
        self.pitch += dpitch

        # clamp to mechanical limits
        self.yaw = self.clamp(self.yaw, self.yaw_min, self.yaw_max)
        self.pitch = self.clamp(self.pitch, self.pitch_min, self.pitch_max)

        # publish command
        msg = Float64MultiArray()
        msg.data = [self.yaw, self.pitch]
        self.pub_cmd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ModelGazeMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
