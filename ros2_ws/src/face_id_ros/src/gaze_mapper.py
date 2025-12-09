# face_track/src/gaze_mapper.py
import rclpy, math, time
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
#face detector just needs to publish /faces/primary_center
class GazeMapper(Node):
    def __init__(self):
        super().__init__('gaze_mapper')
        self.declare_parameters('', [
            ('fov_deg', [62.0, 49.0]), ('deadband', 0.03),
            ('kp', [1.4,1.2]), ('kd', [0.2,0.2]),
            ('limits', [-1.22,1.22,-0.17,0.61]) # yaw_min,max,pitch_min,max
        ])
        self.fov = [math.radians(d) for d in self.get_parameter('fov_deg').value]
        self.db  = self.get_parameter('deadband').value
        self.kp  = self.get_parameter('kp').value
        self.kd  = self.get_parameter('kd').value
        self.lim = self.get_parameter('limits').value
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/head_controller/commands', 10)
        self.sub = self.create_subscription(Point, '/faces/primary_center', self.cb, 10)
        self.W = 640; self.H = 480
        self.exprev = 0.0; self.eyprev = 0.0; self.tprev = time.time()
        self.yaw = 0.0; self.pitch = 0.1  # neutral slight down

    def clamp(self,x,a,b): return min(max(x,a),b)

    def cb(self, p):
        t = time.time(); dt = max(1e-3, t - self.tprev)
        cx, cy = p.x, p.y
        ex = (cx - self.W/2) / (self.W/2)   # -1..1
        ey = (cy - self.H/2) / (self.H/2)

        dex = (ex - self.exprev)/dt; dey = (ey - self.eyprev)/dt
        # Deadband
        if abs(ex) < self.db: ex = 0.0
        if abs(ey) < self.db: ey = 0.0

        # Map normalized error to angle change using FOV
        yaw_err   = -ex * (self.fov[0]/2)  # right pixel -> negative yaw correction
        pitch_err =  ey * (self.fov[1]/2)  # down pixel  -> positive pitch correction

        yaw_cmd   = self.yaw   + (self.kp[0]*yaw_err   + self.kd[0]*(-dex))
        pitch_cmd = self.pitch + (self.kp[1]*pitch_err + self.kd[1]*(-dey))

        # Limits & gentle rate limiting
        max_rate = math.radians(180)  # rad/s
        self.yaw   += self.clamp(yaw_cmd - self.yaw  , -max_rate*dt, max_rate*dt)
        self.pitch += self.clamp(pitch_cmd - self.pitch, -max_rate*dt, max_rate*dt)

        self.yaw   = self.clamp(self.yaw,   self.lim[0], self.lim[1])
        self.pitch = self.clamp(self.pitch, self.lim[2], self.lim[3])

        msg = Float64MultiArray(); msg.data = [self.yaw, self.pitch]
        self.cmd_pub.publish(msg)

        self.exprev, self.eyprev, self.tprev = ex, ey, t

def main():
    rclpy.init(); rclpy.spin(GazeMapper()); rclpy.shutdown()
if __name__ == '__main__': main()
