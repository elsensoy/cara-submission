#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from vision_msgs.msg import Detection2DArray  # adjust if your msg type is different


class FaceIDCenterBridge(Node):
    """
    Bridge from face_id_node detections to a simple face-center Point.

    Subscribes:
      - detections_topic : Detection2DArray from face_id_node

    Publishes:
      - /faces/primary_center : geometry_msgs/Point
          x, y = face center (pixels)
          z    = confidence (0..1)
    """

    def __init__(self):
        super().__init__("face_id_center_bridge")

        self.declare_parameter("detections_topic", "/face_id/detections")
        self.declare_parameter("output_topic", "/faces/primary_center")
        self.declare_parameter("min_confidence", 0.3)

        det_topic = self.get_parameter("detections_topic").value
        out_topic = self.get_parameter("output_topic").value
        self.min_conf = float(self.get_parameter("min_confidence").value)

        self.sub = self.create_subscription(
            Detection2DArray,
            det_topic,
            self.detections_callback,
            10,
        )
        self.pub = self.create_publisher(Point, out_topic, 10)

        self.get_logger().info(
            f"FaceIDCenterBridge listening on {det_topic}, publishing {out_topic}"
        )

    def detections_callback(self, msg: Detection2DArray):
        if not msg.detections:
            return

        # pick detection with highest score
        best_det = None
        best_score = -math.inf

        for det in msg.detections:
            if not det.results:
                continue
            score = det.results[0].hypothesis.score
            if score > best_score:
                best_score = score
                best_det = det

        if best_det is None or best_score < self.min_conf:
            return

        cx = best_det.bbox.center.x
        cy = best_det.bbox.center.y

        pt = Point()
        pt.x = float(cx)
        pt.y = float(cy)
        pt.z = float(best_score)
        self.pub.publish(pt)


def main(args=None):
    rclpy.init(args=args)
    node = FaceIDCenterBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
