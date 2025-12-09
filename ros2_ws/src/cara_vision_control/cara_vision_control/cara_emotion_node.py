#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import torch
import cv2

from cara_vision_control.cara_emotion_core import PersonalizedEmotionViT, InteractiveLearningSystem

class CaraEmotionNode(Node):
    def __init__(self):
        super().__init__('cara_emotion_node')
        self.bridge = CvBridge()
        
        # Load Model
        self.get_logger().info("Loading ViT Model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PersonalizedEmotionViT().to(self.device)
        self.learning_system = InteractiveLearningSystem(self.model)
        self.get_logger().info(f"Model loaded on {self.device}")

        # Subscribers
        self.sub = self.create_subscription(Image, '/cara/face_crop', self.process_face, 10)
        self.sub_feedback = self.create_subscription(String, '/cara/feedback', self.handle_feedback, 10)
        
        # NEW: Training Trigger
        self.sub_train = self.create_subscription(Bool, '/cara/train', self.handle_train, 10)

        # Publishers
        self.pub_emotion = self.create_publisher(String, '/cara/emotion', 10)
        
        self.current_frame = None
        self.is_training = False

    def process_face(self, msg):
        if self.is_training: return # Don't process while training

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_frame = cv_image
            
            pixel_values = self.learning_system.preprocess(cv_image).to(self.device)
            result = self.model.predict(pixel_values)
            
            emotion_str = f"{result['primary_emotion']} ({result['confidence']:.2f})"
            self.pub_emotion.publish(String(data=emotion_str))
            
        except Exception as e:
            pass # Ignore minor glitches

    def handle_feedback(self, msg):
        label = msg.data.lower()
        if self.current_frame is not None and label in self.model.emotion_names:
            self.learning_system.save_labeled_sample(self.current_frame, label)
            self.get_logger().info(f"SAVED: {label}")
        else:
            self.get_logger().warn("Cannot save: No frame or invalid label")

    def handle_train(self, msg):
        if msg.data: # If true received
            self.is_training = True
            self.get_logger().info("--- STARTING TRAINING ---")
            
            try:
                # Train for 5 epochs
                success = self.learning_system.update_model(epochs=10, batch_size=2)
                if success:
                    self.get_logger().info("--- TRAINING SUCCESSFUL ---")
                else:
                    self.get_logger().warn("Training skipped (not enough data)")
            except Exception as e:
                self.get_logger().error(f"Training failed: {e}")
            
            self.is_training = False

def main(args=None):
    rclpy.init(args=args)
    node = CaraEmotionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
