#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class VideoRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',   # << Change topic if needed
            self.image_callback,
            qos_profile
        )
        
        self.bridge = CvBridge()
        self.out = None

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        if self.out is None:
            h, w, _ = frame.shape
            fps = 30
            self.out = cv2.VideoWriter(
                'camera_record.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (w, h)
            )
            self.get_logger().info(f"Recording started at {w}x{h}@{fps}fps")

        self.out.write(frame)

    def destroy_node(self):
        if self.out:
            self.out.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
