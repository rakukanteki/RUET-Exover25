#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
import cv2
import os

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('cam_record_node')

        # Define a QoS profile that matches most camera publishers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create subscriber
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile)

        self.bridge = CvBridge()

        # Prepare video writer (output file)
        output_dir = os.path.expanduser('~/ros2_camera_recordings')
        os.makedirs(output_dir, exist_ok=True)
        self.output_path = os.path.join(output_dir, 'camera_recording.avi')

        self.video_writer = None
        self.get_logger().info('Camera subscriber/recorder node started with BEST_EFFORT QoS!')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if self.video_writer is None:
                height, width, _ = cv_image.shape
                self.video_writer = cv2.VideoWriter(
                    self.output_path,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    30.0,
                    (width, height)
                )

            cv2.imshow("Camera Feed", cv_image)
            cv2.waitKey(1)

            self.video_writer.write(cv_image)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def destroy_node(self):
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
