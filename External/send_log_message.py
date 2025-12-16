#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from space_teams_definitions.srv import String
import time


class LogMessageClient(Node):
    def __init__(self):
        super().__init__('log_message_client')
        self.client = self.create_client(String, '/log_message')

        # Wait for the service to be ready
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /log_message service...')

        # Give the service a short moment to fully initialize
        time.sleep(1.0)

        # Prepare and send the request
        self.req = String.Request()
        self.req.data = "Hello from ROS 2 client!"
        self.get_logger().info(f"Sending message: {self.req.data}")

        self.future = self.client.call_async(self.req)
        self.future.add_done_callback(self.callback_response)

    def callback_response(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"Service responded: {response.response}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
        finally:
            # Shutdown cleanly
            self.get_logger().info("Shutting down node.")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = LogMessageClient()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
