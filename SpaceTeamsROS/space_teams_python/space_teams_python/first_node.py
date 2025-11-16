# #!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from space_teams_definitions.srv import String, Float
from geometry_msgs.msg import Point, Quaternion
import math
import time
from space_teams_python.transformations import *
import logging
import os

# -------------------------------
# Configure Python logging
# -------------------------------
log_dir = os.path.expanduser('/mnt/d/rover_competition/SpaceTeamsROS/python_logs')
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'rover_controller.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()  # also print to console
    ]
)

logger = logging.getLogger(__name__)

# -------------------------------
# Rover Controller Node
# -------------------------------
class RoverController(Node):
    def __init__(self):
        super().__init__('RoverController')
        logger.info("Initializing RoverController node...")

        # Service clients
        self.logger_client = self.create_client(String, 'log_message')
        self.steer_client = self.create_client(Float, 'Steer')
        self.accelerator_client = self.create_client(Float, 'Accelerator')
        self.reverse_client = self.create_client(Float, 'Reverse')
        self.brake_client = self.create_client(Float, 'Brake')
        self.core_sampling_client = self.create_client(Float, 'CoreSample')
        self.change_exposure_client = self.create_client(Float, 'ChangeExposure')

        # Topic subscriptions
        self.current_location_marsFrame = None
        self.current_velocity_marsFrame = None
        self.current_rotation_marsFrame = None
        self.current_location_localFrame = None
        self.current_velocity_localFrame = None
        self.current_rotation_localFrame = None
        self.state = "Driving"

        self.create_subscription(Point, 'LocationMarsFrame', self.location_marsFrame_callback, 10)
        self.create_subscription(Point, 'VelocityMarsFrame', self.velocity_marsFrame_callback, 10)
        self.create_subscription(Quaternion, 'RotationMarsFrame', self.rotation_marsFrame_callback, 10)
        self.create_subscription(Point, 'LocationLocalFrame', self.location_localFrame_callback, 10)
        self.create_subscription(Point, 'VelocityLocalFrame', self.velocity_localFrame_callback, 10)
        self.create_subscription(Quaternion, 'RotationLocalFrame', self.rotation_localFrame_callback, 10)
        self.create_subscription(Point, 'CoreSamplingComplete', self.core_sampling_complete_callback, 1)

        # Control state
        self.target_loc_localFrame = None
        self.tolerance = 5.0  # meters
        self.max_speed = 0.5
        self.navigation_active = False
        self.navigation_iterations = 0
        self.initial_move_end_time = None
        self.initial_move_done = False
        self.waypoints = None
        self.current_waypoint_idx = None

        logger.info("RoverController node initialized successfully.")

    # -------------------------------
    # Callbacks
    # -------------------------------
    def location_marsFrame_callback(self, msg):
        self.current_location_marsFrame = msg
    
    def velocity_marsFrame_callback(self, msg):
        self.current_velocity_marsFrame = msg

    def rotation_marsFrame_callback(self, msg):
        self.current_rotation_marsFrame = msg

    def location_localFrame_callback(self, msg):
        self.current_location_localFrame = msg

    def velocity_localFrame_callback(self, msg):
        self.current_velocity_localFrame = msg

    def rotation_localFrame_callback(self, msg):
        self.current_rotation_localFrame = msg
    
    def core_sampling_complete_callback(self, msg):
        self.state = "Driving"
        logger.info("Core sampling complete. State set to Driving.")

    # -------------------------------
    # Service methods
    # -------------------------------
    def log_message(self, message):
        logger.info(f"[SERVICE CALL] log_message -> '{message}'")
        request = String.Request()
        request.data = message
        future = self.logger_client.call_async(request)
        future.add_done_callback(lambda f: logger.info(f"[SERVICE RESPONSE] log_message completed"))
        return future

    def send_accelerator_command(self, accel_value):
        request = Float.Request()
        request.data = max(0.0, min(1.0, accel_value))
        logger.info(f"[SERVICE CALL] Accelerator -> {request.data}")
        future = self.accelerator_client.call_async(request)
        future.add_done_callback(lambda f: logger.info("[SERVICE RESPONSE] Accelerator completed"))
        return future

    def send_steer_command(self, steer_value):
        request = Float.Request()
        request.data = max(-1.0, min(1.0, steer_value))
        logger.info(f"[SERVICE CALL] Steer -> {request.data}")
        future = self.steer_client.call_async(request)
        future.add_done_callback(lambda f: logger.info("[SERVICE RESPONSE] Steer completed"))
        return future

    def send_reverse_command(self, reverse_value):
        request = Float.Request()
        request.data = max(0.0, min(1.0, reverse_value))
        logger.info(f"[SERVICE CALL] Reverse -> {request.data}")
        future = self.reverse_client.call_async(request)
        future.add_done_callback(lambda f: logger.info("[SERVICE RESPONSE] Reverse completed"))
        return future

    def send_brake_command(self, brake_value):
        request = Float.Request()
        request.data = max(0.0, min(1.0, brake_value))
        logger.info(f"[SERVICE CALL] Brake -> {request.data}")
        future = self.brake_client.call_async(request)
        future.add_done_callback(lambda f: logger.info("[SERVICE RESPONSE] Brake completed"))
        return future

    def send_core_sampling_command(self):
        self.state = "Sampling"
        logger.info("Starting core sampling")
        request = Float.Request()
        request.data = 0.0
        future = self.core_sampling_client.call_async(request)
        future.add_done_callback(lambda f: logger.info("[SERVICE RESPONSE] Core sampling completed"))
        return future


# -------------------------------
# Main function
# -------------------------------
def main(args=None):
    rclpy.init(args=args)
    rover_controller = RoverController()

    rover_controller.log_message("Hello")

    # -------------------------------
    # Driving sequence
    # -------------------------------
    rover_controller.log_message("Starting driving sequence...")
    logger.info("Driving sequence started.")
    rover_controller.send_accelerator_command(1.0)
    time.sleep(5)

    rover_controller.send_steer_command(0.5)
    time.sleep(2)
    rover_controller.send_accelerator_command(0.5)
    rover_controller.send_steer_command(-0.5)
    time.sleep(5)

    rover_controller.send_steer_command(0.0)
    rover_controller.send_brake_command(1.0)

    rover_controller.log_message("Driving sequence completed.")
    logger.info("Driving sequence completed.")

    try:
        rclpy.spin(rover_controller)
    finally:
        rover_controller.destroy_node()
        rclpy.shutdown()
        logger.info("RoverController node shutdown.")


if __name__ == '__main__':
    main()
