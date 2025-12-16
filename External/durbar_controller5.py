#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from space_teams_definitions.srv import String, Float
from geometry_msgs.msg import Point, Quaternion
import math
import time
from space_teams_python.transformations import *

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev



class RoverController(Node):
    def __init__(self):
        super().__init__('RoverController')
        # Service clients
        self.logger_client = self.create_client(String, 'log_message')
        self.steer_client = self.create_client(Float, 'Steer')
        self.accelerator_client = self.create_client(Float, 'Accelerator')
        self.reverse_client = self.create_client(Float, 'Reverse')
        self.brake_client = self.create_client(Float, 'Brake')
        self.core_sampling_client = self.create_client(Float, 'CoreSample')
        self.change_exposure_client = self.create_client(Float, 'ChangeExposure')
        self.change_rgb_freq_client = self.create_client(Float, 'ChangeRGBFreq')
        self.change_depth_freq_client = self.create_client(Float, 'ChangeDepthFreq')



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
        #image subcription
        self.create_subscription(Image, "/camera/image_raw", self.image_callback, qos_profile_sensor_data)
        self.depth_subscription = self.create_subscription(Image,'/camera/depth/image_raw',self.depth_callback,qos_profile_sensor_data)
        self.show_video = False
        self.rock = False
        self.stuck = 0
        self.night = False
        self.center_distance = 99999.0
        self.brightness = 100.0
        # Control state
        self.target_loc_localFrame = None
        self.tolerance = 5.0  # meters
        self.max_speed = 0.5
        self.navigation_active = False
        self.navigation_iterations = 0
        self.initial_move_end_time = None
        self.initial_move_done = False

        # Waypoints
        self.waypoints = None
        self.current_waypoint_idx = None

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('Rover controller is ready. Lets ride on Mars with Team Durbar.')

        #Image
        self.bridge = CvBridge()
        self.show_cam = True
        self.cam_expose = 10.0
        self.last_speed_diff_kph = 15

        self.tm = time.time()


    ## Image_processing  ###########################
    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Night Detection Block ----
        self.brightness = np.mean(gray)  # average brightness 0-255
        if self.brightness < 50 or self.brightness>180:         # threshold (you can tune 50-80 depending on environment)
            self.night = True
        else:
            self.night = False
        # ------------------------------

        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 40, 120)

        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = frame.shape[:2]
        cx_frame = width // 2
        center_dead_zone = 80

        steer_command = "FORWARD"

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000 and area <1500:
                x, y, w, h = cv2.boundingRect(cnt)
                rock_cx = x + w // 2
                rock_cy = y + h // 2
                self.rock = True

                if rock_cy > 420 & rock_cy < 200:
                    continue

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (rock_cx, rock_cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, "Rock", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if area <1200:
                    turn = 0.3
                else:
                    turn = 0.5
                if abs(rock_cx - cx_frame) < center_dead_zone:
                    if rock_cx < cx_frame:
                        steer_command = "TURN RIGHT"
                        self.send_steer_command(turn)
                        time.sleep(0.8)

                    else:
                        steer_command = "TURN LEFT"
                        self.send_steer_command(-turn)
                        time.sleep(0.8)
                else:
                    steer_command = "FORWARD"
            else:
                self.rock = False

        # Visual Display
        if self.show_video:
            cv2.putText(frame, f"Steer: {steer_command}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

            # Display day/night mode
            mode_text = "Night Mode" if self.night else "Day Mode"
            cv2.putText(frame, mode_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 255), 3)

            cv2.imshow("Mars Rock Detection", frame)
            #cv2.imshow("Edges", edges)
            cv2.waitKey(1)
    def depth_callback(self, msg):
        try:
            # Convert to numpy array (depth map)
            depth_image = self.bridge.imgmsg_to_cv2(msg)
            
            # You can access distance values directly from the image
            # For example, to get the distance at the center:
            height, width = depth_image.shape
            self.center_distance = depth_image[height//2, width//2]
            self.get_logger().info(f'Center distance: {self.center_distance} meters')
            
            # Visualize the depth map
            # Note: Need to normalize for visualization
            # depth_colormap = cv2.applyColorMap(
            #     cv2.convertScaleAbs(depth_image, alpha=0.03), 
            #     cv2.COLORMAP_JET
            # )
            # cv2.imshow('Depth Camera', depth_colormap)
            # cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')


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
        self.send_accelerator_command(1.0)
        time.sleep(2)
        self.state = "Driving"
        
        

        

    def log_message(self, message):
        request = String.Request()
        request.data = message
        future = self.logger_client.call_async(request)
        return future

    def send_steer_command(self, steer_value):
        request = Float.Request()
        request.data = max(-1.0, min(1.0, steer_value))
        return self.steer_client.call_async(request)

    def send_accelerator_command(self, accel_value):
        request = Float.Request()
        request.data = max(0.0, min(1.0, accel_value))
        return self.accelerator_client.call_async(request)

    def send_reverse_command(self, reverse_value):
        request = Float.Request()
        request.data = max(0.0, min(1.0, reverse_value))
        return self.reverse_client.call_async(request)

    def send_brake_command(self, brake_value):
        request = Float.Request()
        request.data = max(0.0, min(1.0, brake_value))
        return self.brake_client.call_async(request)
    
    def send_core_sampling_command(self):
        self.state = "Sampling"
        request = Float.Request()
        request.data = 0.0
        return self.core_sampling_client.call_async(request)
    
    def calculate_direction_to_target(self, current_loc_localFrame: npt.NDArray, 
                                      target_loc_localFrame: npt.NDArray) -> npt.NDArray:
        return normalize(target_loc_localFrame - current_loc_localFrame)
    
    def calculate_error_angle_sign(self, vec1: npt.NDArray, vec2: npt.NDArray) -> float:
        return 1.0 if np.dot(np.cross(vec1, vec2), np.array([0.0, 0.0, 1.0])) > 0.0 else -1.0
    
    def error_angle_arctan(self, vec1, vec2):
        up = normalize(np.cross(np.cross(vec2, vec1), vec2))
        x = np.dot(vec1, vec2)
        y = np.dot(vec1, up)
        return np.arctan2(y, x)
    
    def calculate_pointing_error_angle(self, current_loc_localFrame: npt.NDArray, 
                                       target_loc_localFrame: npt.NDArray, current_rot_localFrame: Quat) -> float:
        m = current_rot_localFrame.to_matrix()
        forward = m[:, 0]
        forward = normalize(np.array([forward[0], forward[1], 0.0]))

        target_direction = self.calculate_direction_to_target(current_loc_localFrame, target_loc_localFrame)
        target_direction = normalize(np.array([target_direction[0], target_direction[1], 0.0]))

        # error_angle = self.error_angle_arctan(forward, target_direction)

        error_angle = np.arccos(np.dot(forward, target_direction))
        error_angle_dir = self.calculate_error_angle_sign(target_direction, forward)
        return error_angle_dir * error_angle

    def calculate_distance_to_target(self, current_loc_localFrame: npt.NDArray, target_loc_localFrame: npt.NDArray):
        return np.linalg.norm(target_loc_localFrame - current_loc_localFrame)

    def calculate_speed_difference(self, current_vel_localFrame: npt.NDArray, target_speed_kph: float) -> float:
        return mps_to_kph(kph_to_mps(target_speed_kph) - np.linalg.norm(current_vel_localFrame))

    def start_navigation(self, target_loc_localFrame):
        self.target_loc_localFrame = target_loc_localFrame
        self.navigation_active = True
        self.navigation_iterations = 0
        self.initial_move_done = False
        self.initial_move_end_time = time.time() + 10.0
        self.log_message(f"Starting navigation to target: ({target_loc_localFrame[0]:.2f}, {target_loc_localFrame[1]:.2f})")
        self.send_accelerator_command(0.2)


    def change_exposure(self, exposure_level: float):
        request = Float.Request()
        request.data = exposure_level
        return self.change_exposure_client.call_async(request)
    
    def change_rgb_freq(self, rgb_freq: float):
        request = Float.Request()
        request.data = rgb_freq
        return self.change_rgb_freq_client.call_async(request)
    
    def change_depth_freq(self, depth_freq: float):
        request = Float.Request()
        request.data = depth_freq
        return self.change_depth_freq_client.call_async(request)

    def timer_callback(self):
        if not self.navigation_active:
            return

        # Initial move forward for 4 seconds
        if not self.initial_move_done and self.initial_move_end_time is not None:
            if time.time() < self.initial_move_end_time:
                return
            self.send_accelerator_command(0.0)
            self.initial_move_done = True
        
        # Navigation logic
        if self.current_location_localFrame is None or self.current_rotation_localFrame is None:
            self.get_logger().info("Waiting for location/rotation update...")
            return

        # Get location
        current_x = float(self.current_location_localFrame.x)
        current_y = float(self.current_location_localFrame.y)
        current_z = float(self.current_location_localFrame.z)
        current_loc_localFrame = np.array([current_x, current_y, current_z])

        # Get velocity
        current_vx = float(self.current_velocity_localFrame.x)
        current_vy = float(self.current_velocity_localFrame.y)
        current_vz = float(self.current_velocity_localFrame.z)
        current_vel_localFrame = np.array([current_vx, current_vy, current_vz])

        # Get rotation
        qx = float(self.current_rotation_localFrame.x)
        qy = float(self.current_rotation_localFrame.y)
        qz = float(self.current_rotation_localFrame.z)
        qw = float(self.current_rotation_localFrame.w)
        current_rot_localFrame = Quat(qw, qx, qy, qz)

        # Distance to target
        distance = self.calculate_distance_to_target(current_loc_localFrame, self.target_loc_localFrame)
        if distance < self.tolerance:
            self.send_brake_command(1.0)
            self.send_steer_command(0.0)
            self.send_accelerator_command(0.0)
            self.log_message(f"Target reached! Beginning core sampling at position: ({current_x:.2f}, {current_y:.2f})")
            self.send_core_sampling_command()
            print("core sampling done")

            if self.current_waypoint_idx == len(self.waypoints) - 1:
                self.navigation_active = False
                self.log_message("Navigation completed by Team Drubar: all waypoints reached and all core samples collected.")
            else:
                self.current_waypoint_idx += 1
                self.target_loc_localFrame = self.waypoints[self.current_waypoint_idx]
                next_loc = f"({self.target_loc_localFrame[0]:.2f}, {self.target_loc_localFrame[1]:.2f})"
                self.log_message(f"After sampling, moving to next waypoint at: {next_loc}")
            return
        
        # Velocity error
        #speed_limit_kph = 40

        if self.night:
            speed_limit_kph = min(25,self.center_distance/1000 + 18)
        elif self.rock:
            speed_limit_kph = min(25,self.center_distance/2000 + 20)
        else:
            if self.current_waypoint_idx == 0:
                speed_limit_kph = min(35,self.center_distance/1000 + 22) 
            else:
                speed_limit_kph = min(30,self.center_distance/1000 + 22)
        
        
        # if self.night and self.center_distance<5000:
        #     speed_limit_kph = 15
        # elif self.night:
        #     speed_limit_kph = 18    
        # elif self.rock:
        #     speed_limit_kph = 20 
        # elif self.center_distance >10000:
        #     speed_limit_kph = 28
        # else:
        #     speed_limit_kph = 26

        print("speed set to ", speed_limit_kph)
        
        speed_diff_kph = self.calculate_speed_difference(current_vel_localFrame, speed_limit_kph)  # target - current
        accel_factor = remap_clamp(0.0, speed_limit_kph, 0.0, 1.0, speed_diff_kph)  # 1 if not moving, 0 if too fast
        brake_factor = 1.0 - remap_clamp(-speed_limit_kph, 0.0, 0.0, 1.0, speed_diff_kph)  # 0 if <= speed limit, 1 if 2x over

        # Heading error
        db_heading = np.deg2rad(3.0)  # deadband for heading alignment
        heading_error = self.calculate_pointing_error_angle(current_loc_localFrame, self.target_loc_localFrame, 
                                                            current_rot_localFrame)
        
        # Steering
        steer_command = remap_clamp(-0.25 * np.pi, 0.25 * np.pi, -1.0, 1.0, heading_error)
        if abs(heading_error) < db_heading:
            steer_command = 0.0
        steer_gain = 1.0
        actual_steer_command = -steer_gain * steer_command
        
        # Acceleration
        accel_gain = 2.0
        accel_command = accel_gain * remap_clamp(0.0, 1.0, accel_factor, accel_factor * 0.5, abs(steer_command))

        # Braking
        # If brake, brake_command > 0.5 results in braking (i.e., boolean behavior)
        # If reverse, float value between 0 and 1 is passed, acts as a gradual deceleration
        brake_gain = 1.0
        brake_command = brake_gain * brake_factor

        self.send_steer_command(actual_steer_command)
        self.send_accelerator_command(accel_command)
        self.send_reverse_command(brake_command)  # Send brake command as a float (reverse)
        self.send_brake_command(0.0)

        
        #detect Rock 1st impact
        # if speed_diff_kph > self.last_speed_diff_kph+15 and self.state == "Driving":
        #     print("Impact detected")
        #     self.send_brake_command(1.0)
        #     time.sleep(1)
        #     self.send_brake_command(0.0)
        
        # self.last_speed_diff_kph = speed_diff_kph
        #Stuck by Rock
        if self.center_distance <100 and self.night:
            print('Breaking for close rock')
            self.send_brake_command(1.0)
            time.sleep(1)
            self.send_brake_command(0.0)
        elif self.center_distance<3000 and self.night:
            self.send_steer_command(0.6)
            #self.send_accelerator_command(0.6)
            time.sleep(0.8)

        if speed_diff_kph>speed_limit_kph-0.25 and accel_command>0.9 and brake_command == 0.0 and self.state == "Driving":
            self.stuck +=1
            if self.stuck ==1:
                time.sleep(0.5)
                return
            print('stuct no:',self.stuck)
            #avoid and go from right side
            self.send_accelerator_command(0.0)
            self.send_steer_command(0.0)
            self.send_reverse_command(1.0)
            time.sleep(2)
            self.send_reverse_command(0.0)
            self.send_brake_command(1.0)
            time.sleep(1)
            self.send_brake_command(0.0)
            self.send_accelerator_command(1.0)
            self.send_steer_command(0.8)
            time.sleep(3)
        else: 
            self.stuck =0
        if self.stuck >=3:
            print("stuck over rock")
            #weggle out
            self.send_accelerator_command(1.0)
            self.send_reverse_command(0.0)
            time.sleep(1)
            self.send_reverse_command(0.7)
            self.send_accelerator_command(0.0)
            for i in range(5):
                self.send_steer_command(1)
                time.sleep(0.8)
                self.send_steer_command(-1)
                time.sleep(0.8)
            print("weggled to clear out")
        
        ## Camera Exposure Change
        #print("Brightness: ", self.brightness)
        #self.change_exposure(self.brightness/17+5)
        if time.time() > self.tm +3:

            if self.brightness < 50:
                self.cam_expose = max(5.0, self.cam_expose-0.5) 
                self.change_exposure(self.cam_expose)

            elif self.brightness >150:
                self.cam_expose = min(20.0, self.cam_expose+1)
                self.change_exposure(self.cam_expose) 
            #print("set Exposure: ", self.cam_expose)
            self.tm = time.time()

        
        #Stuck over Rock





        # self.send_brake_command(brake_command)  # Send brake command as a bool

        # Print commands for debugging:
        # if self.navigation_iterations % 10 == 0:
        #     self.log_message(
        #         f"Position: ({current_x:.2f}, {current_y:.2f}), "
        #         f"Distance: {distance:.2f}, "
        #         f"Heading error: {math.degrees(heading_error):.1f} deg, "
        #         f"Steer: {steer_command:.2f}, "
        #         f"Accel: {accel_command:.2f}"
        #     )
        self.navigation_iterations += 1


def main(args=None):
    rclpy.init(args=args)
    rover_controller = RoverController()

    # Test waypoint:
    # waypoint_marsframe = np.array([2193073.87847882, 743984.99629174, -2485667.65565136])
    # waypoint_localframe = np.array([22.0285988, 60.41062071, -4.50449595])

    # Test multiple waypoints:
    """ waypoints_localFrame = [
        np.array([-54.31019727, 191.84449903, -19.54598818]),
        np.array([111.24089259, 427.56166121, -54.81398767]),
        np.array([-349.10709106,  558.01869306, -68.71836618]),
        np.array([1281.36380015, 1647.50529027, -39.35361376]),
        np.array([654.62948546, 1186.61595725, -48.4778713]),
        np.array([-606.74433428, 332.44253661, -20.41775233]),
        np.array([1349.86835614, 1047.23075279, -46.89420337]),
        np.array([231.41034119, -858.69285702, -63.3150879]),
        np.array([45.56236659, 921.05755228, -65.76412603]),
        np.array([1960.32237043, 1423.88737415, -89.97019481]),
        np.array([1098.14343253, 1987.40560248, -45.45757708]),
        np.array([10.15805303, -752.47151722, -68.15878792]),
        np.array([1532.81368707, 1255.13690297, -48.47378546]),
        np.array([-561.74721182, 28.52558036, -29.92751284]),
        np.array([1958.28017108, 1381.24222162, -76.75680176]),
        np.array([-1025.65838348, 274.39353778, -76.31593519]),
        np.array([410.36797363, -956.93367913, -84.31272572]),
        np.array([247.67056987, 579.07900331, -75.04176954]),
        np.array([345.53461945, 1330.35839896, -73.5301525]),
        np.array([1073.3882324, 1613.84763245, -50.72357905])
    ] """
    # Original waypoints (XYZ)
    waypoints_localFrame = np.array([
        [-54.31019727, 191.84449903, -19.54598818],
        [111.24089259, 427.56166121, -54.81398767],
        [-349.10709106, 558.01869306, -68.71836618],
        [1281.36380015, 1647.50529027, -39.35361376],
        [654.62948546, 1186.61595725, -48.4778713],
        [-606.74433428, 332.44253661, -20.41775233],
        [1349.86835614, 1047.23075279, -46.89420337],
        [231.41034119, -858.69285702, -63.3150879],
        [45.56236659, 921.05755228, -65.76412603],
        [1960.32237043, 1423.88737415, -89.97019481],
        [1098.14343253, 1987.40560248, -45.45757708],
        [10.15805303, -752.47151722, -68.15878792],
        [1532.81368707, 1255.13690297, -48.47378546],
        [-561.74721182, 28.52558036, -29.92751284],
        [1958.28017108, 1381.24222162, -76.75680176],
        [-1025.65838348, 274.39353778, -76.31593519],
        [410.36797363, -956.93367913, -84.31272572],
        [247.67056987, 579.07900331, -75.04176954],
        [345.53461945, 1330.35839896, -73.5301525],
        [1073.3882324, 1613.84763245, -50.72357905]
    ])

# ---------- Reorder waypoints using Nearest Neighbor ----------
    """ ordered_waypoints = [waypoints_localFrame[0]]
    remaining = waypoints_localFrame[1:].tolist()

    while remaining:
        distances = cdist([ordered_waypoints[-1]], remaining)
        nearest_idx = np.argmin(distances)
        ordered_waypoints.append(remaining.pop(nearest_idx))

    waypoints_localFrame = np.array(ordered_waypoints) """
    index = [16,7,11,13,15,5,2,0,1,17,8,18,4,19,3,10,9,14,12,6]
    #index = [16,7,11,13,15,5,2,0,1,17,8,18,4,19,3,10,9,14,12,6]
    ord_waypoints = waypoints_localFrame[index]
    waypoints_localFrame = ord_waypoints

    # Wait for initial location and rotation
    while rclpy.ok():
        if rover_controller.current_location_localFrame is not None and rover_controller.current_rotation_localFrame is not None:
            break
        rover_controller.get_logger().info('Waiting for initial location and rotation...')
        rclpy.spin_once(rover_controller, timeout_sec=0.5)

    current_x = rover_controller.current_location_localFrame.x
    current_y = rover_controller.current_location_localFrame.y

    rover_controller.waypoints = waypoints_localFrame
    rover_controller.current_waypoint_idx = 0
    
    # Camera framerate adjustment
    # Set to match defaults but you can adjust here as needed.
    new_rgb_freq = 15.0
    new_depth_freq = 5.0

    rover_controller.log_message(f"Setting RGB frequency to {new_rgb_freq} Hz and Depth frequency to {new_depth_freq} Hz.")
    rover_controller.get_logger().info(f"Setting RGB frequency to {new_rgb_freq} Hz and Depth frequency to {new_depth_freq} Hz.")
    rover_controller.change_rgb_freq(new_rgb_freq)
    rover_controller.change_depth_freq(new_depth_freq)

    rover_controller.log_message(
        f"Starting navigation: moving from ({current_x:.2f}, {current_y:.2f}) to ({waypoints_localFrame[0][0]:.2f}, {waypoints_localFrame[0][1]:.2f})"
    )
    
    rover_controller.start_navigation(waypoints_localFrame[0])

    try:
        rclpy.spin(rover_controller)
    finally:
        rover_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
