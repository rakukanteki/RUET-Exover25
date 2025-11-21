#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from space_teams_definitions.srv import String, Float
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time
from space_teams_python.transformations import *


class IntegratedRoverController(Node):
    def __init__(self):
        super().__init__('IntegratedRoverController')
        
        # ===== STUCK DETECTION =====
        self.last_position = None
        self.last_position_time = None
        self.stuck_start_time = None
        self.stuck_detected = False
        self.reverse_start_time = None
        self.reversing_active = False
        self.position_check_interval = 1.0  # Check position every 1 second


        # ===== NAVIGATION SETUP =====
        # Service clients
        self.logger_client = self.create_client(String, 'log_message')
        self.steer_client = self.create_client(Float, 'Steer')
        self.accelerator_client = self.create_client(Float, 'Accelerator')
        self.reverse_client = self.create_client(Float, 'Reverse')
        self.brake_client = self.create_client(Float, 'Brake')
        self.core_sampling_client = self.create_client(Float, 'CoreSample')
        self.change_exposure_client = self.create_client(Float, 'ChangeExposure')

        # Topic subscriptions for navigation
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

        # ===== CAMERA SETUP =====
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.camera_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile)
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_ready = False

        # ===== ROCK DETECTION PARAMETERS =====
        # Trapezoidal ROI parameters (wider at bottom, narrower at top)
        self.ROI_Y1 = 100      # Top of ROI
        self.ROI_Y2 = 460      # Bottom of ROI  
        self.ROI_X1_TOP = 200  # Narrower at top
        self.ROI_X2_TOP = 300
        self.ROI_X1_BOTTOM = 60  # Wider at bottom
        self.ROI_X2_BOTTOM = 440
        
        # Zone boundaries (calculated for bottom width since that's where rocks matter)
        bottom_width = self.ROI_X2_BOTTOM - self.ROI_X1_BOTTOM
        self.left_end = int(self.ROI_X1_BOTTOM + bottom_width * 0.35)
        self.center_end = int(self.ROI_X1_BOTTOM + bottom_width * 0.65)
        
        self.rock_detected = False
        self.rock_zone = None  # "LEFT", "CENTER", "RIGHT"
        self.rock_count = 0

        # ===== AVOIDANCE STATE =====
        self.avoidance_active = False
        self.avoidance_start_time = 0.0
        self.avoidance_duration = 10.0  # seconds
        self.initial_rock_zone = None  # ADD THIS LINE to store the initial zone

        # ===== EXPOSURE CONTROL =====
        self.current_exposure = 1.0  # Default exposure
        self.current_brightness = 0.0
        self.last_exposure_change = 0.0
        self.exposure_change_interval = 2.0  # seconds between exposure checks
        self.brightness_buffer = []

        # ===== LOGGING THROTTLE =====
        self.brightness_log_counter = 0

        # ===== NAVIGATION CONTROL STATE =====
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

        # ===== MAIN TIMER =====
        self.timer = self.create_timer(0.2, self.timer_callback)  # 5 Hz control loop
        self.get_logger().info('Integrated Rover Controller is ready.')

    # ===== NAVIGATION CALLBACKS =====
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

    def check_if_stuck(self, current_loc_localFrame, current_vel_localFrame):
        """Check if rover is stuck and initiate reverse maneuver"""
        current_time = time.time()
        current_speed = np.linalg.norm(current_vel_localFrame)
        
        # Handle reverse maneuver first (highest priority)
        if self.reversing_active:
            reverse_duration = current_time - self.reverse_start_time

            if reverse_duration < 10.0:  # Reverse for 8 seconds total
                # First 5 seconds: Reverse straight
                if reverse_duration < 5.0:
                    self.send_steer_command(0.0)  # Straight reverse
                    self.get_logger().info(f"🔄 Reversing straight... {reverse_duration:.1f}s / 3.0s")
                    self.send_accelerator_command(0.0)
                    self.send_brake_command(0.0)
                    self.send_reverse_command(0.8)
                
                # Next 5 seconds: Reverse while steering right
                elif reverse_duration >= 5.0 and reverse_duration < 7.0:
                    self.send_steer_command(-0.8)  # Full right steer while reversing
                    self.send_accelerator_command(0.8)
                    self.send_brake_command(0.0)
                    self.send_reverse_command(0.0)
                    steer_time = reverse_duration - 5.0
                    self.get_logger().info(f"🔄 Reversing with right steer... {steer_time:.1f}s / 5.0s")
                
                else:
                    self.send_steer_command(1.0)  # Full left steer while reversing
                    self.send_accelerator_command(0.8)
                    self.send_brake_command(0.0)
                    self.send_reverse_command(0.0)
                    steer_time = reverse_duration - 8.0
                    self.get_logger().info(f"🔄 Reversing with left steer... {steer_time:.1f}s / 5.0s")
                
                  # Moderate reverse speed
                
                return True  # Skip normal navigation
            else:
                # Reverse maneuver complete
                self.log_message("✅ Reverse maneuver complete! Resuming normal navigation")
                self.reversing_active = False
                self.stuck_detected = False
                self.stuck_start_time = None
                self.last_position = None  # Reset position tracking
                self.last_position_time = None
                self.send_reverse_command(0.0)  # Stop reversing
                return False    
        
        # Only check for stuck if we're trying to navigate (not during initial phase, not while avoiding)
        if not self.navigation_active or not self.initial_move_done or self.avoidance_active:
            return False

        # Initialize position tracking
        if self.last_position is None or self.last_position_time is None:
            self.last_position = current_loc_localFrame.copy()
            self.last_position_time = current_time
            return False
        
        # Check position every 1 second (not every 0.1 seconds!)
        time_since_last_check = current_time - self.last_position_time
        
        if time_since_last_check >= self.position_check_interval:
            distance_moved = np.linalg.norm(current_loc_localFrame - self.last_position)
            
            # If we moved very little in the last second AND speed is low
            if distance_moved < 1.0 and current_speed < 1.0:  # Less than 1 meter in 1 second
                if self.stuck_start_time is None:
                    self.stuck_start_time = current_time
                    self.get_logger().info(f"🚨 Possible stuck detected - moved {distance_moved:.2f}m in {time_since_last_check:.1f}s")
                else:
                    stuck_duration = current_time - self.stuck_start_time
                    if stuck_duration > 5.0:  # Stuck for 5 seconds
                        self.stuck_detected = True
                        self.reverse_start_time = current_time
                        self.reversing_active = True
                        self.log_message(f"🚨 ROVER STUCK! Moved only {distance_moved:.2f}m in {stuck_duration:.1f}s. Starting reverse maneuver.")
            else:
                # Moving normally - reset stuck detection
                if self.stuck_start_time is not None:
                    self.get_logger().info(f"✅ Moving normally again - {distance_moved:.2f}m in {time_since_last_check:.1f}s")
                self.stuck_start_time = None
                self.stuck_detected = False
            
            # Update position tracking
            self.last_position = current_loc_localFrame.copy()
            self.last_position_time = current_time
        
        return False

    # ===== CAMERA CALLBACK =====
    def image_callback(self, msg):
        """Store the latest camera frame"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_frame = cv_image
            self.frame_ready = True
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    # ===== EXPOSURE CONTROL FUNCTIONS =====
    def calculate_brightness(self, cv_image):
        """Calculate average brightness of the image"""
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            self.current_brightness = brightness
            
            # Store in buffer (keep last 100 values)
            self.brightness_buffer.append(brightness)
            if len(self.brightness_buffer) > 100:
                self.brightness_buffer.pop(0)
                
            return brightness
        except Exception as e:
            self.get_logger().error(f"Error calculating brightness: {e}")
            self.current_brightness = 0.0
            return 0.0

    def get_average_brightness(self):
        """Get average of last 100 brightness values"""
        if not self.brightness_buffer:
            return self.current_brightness
        return np.mean(self.brightness_buffer)

    def adjust_exposure_based_on_brightness(self, cv_image):
        """Adjust camera exposure based on averaged brightness levels"""
        current_time = time.time()
        
        # Only change exposure every 2 seconds
        if current_time - self.last_exposure_change > self.exposure_change_interval:
            
            # Calculate current brightness and update buffer
            self.calculate_brightness(cv_image)
            
            # Use average of last 100 values for decision making
            avg_brightness = self.get_average_brightness()
            
            # Only make decisions if we have sufficient data
            if len(self.brightness_buffer) < 100:  # Wait for minimum samples
                return
            
            # Enhanced exposure rules with multiple levels
            if avg_brightness <= 20:
                required_exposure = -3.0
            elif avg_brightness <= 80:
                required_exposure = 1.0
            elif avg_brightness <= 150:
                required_exposure = 13.0
            elif avg_brightness > 200:
                required_exposure = 14.0
            else:
                # Between 150-200, keep current exposure
                return
            
            # Only change if exposure level is different
            if abs(required_exposure - self.current_exposure) > 0.1:
                self.send_exposure_command(required_exposure)
                self.current_exposure = required_exposure
                self.last_exposure_change = current_time
                
                # Log the change with both current and averaged values
                if avg_brightness <= 20:
                    condition = "VERY DARK"
                elif avg_brightness <= 80:
                    condition = "DARK" 
                elif avg_brightness <= 150:
                    condition = "MEDIUM"
                elif avg_brightness > 200:
                    condition = "VERY BRIGHT"
                else:
                    condition = "BRIGHT"
                self.log_message(f"Brightness: {self.current_brightness:.1f} (Avg: {avg_brightness:.1f}) - {condition} - Setting exposure: {required_exposure}")

    def send_exposure_command(self, exposure_level):
        """Send exposure command to camera"""
        try:
            future = self.change_exposure(exposure_level)
            self.get_logger().info(f"Sent exposure command: {exposure_level}")
        except Exception as e:
            self.get_logger().error(f"Failed to send exposure command: {e}")

    # ===== SERVICE CALL FUNCTIONS =====
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
    
    def change_exposure(self, exposure_level: float):
        request = Float.Request()
        request.data = exposure_level
        return self.change_exposure_client.call_async(request)

    # ===== NAVIGATION CALCULATIONS =====
    def calculate_direction_to_target(self, current_loc_localFrame: np.ndarray, 
                                      target_loc_localFrame: np.ndarray) -> np.ndarray:
        return normalize(target_loc_localFrame - current_loc_localFrame)
    
    def calculate_error_angle_sign(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return 1.0 if np.dot(np.cross(vec1, vec2), np.array([0.0, 0.0, 1.0])) > 0.0 else -1.0
    
    def error_angle_arctan(self, vec1, vec2):
        up = normalize(np.cross(np.cross(vec2, vec1), vec2))
        x = np.dot(vec1, vec2)
        y = np.dot(vec1, up)
        return np.arctan2(y, x)
    
    def calculate_pointing_error_angle(self, current_loc_localFrame: np.ndarray, 
                                       target_loc_localFrame: np.ndarray, current_rot_localFrame: Quat) -> float:
        m = current_rot_localFrame.to_matrix()
        forward = m[:, 0]
        forward = normalize(np.array([forward[0], forward[1], 0.0]))

        target_direction = self.calculate_direction_to_target(current_loc_localFrame, target_loc_localFrame)
        target_direction = normalize(np.array([target_direction[0], target_direction[1], 0.0]))

        error_angle = np.arccos(np.dot(forward, target_direction))
        error_angle_dir = self.calculate_error_angle_sign(target_direction, forward)
        return error_angle_dir * error_angle

    def calculate_distance_to_target(self, current_loc_localFrame: np.ndarray, target_loc_localFrame: np.ndarray):
        return np.linalg.norm(target_loc_localFrame - current_loc_localFrame)

    def calculate_speed_difference(self, current_vel_localFrame: np.ndarray, target_speed_kph: float) -> float:
        return mps_to_kph(kph_to_mps(target_speed_kph) - np.linalg.norm(current_vel_localFrame))

    # ===== ROCK DETECTION =====
    def detect_rocks(self, cv_image):
        """Detect rocks in the ROI and classify their zones"""
        try:
            h, w = cv_image.shape[:2]
            
            # Create trapezoidal ROI mask (wider at bottom, narrower at top)
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Create polygon points for trapezoid
            pts = np.array([
                [self.ROI_X1_TOP, self.ROI_Y1],      # Top-left
                [self.ROI_X2_TOP, self.ROI_Y1],      # Top-right  
                [self.ROI_X2_BOTTOM, self.ROI_Y2],   # Bottom-right
                [self.ROI_X1_BOTTOM, self.ROI_Y2]    # Bottom-left
            ], np.int32)

            # Fill the trapezoid
            cv2.fillPoly(mask, [pts], 255)
            
            # Preprocessing
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(masked_gray)
            
            blurred = cv2.GaussianBlur(masked_gray, (7, 7), 0)
            edges = cv2.Canny(blurred, 20, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Reset rock detection
            self.rock_detected = False
            self.rock_zone = None
            self.rock_count = 0
            
            rock_data = []
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200:  # Minimum area threshold
                    continue
                    
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Distance estimation
                if area > 2000 or cy > int(self.ROI_Y2 - (self.ROI_Y2 - self.ROI_Y1) * 0.25):
                    distance = "CLOSE"
                elif area > 700 or cy > int(self.ROI_Y1 + (self.ROI_Y2 - self.ROI_Y1) * 0.35):
                    distance = "MID"
                else:
                    distance = "FAR"
                
                # Only consider MID rocks for navigation
                if distance != "MID":
                    continue
                    
                # Zone classification
                if cx < self.left_end:
                    zone = "LEFT"
                elif cx < self.center_end:
                    zone = "CENTER"
                else:
                    zone = "RIGHT"
                    
                rock_data.append({
                    "area": area, 
                    "bbox": (x, y, w_box, h_box), 
                    "centroid": (cx, cy), 
                    "distance": distance,
                    "zone": zone
                })
                
                # Mark rock detected
                self.rock_detected = True
                self.rock_zone = zone
                self.rock_count += 1
            
            return rock_data, edges
            
        except Exception as e:
            self.get_logger().error(f"Error in rock detection: {e}")
            return [], None

    def avoid_obstacle(self):
        """Avoid rocks based on their zone position with timed phases"""
        # If avoidance is active, continue for full duration regardless of current rock detection
        if self.avoidance_active:
            avoidance_time = time.time() - self.avoidance_start_time
            
            # Different durations based on initial rock zone
            avoidance_duration = 4.0 if self.initial_rock_zone == "CENTER" else 2.0
            
            if avoidance_time < avoidance_duration:
                if avoidance_time < (avoidance_duration / 2):  # First half: steering
                    # Steer based on INITIAL rock zone (not current)
                    if self.initial_rock_zone == "CENTER":
                        action = "🔄 TURNING LEFT - Rock in CENTER path!"
                        steer_command = -0.8
                        brake_command = 0.3
                    elif self.initial_rock_zone == "LEFT":
                        action = "↪️ SLIGHT RIGHT - Rock on LEFT"
                        steer_command = 0.8
                        brake_command = 0.0
                    elif self.initial_rock_zone == "RIGHT":
                        action = "↩️ SLIGHT LEFT - Rock on RIGHT"  
                        steer_command = -0.8
                        brake_command = 0.0
                    
                    accel_command = 0.8
                    phase = "STEERING"
                    
                else:  # Second half: straight
                    action = "➡️ DRIVING STRAIGHT - Completing avoidance"
                    steer_command = 0.0  # Straight
                    accel_command = 0.5  # Slightly more acceleration
                    brake_command = 0.0
                    phase = "STRAIGHT"

                # Send commands
                self.send_steer_command(steer_command)
                self.send_accelerator_command(accel_command)
                self.send_brake_command(brake_command)
                self.send_reverse_command(0.0)

                # Log every 1 second (similar to stuck detection)
                if int(avoidance_time) % 1 == 0:
                    duration_type = "LONG" if self.initial_rock_zone == "CENTER" else "SHORT"
                    self.get_logger().info(f"🪨 AVOIDANCE: {action} | Phase: {phase} | Time: {avoidance_time:.1f}s/{avoidance_duration}s ({duration_type})")
                
                return True
            
            else:
                # Avoidance complete
                duration_type = "LONG" if self.initial_rock_zone == "CENTER" else "SHORT"
                self.log_message(f"✅ {avoidance_duration}s Avoidance maneuver complete! Resuming navigation.")
                self.get_logger().info(f"✅ {avoidance_duration}s Avoidance maneuver complete! Resuming navigation.")
                self.avoidance_active = False
                self.initial_rock_zone = None  # Reset the stored zone
                return False
        
        # Start new avoidance only if not already avoiding and rock is detected
        elif self.rock_detected:
            self.avoidance_active = True
            self.avoidance_start_time = time.time()
            self.initial_rock_zone = self.rock_zone  # STORE THE INITIAL ZONE
            
            # Set duration based on rock zone
            avoidance_duration = 5.0 if self.rock_zone == "CENTER" else 2.0
            duration_type = "LONG" if self.rock_zone == "CENTER" else "SHORT"
            
            self.log_message(f"🚨 ROCK DETECTED in {self.initial_rock_zone} zone! Starting {avoidance_duration}s {duration_type} avoidance...")
            return True
        
        else:
            # No rocks detected and no active avoidance
            return False

    # ===== NAVIGATION CONTROL =====
    def start_navigation(self, target_loc_localFrame):
        self.target_loc_localFrame = target_loc_localFrame
        self.navigation_active = True
        self.navigation_iterations = 0
        self.initial_move_done = False
        self.initial_move_end_time = time.time() + 4.0
        self.log_message(f"Starting navigation to target: ({target_loc_localFrame[0]:.2f}, {target_loc_localFrame[1]:.2f})")
        self.send_accelerator_command(0.2)

    # ===== CAMERA PROCESSING =====
    def process_camera_frame(self):
        """Process the latest camera frame for rock detection and exposure control"""
        try:
            cv_image = self.latest_frame
            
            # Adjust exposure based on brightness
            self.adjust_exposure_based_on_brightness(cv_image)
            
            # Detect rocks
            rock_data, edges = self.detect_rocks(cv_image)
            
            # Create visualization - stack original frame with edges
            debug_frame = cv_image.copy()

            # Convert edges to BGR for stacking
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # Resize edges to match debug_frame height
            if edges_bgr.shape[0] != debug_frame.shape[0]:
                edges_bgr = cv2.resize(edges_bgr, (debug_frame.shape[1], debug_frame.shape[0]))

            # Stack horizontally: original frame + edge detection
            stacked_frame = np.hstack((debug_frame, edges_bgr))

            # Use stacked_frame for all further drawing operations
            debug_frame = stacked_frame

            # Adjust all drawing coordinates since frame width is now doubled
            frame_width = cv_image.shape[1]
            
            # Draw trapezoidal ROI on LEFT side (original image)
            pts = np.array([
                [self.ROI_X1_TOP, self.ROI_Y1],
                [self.ROI_X2_TOP, self.ROI_Y1],
                [self.ROI_X2_BOTTOM, self.ROI_Y2],
                [self.ROI_X1_BOTTOM, self.ROI_Y2]
            ], np.int32)
            cv2.polylines(debug_frame, [pts], True, (0, 255, 255), 2)
            
            # Draw zone boundaries on LEFT side
            cv2.line(debug_frame, (self.left_end, self.ROI_Y1), 
                    (self.left_end, self.ROI_Y2), (255, 0, 0), 2)
            cv2.line(debug_frame, (self.center_end, self.ROI_Y1), 
                    (self.center_end, self.ROI_Y2), (255, 0, 0), 2)
            
            # Draw rock detections on LEFT side
            for rock in rock_data:
                x, y, w_box, h_box = rock["bbox"]
                cx, cy = rock["centroid"]
                zone = rock["zone"]
                
                # Choose color based on zone
                if zone == "CENTER":
                    color = (0, 0, 255)  # Red for center rocks
                elif zone == "LEFT":
                    color = (0, 165, 255)  # Orange for left rocks
                else:
                    color = (0, 255, 0)  # Green for right rocks
                
                # Draw bounding box and centroid (on LEFT side)
                cv2.rectangle(debug_frame, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.circle(debug_frame, (cx, cy), 4, color, -1)
                
                # Add labels
                cv2.putText(debug_frame, f"{zone} MID", (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add status text (position on LEFT side)
            brightness_value = self.current_brightness if self.current_brightness is not None else 0.0
            brightness_text = f"Brightness: {brightness_value:.1f} | Exposure: {self.current_exposure}"
            cv2.putText(debug_frame, brightness_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            rock_status = f"Rocks: {self.rock_count} | Zone: {self.rock_zone}" if self.rock_detected else "No rocks"
            status_color = (0, 0, 255) if self.rock_detected else (0, 255, 0)
            cv2.putText(debug_frame, rock_status, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            nav_status = f"Nav: {self.navigation_active} | Avoid: {self.avoidance_active}"
            cv2.putText(debug_frame, nav_status, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            waypoint_info = f"Waypoint: {self.current_waypoint_idx + 1 if self.current_waypoint_idx is not None else 0}/{len(self.waypoints) if self.waypoints else 0}"
            cv2.putText(debug_frame, waypoint_info, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add labels for the display
            cv2.putText(debug_frame, "ORIGINAL + DETECTION", (10, debug_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(debug_frame, "EDGE DETECTION", (frame_width + 10, debug_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add zone labels on LEFT side only
            cv2.putText(debug_frame, "LEFT", (self.ROI_X1_BOTTOM + 10, self.ROI_Y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(debug_frame, "CENTER", (self.left_end + 10, self.ROI_Y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(debug_frame, "RIGHT", (self.center_end + 10, self.ROI_Y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Resize window to fit screen
            debug_frame = cv2.resize(debug_frame, (1200, 600))
            
            # Display
            cv2.imshow("Rover - Navigation & Rock Avoidance", debug_frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing camera frame: {e}")

    # ===== MAIN CONTROL LOOP =====
    def timer_callback(self):
        """Main 10Hz control loop - integrates navigation, camera processing, and exposure control"""
        # Print brightness info with throttling
        if self.frame_ready and self.current_brightness is not None:
            self.brightness_log_counter += 1
            if self.brightness_log_counter >= 20:  # Log every 2 seconds (20 * 0.1s)
                self.get_logger().info(f"Brightness: {self.current_brightness:.1f} | Exposure: {self.current_exposure}")
                self.brightness_log_counter = 0

        # 1. Process camera data if available (includes exposure adjustment and rock detection)
        if self.frame_ready:
            self.process_camera_frame()
            self.frame_ready = False

        # 2. Skip navigation if not active
        if not self.navigation_active:
            return

        # 3. Check if rover is stuck and needs to reverse (HIGHEST PRIORITY)
        if self.current_location_localFrame is None or self.current_velocity_localFrame is None:
            return
            
        current_loc = np.array([self.current_location_localFrame.x, 
                            self.current_location_localFrame.y, 
                            self.current_location_localFrame.z])
        current_vel = np.array([self.current_velocity_localFrame.x,
                            self.current_velocity_localFrame.y,
                            self.current_velocity_localFrame.z])    
            
        if self.check_if_stuck(current_loc, current_vel):
            return  # Skip everything else while reversing
            
        # 4. Check for obstacles and avoid if necessary (priority over navigation)
        if self.avoid_obstacle():
            return  # Skip normal navigation while avoiding

        # 5. Initial move forward phase
        if not self.initial_move_done and self.initial_move_end_time is not None:
            if time.time() < self.initial_move_end_time:
                return
            self.send_accelerator_command(0.0)
            self.initial_move_done = True
        
        # 6. Navigation logic (only if we have sensor data)
        if self.current_rotation_localFrame is None:
            return

        # Extract current state
        current_x = float(self.current_location_localFrame.x)
        current_y = float(self.current_location_localFrame.y)
        current_z = float(self.current_location_localFrame.z)
        current_loc_localFrame = np.array([current_x, current_y, current_z])

        current_vx = float(self.current_velocity_localFrame.x)
        current_vy = float(self.current_velocity_localFrame.y)
        current_vz = float(self.current_velocity_localFrame.z)
        current_vel_localFrame = np.array([current_vx, current_vy, current_vz])

        qx = float(self.current_rotation_localFrame.x)
        qy = float(self.current_rotation_localFrame.y)
        qz = float(self.current_rotation_localFrame.z)
        qw = float(self.current_rotation_localFrame.w)
        current_rot_localFrame = Quat(qw, qx, qy, qz)

        # 7. Check if target reached
        distance = self.calculate_distance_to_target(current_loc_localFrame, self.target_loc_localFrame)
        if distance < self.tolerance:
            self.send_brake_command(1.0)
            self.send_steer_command(0.0)
            self.send_accelerator_command(0.0)
            self.log_message(f"Target reached! Beginning core sampling at position: ({current_x:.2f}, {current_y:.2f})")
            self.send_core_sampling_command()

            if self.current_waypoint_idx == len(self.waypoints) - 1:
                self.navigation_active = False
                self.log_message("Navigation complete: all waypoints reached and all core samples collected.")
            else:
                self.current_waypoint_idx += 1
                self.target_loc_localFrame = self.waypoints[self.current_waypoint_idx]
                next_loc = f"({self.target_loc_localFrame[0]:.2f}, {self.target_loc_localFrame[1]:.2f})"
                self.log_message(f"After sampling, moving to next waypoint at: {next_loc}")
            return
        
        # 8. Normal navigation control
        if not self.avoidance_active:
            speed_limit_kph = 18.0 if self.current_brightness >= 50.0 else 13.0
            speed_diff_kph = self.calculate_speed_difference(current_vel_localFrame, speed_limit_kph)
            accel_factor = remap_clamp(0.0, speed_limit_kph, 0.0, 1.0, speed_diff_kph)
            brake_factor = 1.0 - remap_clamp(-speed_limit_kph, 0.0, 0.0, 1.0, speed_diff_kph)

            db_heading = np.deg2rad(3.0)
            heading_error = self.calculate_pointing_error_angle(current_loc_localFrame, self.target_loc_localFrame, 
                                                                current_rot_localFrame)
            
            steer_command = remap_clamp(-0.25 * np.pi, 0.25 * np.pi, -1.0, 1.0, heading_error)
            if abs(heading_error) < db_heading:
                steer_command = 0.0
            steer_gain = 1.0
            actual_steer_command = -steer_gain * steer_command
            
            accel_gain = 2.0
            accel_command = accel_gain * remap_clamp(0.0, 1.0, accel_factor, accel_factor * 0.5, abs(steer_command))

            brake_gain = 1.0
            brake_command = brake_gain * brake_factor

            # 9. Send final commands
            self.send_steer_command(actual_steer_command)
            self.send_accelerator_command(accel_command)
            self.send_reverse_command(brake_command)
            self.send_brake_command(0.0)    
        
        
        
        self.navigation_iterations += 1

    def destroy_node(self):
        # Send stop commands before shutting down
        self.send_steer_command(0.0)
        self.send_accelerator_command(0.0)
        self.send_brake_command(1.0)
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    rover_controller = IntegratedRoverController()



    waypoints_localFrame = [
        np.array([ -606.74433428,   332.44253661,   -20.41775233]),   # 5
        np.array([-1025.65838348,   274.39353778,   -76.31593519]),   # 15
        np.array([ -561.74721182,    28.52558036,   -29.92751284]),   # 13
        np.array([ -349.10709106,   558.01869306,   -68.71836618]),   # 2
        np.array([  111.24089259,   427.56166121,   -54.81398767]),   # 1
        np.array([  247.67056987,   579.07900331,   -75.04176954]),   # 17
        np.array([  345.53461945,  1330.35839896,   -73.53015250]),   # 18
        np.array([  654.62948546,  1186.61595725,   -48.47787130]),   # 4
        np.array([ 1073.38823240,  1613.84763245,   -50.72357905]),   # 19
        np.array([ 1098.14343253,  1987.40560248,   -45.45757708]),   # 10
        np.array([ 1281.36380015,  1647.50529027,   -39.35361376]),   # 3
        np.array([ 1960.32237043,  1423.88737415,   -89.97019481]),   # 9
        np.array([ 1958.28017108,  1381.24222162,   -76.75680176]),   # 14
        np.array([ 1532.81368707,  1255.13690297,   -48.47378546]),   # 12
        np.array([ 1349.86835614,  1047.23075279,   -46.89420337]),   # 6
        np.array([  410.36797363,  -956.93367913,   -84.31272572]),   # 16
        np.array([  231.41034119,  -858.69285702,   -63.31508790]),   # 7
        np.array([   10.15805303,  -752.47151722,   -68.15878792]),   # 11
        np.array([  -54.31019727,   191.84449903,   -19.54598818]),   # 0**
        np.array([   45.56236659,   921.05755228,   -65.76412603]),   # 8
    ]

    # Wait for initial sensor data
    while rclpy.ok():
        if (rover_controller.current_location_localFrame is not None and 
            rover_controller.current_rotation_localFrame is not None):
            break
        rover_controller.get_logger().info('Waiting for initial location and rotation...')
        rclpy.spin_once(rover_controller, timeout_sec=0.5)

    current_x = rover_controller.current_location_localFrame.x
    current_y = rover_controller.current_location_localFrame.y

    rover_controller.waypoints = waypoints_localFrame
    rover_controller.current_waypoint_idx = 0

    rover_controller.log_message(
        f"Starting integrated navigation with rock avoidance and exposure control"
    )
    
    rover_controller.start_navigation(waypoints_localFrame[0])

    try:
        rclpy.spin(rover_controller)
    except KeyboardInterrupt:
        rover_controller.get_logger().info("Shutting down...")
    finally:
        rover_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()