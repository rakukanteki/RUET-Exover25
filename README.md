# RUET-Exover: Mars Rover Rally 2025

# Approach:
```
**INITIALIZATION**
→ Create ROS2 node "IntegratedRoverController"
→ Set up service clients (logger, steer, accelerator, reverse, brake, core sampling, exposure)
→ Subscribe to location, velocity, rotation topics (Mars frame and local frame)
→ Subscribe to camera image topic
→ Initialize rock detection parameters (ROI boundaries, zone boundaries)
→ Initialize stuck detection variables
→ Initialize exposure control variables
→ Start main timer at 5 Hz (0.2 second interval)

**MAIN CONTROL LOOP (timer_callback - runs every 0.2 seconds)**

→ Log brightness info (throttled to every 2 seconds)
→ If camera frame ready → Process camera frame (exposure adjustment + rock detection)
→ If navigation not active → Return early

→ Check if location/velocity data available → If not, return early

→ **STUCK DETECTION CHECK (HIGHEST PRIORITY)**
→ If reversing is active:
   → If reverse time < 5 seconds → Reverse straight
   → If reverse time 5-8 seconds → Drive forward with right steer
   → If reverse time 8-10 seconds → Drive forward with left steer
   → If reverse time >= 10 seconds → Complete reverse maneuver → Resume normal navigation
   → Return (skip other processing)

→ If not reversing but navigation active:
   → Track position every 1 second
   → If moved < 1 meter in 1 second AND speed < 1 m/s → Increment stuck timer
   → If stuck for > 5 seconds → Start reverse maneuver

→ **OBSTACLE AVOIDANCE CHECK (SECOND PRIORITY)**
→ If avoidance active:
   → Calculate avoidance duration (5 seconds for CENTER rocks, 2 seconds for LEFT/RIGHT)
   → First half of duration → Steer away from rock zone
   → Second half of duration → Drive straight
   → If duration complete → End avoidance → Resume navigation
   → Return (skip normal navigation)

→ If rock detected AND not already avoiding → Start new avoidance maneuver → Return

→ **INITIAL MOVE PHASE**
→ If initial move not done AND time < 4 seconds → Continue moving forward → Return
→ If initial move time elapsed → Stop accelerator → Mark initial move done

→ **NAVIGATION LOGIC**
→ Get current position, velocity, rotation from sensors
→ Calculate distance to target waypoint

→ **TARGET REACHED CHECK**
→ If distance < tolerance (5 meters):
   → Apply brake → Stop steering → Stop accelerator
   → Log "Target reached"
   → Start core sampling
   → If last waypoint → Set navigation inactive → Log "Navigation complete"
   → If not last waypoint → Increment waypoint index → Set new target
   → Return

→ **NORMAL NAVIGATION CONTROL**
→ Calculate speed limit (19 kph if bright, 15 kph if dark)
→ Calculate speed difference from target speed
→ Calculate acceleration factor
→ Calculate brake factor
→ Calculate heading error to target
→ Calculate steer command based on heading error
→ Apply steering deadband (±3 degrees)
→ Send steer command
→ Send accelerator command
→ Send reverse command (for braking)
→ Send brake command (0)
→ Increment navigation iteration counter

**CAMERA PROCESSING (process_camera_frame)**

→ Get latest frame
→ **Adjust exposure:**
   → Calculate current brightness
   → Add to brightness buffer (keep last 100 values)
   → If buffer has 100+ samples AND 2+ seconds since last change:
      → If avg brightness ≤ 20 → Set exposure -3.0 (very dark)
      → If avg brightness ≤ 80 → Set exposure 1.0 (dark)
      → If avg brightness ≤ 150 → Set exposure 13.0 (medium)
      → If avg brightness > 200 → Set exposure 14.0 (very bright)

→ **Detect rocks:**
   → Create ROI mask
   → Convert to grayscale → Apply CLAHE → Blur → Canny edge detection
   → Find contours
   → For each contour with area > 150:
      → Calculate bounding box and centroid
      → Estimate distance (CLOSE/MID/FAR based on area and position)
      → Only process MID distance rocks
      → Classify zone (LEFT/CENTER/RIGHT based on centroid X position)
      → Store rock data

→ **Create debug visualization:**
   → Draw ROI rectangle
   → Draw zone boundary lines
   → Draw rock bounding boxes (red for CENTER, orange for LEFT, green for RIGHT)
   → Add text overlays (brightness, exposure, rock count, nav status, waypoint info)
   → Display frame in OpenCV window

**WAYPOINT SEQUENCE (20 waypoints)**

→ Start at waypoint 0
→ Navigate to each waypoint in order
→ At each waypoint → Perform core sampling → Wait for completion → Move to next
→ After waypoint 19 (last) → Navigation complete

**SHUTDOWN**

→ Send stop commands (steer 0, accelerator 0, brake 1)
→ Close OpenCV windows
→ Destroy ROS2 node
→ Shutdown ROS2

```

# Environment Setup:
1. Install WSL.
2. Install Ubuntu 22.04 (for ROS2 Humble) in the Windows Terminal.
    ```
    wsl.exe --install -d Ubuntu-22.04
    ```
3. Check the existing Ubuntu versions.
    ```
    wsl -l -v
    ```
4. Installation of ROS:
* Set Locale
    ```
    locale  # check for UTF-8
    sudo apt update && sudo apt install locales
    sudo locale-gen en_US en_US.UTF-8
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    export LANG=en_US.UTF-8
    locale  # verify settings
    ```
* Setup Sources
    ```
    sudo apt install software-properties-common
    sudo add-apt-repository universe
    ```
    ```
    sudo apt update && sudo apt install curl -y
    export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
    sudo dpkg -i /tmp/ros2-apt-source.deb
    ```
* Install ROS2 Packages
    ```
    sudo apt update
    ```
    ```
    sudo apt upgrade
    ```
* Desktop Install
    ```
    sudo apt install ros-humble-desktop
    ```
* Development Tools
    ```
    sudo apt install ros-dev-tools
    ```
* Sourcing Setup Script (permanently)
    ```
    Run `nano bash.rc` then paste `source /opt/ros/humble/setup.bash` at the end of the code.
* Build (Check if there are build and install folders present, if present then delete them and run the below command).
    ```
    colcon build
    ```
    ```
    source install/setup.bash
    ```
* Installing ROS Bridge
    ```
    sudo apt install ros-humble-rosbridge-suite -y
    ```

# Running STPro and ROS2:
1. Open terminal in windows and go to Ubuntu.
    ```
    wsl.exe -d Ubuntu-22.04
    ```
2. In a fresh new terminal, source the environment and build the python package.
    ```
    source install/setup.bash
    ```
    ```
    colcon build --packages-select space_teams_python
    ```
3. Then Start the ROS bridge server
    ```
    ros2 launch space_teams_python rosbridge_image_client.launch.py
    ```
    Keep this terminal running. This will: 
        * Launch the rosbridge websocket server.
        * Start the image client for receiving images from STPro.
4. Open STPro.
5. Open a new terminal and run the example client.
    ```
    source install/setup.bash
    colcon build --packages-select space_teams_python
    ```
    ```
    ros2 run space_teams_python example_client
    ```

**If a new node or python file is created then go to setup.py file, then add the file.**
```python
'console_scripts': [
            'example_client = space_teams_python.example_client:main',
            'image_client = space_teams_python.image_client:main',
            'first_node = space_teams_python.first_node:main',
            'cam_record = space_teams_python.cam_record:main',
            'cam_edge_node = space_teams_python.cam_edge_node:main',
        ],
```

# ROS Services and Topics

## Control Services

| Service Name      | Service Type                         | Description |
|-------------------|----------------------------------------|-------------|
| **/log_message**  | `space_teams_definitions/String`       | Sends a log message to the Space Teams PRO console |
| **/Steer**        | `space_teams_definitions/Float`        | Controls rover steering (`-1.0` = left, `1.0` = right) |
| **/Accelerator**  | `space_teams_definitions/Float`        | Controls rover forward acceleration (`0.0` to `1.0`) |
| **/Reverse**      | `space_teams_definitions/Float`        | Controls rover reverse speed (`0.0` to `1.0`) |
| **/Brake**        | `space_teams_definitions/Float`        | Activates brakes (handbrake enabled when value **> 0.5**) |
| **/CoreSample**   | `space_teams_definitions/Float`        | Initiates core sampling at the current rover location |

---

## Camera Control Services

| Service Name        | Service Type                         | Description |
|---------------------|----------------------------------------|-------------|
| **/ChangeExposure** | `space_teams_definitions/Float`        | Sets camera exposure (EV range `5.0` to `20.0`) |

---

## Topics

| Topic Name                | Message Type             | Description |
|---------------------------|---------------------------|-------------|
| **/CoreSamplingComplete** | `geometry_msgs/Point`     | Published when core sampling is completed (location of the core sample) |
| **/camera/image_raw**     | `sensor_msgs/Image`       | RGB camera feed |
| **/camera/depth/image_raw** | `sensor_msgs/Image`     | Depth camera feed |
| **/LocationMarsFrame**    | `geometry_msgs/Point`     | Rover location in **Mars global frame** |
| **/VelocityMarsFrame**    | `geometry_msgs/Point`     | Rover velocity in **Mars global frame** |
| **/RotationMarsFrame**    | `geometry_msgs/Quaternion`| Rover orientation in **Mars global frame** |
| **/LocationLocalFrame**   | `geometry_msgs/Point`     | Rover location in **local frame** |
| **/VelocityLocalFrame**   | `geometry_msgs/Point`     | Rover velocity in **local frame** |
| **/RotationLocalFrame**   | `geometry_msgs/Quaternion`| Rover rotation/orientation in **local frame** |

# Space Teams PRO ROS 2 Integration
All info is on the wiki here: https://github.com/SimDynamX/SpaceTeamsROS/wiki