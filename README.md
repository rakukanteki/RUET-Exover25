# RUET-Exover: Mars Rover Rally 2025

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