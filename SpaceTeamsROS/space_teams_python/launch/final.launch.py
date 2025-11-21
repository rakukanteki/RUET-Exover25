# ros2 launch space_teams_python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction

def generate_launch_description():
    # Topics to record
    topics_to_record = [
        '/LocationLocalFrame',
        '/LocationMarsFrame',
        '/RotationLocalFrame',
        '/RotationMarsFrame',
        '/VelocityLocalFrame',
        '/VelocityMarsFrame',
        '/camera/depth/image_raw',
        '/camera/image_raw',
        '/client_count',
        '/connected_clients'
    ]

    # Start rosbag record (records all selected topics)
    rosbag_record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', 'rover_test_bag'] + topics_to_record,
        output='screen'
    )

    # Start your RoverController node
    controller_node = Node(
        package='space_teams_python',          # <--- package name here
        executable='best_run',               # <--- entry point from setup.py
        name='rover_controller_node',
        output='screen'
    )

    # Optionally delay rosbag start a bit (e.g. 2 s after node starts)
    delayed_rosbag = TimerAction(
        period=2.0,
        actions=[rosbag_record]
    )

    return LaunchDescription([
        controller_node,
        delayed_rosbag
    ])
