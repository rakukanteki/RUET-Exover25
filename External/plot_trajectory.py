#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from geometry_msgs.msg import Point


class TrajectoryPlotter(Node):
    def __init__(self):
        super().__init__('trajectory_plotter')

        # Subscribe to rover location (Point message)
        self.create_subscription(
            Point,
            '/LocationLocalFrame',
            self.location_callback,
            10
        )

        # Waypoints (XYZ), plotted in XY plane
        self.waypoints_localFrame = [
            np.array([-54.31019727, 191.84449903, -19.54598818]),
            np.array([111.24089259, 427.56166121, -54.81398767]),
            np.array([-349.10709106, 558.01869306, -68.71836618]),
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
        ]

        self.wp_x = [w[0] for w in self.waypoints_localFrame]
        self.wp_y = [w[1] for w in self.waypoints_localFrame]

        self.rover_x_list = []
        self.rover_y_list = []

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Rover Trajectory (Live)")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.grid(True)

        # Draw waypoint trail
        self.ax.plot(self.wp_x, self.wp_y, 'bo-', label="Waypoint Path")

        # Label waypoints
        for idx, (x, y) in enumerate(zip(self.wp_x, self.wp_y)):
            self.ax.text(x, y, str(idx+1), fontsize=9, color='blue', ha='right', va='bottom')

        # Rover live line and marker
        self.rover_line, = self.ax.plot([], [], 'r-', linewidth=2, label="Rover Path")
        self.rover_marker, = self.ax.plot([], [], 'ro', markersize=6, label="Rover")

        self.ax.legend()

    def location_callback(self, msg):
        self.rover_x_list.append(msg.x)
        self.rover_y_list.append(msg.y)

    def update_plot(self, i):
        if self.rover_x_list:
            self.rover_line.set_data(self.rover_x_list, self.rover_y_list)
            self.rover_marker.set_data(self.rover_x_list[-1], self.rover_y_list[-1])
        return self.rover_line, self.rover_marker


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlotter()

    ani = FuncAnimation(node.fig, node.update_plot, interval=200)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
