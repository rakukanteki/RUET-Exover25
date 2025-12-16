#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Point
from scipy.interpolate import splprep, splev


class WaypointAndRoverPlot(Node):
    def __init__(self):
        super().__init__("waypoint_and_rover_plot")

        # Subscribe to rover location
        self.create_subscription(Point, "/LocationLocalFrame", self.location_callback, 10)

        # Waypoints (XYZ)
        self.waypoints = np.array([
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

        self.wp_x = self.waypoints[:, 0]
        self.wp_y = self.waypoints[:, 1]

        # Generate spline path
        tck, u = splprep([self.wp_x, self.wp_y], s=0)  # s controls smoothness
        u_fine = np.linspace(0, 1, 800)
        self.spline_x, self.spline_y = splev(u_fine, tck)

        # Rover path history
        self.rover_x = []
        self.rover_y = []

        # Plot setup
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title("Waypoints (Spline) + Live Rover Path")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)
        self.ax.axis('equal')

        # Plot smooth spline line
        self.ax.plot(self.spline_x, self.spline_y, 'g-', linewidth=2, label="Spline Path")

        # Plot waypoint markers + labels
        self.ax.plot(self.wp_x, self.wp_y, 'bo', label="Waypoints")
        for i, (x, y) in enumerate(zip(self.wp_x, self.wp_y)):
            self.ax.text(x, y, str(i + 1), color='blue', fontsize=9, ha='right', va='bottom')

        # Rover display
        self.path_line, = self.ax.plot([], [], 'r-', linewidth=2, label="Rover Path")
        self.rover_marker, = self.ax.plot([], [], 'ro', markersize=8, label="Rover Position")

        self.ax.legend()

    def location_callback(self, msg):
        self.rover_x.append(msg.x)
        self.rover_y.append(msg.y)

    def update_plot(self):
        if self.rover_x:
            self.path_line.set_data(self.rover_x, self.rover_y)
            self.rover_marker.set_data(self.rover_x[-1], self.rover_y[-1])
        plt.pause(0.01)  # required for real-time GUI update


def main():
    rclpy.init()
    node = WaypointAndRoverPlot()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.update_plot()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
