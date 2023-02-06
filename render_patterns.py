from random import choice, random
from time import sleep

import matplotlib.pyplot as plt
import numpy as np


def rotate_point_2d(point, angle):
    return np.array(
        [
            point[0] * np.cos(angle) - point[1] * np.sin(angle),
            point[1] * np.cos(angle) + point[0] * np.sin(angle),
        ]
    )


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def plot(self, ax, **kwargs):
        plt_circle = plt.Circle(self.center, self.radius, fill=False, **kwargs)
        ax.add_patch(plt_circle)

    def pt_at_angle(self, angle):
        return self.center + self.radius * np.array([np.cos(angle), np.sin(angle)])

    def make_polygon(self, sides):
        new_shape = np.zeros([sides, 2])
        new_angles = np.arange(0, np.pi * 2 / sides)
        for i in range(sides):
            new_shape[i] = new_circle.pt_at_angle(new_angles[i])
        return Polygon(new_shape)

    def extents(self):
        return np.array([self.center - self.radius, self.center + self.radius])


class Polygon:
    def __init__(self, points):
        self.points = points

    def degree(self):
        return self.points.shape[0]

    def plot(self, ax, **kwargs):
        ax.plot(
            np.append(self.points[:, 0], self.points[0, 0]), np.append(self.points[:, 1], self.points[0, 1]), **kwargs
        )

    def rotate_by(self, angle):
        # There's a faster way...
        new_shape = np.zeros(self.points.shape)
        for index in range(self.points.shape[0]):
            new_shape[index] = rotate_point_2d(self.points[index], angle)
        return Polygon(new_shape)

    def extents(self):
        max_values = np.max(np.abs(self.points), axis=0)
        return np.array([-max_values, max_values])

    # def extents(self):
    #     return np.array(
    #         [
    #             np.min(self.points,axis=0),
    #             np.max(self.points, axis=0)
    #         ]
    #     )


def inscribe_polygon(Polygon):
    center = np.mean(Polygon.points, axis=0)
    radius = np.linalg.norm(Polygon.points[0] - center)
    return Circle(center, radius)


def circumscribe_circle(circle, sides):
    # Get two control points.
    angle_offset = np.pi * 2 / sides
    tangent1_pt1 = circle.center + np.array([circle.radius, 0])
    tangent2_pt1 = circle.center + np.array(
        [circle.radius * np.cos(angle_offset), circle.radius * np.sin(angle_offset)]
    )

    # Compute tangents at the control points. One of them is always up.
    tangent1_slope = np.array([0, 1])
    tangent2_slope = np.array([-tangent2_pt1[1], tangent2_pt1[0]]) / np.linalg.norm(tangent2_pt1)

    # Using determinate formula.
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

    tangent1_pt2 = tangent1_pt1 + tangent1_slope
    tangent2_pt2 = tangent2_pt1 + tangent2_slope

    x1 = tangent1_pt1[0]
    x2 = tangent1_pt2[0]
    x3 = tangent2_pt1[0]
    x4 = tangent2_pt2[0]
    y1 = tangent1_pt1[1]
    y2 = tangent1_pt2[1]
    y3 = tangent2_pt1[1]
    y4 = tangent2_pt2[1]
    x_intersect = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )
    y_intersect = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )

    radius_at_intersect = np.linalg.norm(np.array([x_intersect, y_intersect]) - circle.center)
    # Create a new circle object.
    new_circle = Circle(center=circle.center, radius=radius_at_intersect)
    new_shape = np.zeros([sides, 2])
    new_angles = np.arange(0, np.pi * 2, angle_offset)
    for i in range(sides):
        new_shape[i] = new_circle.pt_at_angle(new_angles[i])
    return Polygon(new_shape)


# Convert make pattern to a class so it can be called repeatedly.
class PatternRenderer:
    def __init__(
        self,
        center,
        side_choices,
        angle_choices,
        plot_circles=True,
        base_radius=1.0,
    ):
        self.center = center
        self.side_choices = side_choices
        self.angle_choices = angle_choices
        self.plot_circles = plot_circles
        self.base_radius = base_radius
        self.current_shape = None

    def make_next_shape(self):
        if self.current_shape is None:
            self.current_shape = Circle(center=self.center, radius=self.base_radius)
        else:
            if isinstance(self.current_shape, Circle):
                self.current_shape = circumscribe_circle(self.current_shape, choice(self.side_choices))
                self.current_shape = self.current_shape.rotate_by(choice(self.angle_choices))
            else:
                self.current_shape = inscribe_polygon(self.current_shape)

        if isinstance(self.current_shape, Polygon) or self.plot_circles:
            return self.current_shape
        else:
            return self.make_next_shape()


def main():
    iterations = 200
    interval = 1.0
    shape = None
    shape_maker = PatternRenderer(
        np.zeros(2),
        [3, 4, 6, 8, 10, 12, 20],
        [0, np.pi / 3, np.pi / 4, np.pi / 6, np.pi / 8, np.pi / 10],
        plot_circles=False
    )
    fig, ax = plt.subplots()
    plt.ion()
    plt.show()
    for iteration in range(iterations):
        shape = shape_maker.make_next_shape()
        shape.plot(ax)
        extents = shape.extents()
        ax.set_xlim(extents[0, 0], extents[1, 0])
        ax.set_ylim(extents[0, 1], extents[1, 1])
        ax.axis("equal")
        plt.pause(interval)


if __name__ == "__main__":
    main()
