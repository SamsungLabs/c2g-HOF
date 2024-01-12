import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from robot_planner.kinematics_gen3 import forward_kinematics as forward_kinematics7D


def plot_cube_point_length(cube_point, length, ax):

    points = []
    vectors = np.array([[0, length[1], 0], [length[0], 0, 0], [0, 0, length[2]]])
    points += [cube_point]
    points += [cube_point + vectors[0]]
    points += [cube_point + vectors[1]]
    points += [cube_point + vectors[2]]

    points += [cube_point + vectors[0] + vectors[1]]
    points += [cube_point + vectors[0] + vectors[2]]
    points += [cube_point + vectors[1] + vectors[2]]
    points += [cube_point + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]],
    ]

    faces = Poly3DCollection(edges, linewidths=1, edgecolors="k")
    faces.set_facecolor((0, 0, 1, 0.1))

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0)


def plot_traj7D(obstacle_locs, obstacle_dims, waypoints):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for idx in range(np.size(obstacle_locs, 0)):
        plot_cube_point_length(obstacle_locs[idx, :], obstacle_dims[idx, :], ax)
    plot_cube_point_length([0, 0, 28], [64, 64, 4], ax)

    for idx in range(len(waypoints)):
        curr_ang = waypoints[idx, :] * np.pi / 180
        pts = forward_kinematics7D(curr_ang, alpha=1.0)

        gt_x = np.zeros(5)
        gt_y = np.zeros(5)
        gt_z = np.zeros(5)

        for i in range(np.size(pts, 0)):
            gt_x[i + 1] = pts[i, 0]
            gt_y[i + 1] = pts[i, 1]
            gt_z[i + 1] = pts[i, 2]
        max_len = 1187.2 / 1000
        ax.plot(
            (gt_x[0:2] + max_len) * 64 / (2 * max_len),
            (gt_y[0:2] + max_len) * 64 / (2 * max_len),
            (gt_z[0:2] + max_len) * 64 / (2 * max_len),
            color="r",
            marker=".",
            linewidth=6,
            markersize=13,
        )
        ax.plot(
            (gt_x[1:3] + max_len) * 64 / (2 * max_len),
            (gt_y[1:3] + max_len) * 64 / (2 * max_len),
            (gt_z[1:3] + max_len) * 64 / (2 * max_len),
            color="g",
            marker=".",
            linewidth=6,
            markersize=13,
        )
        ax.plot(
            (gt_x[2:4] + max_len) * 64 / (2 * max_len),
            (gt_y[2:4] + max_len) * 64 / (2 * max_len),
            (gt_z[2:4] + max_len) * 64 / (2 * max_len),
            color="b",
            marker=".",
            linewidth=6,
            markersize=13,
        )
        ax.plot(
            (gt_x[3:] + max_len) * 64 / (2 * max_len),
            (gt_y[3:] + max_len) * 64 / (2 * max_len),
            (gt_z[3:] + max_len) * 64 / (2 * max_len),
            color="y",
            marker=".",
            linewidth=6,
            markersize=13,
        )

        ax.plot(
            (gt_x + max_len) * 64 / (2 * max_len),
            (gt_y + max_len) * 64 / (2 * max_len),
            (gt_z + max_len) * 64 / (2 * max_len),
            color="k",
            marker=".",
            markersize=13,
        )

        ax.set_xlabel("x")
    plt.show()
