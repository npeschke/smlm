import scipy.spatial as spat
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

import smlm.voronoi as voronoi


def to_polar(points, new_center: tuple = (0, 0), angle_offset: float = 0.0):
    points[:, 0] = points[:, 0] - new_center[0]
    points[:, 1] = points[:, 1] - new_center[1]

    result = np.zeros_like(points, dtype=np.double)

    result[:, 0] = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
    result[:, 1] = np.rad2deg(np.arctan2(points[:, 1], points[:, 0]) - angle_offset)

    return result


def center_of_mass(points):
    return np.mean(points[:, 0]), np.mean(points[:, 1])


if __name__ == '__main__':
    point_array = []
    size = 8
    for x in range(size):
        for y in range(size):
            point_array.append([x, y])

    # point_array = pd.DataFrame(point_array, columns=["x [nm]", "y [nm]"])
    # point_array.assign()
    point_array = np.array(point_array, dtype=np.double)

    test = voronoi.get_voronoi_density(point_array)
    polar_point_array = to_polar(point_array)

    points_df = pd.concat([pd.DataFrame(point_array), pd.DataFrame(polar_point_array)], axis=1)

    vor = spat.Voronoi(point_array)

    fig = spat.voronoi_plot_2d(vor=vor)
    plt.show()

    print(point_array)
