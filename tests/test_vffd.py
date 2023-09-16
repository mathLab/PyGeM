import filecmp
import os
from unittest import TestCase
import numpy as np
from pygem import VFFD


class TestVFFD(TestCase):
    def test_nothing_happens(self):
        np.random.seed(0)
        points = 0.5 * np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ])

        triangles = np.array([
            [0, 1, 2],
            [3, 1, 0],
            [4, 5, 6],
            [4, 7, 5],
            [7, 4, 3],
            [3, 4, 1],
            [6, 5, 0],
            [6, 0, 2],
            [0, 5, 3],
            [3, 5, 7],
            [6, 2, 1],
            [6, 1, 4],
        ])

        cffd = VFFD(triangles, [2, 2, 2])
        cffd.vweight = np.array([1 / 3, 1 / 3, 1 / 3])
        b = cffd.fun(points)
        cffd.fixval = np.array([b])
        cffd.adjust_control_points(points)
        new_mesh_points = cffd.ffd(points)
        assert np.allclose(np.linalg.norm(points - new_mesh_points), 0.0)

    def test_constraint(self):
        np.random.seed(0)
        points = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ])

        triangles = np.array([
            [0, 1, 2],
            [3, 1, 0],
            [4, 5, 6],
            [4, 7, 5],
            [7, 4, 3],
            [3, 4, 1],
            [6, 5, 0],
            [6, 0, 2],
            [0, 5, 3],
            [3, 5, 7],
            [6, 2, 1],
            [6, 1, 4],
        ])
        cffd = VFFD(triangles, [2, 2, 2])
        b = cffd.fun(points)
        cffd.vweight = np.array([1 / 3, 1 / 3, 1 / 3])
        cffd.read_parameters(
            "tests/test_datasets/parameters_test_ffd_sphere.prm")
        cffd.fixval = np.array([b])
        cffd.adjust_control_points(points)
        new_mesh_points = cffd.ffd(points)
        assert np.isclose(np.linalg.norm(cffd.fun(new_mesh_points) - b),
                          np.array([0.0]))
