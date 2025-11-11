import filecmp
import os
from unittest import TestCase
import numpy as np
from pygem import VFFD
from pygem.vffd import _volume


class TestVFFD(TestCase):

    def test_nothing_happens_vffd(self):
        np.random.seed(0)
        points = 0.5 * np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )

        triangles = np.array(
            [
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
            ]
        )
        b = _volume(points, triangles)
        cffd = VFFD(triangles, b)
        cffd.adjust_control_points(points)
        new_mesh_points = cffd.ffd(points)
        new_fix = cffd.fun(new_mesh_points)
        assert np.allclose(np.linalg.norm(points - new_mesh_points), 0.0)

    def test_constraint_vffd(self):
        np.random.seed(0)
        points = 0.5 * np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )

        triangles = np.array(
            [
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
            ]
        )
        b = _volume(points, triangles) + 0.02 * np.random.rand()
        cffd = VFFD(triangles, b)
        cffd.read_parameters("tests/test_datasets/parameters_test_cffd.prm")
        cffd.adjust_control_points(points)
        new_mesh_points = cffd.ffd(points)
        new_fix = cffd.fun(new_mesh_points)
        assert np.linalg.norm(new_fix - b) / np.linalg.norm(b) < 1e-02
