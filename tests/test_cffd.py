import filecmp
import os
from unittest import TestCase
import numpy as np
from pygem import CFFD, BFFD, VFFD
from pygem.cffd import _volume


class TestCFFD(TestCase):

    def test_nothing_happens_cffd(self):
        np.random.seed(0)
        original_mesh_points = np.random.rand(100, 3)
        A = np.random.rand(3, original_mesh_points.reshape(-1).shape[0])

        def fun(x):
            x = x.reshape(-1)
            return A @ x

        b = fun(original_mesh_points)
        cffd = CFFD(b, fun)
        cffd.adjust_control_points(original_mesh_points)
        new_mesh_points = cffd.ffd(original_mesh_points)
        assert np.linalg.norm(original_mesh_points - new_mesh_points
                              ) / np.linalg.norm(original_mesh_points) < 1e-02

    def test_constraint_cffd(self):
        np.random.seed(0)
        original_mesh_points = np.random.rand(100, 3)
        A = np.random.rand(3, original_mesh_points.reshape(-1).shape[0])

        def fun(x):
            x = x.reshape(-1)
            return A @ x

        b = fun(original_mesh_points) + 0.02 * np.random.rand(3)
        cffd = CFFD(b, fun)
        cffd.read_parameters('tests/test_datasets/parameters_test_cffd.prm')
        cffd.adjust_control_points(original_mesh_points)
        new_mesh_points = cffd.ffd(original_mesh_points)
        assert np.linalg.norm(b -
                              fun(new_mesh_points)) / np.linalg.norm(b) < 1e-02

    def test_nothing_happens_vffd(self):
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
        b = _volume(points, triangles)
        cffd = VFFD(triangles, b)
        cffd.adjust_control_points(points)
        new_mesh_points = cffd.ffd(points)
        new_fix = cffd.fun(new_mesh_points)
        assert np.allclose(np.linalg.norm(points - new_mesh_points), 0.0)

    def test_constraint_vffd(self):
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
        b = _volume(points, triangles) + 0.02 * np.random.rand()
        cffd = VFFD(triangles, b)
        cffd.read_parameters('tests/test_datasets/parameters_test_cffd.prm')
        cffd.adjust_control_points(points)
        new_mesh_points = cffd.ffd(points)
        new_fix = cffd.fun(new_mesh_points)
        assert np.linalg.norm(new_fix - b) / np.linalg.norm(b) < 1e-02

    def test_nothing_happens_bffd(self):
        np.random.seed(0)
        original_mesh_points = np.random.rand(100, 3)
        A = np.random.rand(3, original_mesh_points.reshape(-1).shape[0])

        b = np.mean(original_mesh_points, axis=0)
        cffd = BFFD(b)
        cffd.adjust_control_points(original_mesh_points)
        new_mesh_points = cffd.ffd(original_mesh_points)
        assert np.linalg.norm(original_mesh_points - new_mesh_points
                              ) / np.linalg.norm(original_mesh_points) < 1e-02

    def test_constraint_bffd(self):
        np.random.seed(0)
        original_mesh_points = np.random.rand(100, 3)
        A = np.random.rand(3, original_mesh_points.reshape(-1).shape[0])
        b = np.mean(original_mesh_points, axis=0) + 0.02 * np.random.rand(3)
        cffd = BFFD(b)
        cffd.read_parameters('tests/test_datasets/parameters_test_cffd.prm')
        cffd.adjust_control_points(original_mesh_points)
        new_mesh_points = cffd.ffd(original_mesh_points)
        assert np.linalg.norm(
            b - np.mean(new_mesh_points, axis=0)) / np.linalg.norm(b) < 1e-02
