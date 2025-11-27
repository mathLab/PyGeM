import filecmp
import os
from unittest import TestCase
import numpy as np
from pygem import CFFD


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

