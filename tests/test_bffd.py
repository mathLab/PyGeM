import filecmp
import os
from unittest import TestCase
import numpy as np
from pygem import BFFD


class TestBFFD(TestCase):

    def test_nothing_happens_bffd(self):
        np.random.seed(0)
        original_mesh_points = np.random.rand(100, 3)
        A = np.random.rand(3, original_mesh_points.reshape(-1).shape[0])

        b = np.mean(original_mesh_points, axis=0)
        cffd = BFFD(b)
        cffd.adjust_control_points(original_mesh_points)
        new_mesh_points = cffd.ffd(original_mesh_points)
        assert (
            np.linalg.norm(original_mesh_points - new_mesh_points)
            / np.linalg.norm(original_mesh_points)
            < 1e-02
        )

    def test_constraint_bffd(self):
        np.random.seed(0)
        original_mesh_points = np.random.rand(100, 3)
        A = np.random.rand(3, original_mesh_points.reshape(-1).shape[0])
        b = np.mean(original_mesh_points, axis=0) + 0.02 * np.random.rand(3)
        cffd = BFFD(b)
        cffd.read_parameters("tests/test_datasets/parameters_test_cffd.prm")
        cffd.adjust_control_points(original_mesh_points)
        new_mesh_points = cffd.ffd(original_mesh_points)
        assert (
            np.linalg.norm(b - np.mean(new_mesh_points, axis=0)) / np.linalg.norm(b)
            < 1e-02
        )
