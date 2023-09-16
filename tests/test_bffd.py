import filecmp
import os
from unittest import TestCase
import numpy as np
from pygem import BFFD


class TestBFFD(TestCase):
    def test_nothing_happens(self):
        np.random.seed(0)
        cffd = BFFD()
        original_mesh_points = np.load(
            "tests/test_datasets/meshpoints_sphere_orig.npy")
        A = np.random.rand(3, original_mesh_points.reshape(-1).shape[0])
        b = cffd.fun(original_mesh_points)
        cffd.fixval = b
        cffd.indices = np.arange(np.prod(cffd.n_control_points) * 3).tolist()
        cffd.M = np.eye(len(cffd.indices))
        new_mesh_points = cffd(original_mesh_points)
        assert np.array_equal(original_mesh_points, new_mesh_points)

    def test_constraint(self):
        np.random.seed(0)
        cffd = BFFD()
        cffd.read_parameters(
            "tests/test_datasets/parameters_test_ffd_sphere.prm")
        original_mesh_points = np.load(
            "tests/test_datasets/meshpoints_sphere_orig.npy")
        b = cffd.fun(original_mesh_points)
        cffd.fixval = b
        cffd.indices = np.arange(np.prod(cffd.n_control_points) * 3).tolist()
        cffd.M = np.eye(len(cffd.indices))
        new_mesh_points = cffd(original_mesh_points)
        assert np.isclose(np.linalg.norm(cffd.fun(new_mesh_points) - b),
                          np.array([0.0]))
