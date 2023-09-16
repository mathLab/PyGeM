import filecmp
import os
from unittest import TestCase
import numpy as np
from pygem import CFFD


class TestCFFD(TestCase):
    def test_nothing_happens(self):
        np.random.seed(0)
        cffd = CFFD()
        original_mesh_points = np.load(
            "tests/test_datasets/meshpoints_sphere_orig.npy")
        A = np.random.rand(3, original_mesh_points.reshape(-1).shape[0])

        def fun(x):
            x = x.reshape(-1)
            return A @ x

        b = fun(original_mesh_points)
        cffd.fun = fun
        cffd.fixval = b
        cffd.indices = np.arange(np.prod(cffd.n_control_points) * 3).tolist()
        cffd.M = np.eye(len(cffd.indices))
        new_mesh_points = cffd(original_mesh_points)
        assert np.array_equal(original_mesh_points, new_mesh_points)

    def test_constraint(self):
        np.random.seed(0)
        cffd = CFFD()
        cffd.read_parameters(
            "tests/test_datasets/parameters_test_ffd_sphere.prm")
        original_mesh_points = np.load(
            "tests/test_datasets/meshpoints_sphere_orig.npy")
        A = np.random.rand(3, original_mesh_points.reshape(-1).shape[0])

        def fun(x):
            x = x.reshape(-1)
            return A @ x

        b = fun(original_mesh_points)
        cffd.fun = fun
        cffd.fixval = b
        cffd.indices = np.arange(np.prod(cffd.n_control_points) * 3).tolist()
        cffd.M = np.eye(len(cffd.indices))
        new_mesh_points = cffd(original_mesh_points)
        assert np.isclose(np.linalg.norm(fun(new_mesh_points) - b),
                          np.array([0.0]))

    def test_interpolation(self):
        cffd = CFFD()
        original_mesh_points = np.load(
            "tests/test_datasets/meshpoints_sphere_orig.npy")
        A = np.random.rand(3, original_mesh_points.reshape(-1).shape[0])

        def fun(x):
            x = x.reshape(-1)
            return A @ x

        b = fun(original_mesh_points)
        cffd.fixval = b
        cffd.fun = fun
        cffd.indices = np.arange(np.prod(cffd.n_control_points) * 3).tolist()
        cffd.M = np.eye(len(cffd.indices))
        save_par = cffd._save_parameters()
        C, d = cffd._compute_linear_map(original_mesh_points, save_par.copy())
        for i in range(2 * len(cffd.indices)):
            tmp = np.random.rand(len(cffd.indices))
            save_par[cffd.indices] = tmp
            cffd._load_parameters(tmp)
            assert np.allclose(
                np.linalg.norm(C @ tmp + d -
                               cffd.fun(cffd.ffd(original_mesh_points))),
                0.0,
            )
