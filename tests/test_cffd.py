import filecmp
import os
from unittest import TestCase
import numpy as np
from pygem import CFFD


class TestCFFD(TestCase):
    def test_nothing_happens_0(self):
        np.random.seed(0)
        cffd = CFFD()
        original_mesh_points = np.load(
            "tests/test_datasets/meshpoints_sphere_orig.npy")
        A = np.random.rand(3, original_mesh_points.reshape(-1).shape[0])

        def fun(x):
            x = x.reshape(-1)
            return A @ x
        new_mesh_points = cffd(original_mesh_points)
        assert np.array_equal(original_mesh_points, new_mesh_points)


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
        new_mesh_points = cffd(original_mesh_points)
        assert np.isclose(np.linalg.norm(fun(new_mesh_points) - b),
                          np.array([0.0]),atol=1e-7)

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
        save_par = cffd._save_parameters()
        indices=np.arange(np.prod(cffd.n_control_points)*3)[cffd.mask.reshape(-1)]
        C, d = cffd._compute_linear_map(original_mesh_points, save_par.copy(),indices)
        for i in range(2 * len(indices)):
            tmp = np.random.rand(len(indices))
            save_par[indices] = tmp
            cffd._load_parameters(tmp)
            assert np.allclose(
                np.linalg.norm(C @ tmp + d -
                               cffd.fun(cffd.ffd(original_mesh_points))),
                0.0,
            )
