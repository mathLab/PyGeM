from unittest import TestCase
import unittest
import numpy as np
import filecmp
import os
from pygem import RBFFactory

unit_cube = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.],
                      [0., 1., 1.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]])

class TestRBFFactory(TestCase):
    def get_cube_mesh_points(self):
        # Points that define a cube
        nx, ny, nz = (20, 20, 20)
        mesh = np.zeros((nx * ny * nz, 3))
        xv = np.linspace(0, 1, nx)
        yv = np.linspace(0, 1, ny)
        zv = np.linspace(0, 1, nz)
        z, y, x = np.meshgrid(zv, yv, xv)
        mesh = np.array([x.ravel(), y.ravel(), z.ravel()])
        original_mesh_points = mesh.T
        return original_mesh_points

    def test_gaussian_spline(self):
        test_f = RBFFactory('gaussian_spline')
        value = test_f(np.linalg.norm(np.array([0.5, 1, 2, 0.2])), 0.2)
        np.testing.assert_almost_equal(value, 0.0)

    def test_multi_quadratic_biharmonic_spline(self):
        test_f = RBFFactory('multi_quadratic_biharmonic_spline')
        value = test_f(np.linalg.norm(np.array([0.5, 1, 2, 0.2])), 0.2)
        np.testing.assert_almost_equal(value, 2.30867927612)

    def test_inv_multi_quadratic_biharmonic_spline(self):
        test_f = RBFFactory('inv_multi_quadratic_biharmonic_spline')
        value = test_f(np.linalg.norm(np.array([0.5, 1, 2, 0.2])), 0.2)
        np.testing.assert_almost_equal(value, 0.433148081824)

    def test_thin_plate_spline(self):
        test_f = RBFFactory('thin_plate_spline')
        value = test_f(np.linalg.norm(np.array([0.5, 1, 2, 0.2])), 0.2)
        np.testing.assert_almost_equal(value, 323.000395428)

    def test_beckert_wendland_c2_basis_01(self):
        test_f = RBFFactory('beckert_wendland_c2_basis')
        value = test_f(np.linalg.norm(np.array([0.5, 1, 2, 0.2])), 0.2)
        np.testing.assert_almost_equal(value, 0.0)

    def test_beckert_wendland_c2_basis_02(self):
        test_f = RBFFactory('beckert_wendland_c2_basis')
        value = test_f(np.linalg.norm(np.array([0.1, 0.15, -0.2])), 0.9)
        np.testing.assert_almost_equal(value, 0.529916819595)

    def test_polyharmonic_spline_k_even(self):
        test_f = RBFFactory('polyharmonic_spline')
        value = test_f(np.linalg.norm(np.array([0.1, 0.15, -0.2])), 0.9, 3)
        np.testing.assert_almost_equal(value, 0.02677808)

    def test_polyharmonic_spline_k_odd1(self):
        test_f = RBFFactory('polyharmonic_spline')
        value = test_f(np.linalg.norm(np.array([0.1, 0.15, -0.2])), 0.9, 2)
        np.testing.assert_almost_equal(value, -0.1080092)

    def test_polyharmonic_spline_k_odd2(self):
        test_f = RBFFactory('polyharmonic_spline')
        value = test_f(np.linalg.norm(np.array([0.1, 0.15, -0.2])), 0.2, 2)
        np.testing.assert_almost_equal(value, 0.53895331)
