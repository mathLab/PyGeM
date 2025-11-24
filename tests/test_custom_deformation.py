from unittest import TestCase

import numpy as np

from pygem import CustomDeformation


class TestCustomDeformation(TestCase):
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

    def test_class_members_func(self):
        def move(x):
            return x + x**2

        CustomDeformation(move)

    def test_ffd_sphere_mod(self):
        def move(x):
            x0, x1, x2 = x
            return [x0**2, x1, x2]

        deform = CustomDeformation(move)
        mesh_points = self.get_cube_mesh_points()
        mesh_points_test = deform(mesh_points)
        mesh_points_ref = mesh_points.copy()
        mesh_points_ref[:, 0] **= 2
        np.testing.assert_array_almost_equal(mesh_points_test, mesh_points_ref)
