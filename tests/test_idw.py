import os
import filecmp
import numpy as np
from unittest import TestCase
from pygem import IDW

class TestIDW(TestCase):
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

    def test_idw(self):
        idw = IDW()

    def test_idw_call(self):
        idw = IDW()
        idw.read_parameters('tests/test_datasets/parameters_idw_default.prm')
        idw(self.get_cube_mesh_points())

    def test_idw_perform_deform(self):
        idw = IDW()
        expected_stretch = [1.19541593, 1.36081491, 1.42095073]
        idw.read_parameters('tests/test_datasets/parameters_idw_deform.prm')
        new_pts = idw(self.get_cube_mesh_points())
        np.testing.assert_array_almost_equal(new_pts[-3], expected_stretch)

    def test_class_members_default_p(self):
        idw = IDW()
        assert idw.power == 2

    def test_class_members_default_original_points(self):
        idw = IDW()
        cube_vertices = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
                                  [1., 0., 0.], [0., 1., 1.], [1., 0., 1.],
                                  [1., 1., 0.], [1., 1., 1.]])
        np.testing.assert_equal(idw.original_control_points, cube_vertices)

    def test_class_members_default_deformed_points(self):
        idw = IDW()
        cube_vertices = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
                                  [1., 0., 0.], [0., 1., 1.], [1., 0., 1.],
                                  [1., 1., 0.], [1., 1., 1.]])
        np.testing.assert_equal(idw.deformed_control_points, cube_vertices)

    def test_write_parameters_filename_default(self):
        params = IDW()
        outfilename = 'parameters_rbf.prm'
        outfilename_expected = 'tests/test_datasets/parameters_idw_default.prm'
        params.write_parameters(outfilename)
        self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
        os.remove(outfilename)

    def test_write_not_string(self):
        params = IDW()
        with self.assertRaises(TypeError):
            params.write_parameters(5)

    def test_read_deformed(self):
        params = IDW()
        filename = 'tests/test_datasets/parameters_idw_deform.prm'
        def_vertices = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
                                 [1., 0., 0.], [0., 1., 1.], [1., 0., 1.],
                                 [1., 1., 0.], [1.5, 1.6, 1.7]])
        params.read_parameters(filename)
        np.testing.assert_equal(params.deformed_control_points, def_vertices)

    def test_read_p(self):
        idw = IDW()
        filename = 'tests/test_datasets/parameters_idw_deform.prm'
        idw.read_parameters(filename)
        assert idw.power == 3

    def test_read_not_string(self):
        idw = IDW()
        with self.assertRaises(TypeError):
            idw.read_parameters(5)

    def test_read_not_real_file(self):
        idw = IDW()
        with self.assertRaises(IOError):
            idw.read_parameters('not_real_file')

    def test_print(self):
        idw = IDW()
        print(idw)
