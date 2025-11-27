from unittest import TestCase
import unittest
import numpy as np
import filecmp
import os
from pygem import RBF
from pygem import RBFFactory

unit_cube = np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.],
                      [0., 1., 1.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]])

class TestRBF(TestCase):
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

    def test_rbf_weights_member(self):
        rbf = RBF()
        rbf.read_parameters('tests/test_datasets/parameters_rbf_cube.prm')
        rbf.compute_weights()
        weights_true = np.load('tests/test_datasets/weights_rbf_cube.npy')
        np.testing.assert_array_almost_equal(rbf.weights, weights_true)

    def test_rbf_cube_mod(self):
        mesh_points_ref = np.load(
            'tests/test_datasets/meshpoints_cube_mod_rbf.npy')
        rbf = RBF()
        rbf.read_parameters('tests/test_datasets/parameters_rbf_cube.prm')
        rbf.radius = 0.5
        deformed_mesh = rbf(self.get_cube_mesh_points())
        np.testing.assert_array_almost_equal(deformed_mesh, mesh_points_ref)

    def test_wrong_basis(self):
        rbf = RBF()
        with self.assertRaises(NameError):
            rbf.read_parameters(
                    'tests/test_datasets/parameters_rbf_bugged_02.prm')

    def test_class_members_default_basis(self):
        rbf = RBF()

    def test_class_members_default_radius(self):
        rbf = RBF()
        assert rbf.radius == 0.5

    def test_class_members_default_extra(self):
        rbf = RBF()
        assert rbf.extra == {}

    def test_class_members_default_n_control_points(self):
        rbf = RBF()
        assert rbf.n_control_points == 8

    def test_class_members_default_original_control_points(self):
        rbf = RBF()
        np.testing.assert_array_equal(rbf.original_control_points, unit_cube)

    def test_class_members_default_deformed_control_points(self):
        rbf = RBF()
        np.testing.assert_array_equal(rbf.deformed_control_points, unit_cube)

    def test_read_parameters_basis(self):
        rbf = RBF()
        rbf.read_parameters('tests/test_datasets/parameters_rbf_default.prm')
        assert rbf.basis == RBFFactory('gaussian_spline')

    def test_read_parameters_basis2(self):
        rbf = RBF()
        rbf.read_parameters('tests/test_datasets/parameters_rbf_extra.prm')
        assert rbf.basis == RBFFactory('polyharmonic_spline')

    def test_read_parameters_radius(self):
        rbf = RBF()
        rbf.read_parameters('tests/test_datasets/parameters_rbf_radius.prm')
        assert rbf.radius == 2.0

    def test_read_extra_parameters(self):
        rbf = RBF()
        rbf.read_parameters('tests/test_datasets/parameters_rbf_extra.prm')
        assert rbf.extra == {'k': 4}

    def test_read_parameters_n_control_points(self):
        rbf = RBF()
        rbf.read_parameters('tests/test_datasets/parameters_rbf_default.prm')
        assert rbf.n_control_points == 8

    def test_read_parameters_original_control_points(self):
        params = RBF()
        params.read_parameters('tests/test_datasets/parameters_rbf_default.prm')
        np.testing.assert_array_almost_equal(params.original_control_points,
                                             unit_cube)

    def test_read_parameters_deformed_control_points(self):
        params = RBF()
        params.read_parameters('tests/test_datasets/parameters_rbf_default.prm')
        np.testing.assert_array_almost_equal(params.deformed_control_points,
                                             unit_cube)

    def test_read_parameters_failing_filename_type(self):
        params = RBF()
        with self.assertRaises(TypeError):
            params.read_parameters(3)

    def test_read_parameters_failing_number_deformed_control_points(self):
        params = RBF()
        with self.assertRaises(TypeError):
            params.read_parameters(
                'tests/test_datasets/parameters_rbf_bugged_01.prm')

    def test_write_parameters_failing_filename_type(self):
        params = RBF()
        with self.assertRaises(TypeError):
            params.write_parameters(5)

    def test_write_parameters_filename_default_existance(self):
        params = RBF()
        params.basis = 'inv_multi_quadratic_biharmonic_spline'
        params.radius = 0.1
        params.original_control_points = np.array(
            [0., 0., 0., 0., 0., 1., 0., 1., 0.]).reshape((3, 3))
        params.deformed_control_points = np.array(
            [0., 0., 0., 0., 0., 1., 0., 1., 0.]).reshape((3, 3))
        params.write_parameters()
        outfilename = 'parameters_rbf.prm'
        assert os.path.isfile(outfilename)
        os.remove(outfilename)
    """
    def test_write_parameters_filename_default(self):
        params = RBF()
        params.basis = 'gaussian_spline'
        params.radius = 0.5
        params.power = 2
        params.original_control_points = unit_cube
        params.deformed_control_points = unit_cube
        outfilename = 'test.prm'
        params.write_parameters(outfilename)
        outfilename_expected = 'tests/test_datasets/parameters_rbf_default.prm'

        print(filecmp.cmp(outfilename, outfilename_expected))
        self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
        os.remove(outfilename)

    def test_write_parameters(self):
        params = RBF()
        params.read_parameters('tests/test_datasets/parameters_rbf_cube.prm')

        outfilename = 'ters_rbf_cube_out.prm'
        #outfilename = 'tests/test_datasets/parameters_rbf_cube_out.prm'
        outfilename_expected = 'tests/test_datasets/parameters_rbf_cube_out_true.prm'
        params.write_parameters(outfilename)
        
        self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
        #os.remove(outfilename)
    """

    def test_print_info(self):
        rbf = RBF()
        print(rbf)

    def test_call_dummy_transformation(self):
        rbf = RBF()
        rbf.read_parameters('tests/test_datasets/parameters_rbf_default.prm')
        mesh = self.get_cube_mesh_points()
        new = rbf(mesh)
        np.testing.assert_array_almost_equal(new[17], mesh[17])

    def test_call(self):
        rbf = RBF()
        rbf.read_parameters('tests/test_datasets/parameters_rbf_extra.prm')
        mesh = self.get_cube_mesh_points()
        new = rbf(mesh)
        np.testing.assert_array_almost_equal(new[17], [8.947368e-01, 5.353524e-17, 8.845331e-03])

