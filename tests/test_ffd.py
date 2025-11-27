import filecmp
import os
from unittest import TestCase

import numpy as np
from pygem import FFD



class TestFFD(TestCase):
    def test_class_members_default_n_control_points(self):
        params = FFD()
        assert np.array_equal(params.n_control_points, [2, 2, 2])

    def test_class_members_default_conversion_unit(self):
        params = FFD()
        assert params.conversion_unit == 1.

    def test_class_members_default_box_length(self):
        params = FFD()
        assert np.array_equal(params.box_length, np.ones(3))

    def test_class_members_default_box_origin(self):
        params = FFD()
        assert np.array_equal(params.box_origin, np.zeros(3))

    def test_class_members_default_rot_angle(self):
        params = FFD()
        assert np.array_equal(params.rot_angle, np.zeros(3))

    def test_class_members_default_array_mu_x(self):
        params = FFD()
        np.testing.assert_array_almost_equal(params.array_mu_x,
                                             np.zeros((2, 2, 2)))

    def test_class_members_default_array_mu_y(self):
        params = FFD()
        np.testing.assert_array_almost_equal(params.array_mu_y,
                                             np.zeros((2, 2, 2)))

    def test_class_members_default_array_mu_z(self):
        params = FFD()
        np.testing.assert_array_almost_equal(params.array_mu_z,
                                             np.zeros((2, 2, 2)))

    def test_class_members_default_rotation_matrix(self):
        params = FFD()
        np.testing.assert_array_almost_equal(params.rotation_matrix, np.eye(3))

    def test_class_members_default_position_vertices(self):
        params = FFD()
        expected_matrix = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                                    [0., 0., 1.]])
        np.testing.assert_array_almost_equal(params.position_vertices,
                                             expected_matrix)

    def test_class_members_generic_n_control_points(self):
        params = FFD([2, 3, 5])
        assert np.array_equal(params.n_control_points, [2, 3, 5])

    def test_class_members_generic_array_mu_x(self):
        params = FFD([2, 3, 5])
        np.testing.assert_array_almost_equal(params.array_mu_x,
                                             np.zeros((2, 3, 5)))

    def test_class_members_generic_array_mu_y(self):
        params = FFD([2, 3, 5])
        np.testing.assert_array_almost_equal(params.array_mu_y,
                                             np.zeros((2, 3, 5)))

    def test_class_members_generic_array_mu_z(self):
        params = FFD([2, 3, 5])
        np.testing.assert_array_almost_equal(params.array_mu_z,
                                             np.zeros((2, 3, 5)))

    def test_reflect_n_control_points_1(self):
        params = FFD([2, 3, 5])
        params.reflect(axis=0)
        assert np.array_equal(params.n_control_points, [3, 3, 5])

    def test_reflect_n_control_points_2(self):
        params = FFD([2, 3, 5])
        params.reflect(axis=1)
        assert np.array_equal(params.n_control_points, [2, 5, 5])

    def test_reflect_n_control_points_3(self):
        params = FFD([2, 3, 5])
        params.reflect(axis=2)
        assert np.array_equal(params.n_control_points, [2, 3, 9])

    def test_reflect_box_length_1(self):
        params = FFD([2, 3, 5])
        params.reflect(axis=0)
        assert params.box_length[0] == 2

    def test_reflect_box_length_2(self):
        params = FFD([2, 3, 5])
        params.reflect(axis=1)
        assert params.box_length[1] == 2

    def test_reflect_box_length_3(self):
        params = FFD([2, 3, 5])
        params.reflect(axis=2)
        assert params.box_length[2] == 2

    def test_reflect_wrong_axis(self):
        params = FFD([2, 3, 5])
        with self.assertRaises(ValueError):
            params.reflect(axis=4)

    def test_reflect_wrong_symmetry_plane_1(self):
        params = FFD([3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        params.array_mu_x = np.array(
            [0.2, 0., 0., 0., 0.5, 0., 0., 0., 1., 0., 0.3, 0.]).reshape((3, 2,
                                                                         2))
        with self.assertRaises(RuntimeError):
            params.reflect(axis=0)

    def test_reflect_wrong_symmetry_plane_2(self):
        params = FFD([3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        params.array_mu_y = np.array(
            [0.2, 0., 0., 0., 0.5, 0., 0., 0., 1., 0., 0.3, 0.]).reshape((3, 2,
                                                                         2))
        with self.assertRaises(RuntimeError):
            params.reflect(axis=1)

    def test_reflect_wrong_symmetry_plane_3(self):
        params = FFD([3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        params.array_mu_z = np.array(
            [0.2, 0., 0., 0., 0.5, 0., 0., 0., 1., 0., 0.3, 0.1]).reshape((3, 2,
                                                                         2))
        with self.assertRaises(RuntimeError):
            params.reflect(axis=2)

    def test_reflect_axis_0(self):
        params = FFD([3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        params.array_mu_x = np.array(
            [0.2, 0., 0., 0., 0.5, 0., 0., .2, 0., 0., 0., 0.]).reshape((3, 2,
                                                                         2))
        params.reflect(axis=0)
        array_mu_x_exact = np.array([0.2, 0., 0., 0., 0.5, 0., 0., 0.2, 0.,
            0., 0., 0., -0.5, -0., -0., -0.2, -0.2, -0., -0., -0.]).reshape((5, 2,
                                                                         2))
        np.testing.assert_array_almost_equal(params.array_mu_x,
                                             array_mu_x_exact)

    def test_reflect_axis_1(self):
        params = FFD([3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        params.array_mu_y = np.array(
            [0.2, 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.]).reshape((3, 2,
                                                                         2))
        params.reflect(axis=1)
        array_mu_y_exact = np.array([0.2, 0., 0., 0., -0.2, -0., 0.5, 0., 0., 0.,
            -0.5, -0., 0., 0., 0., 0., 0., 0.]).reshape((3, 3, 2))
        np.testing.assert_array_almost_equal(params.array_mu_y,
                                             array_mu_y_exact)

    def test_reflect_axis_2(self):
        params = FFD([3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        params.array_mu_z = np.array(
            [0.2, 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0.]).reshape((3, 2,
                                                                         2))
        params.reflect(axis=2)
        array_mu_z_exact = np.array([0.2, 0., -0.2, 0., 0., 0., 0.5, 0., -0.5,
            0., 0., -0., 0., 0., -0., 0., 0., -0.]).reshape((3, 2, 3))
        np.testing.assert_array_almost_equal(params.array_mu_z,
                                             array_mu_z_exact)

    def test_read_parameters_conversion_unit(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        assert params.conversion_unit == 1.

    def test_read_parameters_n_control_points(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        assert np.array_equal(params.n_control_points, [3, 2, 2])

    def test_read_parameters_box_length_x(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        assert np.array_equal(params.box_length, [45.0, 90.0, 90.0])

    def test_read_parameters_box_origin(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        box_origin_exact = np.array([-20.0, -55.0, -45.0])
        np.testing.assert_array_almost_equal(params.box_origin,
                                             box_origin_exact)

    def test_read_parameters_rot_angle_x(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        assert np.array_equal(params.rot_angle, [20.3, 11.0, 0.])

    def test_read_parameters_array_mu_x(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        array_mu_x_exact = np.array(
            [0.2, 0., 0., 0., 0.5, 0., 0., 0., 1., 0., 0., 0.]).reshape((3, 2,
                                                                         2))
        np.testing.assert_array_almost_equal(params.array_mu_x,
                                             array_mu_x_exact)

    def test_read_parameters_array_mu_y(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        array_mu_y_exact = np.array(
            [0., 0., 0.5555555555, 0., 0., 0., 0., 0., -1., 0., 0.,
             0.]).reshape((3, 2, 2))
        np.testing.assert_array_almost_equal(params.array_mu_y,
                                             array_mu_y_exact)

    def test_read_parameters_array_mu_z(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        array_mu_z_exact = np.array(
            [0., -0.2, 0., -0.45622985, 0., 0., 0., 0., -1.22, 0., -1.,
             0.]).reshape((3, 2, 2))
        np.testing.assert_array_almost_equal(params.array_mu_z,
                                             array_mu_z_exact)

    def test_read_parameters_rotation_matrix(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        rotation_matrix_exact = np.array(
            [[0.98162718, 0., 0.190809], [0.06619844, 0.93788893, -0.34056147],
             [-0.17895765, 0.34693565, 0.92065727]])
        np.testing.assert_array_almost_equal(params.rotation_matrix,
                                             rotation_matrix_exact)

    def test_read_parameters_position_vertex_0_origin(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        np.testing.assert_array_almost_equal(params.position_vertices[0],
                                             params.box_origin)

    def test_read_parameters_position_vertex_0(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')
        position_vertices = np.array(
            [[-20.0, -55.0, -45.0], [24.17322326, -52.02107006, -53.05309404],
             [-20., 29.41000412,
              -13.77579136], [-2.82719042, -85.65053198, 37.85915459]])

        np.testing.assert_array_almost_equal(params.position_vertices,
                                             position_vertices)

    def test_read_parameters_failing_filename_type(self):
        params = FFD(n_control_points=[3, 2, 2])
        with self.assertRaises(TypeError):
            params.read_parameters(3)

    def test_read_parameters_filename_default_existance(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.read_parameters()
        outfilename = 'parameters.prm'
        assert os.path.isfile(outfilename)
        os.remove(outfilename)

    def test_read_parameters_filename_default(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.read_parameters()
        outfilename = 'parameters.prm'
        outfilename_expected = 'tests/test_datasets/parameters_default.prm'

        self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
        os.remove(outfilename)

    def test_write_parameters_failing_filename_type(self):
        params = FFD(n_control_points=[3, 2, 2])
        with self.assertRaises(TypeError):
            params.write_parameters(5)

    def test_write_parameters_filename_default_existance(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.write_parameters()
        outfilename = 'parameters.prm'
        assert os.path.isfile(outfilename)
        os.remove(outfilename)

    def test_write_parameters_filename_default(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.write_parameters()
        outfilename = 'parameters.prm'
        outfilename_expected = 'tests/test_datasets/parameters_default.prm'

        self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
        os.remove(outfilename)

    def test_write_parameters(self):
        params = FFD(n_control_points=[3, 2, 2])
        params.read_parameters('tests/test_datasets/parameters_sphere.prm')

        outfilename = 'tests/test_datasets/parameters_sphere_out.prm'
        outfilename_expected = 'tests/test_datasets/parameters_sphere_out_true.prm'
        params.write_parameters(outfilename)
        self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
        os.remove(outfilename)

    """
    def test_save_points(self):
        params = FFD()
        params.read_parameters(
            filename='tests/test_datasets/parameters_test_ffd_sphere.prm')
        outfilename = 'tests/test_datasets/box_test_sphere_out.vtk'
        outfilename_expected = 'tests/test_datasets/box_test_sphere.vtk'
        params.save_points(outfilename, False)
        with open(outfilename, 'r') as out, open(outfilename_expected, 'r') as exp:
            self.assertTrue(out.readlines()[1:] == exp.readlines()[1:])
        os.remove(outfilename)

    def test_save_points_deformed(self):
        params = FFD()
        params.read_parameters(
            filename='tests/test_datasets/parameters_test_ffd_sphere.prm')
        outfilename = 'tests/test_datasets/box_test_sphere_deformed_out.vtk'
        outfilename_expected = 'tests/test_datasets/box_test_sphere_deformed.vtk'
        params.save_points(outfilename, True)
        with open(outfilename, 'r') as out, open(outfilename_expected, 'r') as exp:
            self.assertTrue(out.readlines()[1:] == exp.readlines()[1:])
        os.remove(outfilename)
    """

    def test_print(self):
        params = FFD(n_control_points=[3, 2, 2])
        print(params)

#    def test_build_bounding_box_1(self):
#        origin = np.array([0., 0., 0.])
#        tops = np.array([1., 1., 1.])
#        cube = BRepPrimAPI_MakeBox(*tops).Shape()
#        params = FFD()
#        params.build_bounding_box(cube)
#
#        np.testing.assert_array_almost_equal(params.box_length, tops, decimal=5)
#
#    def test_build_bounding_box_2(self):
#        origin = np.array([0., 0., 0.])
#        tops = np.array([1., 1., 1.])
#        cube = BRepPrimAPI_MakeBox(*tops).Shape()
#        params = FFD()
#        params.build_bounding_box(cube)
#
#        expected_matrix = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
#                                    [0., 0., 1.]])
#        np.testing.assert_almost_equal(
#            params.position_vertices, expected_matrix, decimal=5)

    def test_set_position_of_vertices(self):
        expected_matrix = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                                    [0., 0., 1.]])
        tops = np.array([1., 1., 1.])
        params = FFD()
        params.box_origin = expected_matrix[0]
        params.box_length = tops - expected_matrix[0]
        np.testing.assert_almost_equal(
            params.position_vertices, expected_matrix, decimal=5)

    def test_set_modification_parameters_to_zero(self):
        params = FFD([5, 5, 5])
        params.reset_weights()
        np.testing.assert_almost_equal(
            params.array_mu_x, np.zeros(shape=(5, 5, 5)))
        np.testing.assert_almost_equal(
            params.array_mu_y, np.zeros(shape=(5, 5, 5)))
        np.testing.assert_almost_equal(
            params.array_mu_z, np.zeros(shape=(5, 5, 5)))

    def test_ffd_sphere_mod(self):
        ffd = FFD()
        ffd.read_parameters(
            filename='tests/test_datasets/parameters_test_ffd_sphere.prm')
        mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
        mesh_points_ref = np.load(
            'tests/test_datasets/meshpoints_sphere_mod.npy')
        mesh_points_test = ffd(mesh_points)
        np.testing.assert_array_almost_equal(mesh_points_test, mesh_points_ref)
