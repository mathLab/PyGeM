from unittest import TestCase
import unittest
import pygem.params as ffdp
import numpy as np
import filecmp
import os


class TestFFDParameters(TestCase):
	def test_class_members_default_n_control_points(self):
		params = ffdp.FFDParameters()
		assert params.n_control_points == [2, 2, 2]

	def test_class_members_default_conversion_unit(self):
		params = ffdp.FFDParameters()
		assert params.conversion_unit == 1.

	def test_class_members_default_lenght_box_x(self):
		params = ffdp.FFDParameters()
		assert params.lenght_box_x == 1.

	def test_class_members_default_lenght_box_y(self):
		params = ffdp.FFDParameters()
		assert params.lenght_box_y == 1.

	def test_class_members_default_lenght_box_z(self):
		params = ffdp.FFDParameters()
		assert params.lenght_box_z == 1.

	def test_class_members_default_origin_box(self):
		params = ffdp.FFDParameters()
		np.testing.assert_array_almost_equal(params.origin_box, np.zeros(3))

	def test_class_members_default_rot_angle_x(self):
		params = ffdp.FFDParameters()
		assert params.rot_angle_x == 0

	def test_class_members_default_rot_angle_y(self):
		params = ffdp.FFDParameters()
		assert params.rot_angle_y == 0

	def test_class_members_default_rot_angle_z(self):
		params = ffdp.FFDParameters()
		assert params.rot_angle_z == 0

	def test_class_members_default_array_mu_x(self):
		params = ffdp.FFDParameters()
		np.testing.assert_array_almost_equal(
			params.array_mu_x, np.zeros((2, 2, 2))
		)

	def test_class_members_default_array_mu_y(self):
		params = ffdp.FFDParameters()
		np.testing.assert_array_almost_equal(
			params.array_mu_y, np.zeros((2, 2, 2))
		)

	def test_class_members_default_array_mu_z(self):
		params = ffdp.FFDParameters()
		np.testing.assert_array_almost_equal(
			params.array_mu_z, np.zeros((2, 2, 2))
		)

	def test_class_members_default_psi_mapping(self):
		params = ffdp.FFDParameters()
		np.testing.assert_array_almost_equal(
			params.psi_mapping, np.diag([1, 1, 1])
		)

	def test_class_members_default_inv_psi_mapping(self):
		params = ffdp.FFDParameters()
		np.testing.assert_array_almost_equal(
			params.inv_psi_mapping, np.diag([1, 1, 1])
		)

	def test_class_members_default_rotation_matrix(self):
		params = ffdp.FFDParameters()
		np.testing.assert_array_almost_equal(params.rotation_matrix, np.eye(3))

	def test_class_members_default_position_vertex_0(self):
		params = ffdp.FFDParameters()
		np.testing.assert_array_almost_equal(
			params.position_vertex_0, np.zeros(3)
		)

	def test_class_members_default_position_vertex_1(self):
		params = ffdp.FFDParameters()
		np.testing.assert_array_almost_equal(
			params.position_vertex_1, np.array([1., 0., 0.])
		)

	def test_class_members_default_position_vertex_2(self):
		params = ffdp.FFDParameters()
		np.testing.assert_array_almost_equal(
			params.position_vertex_2, np.array([0., 1., 0.])
		)

	def test_class_members_default_position_vertex_3(self):
		params = ffdp.FFDParameters()
		np.testing.assert_array_almost_equal(
			params.position_vertex_3, np.array([0., 0., 1.])
		)

	def test_class_members_generic_n_control_points(self):
		params = ffdp.FFDParameters([2, 3, 5])
		assert params.n_control_points == [2, 3, 5]

	def test_class_members_generic_array_mu_x(self):
		params = ffdp.FFDParameters([2, 3, 5])
		np.testing.assert_array_almost_equal(
			params.array_mu_x, np.zeros((2, 3, 5))
		)

	def test_class_members_generic_array_mu_y(self):
		params = ffdp.FFDParameters([2, 3, 5])
		np.testing.assert_array_almost_equal(
			params.array_mu_y, np.zeros((2, 3, 5))
		)

	def test_class_members_generic_array_mu_z(self):
		params = ffdp.FFDParameters([2, 3, 5])
		np.testing.assert_array_almost_equal(
			params.array_mu_z, np.zeros((2, 3, 5))
		)

	def test_read_parameters_conversion_unit(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		assert params.conversion_unit == 1.

	def test_read_parameters_n_control_points(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		assert params.n_control_points == [3, 2, 2]

	def test_read_parameters_lenght_box_x(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		assert params.lenght_box_x == 45.0

	def test_read_parameters_lenght_box_y(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		assert params.lenght_box_y == 90.0

	def test_read_parameters_lenght_box_z(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		assert params.lenght_box_z == 90.0

	def test_read_parameters_origin_box(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		origin_box_exact = np.array([-20.0, -55.0, -45.0])
		np.testing.assert_array_almost_equal(
			params.origin_box, origin_box_exact
		)

	def test_read_parameters_rot_angle_x(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		assert params.rot_angle_x == 20.3

	def test_read_parameters_rot_angle_y(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		assert params.rot_angle_y == 11.0

	def test_read_parameters_rot_angle_z(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		assert params.rot_angle_z == 0.

	def test_read_parameters_array_mu_x(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		array_mu_x_exact = np.array(
			[0.2, 0., 0., 0., 0.5, 0., 0., 0., 1., 0., 0., 0.]
		).reshape((3, 2, 2))
		np.testing.assert_array_almost_equal(
			params.array_mu_x, array_mu_x_exact
		)

	def test_read_parameters_array_mu_y(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		array_mu_y_exact = np.array(
			[0., 0., 0.5555555555, 0., 0., 0., 0., 0., -1., 0., 0., 0.]
		).reshape((3, 2, 2))
		np.testing.assert_array_almost_equal(
			params.array_mu_y, array_mu_y_exact
		)

	def test_read_parameters_array_mu_z(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		array_mu_z_exact = np.array(
			[0., -0.2, 0., -0.45622985, 0., 0., 0., 0., -1.22, 0., -1., 0.]
		).reshape((3, 2, 2))
		np.testing.assert_array_almost_equal(
			params.array_mu_z, array_mu_z_exact
		)

	def test_read_parameters_psi_mapping(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		psi_mapping_exact = np.diag([0.02222222, 0.01111111, 0.01111111])
		np.testing.assert_array_almost_equal(
			params.psi_mapping, psi_mapping_exact
		)

	def test_read_parameters_inv_psi_mapping(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		inv_psi_mapping_exact = np.diag([45., 90., 90.])
		np.testing.assert_array_almost_equal(
			params.inv_psi_mapping, inv_psi_mapping_exact
		)

	def test_read_parameters_rotation_matrix(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		rotation_matrix_exact = np.array([0.98162718, 0., 0.190809, 0.06619844, 0.93788893, \
		 -0.34056147, -0.17895765, 0.34693565, 0.92065727]).reshape((3,3))
		np.testing.assert_array_almost_equal(
			params.rotation_matrix, rotation_matrix_exact
		)

	def test_read_parameters_position_vertex_0_origin(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		np.testing.assert_array_almost_equal(
			params.position_vertex_0, params.origin_box
		)

	def test_read_parameters_position_vertex_0(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		position_vertex_0_exact = np.array([-20.0, -55.0, -45.0])
		np.testing.assert_array_almost_equal(
			params.position_vertex_0, position_vertex_0_exact
		)

	def test_read_parameters_position_vertex_1(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		position_vertex_1_exact = np.array(
			[24.17322326, -52.02107006, -53.05309404]
		)
		np.testing.assert_array_almost_equal(
			params.position_vertex_1, position_vertex_1_exact
		)

	def test_read_parameters_position_vertex_2(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		position_vertex_2_exact = np.array([-20., 29.41000412, -13.77579136])
		np.testing.assert_array_almost_equal(
			params.position_vertex_2, position_vertex_2_exact
		)

	def test_read_parameters_position_vertex_3(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		position_vertex_3_exact = np.array(
			[-2.82719042, -85.65053198, 37.85915459]
		)
		np.testing.assert_array_almost_equal(
			params.position_vertex_3, position_vertex_3_exact
		)

	def test_read_parameters_failing_filename_type(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		with self.assertRaises(TypeError):
			params.read_parameters(3)

	def test_read_parameters_filename_default_existance(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters()
		outfilename = 'parameters.prm'
		assert os.path.isfile(outfilename)
		os.remove(outfilename)

	def test_read_parameters_filename_default(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters()
		outfilename = 'parameters.prm'
		outfilename_expected = 'tests/test_datasets/parameters_default.prm'

		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)

	def test_write_parameters_failing_filename_type(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		with self.assertRaises(TypeError):
			params.write_parameters(5)

	def test_write_parameters_filename_default_existance(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.write_parameters()
		outfilename = 'parameters.prm'
		assert os.path.isfile(outfilename)
		os.remove(outfilename)

	def test_write_parameters_filename_default(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.write_parameters()
		outfilename = 'parameters.prm'
		outfilename_expected = 'tests/test_datasets/parameters_default.prm'

		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)

	def test_write_parameters(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')

		outfilename = 'tests/test_datasets/parameters_sphere_out.prm'
		outfilename_expected = 'tests/test_datasets/parameters_sphere_out_true.prm'
		params.write_parameters(outfilename)

		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)

	def test_print_info(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.print_info()
