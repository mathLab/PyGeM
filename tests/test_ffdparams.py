import filecmp
import os
from unittest import TestCase

import numpy as np
from OCC.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeBox
from OCC.gp import gp_Pnt

import pygem.params as ffdp


class TestFFDParameters(TestCase):
	def test_class_members_default_n_control_points(self):
		params = ffdp.FFDParameters()
		assert np.array_equal(params.n_control_points, [2, 2, 2])

	def test_class_members_default_conversion_unit(self):
		params = ffdp.FFDParameters()
		assert params.conversion_unit == 1.

	def test_class_members_default_lenght_box(self):
		params = ffdp.FFDParameters()
		assert np.array_equal(params.lenght_box, np.ones(3))

	def test_class_members_default_origin_box(self):
		params = ffdp.FFDParameters()
		assert np.array_equal(params.origin_box, np.zeros(3))

	def test_class_members_default_rot_angle(self):
		params = ffdp.FFDParameters()
		assert np.array_equal(params.rot_angle, np.zeros(3))

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
		assert np.array_equal(params.n_control_points, [2, 3, 5])

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
		assert np.array_equal(params.n_control_points, [3, 2, 2])

	def test_read_parameters_lenght_box_x(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		assert np.array_equal(params.lenght_box, [45.0, 90.0, 90.0])

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
		assert np.array_equal(params.rot_angle, [20.3, 11.0, 0.])

	def test_read_parameters_array_mu_x(self):
		params = ffdp.FFDParameters(n_control_points=[3, 2, 2])
		params.read_parameters('tests/test_datasets/parameters_sphere.prm')
		array_mu_x_exact = np.array(
			[0.2, 0., 0., 0., 0.5, 0., 0., 0., 1., 0., 0., 0.]
		).reshape((3, 2, 2))
		print(params.array_mu_x)
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
		rotation_matrix_exact = np.array([
			[ 0.98162718, 0.        ,  0.190809  ],
			[ 0.06619844, 0.93788893, -0.34056147],
			[-0.17895765, 0.34693565,  0.92065727]
		])
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
		print(params)

	def test_build_bounding_box(self):
		origin = np.array([0., 0., 0.])
		tops = np.array([1., 1., 1.])
		cube = BRepPrimAPI_MakeBox(*tops).Shape()
		params = ffdp.FFDParameters()
		params.build_bounding_box(cube)

		self.assertAlmostEqual(params.lenght_box_x, tops[0], places=5)
		self.assertAlmostEqual(params.lenght_box_y, tops[1], places=5)
		self.assertAlmostEqual(params.lenght_box_z, tops[2], places=5)
		np.testing.assert_almost_equal(params.position_vertex_0, origin, decimal=5)
		np.testing.assert_almost_equal(params.position_vertex_1, [1., 0., 0.], decimal=5)
		np.testing.assert_almost_equal(params.position_vertex_2, [0., 1., 0.], decimal=5)
		np.testing.assert_almost_equal(params.position_vertex_3, [0., 0., 1.], decimal=5)

	def test_set_box_dimension(self):
		origin = np.array([0., 0., 0.])
		tops = np.array([10., 10., 10.])
		params = ffdp.FFDParameters()
		params.origin_box = origin
		params._set_box_dimensions(origin, tops)
		self.assertEqual(params.lenght_box_x, tops[0])
		self.assertEqual(params.lenght_box_y, tops[1])
		self.assertEqual(params.lenght_box_z, tops[2])

	def test_set_position_of_vertices(self):
		vertex_0 = [0., 0., 0.]
		vertex_1 = [1., 0., 0.]
		vertex_2 = [0., 1., 0.]
		vertex_3 = [0., 0., 1.]
		tops = np.array([1., 1., 1.])
		params = ffdp.FFDParameters()
		params.origin_box = vertex_0
		params._set_box_dimensions(vertex_0, tops)
		params._set_position_of_vertices()
		np.testing.assert_equal(params.position_vertex_0, vertex_0)
		np.testing.assert_equal(params.position_vertex_1, vertex_1)
		np.testing.assert_equal(params.position_vertex_2, vertex_2)
		np.testing.assert_equal(params.position_vertex_3, vertex_3)

	def test_set_mapping(self):
		origin = np.array([0., 0., 0.])
		tops = np.array([10., 10., 10.])
		params = ffdp.FFDParameters()
		params.origin_box = origin
		params._set_box_dimensions(origin, tops)
		params._set_mapping()
		for i in range(3):
			self.assertEqual(params.psi_mapping[i][i], 1. / tops[i])
			self.assertEqual(params.inv_psi_mapping[i][i], tops[i])

	def test_set_modification_parameters_to_zero(self):
		params = ffdp.FFDParameters([5, 5, 5])
		params._set_transformation_params_to_zero()
		np.testing.assert_almost_equal(params.array_mu_x, np.zeros(shape=(5, 5, 5)))
		np.testing.assert_almost_equal(params.array_mu_y, np.zeros(shape=(5, 5, 5)))
		np.testing.assert_almost_equal(params.array_mu_z, np.zeros(shape=(5, 5, 5)))

	def test_calculate_bb_dimensions(self):
		min_vals = np.zeros(3)
		max_vals = np.ones(3)
		cube = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
		params = ffdp.FFDParameters()
		xyz_min, xyz_max = params._calculate_bb_dimension(cube)
		np.testing.assert_almost_equal(xyz_min, min_vals, decimal=5)
		np.testing.assert_almost_equal(xyz_max, max_vals, decimal=5)

	def test_calculate_bb_dimensions_triangulate(self):
		a = gp_Pnt(-1, -1, -1)
		b = gp_Pnt(3, 3, 3)

		box = BRepPrimAPI_MakeBox(a, b).Shape()
		sphere = BRepPrimAPI_MakeSphere(3).Shape()
		section = BRepAlgoAPI_Cut(box, sphere).Shape()
		params = ffdp.FFDParameters()
		xyz_min, xyz_max = params._calculate_bb_dimension(section, triangulate=True)
		correct_min = -1 * np.ones(3)
		correct_max = 3 * np.ones(3)
		np.testing.assert_almost_equal(xyz_min, correct_min, decimal=1)
		np.testing.assert_almost_equal(xyz_max, correct_max, decimal=1)
