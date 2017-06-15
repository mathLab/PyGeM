from unittest import TestCase
import unittest
import pygem.params as rbfp
import numpy as np
import filecmp
import os


class TestRBFParameters(TestCase):
	def test_class_members_default_basis(self):
		params = rbfp.RBFParameters()
		assert params.basis == None

	def test_class_members_default_radius(self):
		params = rbfp.RBFParameters()
		assert params.radius == None

	def test_class_members_default_n_control_points(self):
		params = rbfp.RBFParameters()
		assert params.n_control_points == None

	def test_class_members_default_original_control_points(self):
		params = rbfp.RBFParameters()
		assert params.original_control_points == None

	def test_class_members_default_deformed_control_points(self):
		params = rbfp.RBFParameters()
		assert params.deformed_control_points == None

	def test_read_parameters_basis(self):
		params = rbfp.RBFParameters()
		params.read_parameters('tests/test_datasets/parameters_rbf_default.prm')
		assert params.basis == 'gaussian_spline'

	def test_read_parameters_radius(self):
		params = rbfp.RBFParameters()
		params.read_parameters('tests/test_datasets/parameters_rbf_default.prm')
		assert params.radius == 0.5

	def test_read_parameters_n_control_points(self):
		params = rbfp.RBFParameters()
		params.read_parameters('tests/test_datasets/parameters_rbf_default.prm')
		assert params.n_control_points == 8

	def test_read_parameters_original_control_points(self):
		params = rbfp.RBFParameters()
		params.read_parameters('tests/test_datasets/parameters_rbf_default.prm')
		original_control_points_exact = np.array([0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., \
		  0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1.]).reshape((8, 3))
		np.testing.assert_array_almost_equal(
			params.original_control_points, original_control_points_exact
		)

	def test_read_parameters_deformed_control_points(self):
		params = rbfp.RBFParameters()
		params.read_parameters('tests/test_datasets/parameters_rbf_default.prm')
		deformed_control_points_exact = np.array([0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., \
		  0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1.]).reshape((8, 3))
		np.testing.assert_array_almost_equal(
			params.deformed_control_points, deformed_control_points_exact
		)

	def test_read_parameters_failing_filename_type(self):
		params = rbfp.RBFParameters()
		with self.assertRaises(TypeError):
			params.read_parameters(3)

	def test_read_parameters_failing_number_deformed_control_points(self):
		params = rbfp.RBFParameters()
		with self.assertRaises(TypeError):
			params.read_parameters(
				'tests/test_datasets/parameters_rbf_bugged_01.prm'
			)

	def test_write_parameters_failing_filename_type(self):
		params = rbfp.RBFParameters()
		with self.assertRaises(TypeError):
			params.write_parameters(5)

	def test_write_parameters_filename_default_existance(self):
		params = rbfp.RBFParameters()
		params.basis = 'inv_multi_quadratic_biharmonic_spline'
		params.radius = 0.1
		params.n_control_points = 3
		params.original_control_points = np.array(
			[0., 0., 0., 0., 0., 1., 0., 1., 0.]
		).reshape((3, 3))
		params.deformed_control_points = np.array(
			[0., 0., 0., 0., 0., 1., 0., 1., 0.]
		).reshape((3, 3))
		params.write_parameters()
		outfilename = 'parameters_rbf.prm'
		assert os.path.isfile(outfilename)
		os.remove(outfilename)

	def test_write_parameters_filename_default(self):
		params = rbfp.RBFParameters()
		params.basis = 'gaussian_spline'
		params.radius = 0.5
		params.n_control_points = 8
		params.power = 2
		params.original_control_points = np.array([0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., \
		 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1.]).reshape((8, 3))
		params.deformed_control_points = np.array([0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., \
		 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1.]).reshape((8, 3))
		outfilename = 'test.prm'
		params.write_parameters(outfilename)
		outfilename_expected = 'tests/test_datasets/parameters_rbf_default.prm'

		print(filecmp.cmp(outfilename, outfilename_expected))
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)

	def test_write_parameters(self):
		params = rbfp.RBFParameters()
		params.read_parameters('tests/test_datasets/parameters_rbf_cube.prm')

		outfilename = 'tests/test_datasets/parameters_rbf_cube_out.prm'
		outfilename_expected = 'tests/test_datasets/parameters_rbf_cube_out_true.prm'
		params.write_parameters(outfilename)

		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)

	def test_read_parameters_filename_default(self):
 		params = rbfp.RBFParameters()
		params.read_parameters()
 		outfilename = 'parameters_rbf.prm'
 		outfilename_expected = 'tests/test_datasets/parameters_rbf_default.prm'
 
 		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)

	def test_print_info(self):
		params = rbfp.RBFParameters()
		print(params)
