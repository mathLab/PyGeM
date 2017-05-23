from unittest import TestCase
import unittest
from pygem.params_idw import IDWParameters
import numpy as np
import filecmp
import os


class TestIDWParameters(TestCase):
	def test_class_members_default_p(self):
		params = IDWParameters()
		assert params.power == 2

	def test_class_members_default_original_points(self):
		params = IDWParameters()
		cube_vertices = np.array([
				[0., 0., 0.], 
				[0., 0., 1.],
				[0., 1., 0.],
				[1., 0., 0.],
				[0., 1., 1.],
				[1., 0., 1.],
				[1., 1., 0.],
				[1., 1., 1.]
		])
		np.testing.assert_equal(
				params.original_control_points, cube_vertices
		)

	def test_class_members_default_deformed_points(self):
		params = IDWParameters()
		cube_vertices = np.array([
				[0., 0., 0.], 
				[0., 0., 1.],
				[0., 1., 0.],
				[1., 0., 0.],
				[0., 1., 1.],
				[1., 0., 1.],
				[1., 1., 0.],
				[1., 1., 1.]
		])
		np.testing.assert_equal(
				params.deformed_control_points, cube_vertices
		)
	
	def test_write_parameters_filename_default(self):
		params = IDWParameters()
		outfilename = 'parameters_rbf.prm'
		outfilename_expected = 'tests/test_datasets/parameters_idw_default.prm'
		params.write_parameters(outfilename)
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)

	def test_write_not_string(self):
		params = IDWParameters()
		with self.assertRaises(TypeError):
			params.write_parameters(5)

	def test_read_deformed(self):
		params = IDWParameters()
		filename = 'tests/test_datasets/parameters_idw_deform.prm'
		def_vertices = np.array([
				[0., 0., 0.], 
				[0., 0., 1.],
				[0., 1., 0.],
				[1., 0., 0.],
				[0., 1., 1.],
				[1., 0., 1.],
				[1., 1., 0.],
				[1.5, 1.6, 1.7]
		])
		params.read_parameters(filename)
		np.testing.assert_equal(
				params.deformed_control_points, def_vertices
		)

	def test_read_p(self):
		params = IDWParameters()
		filename = 'tests/test_datasets/parameters_idw_deform.prm'
		params.read_parameters(filename)
		assert params.power == 3

	def test_read_not_string(self):
		params = IDWParameters()
		with self.assertRaises(TypeError):
			params.read_parameters(5)

	def test_read_not_real_file(self):
		params = IDWParameters()
		with self.assertRaises(IOError):
			params.read_parameters('not_real_file')
		
	def test_print(self):
		params = IDWParameters()
		print(params)
