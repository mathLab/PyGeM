from unittest import TestCase
import unittest
import pygem.unvhandler as uh
import numpy as np
import filecmp
import os


class TestUnvHandler(TestCase):
	def test_unv_instantiation(self):
		unv_handler = uh.UnvHandler()

	def test_unv_default_infile_member(self):
		unv_handler = uh.UnvHandler()
		self.assertIsNone(unv_handler.infile)

	def test_unv_default_outfile_member(self):
		unv_handler = uh.UnvHandler()
		self.assertIsNone(unv_handler.outfile)

	def test_unv_default_extension_member(self):
		unv_handler = uh.UnvHandler()
		self.assertListEqual(unv_handler.extensions, ['.unv'])

	def test_unv_parse_failing_filename_type(self):
		unv_handler = uh.UnvHandler()
		with self.assertRaises(TypeError):
			mesh_points = unv_handler.parse(5.2)

	def test_unv_parse_failing_check_extension(self):
		unv_handler = uh.UnvHandler()
		with self.assertRaises(ValueError):
			mesh_points = unv_handler.parse(
				'tests/test_datasets/test_square.iges'
			)

	def test_unv_parse_infile(self):
		unv_handler = uh.UnvHandler()
		mesh_points = unv_handler.parse('tests/test_datasets/test_square.unv')
		self.assertEqual(unv_handler.infile, 'tests/test_datasets/test_square.unv')

	def test_unv_parse_shape(self):
		unv_handler = uh.UnvHandler()
		mesh_points = unv_handler.parse('tests/test_datasets/test_square.unv')
		self.assertTupleEqual(mesh_points.shape, (256, 3))

	def test_unv_parse_coords_1(self):
		unv_handler = uh.UnvHandler()
		mesh_points = unv_handler.parse('tests/test_datasets/test_square.unv')
		np.testing.assert_almost_equal(mesh_points[33][0], 1.0)

	def test_unv_parse_coords_2(self):
		unv_handler = uh.UnvHandler()
		mesh_points = unv_handler.parse('tests/test_datasets/test_square.unv')
		np.testing.assert_almost_equal(mesh_points[178][1], 0.4)

	def test_unv_parse_coords_3(self):
		unv_handler = uh.UnvHandler()
		mesh_points = unv_handler.parse('tests/test_datasets/test_square.unv')
		np.testing.assert_almost_equal(mesh_points[100][2], 0.0)

	def test_unv_parse_coords_4(self):
		unv_handler = uh.UnvHandler()
		mesh_points = unv_handler.parse('tests/test_datasets/test_square.unv')
		np.testing.assert_almost_equal(mesh_points[0][0], 0.0)

	def test_unv_parse_coords_5(self):
		unv_handler = uh.UnvHandler()
		mesh_points = unv_handler.parse('tests/test_datasets/test_square.unv')
		np.testing.assert_almost_equal(mesh_points[-1][2], 0.0)

	def test_unv_write_failing_filename_type(self):
		unv_handler = uh.UnvHandler()
		mesh_points = unv_handler.parse('tests/test_datasets/test_square.unv')
		with self.assertRaises(TypeError):
			unv_handler.write(mesh_points, -2)

	def test_unv_write_failing_check_extension(self):
		unv_handler = uh.UnvHandler()
		mesh_points = unv_handler.parse('tests/test_datasets/test_square.unv')
		with self.assertRaises(ValueError):
			unv_handler.write(
				mesh_points, 'tests/test_datasets/test_square.iges'
			)

	def test_unv_write_failing_infile_instantiation(self):
		unv_handler = uh.UnvHandler()
		mesh_points = np.zeros((20, 3))
		with self.assertRaises(RuntimeError):
			unv_handler.write(
				mesh_points, 'tests/test_datasets/test_square_out.unv'
			)

	def test_unv_write_outfile(self):
		unv_handler = uh.UnvHandler()
		mesh_points = unv_handler.parse('tests/test_datasets/test_square.unv')
		outfilename = 'tests/test_datasets/test_square_out.unv'
		unv_handler.write(mesh_points, outfilename)
		self.assertEqual(unv_handler.outfile, outfilename)
		self.addCleanup(os.remove, outfilename)

	def test_unv_write_comparison_1(self):
		unv_handler = uh.UnvHandler()
		mesh_points = unv_handler.parse('tests/test_datasets/test_square.unv')

		outfilename = 'tests/test_datasets/test_square_out.unv'
		outfilename_expected = 'tests/test_datasets/test_square.unv'

		unv_handler.write(mesh_points, outfilename)
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		self.addCleanup(os.remove, outfilename)

	def test_unv_write_comparison_2(self):
		unv_handler = uh.UnvHandler()
		mesh_points = unv_handler.parse('tests/test_datasets/test_square.unv')

		mesh_points[0][0] = 2.2
		mesh_points[5][1] = 4.3
		mesh_points[9][2] = 0.5
		mesh_points[45][0] = 7.2
		mesh_points[132][1] = -1.2
		mesh_points[255][2] = -3.6

		outfilename = 'tests/test_datasets/test_square_out.unv'
		outfilename_expected = 'tests/test_datasets/test_square_out_true.unv'

		unv_handler.write(mesh_points, outfilename)
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		self.addCleanup(os.remove, outfilename)
