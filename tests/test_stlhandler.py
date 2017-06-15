from unittest import TestCase
import unittest
import pygem.stlhandler as sh
import numpy as np
import filecmp
import os


class TestStlHandler(TestCase):
	def test_stl_instantiation(self):
		stl_handler = sh.StlHandler()

	def test_stl_default_infile_member(self):
		stl_handler = sh.StlHandler()
		self.assertIsNone(stl_handler.infile)

	def test_stl_default_outfile_member(self):
		stl_handler = sh.StlHandler()
		self.assertIsNone(stl_handler.outfile)

	def test_stl_default_extension_member(self):
		stl_handler = sh.StlHandler()
		self.assertListEqual(stl_handler.extensions, ['.stl'])

	def test_stl_parse_failing_filename_type(self):
		stl_handler = sh.StlHandler()
		with self.assertRaises(TypeError):
			mesh_points = stl_handler.parse(5.2)

	def test_stl_parse_failing_check_extension(self):
		stl_handler = sh.StlHandler()
		with self.assertRaises(ValueError):
			mesh_points = stl_handler.parse(
				'tests/test_datasets/test_square.iges'
			)

	def test_stl_parse_infile(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse('tests/test_datasets/test_sphere.stl')
		self.assertEqual(stl_handler.infile, 'tests/test_datasets/test_sphere.stl')

	def test_stl_parse_shape(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse('tests/test_datasets/test_sphere.stl')
		self.assertTupleEqual(mesh_points.shape, (1202, 3))

	def test_stl_parse_coords_1(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse('tests/test_datasets/test_sphere.stl')
		np.testing.assert_almost_equal(mesh_points[33][0], -17.51774978)

	def test_stl_parse_coords_2(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse('tests/test_datasets/test_sphere.stl')
		np.testing.assert_almost_equal(mesh_points[998][1], 11.18433952)

	def test_stl_parse_coords_3(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse('tests/test_datasets/test_sphere.stl')
		np.testing.assert_almost_equal(mesh_points[557][2], 2.47205805)

	def test_stl_parse_coords_4(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse('tests/test_datasets/test_sphere.stl')
		np.testing.assert_almost_equal(mesh_points[0][0], -21.31975937)

	def test_stl_parse_coords_5(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse('tests/test_datasets/test_sphere.stl')
		np.testing.assert_almost_equal(mesh_points[-2][2], -39.05963898)

	def test_stl_parse_coords_5_bin(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse(
			'tests/test_datasets/test_sphere_bin.stl'
		)
		np.testing.assert_almost_equal(mesh_points[-2][2], -39.05963898)

	def test_stl_write_failing_filename_type(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse('tests/test_datasets/test_sphere.stl')
		with self.assertRaises(TypeError):
			stl_handler.write(mesh_points, 4.)

	def test_stl_write_failing_check_extension(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse('tests/test_datasets/test_sphere.stl')
		with self.assertRaises(ValueError):
			stl_handler.write(
				mesh_points, 'tests/test_datasets/test_square.iges'
			)

	def test_stl_write_failing_infile_instantiation(self):
		stl_handler = sh.StlHandler()
		mesh_points = np.zeros((40, 3))
		with self.assertRaises(RuntimeError):
			stl_handler.write(
				mesh_points, 'tests/test_datasets/test_sphere_out.stl'
			)

	def test_stl_write_outfile(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse('tests/test_datasets/test_sphere.stl')
		outfilename = 'tests/test_datasets/test_sphere_out.stl'
		stl_handler.write(mesh_points, outfilename)
		self.assertEqual(stl_handler.outfile, outfilename)
		self.addCleanup(os.remove, outfilename)

	def test_stl_write_comparison(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse('tests/test_datasets/test_sphere.stl')
		mesh_points[0] = [-40.2, -20.5, 60.9]
		mesh_points[1] = [-40.2, -10.5, 60.9]
		mesh_points[2] = [-40.2, -10.5, 60.9]
		mesh_points[500] = [-40.2, -20.5, 60.9]
		mesh_points[501] = [-40.2, -10.5, 60.9]
		mesh_points[502] = [-40.2, -10.5, 60.9]
		mesh_points[1000] = [-40.2, -20.5, 60.9]
		mesh_points[1001] = [-40.2, -10.5, 60.9]
		mesh_points[1002] = [-40.2, -10.5, 60.9]

		outfilename = 'tests/test_datasets/test_sphere_out.stl'
		outfilename_expected = 'tests/test_datasets/test_sphere_out_true.stl'

		stl_handler.write(mesh_points, outfilename)
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		self.addCleanup(os.remove, outfilename)

	def test_stl_write_binary_from_binary(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse(
			'tests/test_datasets/test_sphere_bin.stl'
		)
		mesh_points[0] = [-40.2, -20.5, 60.9]
		mesh_points[1] = [-40.2, -10.5, 60.9]
		mesh_points[2] = [-40.2, -10.5, 60.9]
		mesh_points[500] = [-40.2, -20.5, 60.9]
		mesh_points[501] = [-40.2, -10.5, 60.9]
		mesh_points[502] = [-40.2, -10.5, 60.9]
		mesh_points[1000] = [-40.2, -20.5, 60.9]
		mesh_points[1001] = [-40.2, -10.5, 60.9]
		mesh_points[1002] = [-40.2, -10.5, 60.9]

		outfilename = 'tests/test_datasets/test_sphere_out.stl'

		stl_handler.write(mesh_points, outfilename, write_bin=True)
		self.addCleanup(os.remove, outfilename)

	def test_stl_write_binary_from_ascii(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse('tests/test_datasets/test_sphere.stl')
		mesh_points[0] = [-40.2, -20.5, 60.9]
		mesh_points[1] = [-40.2, -10.5, 60.9]
		mesh_points[2] = [-40.2, -10.5, 60.9]
		mesh_points[500] = [-40.2, -20.5, 60.9]
		mesh_points[501] = [-40.2, -10.5, 60.9]
		mesh_points[502] = [-40.2, -10.5, 60.9]
		mesh_points[1000] = [-40.2, -20.5, 60.9]
		mesh_points[1001] = [-40.2, -10.5, 60.9]
		mesh_points[1002] = [-40.2, -10.5, 60.9]

		outfilename = 'tests/test_datasets/test_sphere_out.stl'

		stl_handler.write(mesh_points, outfilename, write_bin=True)
		self.addCleanup(os.remove, outfilename)

	def test_stl_write_ascii_from_binary(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse(
			'tests/test_datasets/test_sphere_bin.stl'
		)
		mesh_points[0] = [-40.2, -20.5, 60.9]
		mesh_points[1] = [-40.2, -10.5, 60.9]
		mesh_points[2] = [-40.2, -10.5, 60.9]
		mesh_points[500] = [-40.2, -20.5, 60.9]
		mesh_points[501] = [-40.2, -10.5, 60.9]
		mesh_points[502] = [-40.2, -10.5, 60.9]
		mesh_points[1000] = [-40.2, -20.5, 60.9]
		mesh_points[1001] = [-40.2, -10.5, 60.9]
		mesh_points[1002] = [-40.2, -10.5, 60.9]

		outfilename = 'tests/test_datasets/test_sphere_out.stl'
		outfilename_expected = 'tests/test_datasets/test_sphere_out_true.stl'

		stl_handler.write(mesh_points, outfilename, write_bin=False)
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		self.addCleanup(os.remove, outfilename)

	def test_stl_plot_save_fig(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse('tests/test_datasets/test_sphere.stl')
		stl_handler.plot(save_fig=True)
		self.assertTrue(os.path.isfile('tests/test_datasets/test_sphere.png'))
		self.addCleanup(os.remove, 'tests/test_datasets/test_sphere.png')

	def test_stl_plot_save_fig_bin(self):
		stl_handler = sh.StlHandler()
		mesh_points = stl_handler.parse(
			'tests/test_datasets/test_sphere_bin.stl'
		)
		stl_handler.plot(save_fig=True)
		self.assertTrue(
			os.path.isfile('tests/test_datasets/test_sphere_bin.png')
		)
		self.addCleanup(os.remove, 'tests/test_datasets/test_sphere_bin.png')

	def test_stl_plot_save_fig_plot_file(self):
		stl_handler = sh.StlHandler()
		stl_handler.plot(
			plot_file='tests/test_datasets/test_sphere.stl', save_fig=True
		)
		self.assertTrue(os.path.isfile('tests/test_datasets/test_sphere.png'))
		self.addCleanup(os.remove, 'tests/test_datasets/test_sphere.png')

	def test_stl_plot_failing_outfile_type(self):
		stl_handler = sh.StlHandler()
		with self.assertRaises(TypeError):
			stl_handler.plot(plot_file=3)

	def test_stl_show_fail(self):
		stl_handler = sh.StlHandler()
		with self.assertRaises(TypeError):
			stl_handler.show(show_file=3)
