from unittest import TestCase
import unittest
import pygem.igeshandler as ih
import numpy as np
import filecmp
import os


class TestIgesHandler(TestCase):
	def test_iges_instantiation(self):
		iges_handler = ih.IgesHandler()

	def test_iges_default_infile_member(self):
		iges_handler = ih.IgesHandler()
		assert iges_handler.infile == None

	def test_iges_default_control_point_position_member(self):
		iges_handler = ih.IgesHandler()
		assert iges_handler._control_point_position == None

	def test_iges_default_outfile_member(self):
		iges_handler = ih.IgesHandler()
		assert iges_handler.outfile == None

	def test_iges_default_tolerance(self):
		iges_handler = ih.IgesHandler()
		assert iges_handler.tolerance == 1e-6

	def test_iges_default_extension_member(self):
		iges_handler = ih.IgesHandler()
		assert iges_handler.extension == ['.iges', '.igs']

	def test_iges_parse_failing_filename_type(self):
		iges_handler = ih.IgesHandler()
		with self.assertRaises(TypeError):
			mesh_points = iges_handler.parse(5.2)

	def test_iges_parse_failing_check_extension(self):
		iges_handler = ih.IgesHandler()
		with self.assertRaises(ValueError):
			mesh_points = iges_handler.parse(
				'tests/test_datasets/test_pipe.vtk'
			)

	def test_iges_parse_infile(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		assert iges_handler.infile == 'tests/test_datasets/test_pipe.iges'

	def test_iges_parse_control_point_position_member(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		assert iges_handler._control_point_position == [
			0, 6, 12, 18, 24, 28, 32
		]

	def test_iges_parse_shape(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		assert mesh_points.shape == (32, 3)

	def test_iges_parse_shape_igs(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.igs')
		assert mesh_points.shape == (32, 3)

	def test_iges_parse_coords_1(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		np.testing.assert_almost_equal(mesh_points[6][0], -1000.0)

	def test_iges_parse_coords_2(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		np.testing.assert_almost_equal(mesh_points[8][1], 999.99999997448208)

	def test_iges_parse_coords_3(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		np.testing.assert_almost_equal(mesh_points[30][2], 10000.0)

	def test_iges_parse_coords_4(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		np.testing.assert_almost_equal(mesh_points[0][0], 0.0)

	def test_iges_parse_coords_5(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		np.testing.assert_almost_equal(mesh_points[-1][2], 10000.0)

	def test_iges_parse_coords_5_igs(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.igs')
		np.testing.assert_almost_equal(mesh_points[-1][2], 10000.0)

	def test_iges_write_failing_filename_type(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		with self.assertRaises(TypeError):
			iges_handler.write(mesh_points, -2)

	def test_iges_write_failing_check_extension(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		with self.assertRaises(ValueError):
			iges_handler.write(
				mesh_points, 'tests/test_datasets/test_square.stl'
			)

	def test_iges_write_failing_infile_instantiation(self):
		iges_handler = ih.IgesHandler()
		mesh_points = np.zeros((20, 3))
		with self.assertRaises(RuntimeError):
			iges_handler.write(
				mesh_points, 'tests/test_datasets/test_pipe_out.iges'
			)

	def test_iges_write_outfile(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		outfilename = 'tests/test_datasets/test_pipe_out.iges'
		iges_handler.write(mesh_points, outfilename)
		assert iges_handler.outfile == outfilename
		os.remove(outfilename)

	def test_iges_write_outfile_tolerance(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		outfilename = 'tests/test_datasets/test_pipe_out.iges'
		iges_handler.write(mesh_points, outfilename, 1e-3)
		assert iges_handler.tolerance == 1e-3
		os.remove(outfilename)

	def test_iges_write_modified_tolerance(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		outfilename = 'tests/test_datasets/test_pipe_out.iges'
		iges_handler.write(mesh_points, outfilename, 1e-3)
		assert iges_handler.outfile == outfilename
		os.remove(outfilename)

	def test_iges_write_comparison_iges(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		mesh_points[0][0] = 2.2
		mesh_points[5][1] = 4.3
		mesh_points[9][2] = 0.5
		mesh_points[12][0] = 7.2
		mesh_points[16][1] = -1.2
		mesh_points[31][2] = -3.6

		outfilename = 'tests/test_datasets/test_pipe_out.iges'
		outfilename_expected = 'tests/test_datasets/test_pipe_out_true.iges'

		iges_handler.write(mesh_points, outfilename)

		mesh_points = iges_handler.parse(outfilename)
		mesh_points_expected = iges_handler.parse(outfilename_expected)
		np.testing.assert_array_almost_equal(mesh_points, mesh_points_expected)
		os.remove(outfilename)

	def test_iges_write_comparison_igs(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.igs')
		mesh_points[0][0] = 2.2
		mesh_points[5][1] = 4.3
		mesh_points[9][2] = 0.5
		mesh_points[12][0] = 7.2
		mesh_points[16][1] = -1.2
		mesh_points[31][2] = -3.6

		outfilename = 'tests/test_datasets/test_pipe_out.igs'
		outfilename_expected = 'tests/test_datasets/test_pipe_out_true.igs'

		iges_handler.write(mesh_points, outfilename)

		mesh_points = iges_handler.parse(outfilename)
		mesh_points_expected = iges_handler.parse(outfilename_expected)
		np.testing.assert_array_almost_equal(mesh_points, mesh_points_expected)
		os.remove(outfilename)

	def test_iges_plot_save_fig(self):
		iges_handler = ih.IgesHandler()
		mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		iges_handler.plot(save_fig=True)
		self.assertTrue(os.path.isfile('tests/test_datasets/test_pipe.png'))
		os.remove('tests/test_datasets/test_pipe.png')

	def test_iges_plot_failing_outfile_type(self):
		iges_handler = ih.IgesHandler()
		with self.assertRaises(TypeError):
			iges_handler.plot(plot_file=3)

	def test_iges_show_failing_outfile_type(self):
		iges_handler = ih.IgesHandler()
		with self.assertRaises(TypeError):
			iges_handler.show(show_file=1.1)
