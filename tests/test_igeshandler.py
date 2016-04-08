
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


	def test_iges_default_outfile_member(self):
		iges_handler = ih.IgesHandler()
		assert iges_handler.outfile == None


	def test_iges_default_extension_member(self):
		iges_handler = ih.IgesHandler()
		assert iges_handler.extension == '.iges'
	

	def test_iges_parse_failing_filename_type(self):
		iges_handler = ih.IgesHandler()
		with self.assertRaises(TypeError):
			mesh_points = iges_handler.parse(5.2)

	
	def test_iges_parse_failing_check_extension(self):
		iges_handler = ih.IgesHandler()
		with self.assertRaises(ValueError):
			mesh_points = iges_handler.parse('tests/test_datasets/test_pipe.vtk')


	def test_iges_parse_infile(self):
		iges_handler = ih.IgesHandler()
		mesh_points, __ = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		assert iges_handler.infile == 'tests/test_datasets/test_pipe.iges'


	def test_iges_parse_shape(self):
		iges_handler = ih.IgesHandler()
		mesh_points, __ = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		assert mesh_points.shape == (32, 3)
		
	
	def test_iges_parse_shape_position(self):
		iges_handler = ih.IgesHandler()
		__, control_point_position = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		assert control_point_position == [0, 6, 12, 18, 24, 28, 32]


	def test_iges_parse_coords_1(self):
		iges_handler = ih.IgesHandler()
		mesh_points, __ = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		np.testing.assert_almost_equal(mesh_points[6][0], -1000.0)


	def test_iges_parse_coords_2(self):
		iges_handler = ih.IgesHandler()
		mesh_points, __ = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		np.testing.assert_almost_equal(mesh_points[8][1], 999.99999997448208)


	def test_iges_parse_coords_3(self):
		iges_handler = ih.IgesHandler()
		mesh_points, __ = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		np.testing.assert_almost_equal(mesh_points[30][2], 10000.0)


	def test_iges_parse_coords_4(self):
		iges_handler = ih.IgesHandler()
		mesh_points, __ = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		np.testing.assert_almost_equal(mesh_points[0][0], 0.0)
		
		
	def test_iges_parse_coords_5(self):
		iges_handler = ih.IgesHandler()
		mesh_points, __ = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		np.testing.assert_almost_equal(mesh_points[-1][2], 10000.0)		


	'''def test_iges_write_failing_filename_type(self):
		iges_handler = ih.IgesHandler()
		mesh_points, __ = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		with self.assertRaises(TypeError):
			iges_handler.write(mesh_points, -2)


	def test_iges_write_failing_check_extension(self):
		iges_handler = ih.IgesHandler()
		mesh_points, __ = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		with self.assertRaises(ValueError):
			iges_handler.write(mesh_points, 'tests/test_datasets/test_square.iges')


	def test_iges_write_failing_infile_instantiation(self):
		iges_handler = ih.IgesHandler()
		mesh_points, __ = np.zeros((20, 3))
		with self.assertRaises(RuntimeError):
 			iges_handler.write(mesh_points, 'tests/test_datasets/test_red_blood_cell_out.iges')


	def test_iges_write_outfile(self):
		iges_handler = ih.IgesHandler()
		mesh_points, __ = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		outfilename = 'tests/test_datasets/test_red_blood_cell_out.iges'
		iges_handler.write(mesh_points, outfilename)
		assert iges_handler.outfile == outfilename
		os.remove(outfilename)


	def test_iges_write_comparison(self):
		import iges
		iges_handler = ih.IgesHandler()
		mesh_points, __ = iges_handler.parse('tests/test_datasets/test_pipe.iges')
		mesh_points[0][0] = 2.2
		mesh_points[5][1] = 4.3
		mesh_points[9][2] = 0.5
		mesh_points[45][0] = 7.2
		mesh_points[132][1] = -1.2
		mesh_points[255][2] = -3.6

		outfilename = 'tests/test_datasets/test_red_blood_cell_out.iges'
		if iges.VTK_MAJOR_VERSION <= 5:
			outfilename_expected = 'tests/test_datasets/test_red_blood_cell_out_true_version5.iges'
		else:
			outfilename_expected = 'tests/test_datasets/test_red_blood_cell_out_true_version6.iges'
		
		iges_handler.write(mesh_points, outfilename)
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)'''
		
	def test_iges_plot_failing_outfile_type(self):
		iges_handler = ih.IgesHandler()
		with self.assertRaises(TypeError):
			iges_handler.plot(plot_file=3)
