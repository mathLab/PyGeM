
from unittest import TestCase
import unittest
import pygem.vtkhandler as vh
import numpy as np
import filecmp
import os


class TestVtkHandler(TestCase):


	def test_vtk_instantiation(self):
		vtk_handler = vh.VtkHandler()
	

	def test_vtk_default_infile_member(self):
		vtk_handler = vh.VtkHandler()
		assert vtk_handler.infile == None


	def test_vtk_default_outfile_member(self):
		vtk_handler = vh.VtkHandler()
		assert vtk_handler.outfile == None


	def test_vtk_default_extension_member(self):
		vtk_handler = vh.VtkHandler()
		assert vtk_handler.extension == '.vtk'
	

	def test_vtk_parse_failing_filename_type(self):
		vtk_handler = vh.VtkHandler()
		with self.assertRaises(TypeError):
			mesh_points = vtk_handler.parse(5.2)

	
	def test_vtk_parse_failing_check_extension(self):
		vtk_handler = vh.VtkHandler()
		with self.assertRaises(ValueError):
			mesh_points = vtk_handler.parse('tests/test_datasets/test_square.iges')


	def test_vtk_parse_infile(self):
		vtk_handler = vh.VtkHandler()
		mesh_points = vtk_handler.parse('tests/test_datasets/test_red_blood_cell.vtk')
		assert vtk_handler.infile == 'tests/test_datasets/test_red_blood_cell.vtk'


	def test_vtk_parse_shape(self):
		vtk_handler = vh.VtkHandler()
		mesh_points = vtk_handler.parse('tests/test_datasets/test_red_blood_cell.vtk')
		assert mesh_points.shape == (500, 3)


	def test_vtk_parse_coords_1(self):
		vtk_handler = vh.VtkHandler()
		mesh_points = vtk_handler.parse('tests/test_datasets/test_red_blood_cell.vtk')
		np.testing.assert_almost_equal(mesh_points[33][0], -2.2977099)


	def test_vtk_parse_coords_2(self):
		vtk_handler = vh.VtkHandler()
		mesh_points = vtk_handler.parse('tests/test_datasets/test_red_blood_cell.vtk')
		np.testing.assert_almost_equal(mesh_points[178][1], 0.143506)


	def test_vtk_parse_coords_3(self):
		vtk_handler = vh.VtkHandler()
		mesh_points = vtk_handler.parse('tests/test_datasets/test_red_blood_cell.vtk')
		np.testing.assert_almost_equal(mesh_points[100][2], 2.3306999)


	def test_vtk_parse_coords_4(self):
		vtk_handler = vh.VtkHandler()
		mesh_points = vtk_handler.parse('tests/test_datasets/test_red_blood_cell.vtk')
		np.testing.assert_almost_equal(mesh_points[0][0], -3.42499995)
		
		
	def test_vtk_parse_coords_5(self):
		vtk_handler = vh.VtkHandler()
		mesh_points = vtk_handler.parse('tests/test_datasets/test_red_blood_cell.vtk')
		np.testing.assert_almost_equal(mesh_points[-1][2], -2.8480699)		


	def test_vtk_write_failing_filename_type(self):
		vtk_handler = vh.VtkHandler()
		mesh_points = vtk_handler.parse('tests/test_datasets/test_red_blood_cell.vtk')
		with self.assertRaises(TypeError):
			vtk_handler.write(mesh_points, -2)


	def test_vtk_write_failing_check_extension(self):
		vtk_handler = vh.VtkHandler()
		mesh_points = vtk_handler.parse('tests/test_datasets/test_red_blood_cell.vtk')
		with self.assertRaises(ValueError):
			vtk_handler.write(mesh_points, 'tests/test_datasets/test_square.iges')


	def test_vtk_write_failing_infile_instantiation(self):
		vtk_handler = vh.VtkHandler()
		mesh_points = np.zeros((20, 3))
		with self.assertRaises(RuntimeError):
 			vtk_handler.write(mesh_points, 'tests/test_datasets/test_red_blood_cell_out.vtk')


	def test_vtk_write_outfile(self):
		vtk_handler = vh.VtkHandler()
		mesh_points = vtk_handler.parse('tests/test_datasets/test_red_blood_cell.vtk')
		outfilename = 'tests/test_datasets/test_red_blood_cell_out.vtk'
		vtk_handler.write(mesh_points, outfilename)
		assert vtk_handler.outfile == outfilename
		os.remove(outfilename)


	def test_vtk_write_comparison(self):
		import vtk
		vtk_handler = vh.VtkHandler()
		mesh_points = vtk_handler.parse('tests/test_datasets/test_red_blood_cell.vtk')
		mesh_points[0][0] = 2.2
		mesh_points[5][1] = 4.3
		mesh_points[9][2] = 0.5
		mesh_points[45][0] = 7.2
		mesh_points[132][1] = -1.2
		mesh_points[255][2] = -3.6

		outfilename = 'tests/test_datasets/test_red_blood_cell_out.vtk'
		if vtk.VTK_MAJOR_VERSION <= 5:
			outfilename_expected = 'tests/test_datasets/test_red_blood_cell_out_true_version5.vtk'
		else:
			outfilename_expected = 'tests/test_datasets/test_red_blood_cell_out_true_version6.vtk'
		
		vtk_handler.write(mesh_points, outfilename)
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)


	def test_vtk_plot_failing_outfile_type(self):
		vtk_handler = vh.VtkHandler()
		with self.assertRaises(TypeError):
			vtk_handler.plot(plot_file=1.1)


	def test_vtk_plot_save_fig_infile(self):
		vtk_handler = vh.VtkHandler()
		mesh_points = vtk_handler.parse('tests/test_datasets/test_red_blood_cell.vtk')
		vtk_handler.plot(save_fig=True)
		self.assertTrue(os.path.isfile('tests/test_datasets/test_red_blood_cell.png'))
		os.remove('tests/test_datasets/test_red_blood_cell.png')


	def test_vtk_plot_save_fig_plot_file(self):
		vtk_handler = vh.VtkHandler()
		vtk_handler.plot(plot_file='tests/test_datasets/test_red_blood_cell.vtk', save_fig=True)
		self.assertTrue(os.path.isfile('tests/test_datasets/test_red_blood_cell.png'))
		os.remove('tests/test_datasets/test_red_blood_cell.png')


	def test_vtk_show_failing_outfile_type(self):
		vtk_handler = vh.VtkHandler()
		with self.assertRaises(TypeError):
			vtk_handler.show(show_file=1.1)

