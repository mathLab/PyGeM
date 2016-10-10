from unittest import TestCase
import unittest
import pygem.utils as ut
import pygem.params as pars
import numpy as np
import filecmp
import os
import vtk


class TestUtils(TestCase):
	def test_utils_write_original_box(self):
		params = pars.FFDParameters()
		params.read_parameters(
			filename='tests/test_datasets/parameters_test_ffd_sphere.prm'
		)

		outfilename = 'tests/test_datasets/box_test_sphere.vtk'

		ut.write_bounding_box(params, outfilename, write_deformed=False)
		os.remove(outfilename)

	def test_utils_write_modified_box(self):
		params = pars.FFDParameters()
		params.read_parameters(
			filename='tests/test_datasets/parameters_test_ffd_sphere.prm'
		)

		outfilename = 'tests/test_datasets/box_test_sphere.vtk'

		ut.write_bounding_box(params, outfilename)
		os.remove(outfilename)

	def test_utils_check_vtk_original_box(self):
		params = pars.FFDParameters()
		params.read_parameters(
			filename='tests/test_datasets/parameters_test_ffd_sphere.prm'
		)

		outfilename = 'tests/test_datasets/box_test_sphere.vtk'
		if vtk.VTK_MAJOR_VERSION <= 5:
			outfilename_expected = 'tests/test_datasets/box_test_sphere_true_version5.vtk'
		else:
			outfilename_expected = 'tests/test_datasets/box_test_sphere_true_version6.vtk'

		ut.write_bounding_box(params, outfilename, write_deformed=False)

		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)

	def test_utils_check_vtk_modified_box(self):
		params = pars.FFDParameters()
		params.read_parameters(
			filename='tests/test_datasets/parameters_test_ffd_sphere.prm'
		)

		outfilename = 'tests/test_datasets/box_test_sphere.vtk'
		if vtk.VTK_MAJOR_VERSION <= 5:
			outfilename_expected = 'tests/test_datasets/box_modified_test_sphere_true_version5.vtk'
		else:
			outfilename_expected = 'tests/test_datasets/box_modified_test_sphere_true_version6.vtk'

		ut.write_bounding_box(params, outfilename)

		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)

	def test_utils_plot_rbf_control_points(self):
		params = pars.RBFParameters()
		params.read_parameters(
			filename='tests/test_datasets/parameters_rbf_cube.prm'
		)
		ut.plot_rbf_control_points(params, save_fig=False)

	def test_utils_plot_rbf_control_points_save_fig(self):
		params = pars.RBFParameters()
		params.read_parameters(
			filename='tests/test_datasets/parameters_rbf_cube.prm'
		)
		ut.plot_rbf_control_points(params, save_fig=True)
		self.assertTrue(os.path.isfile('RBF_control_points.png'))
		os.remove('RBF_control_points.png')
