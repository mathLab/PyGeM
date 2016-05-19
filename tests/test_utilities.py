
from unittest import TestCase
import unittest
import pygem.utilities as util
import pygem.params as pars
import numpy as np
import filecmp
import os


class TestVtkHandler(TestCase):

	
	def test_utilities_write_original_box(self):
		params = pars.FFDParameters()
		params.read_parameters(filename='tests/test_datasets/parameters_test_ffd_sphere.prm')
		
		outfilename = 'tests/test_datasets/box_test_sphere.vtk'
		
		util.write_bounding_box(params, outfilename, False)
		os.remove(outfilename)
		
	
	def test_utilities_write_modified_box(self):
		params = pars.FFDParameters()
		params.read_parameters(filename='tests/test_datasets/parameters_test_ffd_sphere.prm')
		
		outfilename = 'tests/test_datasets/box_test_sphere.vtk'
		
		util.write_bounding_box(params, outfilename)
		os.remove(outfilename)


	def test_utilities_check_vtk_original_box(self):
		import vtk
		
		params = pars.FFDParameters()
		params.read_parameters(filename='tests/test_datasets/parameters_test_ffd_sphere.prm')
		
		outfilename = 'tests/test_datasets/box_test_sphere.vtk'
		if vtk.VTK_MAJOR_VERSION <= 5:
			outfilename_expected = 'tests/test_datasets/box_test_sphere_true_version5.vtk'
		else:
			outfilename_expected = 'tests/test_datasets/box_test_sphere_true_version6.vtk'
		
		util.write_bounding_box(params, outfilename, False)
		
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)
		
		
	def test_utilities_check_vtk_modified_box(self):
		import vtk
		
		params = pars.FFDParameters()
		params.read_parameters(filename='tests/test_datasets/parameters_test_ffd_sphere.prm')
		
		outfilename = 'tests/test_datasets/box_test_sphere.vtk'
		if vtk.VTK_MAJOR_VERSION <= 5:
			outfilename_expected = 'tests/test_datasets/box_modified_test_sphere_true_version5.vtk'
		else:
			outfilename_expected = 'tests/test_datasets/box_modified_test_sphere_true_version6.vtk'
		
		util.write_bounding_box(params, outfilename)
		
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)
		
