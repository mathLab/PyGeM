
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
		
		util.write_bounding_box(params, outfilename)
		os.remove('tests/test_datasets/box_test_sphere.vtk')
		
	
	def test_utilities_write_modified_box(self):
		params = pars.FFDParameters()
		params.read_parameters(filename='tests/test_datasets/parameters_test_ffd_sphere.prm')
		
		outfilename = 'tests/test_datasets/box_test_sphere.vtk'
		
		util.write_bounding_box(params, outfilename, 'modified')
		os.remove('tests/test_datasets/box_test_sphere.vtk')


	def test_utilities_check_vtk_box(self):
		params = pars.FFDParameters()
		params.read_parameters(filename='tests/test_datasets/parameters_test_ffd_sphere.prm')
		
		outfilename = 'tests/test_datasets/box_test_sphere.vtk'
		outfilename_expected = 'tests/test_datasets/box_test_sphere_true.vtk'
		
		util.write_bounding_box(params, outfilename)
		
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)
		
