
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
		
		util.write_initial_box(params, 'test_on_sphere')
		os.remove('originalBox_test_on_sphere.vtk')
		
	
	def test_utilities_write_modified_box(self):
		params = pars.FFDParameters()
		params.read_parameters(filename='tests/test_datasets/parameters_test_ffd_sphere.prm')
		
		util.write_modified_box(params, 'test_on_sphere')
		os.remove('modifiedBox_test_on_sphere.vtk')
		
		
	def test_utilities_write_vtk_box(self):
		box_points = np.array([[-20., -4.37166401, -42.9398302, -27.31149421, 1.40315296, 17.03148895, \
		-21.53667724, -5.90834125, 22.80630592, 38.43464191, -0.13352429, 15.4948117], \
		[-55.      ,        -55.,  31.93332437,  31.93332437, -49.17657149, -49.17657149, \
		37.75675288, 37.75675288, -43.35314297, -43.35314297,   43.5801814,  43.5801814], \
		[-45.       , 43.63269777, -40.955089  , 47.67760877, -48.77395334, 39.85874443 , \
		-44.72904234, 43.90365543, -52.54790668, 36.08479109, -48.50299568, 40.12970209]])
		
		outfilename = 'tests/test_datasets/originalBox_test_on_sphere.vtk'
		
		util.write_vtk_box(box_points, outfilename)
		os.remove(outfilename)


	def test_utilities_check_vtk_box(self):
		box_points = np.array([[-20., -4.37166401, -42.9398302, -27.31149421, 1.40315296, 17.03148895, \
		-21.53667724, -5.90834125, 22.80630592, 38.43464191, -0.13352429, 15.4948117], \
		[-55.      ,        -55.,  31.93332437,  31.93332437, -49.17657149, -49.17657149, \
		37.75675288, 37.75675288, -43.35314297, -43.35314297,   43.5801814,  43.5801814], \
		[-45.       , 43.63269777, -40.955089  , 47.67760877, -48.77395334, 39.85874443 , \
		-44.72904234, 43.90365543, -52.54790668, 36.08479109, -48.50299568, 40.12970209]])
		
		outfilename = 'tests/test_datasets/box_test_sphere.vtk'
		outfilename_expected = 'tests/test_datasets/box_test_sphere_true.vtk'
		
		util.write_vtk_box(box_points, outfilename)
		
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)
		
