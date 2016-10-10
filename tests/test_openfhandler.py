from unittest import TestCase
import unittest
import pygem.openfhandler as ofh
import numpy as np
import filecmp
import os


class TestOpenFoamHandler(TestCase):
	def test_open_foam_instantiation(self):
		open_foam_handler = ofh.OpenFoamHandler()

	def test_open_foam_default_infile_member(self):
		open_foam_handler = ofh.OpenFoamHandler()
		assert open_foam_handler.infile == None

	def test_open_foam_default_outfile_member(self):
		open_foam_handler = ofh.OpenFoamHandler()
		assert open_foam_handler.outfile == None

	def test_open_foam_default_extension_member(self):
		open_foam_handler = ofh.OpenFoamHandler()
		assert open_foam_handler.extension == ''

	def test_open_foam_parse_failing_filename_type(self):
		open_foam_handler = ofh.OpenFoamHandler()
		with self.assertRaises(TypeError):
			mesh_points = open_foam_handler.parse(.2)

	def test_open_foam_parse_failing_check_extension(self):
		open_foam_handler = ofh.OpenFoamHandler()
		with self.assertRaises(ValueError):
			mesh_points = open_foam_handler.parse(
				'tests/test_datasets/test_square.iges'
			)

	def test_open_foam_parse_infile(self):
		open_foam_handler = ofh.OpenFoamHandler()
		mesh_points = open_foam_handler.parse(
			'tests/test_datasets/test_openFOAM'
		)
		assert open_foam_handler.infile == 'tests/test_datasets/test_openFOAM'

	def test_open_foam_parse_shape(self):
		open_foam_handler = ofh.OpenFoamHandler()
		mesh_points = open_foam_handler.parse(
			'tests/test_datasets/test_openFOAM'
		)
		assert mesh_points.shape == (21812, 3)

	def test_open_foam_parse_coords_1(self):
		open_foam_handler = ofh.OpenFoamHandler()
		mesh_points = open_foam_handler.parse(
			'tests/test_datasets/test_openFOAM'
		)
		np.testing.assert_almost_equal(mesh_points[33][0], 1.42254)

	def test_open_foam_parse_coords_2(self):
		open_foam_handler = ofh.OpenFoamHandler()
		mesh_points = open_foam_handler.parse(
			'tests/test_datasets/test_openFOAM'
		)
		np.testing.assert_almost_equal(mesh_points[1708][1], -3.13059)

	def test_open_foam_parse_coords_3(self):
		open_foam_handler = ofh.OpenFoamHandler()
		mesh_points = open_foam_handler.parse(
			'tests/test_datasets/test_openFOAM'
		)
		np.testing.assert_almost_equal(mesh_points[3527][2], .0)

	def test_open_foam_parse_coords_4(self):
		open_foam_handler = ofh.OpenFoamHandler()
		mesh_points = open_foam_handler.parse(
			'tests/test_datasets/test_openFOAM'
		)
		np.testing.assert_almost_equal(mesh_points[0][0], -17.5492)

	def test_open_foam_parse_coords_5(self):
		open_foam_handler = ofh.OpenFoamHandler()
		mesh_points = open_foam_handler.parse(
			'tests/test_datasets/test_openFOAM'
		)
		np.testing.assert_almost_equal(mesh_points[-1][2], 0.05)

	def test_open_foam_write_failing_filename_type(self):
		open_foam_handler = ofh.OpenFoamHandler()
		mesh_points = open_foam_handler.parse(
			'tests/test_datasets/test_openFOAM'
		)
		with self.assertRaises(TypeError):
			open_foam_handler.write(mesh_points, -1.)

	def test_open_foam_write_failing_check_extension(self):
		open_foam_handler = ofh.OpenFoamHandler()
		mesh_points = open_foam_handler.parse(
			'tests/test_datasets/test_openFOAM'
		)
		with self.assertRaises(ValueError):
			open_foam_handler.write(
				mesh_points, 'tests/test_datasets/test_square.iges'
			)

	def test_open_foam_write_failing_infile_instantiation(self):
		open_foam_handler = ofh.OpenFoamHandler()
		mesh_points = np.zeros((40, 3))
		with self.assertRaises(RuntimeError):
			open_foam_handler.write(
				mesh_points, 'tests/test_datasets/test_openFOAM_out'
			)

	def test_open_foam_write_outfile(self):
		open_foam_handler = ofh.OpenFoamHandler()
		mesh_points = open_foam_handler.parse(
			'tests/test_datasets/test_openFOAM'
		)
		outfilename = 'tests/test_datasets/test_openFOAM_out'
		open_foam_handler.write(mesh_points, outfilename)
		assert open_foam_handler.outfile == outfilename
		os.remove(outfilename)

	def test_open_foam_write_comparison(self):
		open_foam_handler = ofh.OpenFoamHandler()
		mesh_points = open_foam_handler.parse(
			'tests/test_datasets/test_openFOAM'
		)
		mesh_points[0] = [-14., 1.55, 0.2]
		mesh_points[1] = [-14.3, 2.55, 0.3]
		mesh_points[2] = [-14.3, 2.55, 0.3]
		mesh_points[2000] = [7.8, -42.8, .0]
		mesh_points[2001] = [8.8, -41.8, .1]
		mesh_points[2002] = [9.8, -40.8, .0]
		mesh_points[-3] = [236.3, 183.7, 0.06]
		mesh_points[-2] = [237.3, 183.7, 0.06]
		mesh_points[-1] = [236.3, 185.7, 0.06]

		outfilename = 'tests/test_datasets/test_openFOAM_out'
		outfilename_expected = 'tests/test_datasets/test_openFOAM_out_true'

		open_foam_handler.write(mesh_points, outfilename)
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)
