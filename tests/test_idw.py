from unittest import TestCase
import unittest
from pygem.idw import IDW
from pygem.params_idw import IDWParameters
import numpy as np

class TestIDW(TestCase):
	def get_cube_mesh_points(self):
		# Points that define a cube
		nx, ny, nz = (20, 20, 20)
		mesh = np.zeros((nx * ny * nz, 3))
		xv = np.linspace(0, 1, nx)
		yv = np.linspace(0, 1, ny)
		zv = np.linspace(0, 1, nz)
		z, y, x = np.meshgrid(zv, yv, xv)
		mesh = np.array([x.ravel(), y.ravel(), z.ravel()])
		original_mesh_points = mesh.T
		return original_mesh_points

	def test_idw(self):
		params = IDWParameters()
		params.read_parameters('tests/test_datasets/parameters_idw_default.prm')
		idw = IDW(params, self.get_cube_mesh_points())

	def test_idw_perform(self):
		params = IDWParameters()
		params.read_parameters('tests/test_datasets/parameters_idw_default.prm')
		IDW(params, self.get_cube_mesh_points()).perform()

	def test_idw_perform_deform(self):
		params = IDWParameters()
		expected_stretch = [ 1.19541593, 1.36081491, 1.42095073]
		params.read_parameters('tests/test_datasets/parameters_idw_deform.prm')
		idw = IDW(params, self.get_cube_mesh_points())
		idw.perform()
		np.testing.assert_array_almost_equal(
				idw.modified_mesh_points[-3], expected_stretch
		)
