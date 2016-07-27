
from unittest import TestCase
import unittest
import pygem.radial as rad
import pygem.params as rbfp
import numpy as np

#parameters_rbf_cube.prm
#parameters_rbf_default.prm


class TestRadial(TestCase):

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
	
	def test_rbf_parameters_member(self):
		params = rbfp.RBFParameters()
		params.read_parameters(filename='tests/test_datasets/parameters_rbf_default.prm')
		rbf = rad.RBF(params, self.get_cube_mesh_points())
		assert rbf.parameters == params


	def test_rbf_original_mesh_points_member(self):
		params = rbfp.RBFParameters()
		params.read_parameters(filename='tests/test_datasets/parameters_rbf_default.prm')
		rbf = rad.RBF(params, self.get_cube_mesh_points())
		np.testing.assert_array_almost_equal(rbf.original_mesh_points, self.get_cube_mesh_points())


	def test_rbf_default_modified_mesh_points_member(self):
		params = rbfp.RBFParameters()
		params.read_parameters(filename='tests/test_datasets/parameters_rbf_default.prm')
		rbf = rad.RBF(params, self.get_cube_mesh_points())
		assert rbf.modified_mesh_points == None


	def test_rbf_modified_mesh_points_member(self):
		params = rbfp.RBFParameters()
		params.read_parameters(filename='tests/test_datasets/parameters_rbf_default.prm')
		rbf = rad.RBF(params, self.get_cube_mesh_points())
		rbf.perform()
		np.testing.assert_array_almost_equal(rbf.modified_mesh_points, self.get_cube_mesh_points())


	def test_rbf_original_mesh_points_member(self):
		params = rbfp.RBFParameters()
		params.read_parameters(filename='tests/test_datasets/parameters_rbf_cube.prm')
		rbf = rad.RBF(params, self.get_cube_mesh_points())
		weights_true = np.load('tests/test_datasets/weights_rbf_cube.npy')
		np.testing.assert_array_almost_equal(rbf.weights, weights_true)


	def test_rbf_cube_mod(self):
		params = rbfp.RBFParameters()
		params.read_parameters(filename='tests/test_datasets/parameters_rbf_cube.prm')
		mesh_points_ref = np.load('tests/test_datasets/meshpoints_cube_mod_rbf.npy')
		rbf = rad.RBF(params, self.get_cube_mesh_points())
		rbf.perform()
		mesh_points_test = rbf.modified_mesh_points
		np.testing.assert_array_almost_equal(mesh_points_test, mesh_points_ref)


	def test_wrong_basis(self):
		params = rbfp.RBFParameters()
		params.read_parameters('tests/test_datasets/parameters_rbf_bugged_02.prm')
		with self.assertRaises(NameError):
			rbf = rad.RBF(params, self.get_cube_mesh_points())

