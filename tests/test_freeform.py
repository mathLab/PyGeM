from unittest import TestCase
import unittest
import pygem.freeform as ffd
import pygem.params as ffdp
import numpy as np


class TestFreeform(TestCase):
	def test_ffd_parameters_member(self):
		params = ffdp.FFDParameters()
		params.read_parameters(
			filename='tests/test_datasets/parameters_test_ffd_identity.prm'
		)
		mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
		free_form = ffd.FFD(params, mesh_points)
		assert free_form.parameters == params

	def test_ffd_original_mesh_points_member(self):
		params = ffdp.FFDParameters()
		params.read_parameters(
			filename='tests/test_datasets/parameters_test_ffd_identity.prm'
		)
		mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
		free_form = ffd.FFD(params, mesh_points)
		np.testing.assert_array_almost_equal(
			free_form.original_mesh_points, mesh_points
		)

	def test_ffd_default_modified_mesh_points_member(self):
		params = ffdp.FFDParameters()
		params.read_parameters(
			filename='tests/test_datasets/parameters_test_ffd_identity.prm'
		)
		mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
		free_form = ffd.FFD(params, mesh_points)
		assert free_form.modified_mesh_points == None

	def test_ffd_modified_mesh_points_member(self):
		params = ffdp.FFDParameters()
		params.read_parameters(
			filename='tests/test_datasets/parameters_test_ffd_identity.prm'
		)
		mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
		free_form = ffd.FFD(params, mesh_points)
		free_form.perform()
		np.testing.assert_array_almost_equal(
			free_form.modified_mesh_points, mesh_points
		)

	def test_ffd_sphere_mod(self):
		params = ffdp.FFDParameters()
		params.read_parameters(
			filename='tests/test_datasets/parameters_test_ffd_sphere.prm'
		)
		mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
		mesh_points_ref = np.load(
			'tests/test_datasets/meshpoints_sphere_mod.npy'
		)
		free_form = ffd.FFD(params, mesh_points)
		free_form.perform()
		mesh_points_test = free_form.modified_mesh_points
		np.testing.assert_array_almost_equal(mesh_points_test, mesh_points_ref)
