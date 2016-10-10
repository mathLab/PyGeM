from unittest import TestCase
import unittest
import pygem.affine as at
import numpy as np


class TestAffine(TestCase):
	def test_angles2matrix_rot_default(self):
		mat_exact = np.eye(3)
		mat_test = at.angles2matrix()
		np.testing.assert_array_almost_equal(mat_exact, mat_test)

	def test_angles2matrix_rot_x(self):
		rotz = 0
		roty = 0
		rotx = 50 * np.pi / 180
		mat_exact = np.array([
			1., 0., 0., 0., 0.64278761, -0.76604444, 0., 0.76604444, 0.64278761
		]).reshape((3, 3))

		mat_test = at.angles2matrix(rotz, roty, rotx)
		np.testing.assert_array_almost_equal(mat_exact, mat_test)

	def test_angles2matrix_rot_y(self):
		rotz = 0
		roty = 23 * np.pi / 180
		rotx = 0
		mat_exact = np.array([
			0.92050485, 0., 0.39073113, 0., 1., 0., -0.39073113, 0., 0.92050485
		]).reshape((3, 3))

		mat_test = at.angles2matrix(rotz, roty, rotx)
		np.testing.assert_array_almost_equal(mat_exact, mat_test)

	def test_angles2matrix_rot_z(self):
		rotz = -57 * np.pi / 180
		roty = 0
		rotx = 0
		mat_exact = np.array([
			0.54463904, 0.83867057, 0., -0.83867057, 0.54463904, 0., 0., 0., 1.
		]).reshape((3, 3))

		mat_test = at.angles2matrix(rotz, roty, rotx)
		np.testing.assert_array_almost_equal(mat_exact, mat_test)

	def test_angles2matrix_rot_xyz(self):
		rotz = 10 * np.pi / 180
		roty = 20 * np.pi / 180
		rotx = 30 * np.pi / 180
		mat_exact = np.array([0.92541658, -0.16317591, 0.34202014, 0.31879578, \
		 0.82317294, -0.46984631, -0.20487413, 0.54383814, 0.81379768]).reshape((3,3))

		mat_test = at.angles2matrix(rotz, roty, rotx)
		np.testing.assert_array_almost_equal(mat_exact, mat_test)

	def test_to_reduced_row_echelon_form_1(self):
		matrix = [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]
		rref_matrix = at.to_reduced_row_echelon_form(matrix)
		rref_matrix_exact = [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
		assert rref_matrix == rref_matrix_exact

	def test_to_reduced_row_echelon_form_2(self):
		matrix = [[1., -1., 1.], [-1., 1., -1.], [3., 4., 5.]]
		rref_matrix = at.to_reduced_row_echelon_form(matrix)
		rref_matrix_exact = [[1.0, 0.0, 1.2857142857142856],
							 [0.0, 1.0, 0.2857142857142857], [0.0, 0.0, 0.0]]
		assert rref_matrix == rref_matrix_exact

	def test_to_reduced_row_echelon_form_3(self):
		matrix = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
		rref_matrix = at.to_reduced_row_echelon_form(matrix)
		rref_matrix_exact = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
		assert rref_matrix == rref_matrix_exact

	def test_to_reduced_row_echelon_form_4(self):
		matrix = [[0., 0., 0.], [-3., 6., -3.], [0., 0., 0.]]
		rref_matrix = at.to_reduced_row_echelon_form(matrix)
		rref_matrix_exact = [[1.0, -2.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
		assert rref_matrix == rref_matrix_exact

	def test_to_reduced_row_echelon_form_5(self):
		matrix = [[0., 0., 0.], [0., 0., 0.], [2., 4., 6.]]
		rref_matrix = at.to_reduced_row_echelon_form(matrix)
		rref_matrix_exact = [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
		assert rref_matrix == rref_matrix_exact

	def test_to_reduced_row_echelon_form_6(self):
		matrix = [[0., 0., 0.], [0., 0., 0.], [2., 4., 6.], [1., 0., 0.]]
		rref_matrix = at.to_reduced_row_echelon_form(matrix)
		rref_matrix_exact = [[1.0, 0.0, 0.0], [-0.0, 1.0, 1.5], [0.0, 0.0, 0.0],
							 [0.0, 0.0, 0.0]]
		assert rref_matrix == rref_matrix_exact

	def test_to_reduced_row_echelon_form_7(self):
		matrix = [[0., 0., 0., 4], [0., 0., -3., 2], [2., 4., 6., 0]]
		rref_matrix = at.to_reduced_row_echelon_form(matrix)
		rref_matrix_exact = [[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
							 [0.0, 0.0, 0.0, 1.0]]
		assert rref_matrix == rref_matrix_exact

	def test_affine_points_fit_identity_1(self):
		p_start = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
		p_end = p_start
		v_test = np.array([1., 2., 3.])
		v_exact = v_test

		transformation = at.affine_points_fit(p_start, p_end)
		v_trans = transformation(v_test)
		np.testing.assert_array_almost_equal(v_exact, v_trans)

	def test_affine_points_fit_identity_2(self):
		p_start = np.array(
			[[1, .5, -.3], [0, 2, 4], [-1, 0., -1.5], [1, -4, .5]]
		)
		p_end = p_start
		v_test = np.array([-1., 2.5, .3])
		v_exact = v_test

		transformation = at.affine_points_fit(p_start, p_end)
		v_trans = transformation(v_test)
		np.testing.assert_array_almost_equal(v_exact, v_trans)

	def test_affine_points_fit_rotation(self):
		p_start = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
		p_end = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, 0]])
		v_test = np.array([1., 2., 3.])
		v_exact = np.array([-2., 1., 3.])

		transformation = at.affine_points_fit(p_start, p_end)
		v_trans = transformation(v_test)
		np.testing.assert_array_almost_equal(v_exact, v_trans)

	def test_affine_points_fit_generic(self):
		p_start = np.array(
			[[1, .5, -.3], [0, 2, 4], [-1, 0., -1.5], [1, -4, .5]]
		)
		p_end = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, 0]])
		v_test = np.array([1., 2., 3.])
		v_exact = np.array([-0.68443497, 0.7249467, -0.34221748])

		transformation = at.affine_points_fit(p_start, p_end)
		v_trans = transformation(v_test)
		np.testing.assert_array_almost_equal(v_exact, v_trans)

	def test_affine_points_fit_coplanar(self):
		p_start = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]])
		p_end = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, 0]])
		with self.assertRaises(RuntimeError):
			transformation = at.affine_points_fit(p_start, p_end)

	def test_affine_points_fit_right_points_size(self):
		p_start = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
		p_end = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
		with self.assertRaises(RuntimeError):
			transformation = at.affine_points_fit(p_start, p_end)

	def test_affine_points_fit_under_determined_system_1(self):
		p_start = np.array([[1, 0, 0]])
		p_end = np.array([[0, 1, 0]])
		with self.assertRaises(RuntimeError):
			transformation = at.affine_points_fit(p_start, p_end)

	def test_affine_points_fit_under_determined_system_2(self):
		p_start = np.array([[1, 0, 0], [0, 1, 0]])
		p_end = np.array([[0, 1, 0], [-1, 0, 0]])
		with self.assertRaises(RuntimeError):
			transformation = at.affine_points_fit(p_start, p_end)
