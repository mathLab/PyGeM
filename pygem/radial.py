"""
Module focused on the implementation of the Radial Basis Functions interpolation technique.
This technique is still based on the use of a set of parameters, the so-called control points,
as for FFD, but RBF is interpolatory. Another important key point of RBF strategy relies in the
way we can locate the control points: in fact, instead of FFD where control points need to be
placed inside a regular lattice, with RBF we hano no more limitations. So we have the possibility
to perform localized control points refiniments.
The module is analogous to the freeform one.

:Theoretical Insight:

	As reference please consult M. D. Buhmann. Radial Basis Functions, volume 12 of Cambridge
	monographs on applied and computational mathematics. Cambridge University Press, UK, 2003.
	RBF shape parametrization technique is based on the definition of a map, 
	:math:`\\mathcal{M}(\\boldsymbol{x}) : \\mathbb{R}^n \\rightarrow \\mathbb{R}^n`, that allows the
	possibility of transferring data across non-matching grids and facing the dynamic mesh handling. 
	The map	introduced is defines as follows

	.. math::
		\\mathcal{M}(\\boldsymbol{x}) = p(\\boldsymbol{x}) + \\sum_{i=1}^{\\mathcal{N}_C} \\gamma_i
		\\varphi(\\| \\boldsymbol{x} - \\boldsymbol{x_{C_i}} \\|)

	where :math:`p(\\boldsymbol{x})` is a low_degree polynomial term, :math:`\\gamma_i` is the weight,
	corresponding to the a-priori selected :math:`\\mathcal{N}_C` control points, associated to the
	:math:`i`-th basis function, and :math:`\\varphi(\\| \\boldsymbol{x} - \\boldsymbol{x_{C_i}} \\|)`
	a radial function based on the Euclidean distance between the control points position 
	:math:`\\boldsymbol{x_{C_i}}` and :math:`\\boldsymbol{x}`. A radial basis function, generally, is
	a real-valued function whose value depends only on the distance from the origin, so that 
	:math:`\\varphi(\\boldsymbol{x}) = \\tilde{\\varphi}(\\| \\boldsymbol{x} \\|)`. 

	The matrix version of the formula above is:

	.. math::
		\\mathcal{M}(\\boldsymbol{x}) = \\boldsymbol{c} + \\boldsymbol{Q}\\boldsymbol{x} +
		\\boldsymbol{W^T}\\boldsymbol{d}(\\boldsymbol{x})

	The idea is that after the computation of the weights and the polynomial terms from the coordinates
	of the control points before and after the deformation, we can deform all the points of the mesh
	accordingly.
	Among the most common used radial basis functions for modelling 2D and 3D shapes, we consider
	Gaussian splines, Multi-quadratic biharmonic splines, Inverted multi-quadratic biharmonic splines,
	Thin-plate splines and Beckert and Wendland :math:`C^2` basis all defined and implemented below.
"""
import os
import params as rbfp
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


class RBF(object):
	"""
	Class that handles the Radial Basis Functions interpolation on the mesh points.

	:param RBFParameters rbf_parameters: parameters of the RBF.
	:param numpy.ndarray original_mesh_points: coordinates of the original points of the mesh.

	:cvar RBFParameters parameters: parameters of the RBF.
	:cvar numpy.ndarray original_mesh_points: coordinates of the original points of the mesh.
		The shape is `n_points`-by-3.
	:cvar numpy.ndarray modified_mesh_points: coordinates of the points of the deformed mesh.
		The shape is `n_points`-by-3.
	:cvar dict bases: a dictionary that associates the names of the basis functions
		implemented to the actual implementation.
	:cvar numpy.matrix weights: the matrix formed by the weights corresponding to the a-priori
		selected N control points, associated to the basis functions and c and Q terms that
		describe the polynomial of order one p(x) = c + Qx. The shape is 
		(n_control_points+1+3)-by-3. It is computed internally.

	:Example:

	>>> import pygem.radial as rbf
	>>> import pygem.params as rbfp
	>>> import numpy as np

	>>> rbf_parameters = rbfp.FFDParameters()
	>>> rbf_parameters.read_parameters('tests/test_datasets/parameters_rbf_cube.prm')

	>>> nx, ny, nz = (20, 20, 20)
	>>> mesh = np.zeros((nx * ny * nz, 3))
	>>> xv = np.linspace(0, 1, nx)
	>>> yv = np.linspace(0, 1, ny)
	>>> zv = np.linspace(0, 1, nz)
	>>> z, y, x = np.meshgrid(zv, yv, xv)
	>>> mesh = np.array([x.ravel(), y.ravel(), z.ravel()])
	>>> original_mesh_points = mesh.T
	
	>>> radial_trans = rbf.RBF(rbf_parameters, original_mesh_points)
	>>> radial_trans.perform()
	>>> new_mesh_points = radial_trans.modified_mesh_points
	"""
	def __init__(self, rbf_parameters, original_mesh_points):
		self.parameters = rbf_parameters
		self.original_mesh_points = original_mesh_points
		self.modified_mesh_points = None
		
		self.bases = {
			'gaussian_spline': self.gaussian_spline,
			'multi_quadratic_biharmonic_spline': self.multi_quadratic_biharmonic_spline,
			'inv_multi_quadratic_biharmonic_spline': self.inv_multi_quadratic_biharmonic_spline,
			'thin_plate_spline': self.thin_plate_spline,
			'beckert_wendland_c2_basis': self.beckert_wendland_c2_basis
		}

		# to make the str callable we have to use a dictionary with all the implemented radial basis functions
		if params.basis in self.bases:
			self.basis = self.bases[params.basis]
		else:
			raise NameError('The name of the basis function in the parameters file is not correct ' + \
					'or not implemented. Check the documentation for all the available functions.')

		self.weights = self._get_weights(self.parameters.original_control_points, \
			self.parameters.deformed_control_points)

	
	@staticmethod
	def gaussian_spline(X, r):
		"""
		It implements the following formula:

		.. math::
			\\varphi(\\| \\boldsymbol{x} \\|) = e^{-\\frac{\\| \\boldsymbol{x} \\|^2}{r^2}}

		:param numpy.ndarray X: the vector x in the formula above.
		:param float r: the parameter r in the formula above.

		:return: result: the result of the formula above.
		:rtype: float
		"""
		norm = np.linalg.norm(X)
		result = np.exp( -(norm * norm) / (r * r) )
		return result
	
	
	@staticmethod
	def multi_quadratic_biharmonic_spline(X, r):
		"""
		It implements the following formula:

		.. math::
			\\varphi(\\| \\boldsymbol{x} \\|) = \\sqrt{\\| \\boldsymbol{x} \\|^2 + r^2}

		:param numpy.ndarray X: the vector x in the formula above.
		:param float r: the parameter r in the formula above.

		:return: result: the result of the formula above.
		:rtype: float
		"""
		norm = np.linalg.norm(X)
		result = np.sqrt( (norm * norm) + (r * r) )
		return result
	
	
	@staticmethod
	def inv_multi_quadratic_biharmonic_spline(X, r):
		"""
		It implements the following formula:

		.. math::
			\\varphi(\\| \\boldsymbol{x} \\|) = (\\| \\boldsymbol{x} \\|^2 + r^2 )^{-\\frac{1}{2}}

		:param numpy.ndarray X: the vector x in the formula above.
		:param float r: the parameter r in the formula above.

		:return: result: the result of the formula above.
		:rtype: float
		"""
		result = 1.0/multi_quadratic_biharmonic_spline(X, r)
		return result
	
	
	@staticmethod
	def thin_plate_spline(X, r):
		"""
		It implements the following formula:

		.. math::
			\\varphi(\\| \\boldsymbol{x} \\|) = \\left\\| \\frac{\\boldsymbol{x} }{r} \\right\\|^2 
			\\ln \\left\\| \\frac{\\boldsymbol{x} }{r} \\right\\|

		:param numpy.ndarray X: the vector x in the formula above.
		:param float r: the parameter r in the formula above.

		:return: result: the result of the formula above.
		:rtype: float
		"""
		arg = X/r
		norm = np.linalg.norm(arg)
		result = norm * norm
		if norm > 0:
			result *= np.log(norm)
		return result
	
	
	@staticmethod
	def beckert_wendland_c2_basis(X, r):
		"""
		It implements the following formula:

		.. math::
			\\varphi(\\| \\boldsymbol{x} \\|) = \\left( 1 - \\frac{\\| \\boldsymbol{x} \\|}{r} \\right)^4_+
			\\left( 4 \\frac{\\| \\boldsymbol{x} \\|}{r} + 1 \\right)

		:param numpy.ndarray X: the vector x in the formula above.
		:param float r: the parameter r in the formula above.

		:return: result: the result of the formula above.
		:rtype: float
		"""
		norm = np.linalg.norm(X)
		arg = norm / r
		first = 0
		if (1 - arg) > 0:
			first = np.power((1 - arg), 4)
		second = (4 * arg) + 1
		result = first * second
		return result
	

	def _distance_matrix(self, X1, X2):
		"""
		This private method returns the following matrix: 
		:math:`\\boldsymbol{D_{ij}} = \\varphi(\\| \\boldsymbol{x_i} - \\boldsymbol{y_j} \\|)`

		:param numpy.ndarray X1: the vector x in the formula above.
		:param numpy.ndarray X2: the vector y in the formula above.

		:return: matrix: the matrix D.
		:rtype: numpy.ndarray
		"""
		m, n = X1.shape[0], X2.shape[0]
		matrix = np.zeros(shape=(m, n))
		for i in range(0, m):
			for j in range(0, n):
				matrix[i][j] = self.basis(X1[i] - X2[j], self.parameters.radius)
		return matrix
	
	
	def _get_weights(self, X, Y):
		"""
		This private method, given the original control points and the deformed ones, returns the matrix
		with the weights and the polynomial terms, that is :math:`W`, :math:`c^T` and :math:`Q^T`.
		The shape is (n_control_points+1+3)-by-3.

		:param numpy.ndarray X: it is an n_control_points-by-3 array with the
			coordinates of the original interpolation control points before the deformation.
		:param numpy.ndarray Y: it is an n_control_points-by-3 array with the
			coordinates of the interpolation control points after the deformation.

		:return: weights: the matrix with the weights and the polynomial terms.
		:rtype: numpy.matrix
		"""
		n_points = X.shape[0]
		dim = X.shape[1]
		identity = np.ones(n_points).reshape(n_points, 1)
		dist = self._distance_matrix(X, X)
		H = np.bmat([[dist, identity, X], [identity.T, np.zeros((1, 1)), np.zeros((1, dim))], \
			[X.T, np.zeros((dim, 1)), np.zeros((dim, dim))]])
		rhs = np.bmat([[Y], [np.zeros((1, dim))], [np.zeros((dim, dim))]])
		inv_H = np.linalg.inv(H)
		weights = np.dot(inv_H, rhs)
		return weights
	
	
	def perform(self):
		"""
		This method performs the deformation of the mesh points. After the execution
		it sets `self.modified_mesh_points`.
		"""
		n_points = self.original_mesh_points.shape[0]
		dim = self.original_mesh_points.shape[1]
		dist = self._distance_matrix(self.original_mesh_points, self.parameters.original_control_points)
		identity = np.ones(n_points).reshape(n_points, 1)
		H = np.bmat([[dist, identity, self.original_mesh_points]])
		self.modified_mesh_points = np.asarray(np.dot(H, self.weights))

