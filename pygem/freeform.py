"""
Utilities for performing Free Form Deformation (FFD)

:Theoretical Insight:

	Free Form Deformation is a technique for the efficient, smooth and accurate geometrical 
	parametrization. It has been proposed the first time in *Sederberg, Thomas W., and Scott 
	R. Parry. "Free-form deformation of solid geometric models." ACM SIGGRAPH computer 
	graphics 20.4 (1986): 151-160*. It consists in three different step:
	
	- Mapping the physical domain to the reference one with map :math:`\\boldsymbol{\psi}`. 
	  In the code it is named *transformation*.
	  
	- Moving some control points to deform the lattice with :math:`\\hat{T}`.
	  The movement of the control points is basically the weight (or displacement) 
	  :math:`\\boldsymbol{\mu}` we set in the *parameters file*.
	  
	- Mapping back to the physical domain with map :math:`\\boldsymbol{\psi}^-1`.
	  In the code it is named *inverse_transformation*.
	  
	FFD map (:math:`T`) is the composition of the three maps, that is
	
	.. math::
		T(\\cdot, \\boldsymbol{\\mu}) = (\\Psi^{-1} \\circ \\hat{T} \\circ \\Psi) 
		(\\cdot, \\boldsymbol{\\mu})
			
	In this way, every point inside the FFD box changes it position according to
	
	.. math::
		\\boldsymbol{P} = \\boldsymbol{\psi}^-1 \\left( \\sum_{l=0} ^L \\sum_{m=0} ^M 
		\\sum_{n=0} ^N \\mathsf{b}_{lmn}(\\boldsymbol{\\psi}(\\boldsymbol{P}_0)) 
		\\boldsymbol{\\mu}_{lmn} \\right)
		
	where :math:`\\mathsf{b}_{lmn}` are Bernstein polynomials.
	We improve the traditional version by allowing a rotation of the FFD lattice in order
	to give more flexibility to the tool.
	
	You can try to add more shapes to the lattice to allow more and more involved transformations.

"""
import numpy as np
from scipy import special
import pygem.affine as at


class FFD(object):
	"""
	Class that handles the Free Form Deformation on the mesh points.

	:param FFDParameters ffd_parameters: parameters of the Free Form Deformation.
	:param numpy.ndarray original_mesh_points: coordinates of the original points of the mesh.

	:cvar FFDParameters parameters: parameters of the Free Form Deformation.
	:cvar numpy.ndarray original_mesh_points: coordinates of the original points of the mesh.
		The shape is `n_points`-by-3.
	:cvar numpy.ndarray modified_mesh_points: coordinates of the points of the deformed mesh.
		The shape is `n_points`-by-3.

	:Example:

	>>> import pygem.freeform as ffd
	>>> import pygem.params as ffdp
	>>> import numpy as np

	>>> ffd_parameters = ffdp.FFDParameters()
	>>> ffd_parameters.read_parameters('tests/test_datasets/parameters_test_ffd_sphere.prm')
	>>> original_mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
	>>> free_form = ffd.FFD(ffd_parameters, original_mesh_points)
	>>> free_form.perform()
	>>> new_mesh_points = free_form.modified_mesh_points
			
    """
	def __init__(self, ffd_parameters, original_mesh_points):
		self.parameters = ffd_parameters
		self.original_mesh_points = original_mesh_points
		self.modified_mesh_points = None


	def perform(self):
		"""
		This method performs the deformation on the mesh points. After the execution
		it sets `self.modified_mesh_points`.

		.. todo::
			In order to improve the performances, we need to perform the FFD only on the points inside
			the lattice.
		"""
		(n_rows_mesh, n_cols_mesh) = self.original_mesh_points.shape

		# translation and then affine transformation
		translation = self.parameters.origin_box

		fisical_frame = np.array([self.parameters.position_vertex_1 - translation, \
			  self.parameters.position_vertex_2 - translation, \
			  self.parameters.position_vertex_3 - translation, \
			  [0, 0, 0]])
		reference_frame = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])

		transformation = at.affine_points_fit(fisical_frame, reference_frame)
		inverse_transformation = at.affine_points_fit(reference_frame, fisical_frame)

		# Initialization. In order to exploit the contiguity in memory the following are transposed
		(dim_n_mu, dim_m_mu, dim_t_mu) = self.parameters.array_mu_x.shape
		bernstein_x = np.zeros((dim_n_mu, n_rows_mesh))
		bernstein_y = np.zeros((dim_m_mu, n_rows_mesh))
		bernstein_z = np.zeros((dim_t_mu, n_rows_mesh))
		shift_mesh_points = np.zeros((n_cols_mesh, n_rows_mesh))

		reference_frame_mesh_points = self._transform_points(self.original_mesh_points - translation, \
															 transformation)

		for i in range(0, dim_n_mu):
			aux1 = np.power((1-reference_frame_mesh_points[:, 0]), dim_n_mu-1-i)
			aux2 = np.power(reference_frame_mesh_points[:, 0], i)
			bernstein_x[i, :] = special.binom(dim_n_mu-1, i) * np.multiply(aux1, aux2)

		for i in range(0, dim_m_mu):
			aux1 = np.power((1-reference_frame_mesh_points[:, 1]), dim_m_mu-1-i)
			aux2 = np.power(reference_frame_mesh_points[:, 1], i)
			bernstein_y[i, :] = special.binom(dim_m_mu-1, i) * np.multiply(aux1, aux2)

		for i in range(0, dim_t_mu):
			aux1 = np.power((1-reference_frame_mesh_points[:, 2]), dim_t_mu-1-i)
			aux2 = np.power(reference_frame_mesh_points[:, 2], i)
			bernstein_z[i, :] = special.binom(dim_t_mu-1, i) * np.multiply(aux1, aux2)


		aux_x = 0.
		aux_y = 0.
		aux_z = 0.
		for j in range(0, dim_m_mu):
			for k in range(0, dim_t_mu):
				bernstein_yz = np.multiply(bernstein_y[j, :], bernstein_z[k, :])
				for i in range(0, dim_n_mu):
					aux = np.multiply(bernstein_x[i, :], bernstein_yz)
					aux_x += aux * self.parameters.array_mu_x[i, j, k]
					aux_y += aux * self.parameters.array_mu_y[i, j, k]
					aux_z += aux * self.parameters.array_mu_z[i, j, k]
		shift_mesh_points[0, :] += aux_x
		shift_mesh_points[1, :] += aux_y
		shift_mesh_points[2, :] += aux_z

		# Splitting points inside and outside the lattice: TODO not very efficient
		for i in range(0, n_rows_mesh):
			if (reference_frame_mesh_points[i, 0] < 0) or (reference_frame_mesh_points[i, 1] < 0) \
			or (reference_frame_mesh_points[i, 2] < 0) or (reference_frame_mesh_points[i, 0] > 1) \
			or (reference_frame_mesh_points[i, 1] > 1) or (reference_frame_mesh_points[i, 2] > 1):
				shift_mesh_points[:, i] = 0

		# Here shift_mesh_points needs to be transposed to be summed with reference_frame_mesh_points
		self.modified_mesh_points = self._transform_points(np.transpose(shift_mesh_points) + \
									reference_frame_mesh_points, inverse_transformation) + translation


	def _transform_points(self, original_points, transformation):
		"""
		This method transforms the points according to the affine transformation taken from
		affine_points_fit method.

		:param numpy.ndarray original_points: coordinates of the original points.
		:param function transformation: affine transformation taken from affine_points_fit method.

		:return: modified_points: coordinates of the modified points.
		:rtype: numpy.ndarray
		"""
		n_rows_mesh, n_cols_mesh = original_points.shape
		modified_points = np.zeros((n_rows_mesh, n_cols_mesh))

		for i in range(0, n_rows_mesh):
			modified_points[i, :] = transformation(original_points[i])

		return modified_points
