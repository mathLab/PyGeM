"""
Module focused on the Inverse Distance Weighting interpolation technique.
The IDW algorithm is an average moving interpolation that is usually applied to
highly variable data. The main idea of this interpolation strategy lies in
fact that it is not desirable to honour local high/low values but rather to look
at a moving average of nearby data points and estimate the local trends.
The node value is calculated by averaging the weighted sum of all the points.
Data points that lie progressively farther from the node inuence much less the
computed value than those lying closer to the node.

:Theoretical Insight:

	This implementation is based on the simplest form of inverse distance
	weighting interpolation, proposed by D. Shepard, A two-dimensional
	interpolation function for irregularly-spaced data, Proceedings of the 23 rd
	ACM National Conference.

	The interpolation value :math:`u` of a given point :math:`\\mathrm{x}`
	from a set of samples :math:`u_k = u(\\mathrm{x}_k)`, with
	:math:`k = 1,2,\dotsc,\\mathcal{N}`, is given by:

	.. math::
		u(\\mathrm{x}) = \\displaystyle\\sum_{k=1}^\\mathcal{N}
		\\frac{w(\\mathrm{x},\\mathrm{x}_k)}
		{\\displaystyle\\sum_{j=1}^\\mathcal{N} w(\\mathrm{x},\\mathrm{x}_j)}
		u_k


	where, in general, :math:`w(\\mathrm{x}, \\mathrm{x}_i)` represents the
	weighting function:

	.. math::
		w(\\mathrm{x}, \\mathrm{x}_i) = \\| \\mathrm{x} - \\mathrm{x}_i \\|^{-p}
	
	being :math:`\\| \\mathrm{x} - \\mathrm{x}_i \\|^{-p} \\ge 0` is the
	Euclidean distance between :math:`\\mathrm{x}` and data point
	:math:`\\mathrm{x}_i` and :math:`p` is a power parameter, typically equal to
	2.

"""
import numpy as np

from scipy.spatial.distance import cdist


class IDW(object):
	"""
	Class that handles the IDW technique.

	:param idw_parameters: the parameters of the IDW
	:type idw_parameters: :class:`IDWParameters`
	:param numpy.ndarray original_mesh_points: coordinates of the original
		points of the mesh.

	:cvar parameters: the parameters of the IDW.
	:vartype parameters: :class:`~pygem.params_idw.IDWParameters`
	:cvar numpy.ndarray original_mesh_points: coordinates of the original
		points of the mesh.
	:cvar numpy.ndarray modified_mesh_points: coordinates of the deformed
		points of the mesh.

	:Example:

	>>> from pygem.idw import IDW
	>>> from pygem.params_idw import IDWParameters
	>>> import numpy as np
	>>> params = IDWParameters()
	>>> params.read_parameters('tests/test_datasets/parameters_idw_cube.prm')
	>>> nx, ny, nz = (20, 20, 20)
	>>> mesh = np.zeros((nx * ny * nz, 3))
	>>> xv = np.linspace(0, 1, nx)
	>>> yv = np.linspace(0, 1, ny)
	>>> zv = np.linspace(0, 1, nz)
	>>> z, y, x = np.meshgrid(zv, yv, xv)
	>>> mesh = np.array([x.ravel(), y.ravel(), z.ravel()])
	>>> original_mesh_points = mesh.T
	>>> idw = IDW(rbf_parameters, original_mesh_points)
	>>> idw.perform()
	>>> new_mesh_points = idw.modified_mesh_points
	"""

	def __init__(self, idw_parameters, original_mesh_points):
		self.parameters = idw_parameters
		self.original_mesh_points = original_mesh_points
		self.modified_mesh_points = None

	def perform(self):
		"""
		This method performs the deformation of the mesh points. After the
		execution it sets `self.modified_mesh_points`.
		"""

		def distance(u, v):
			return np.linalg.norm(u - v, self.parameters.power)

		# Compute displacement of the control points
		displ = (
			self.parameters.deformed_control_points -
			self.parameters.original_control_points
		)
	
		# Compute the distance between the mesh points and the control points
		dist = cdist(
			self.original_mesh_points, 
			self.parameters.original_control_points,
			distance
		)

		# Weights are set as the reciprocal of the distance if the distance is
		# not zero, otherwise 1.0 where distance is zero.
		weights = np.zeros(dist.shape)
		for i, d in enumerate(dist):
			weights[i] = 1. / d if d.all() else np.where(d == 0.0, 1.0, 0.0)
		
		
		offset = np.array([
			np.sum(displ * wi[:, np.newaxis] / np.sum(wi), axis=0)
			for wi in weights
		])

		self.modified_mesh_points = self.original_mesh_points + offset
