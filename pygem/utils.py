"""
Auxiliary utilities for PyGeM.
"""
import vtk
import numpy as np
import matplotlib.pyplot as plt


def write_bounding_box(parameters, outfile, write_deformed=True):
	"""
	Method that writes a vtk file containing the FFD lattice. This method allows
	to visualize where the FFD control points are located before the geometrical
	morphing. If the `write_deformed` flag is set to True the method writes out
	the deformed lattice, otherwise it writes one the original undeformed
	lattice.

	:param FFDParameters parameters: parameters of the Free Form Deformation.
	:param string outfile: name of the output file.
	:param bool write_deformed: flag to write the original or modified FFD
		control lattice.  The default is set to True.

	:Example:

	>>> import pygem.utils as ut
	>>> import pygem.params as pars
	>>> import numpy as np

	>>> params = pars.FFDParameters()
	>>> params.read_parameters(filename='tests/test_datasets/parameters_test_ffd_sphere.prm')
	>>> ut.write_bounding_box(params, 'tests/test_datasets/box_test_sphere.vtk')
	"""
	aux_x = np.linspace(
		0, parameters.lenght_box[0], parameters.n_control_points[0]
	)
	aux_y = np.linspace(
		0, parameters.lenght_box[1], parameters.n_control_points[1]
	)
	aux_z = np.linspace(
		0, parameters.lenght_box[2], parameters.n_control_points[2]
	)
	lattice_y_coords, lattice_x_coords, lattice_z_coords = np.meshgrid(
		aux_y, aux_x, aux_z
	)

	if write_deformed:
		box_points = np.array([ \
		 lattice_x_coords.ravel() + parameters.array_mu_x.ravel() * parameters.lenght_box[0], \
		 lattice_y_coords.ravel() + parameters.array_mu_y.ravel() * parameters.lenght_box[1], \
		 lattice_z_coords.ravel() + parameters.array_mu_z.ravel() * parameters.lenght_box[2]])
	else:
		box_points = np.array([lattice_x_coords.ravel(), lattice_y_coords.ravel(), \
		 lattice_z_coords.ravel()])

	n_rows = box_points.shape[1]

	box_points = np.dot(parameters.rotation_matrix, box_points) + \
	 np.transpose(np.tile(parameters.origin_box, (n_rows, 1)))

	# step necessary to set the correct order to the box points for vtkStructuredGrid:
	# Data in vtkStructuredGrid are ordered with x increasing fastest, then y, then z
	dims = lattice_y_coords.shape
	aux_xx = box_points[0, :].reshape(dims).ravel(order='f')
	aux_yy = box_points[1, :].reshape(dims).ravel(order='f')
	aux_zz = box_points[2, :].reshape(dims).ravel(order='f')
	reordered_box_points = np.array((aux_xx, aux_yy, aux_zz))

	_write_vtk_box(reordered_box_points, outfile, parameters.n_control_points)


def _write_vtk_box(box_points, filename, dimensions):
	"""
	Private method that writes a vtk file containing FFD control points.
	
	:param numpy.ndarray box_points: coordinates of the FFD control points.
	:param string filename: name of the output file.
	:param list dimensions: dimension of the lattice in (x, y, z) directions.
	
	.. warning::
			If you want to visualize in paraview the inner points, 
			you have to slice the lattice because paraview does not visualize them automatically
			even in the wireframe visualization.
	"""
	# setup points and vertices
	points = vtk.vtkPoints()

	for index in range(0, box_points.shape[1]):
		points.InsertNextPoint(
			box_points[0, index], box_points[1, index], box_points[2, index]
		)

	grid = vtk.vtkStructuredGrid()

	grid.SetPoints(points)
	grid.SetDimensions(dimensions)
	grid.Modified()

	writer = vtk.vtkStructuredGridWriter()
	writer.SetFileName(filename)

	if vtk.VTK_MAJOR_VERSION <= 5:
		grid.Update()
		writer.SetInput(grid)
	else:
		writer.SetInputData(grid)

	writer.Write()


def plot_rbf_control_points(parameters, save_fig=False):
	"""
	Method to plot the control points of a RBFParameters class. It is possible to save the
	resulting figure.

	:param RBFParameters parameters: parameters of the Radial Basis Functions interpolation.
	:param bool save_fig: a flag to save the figure in png or not. If True the
		plot is not shown and the figure is saved with the name 'RBF_control_points.png'.
		The default value is False.
	"""
	fig = plt.figure(1)
	axes = fig.add_subplot(111, projection='3d')
	orig = axes.scatter(parameters.original_control_points[:, 0], \
	 parameters.original_control_points[:, 1], \
	 parameters.original_control_points[:, 2], c='blue', marker='o')
	defor = axes.scatter(parameters.deformed_control_points[:, 0], \
	 parameters.deformed_control_points[:, 1], \
	 parameters.deformed_control_points[:, 2], c='red', marker='x')

	axes.set_xlabel('X axis')
	axes.set_ylabel('Y axis')
	axes.set_zlabel('Z axis')

	plt.legend((orig, defor), \
	 ('Original', 'Deformed'), \
	 scatterpoints=1, \
	 loc='lower left', \
	 ncol=2, \
	 fontsize=10)

	# Show the plot to the screen
	if not save_fig:
		plt.show()
	else:
		fig.savefig('RBF_control_points.png')
