"""
Auxiliary utilities for PyGeM.
"""

import vtk
import vtk.util.numpy_support as ns
import numpy as np

# TODO: add the connectivity to the ffd control points to visualize the lattice.



def write_bounding_box(parameters, outfile, write_deformed=True):
	"""
	Method that writes a vtk file containing the FFD lattice. This method
	allows to visualize where the FFD control points are located before the geometrical morphing.
	If the flag is set to original (default) the method writes out the undeformed lattice, 
	if it is set to modified it writes out the deformed lattice.
	
	:param FFDParameters parameters: parameters of the Free Form Deformation.
	:param string outfile: name of the output file.
	:param bool write_deformed: flag to write the original or modified FFD control lattice. 
		The default is set to deformed.
		
	:Example:
	
	>>> import pygem.utilities as util
	>>> import pygem.params as pars
	>>> import numpy as np
	
	>>> params = pars.FFDParameters()
	>>> params.read_parameters(filename='tests/test_datasets/parameters_test_ffd_sphere.prm')
	>>> util.write_bounding_box(params, 'tests/test_datasets/box_test_sphere.vtk')
	"""

	aux_x = np.linspace(0, parameters.lenght_box_x, parameters.n_control_points[0])
	aux_y = np.linspace(0, parameters.lenght_box_y, parameters.n_control_points[1])
	aux_z = np.linspace(0, parameters.lenght_box_z, parameters.n_control_points[2])
	lattice_y_coords, lattice_x_coords, lattice_z_coords = np.meshgrid(aux_y, aux_x, aux_z)

	if write_deformed == False:
		box_points = np.array([lattice_x_coords.ravel(), lattice_y_coords.ravel(), lattice_z_coords.ravel()])
	if write_deformed == True:
		box_points = np.array([lattice_x_coords.ravel() + parameters.array_mu_x.ravel()*parameters.lenght_box_x, \
		lattice_y_coords.ravel()  + parameters.array_mu_y.ravel()*parameters.lenght_box_y, \
		lattice_z_coords.ravel()  + parameters.array_mu_z.ravel()*parameters.lenght_box_z])
		
	n_rows = box_points.shape[1]

	box_points = np.dot(parameters.rotation_matrix,box_points) + np.transpose(np.tile(parameters.origin_box, (n_rows,1)))
	
	_write_vtk_box(box_points, outfile)
	
	
def _write_vtk_box(box_points, filename):
	"""
	Method that writes a vtk file containing FFD control points.
	
	:param numpy.ndarray box_points: coordinates of the FFD control points.
	:param string filename: name of the output file.
	"""
	# setup points and vertices
	points = vtk.vtkPoints()
	vertices = vtk.vtkCellArray()
	
	for index in range(0, box_points.shape[1]):
		id = points.InsertNextPoint(box_points[0, index], box_points[1, index], box_points[2, index])
		vertices.InsertNextCell(1)
		vertices.InsertCellPoint(id)
		
	polydata = vtk.vtkPolyData()
	polydata.SetPoints(points)
	polydata.SetVerts(vertices)
	
	polydata.Modified()
	writer = vtk.vtkDataSetWriter()
	writer.SetFileName(filename)
	
	if vtk.VTK_MAJOR_VERSION <= 5:
		polydata.Update()
		writer.SetInput(polydata)
	else:
		writer.SetInputData(polydata)
		
	writer.Write()

