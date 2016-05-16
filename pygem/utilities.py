import vtk
import vtk.util.numpy_support as ns
import numpy as np

def write_initial_box(parameters, latticeName):
	"""
	Writes a vtk files for the undeformed FFD lattice.
	"""

	x = np.linspace(0, parameters.lenght_box_x, parameters.n_control_points[0])
	y = np.linspace(0, parameters.lenght_box_y, parameters.n_control_points[1])
	z = np.linspace(0, parameters.lenght_box_z, parameters.n_control_points[2])
	yy, xx, zz = np.meshgrid(y, x, z)

	boxPoints = np.array([xx.ravel(), yy.ravel(), zz.ravel()])
	n_row = boxPoints.shape[1]

	boxPoints = np.dot(parameters.rotation_matrix,boxPoints) + np.transpose(np.tile(parameters.origin_box, (n_row,1)))
	
	write_vtk_box(boxPoints, 'originalBox_' + latticeName + '.vtk')

def write_modified_box(parameters, latticeName):
	"""
	Writes a vtk files for the deformed FFD lattice.
	"""

	x = np.linspace(0, parameters.lenght_box_x, parameters.n_control_points[0])
	y = np.linspace(0, parameters.lenght_box_y, parameters.n_control_points[1])
	z = np.linspace(0, parameters.lenght_box_z, parameters.n_control_points[2])
	yy, xx, zz = np.meshgrid(y, x, z)

	boxPoints = np.array([xx.ravel() + parameters.array_mu_x.ravel()*parameters.lenght_box_x, \
	yy.ravel()  + parameters.array_mu_y.ravel()*parameters.lenght_box_y, \
	zz.ravel()  + parameters.array_mu_z.ravel()*parameters.lenght_box_z])

	n_row = boxPoints.shape[1]

	boxPoints = np.dot(parameters.rotation_matrix,boxPoints) + np.transpose(np.tile(parameters.origin_box, (n_row,1)))
	
	write_vtk_box(boxPoints, 'modifiedBox_' + latticeName + '.vtk')	
	
def write_vtk_box(boxPoints, filename):
	"""
	Documentation
	"""
	# setup points and vertices
	points = vtk.vtkPoints()
	vertices = vtk.vtkCellArray()
	
	for index in range(0, boxPoints.shape[1]):
		id = points.InsertNextPoint(boxPoints[0, index], boxPoints[1, index], boxPoints[2, index])
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

