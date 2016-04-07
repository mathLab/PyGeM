"""
Utilities for reading and writing different CAD files.
"""
import numpy as np
import pygem.filehandler as fh


class IgesHandler(fh.FileHandler):
	"""
	Iges file handler class

	:cvar string infile: name of the input file to be processed.
	:cvar string outfile: name of the output file where to write in.
	:cvar string extension: extension of the input/output files. It is equal to '.iges'.
	"""
	def __init__(self):
		super(IgesHandler, self).__init__()
		self.extension = '.iges'	# TODO: also igs could be accepted


	def parse(self, filename):
		"""
		Method to parse the file `filename`. It returns a matrix with all the coordinates.

		:return: mesh_points: it is a `n_points`-by-3 matrix containing the coordinates of
			the points of the mesh
		:rtype: numpy.ndarray

		.. todo::

			- specify when it works
		"""
		self._check_filename_type(filename)
		self._check_extension(filename)

		self.infile = filename

		reader = vtk.vtkDataSetReader()
		reader.SetFileName(self.infile)
		reader.ReadAllVectorsOn()
		reader.ReadAllScalarsOn()
		reader.Update()
		data = reader.GetOutput()

		n_points = data.GetNumberOfPoints()
		mesh_points = np.zeros([n_points, 3])

		for i in range(n_points):
			mesh_points[i, 0], mesh_points[i, 1], mesh_points[i, 2] = data.GetPoint(i)

		return mesh_points


	def write(self, mesh_points, filename):
		"""
		Writes a vtk file, called filename, copying all the structures from self.filename but
		the coordinates. mesh_points is a matrix that contains the new coordinates to
		write in the vtk file.

		:param numpy.ndarray mesh_points: it is a `n_points`-by-3 matrix containing
			the coordinates of the points of the mesh
		:param string filename: name of the output file.

		.. todo:: DOCS
		"""
		self._check_filename_type(filename)
		self._check_extension(filename)
		self._check_infile_instantiation(self.infile)

		self.outfile = filename

		reader = vtk.vtkDataSetReader()
		reader.SetFileName(self.infile)
		reader.ReadAllVectorsOn()
		reader.ReadAllScalarsOn()
		reader.Update()
		data = reader.GetOutput()

		points = vtk.vtkPoints()

		for i in range(data.GetNumberOfPoints()):
			points.InsertNextPoint(mesh_points[i, :])

		data.SetPoints(points)

		writer = vtk.vtkDataSetWriter()
		writer.SetFileName(self.outfile)

		if vtk.VTK_MAJOR_VERSION <= 5:
			writer.SetInput(data)
		else:
			writer.SetInputData(data)

		writer.Write()
