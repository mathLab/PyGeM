"""
Utilities for reading and writing different CAD files.
"""
import numpy as np
import pygem.filehandler as fh
from OCC.IGESControl import IGESControl_Reader
from OCC.BRep import BRep_Tool
from OCC.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCC.GeomConvert import geomconvert_SurfaceToBSplineSurface
import OCC.TopoDS
from OCC.TopAbs import TopAbs_FACE
from OCC.TopExp import TopExp_Explorer


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
		
		## read in the IGES file
		reader = IGESControl_Reader()
		reader.ReadFile(self.infile)
		reader.TransferRoots()
		shape = reader.Shape()

		## cycle on the faces to get the control points
		# init some quantities
		n_faces = 0
		control_point_position = [0]
		faces_explorer = TopExp_Explorer(shape, TopAbs_FACE)
		mesh_points = np.zeros(shape=(0,3)) # inizializzare
		
		while faces_explorer.More():

			# performing some conversions to get the right format (BSplineSurface)
			face = OCC.TopoDS.topods_Face(faces_explorer.Current())
			converter = BRepBuilderAPI_NurbsConvert(face)
			converter.Perform(face)
			face = converter.Shape()
			brep_surf = BRep_Tool.Surface(OCC.TopoDS.topods_Face(face))
			bspline_surf_handle = geomconvert_SurfaceToBSplineSurface(brep_surf)

			# openCascade object
			occ_object = bspline_surf_handle.GetObject()

			# extract the Control Points
			n_poles_u = occ_object.NbUPoles()
			n_poles_v = occ_object.NbVPoles()
			controlPolygonCoordinates = np.zeros(shape=(n_poles_u*n_poles_v,3))

			# cycle over the poles to get their coordinates
			i = 0
			for poleU in xrange(n_poles_u):
				for poleV in xrange(n_poles_v):
					pnt = occ_object.Pole(poleU+1,poleV+1)
					weight = occ_object.Weight(poleU+1,poleV+1)
					controlPolygonCoordinates[i,:] = [pnt.X(), pnt.Y(), pnt.Z()]
					i += 1

			# pushing the control points coordinates to the meshPoints array (used for FFD)
			mesh_points = np.append(mesh_points, controlPolygonCoordinates, axis=0)
			control_point_position.append(control_point_position[-1] + n_poles_u*n_poles_v)

			n_faces += 1
			faces_explorer.Next()


		return mesh_points, control_point_position


	'''def write(self, mesh_points, filename):
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

		writer.Write()'''
