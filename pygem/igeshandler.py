"""
Utilities for reading and writing different CAD files.
"""
import numpy as np
import pygem.filehandler as fh
from OCC.IGESControl import (IGESControl_Reader, IGESControl_Writer)
from OCC.BRep import BRep_Tool
from OCC.BRepBuilderAPI import (BRepBuilderAPI_NurbsConvert, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace)
from OCC.GeomConvert import geomconvert_SurfaceToBSplineSurface
import OCC.TopoDS
from OCC.TopAbs import (TopAbs_FACE, TopAbs_EDGE)
from OCC.TopExp import TopExp_Explorer
from OCC.Geom import Geom_BSplineSurface
from OCC.gp import (gp_Pnt, gp_XYZ)
from OCC.Display.SimpleGui import init_display
from OCC.Graphic3d import (Graphic3d_MaterialAspect, Graphic3d_NOM_GOLD, Graphic3d_NOM_SILVER) #maybe to remove
from OCC.ShapeFix import ShapeFix_ShapeTolerance


class IgesHandler(fh.FileHandler):
	"""
	Iges file handler class

	:cvar string infile: name of the input file to be processed.
	:cvar string outfile: name of the output file where to write in.
	:cvar string extension: extension of the input/output files. It is equal to '.iges'.
	:cvar list control_point_position: index of the first NURBS control point (or pole) of each face of the iges file.
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

			# extract the Control Points of each face
			n_poles_u = occ_object.NbUPoles()
			n_poles_v = occ_object.NbVPoles()
			control_polygon_coordinates = np.zeros(shape=(n_poles_u*n_poles_v,3))

			# cycle over the poles to get their coordinates
			i = 0
			for pole_u_direction in xrange(n_poles_u):
				for pole_v_direction in xrange(n_poles_v):
					pnt = occ_object.Pole(pole_u_direction+1,pole_v_direction+1)
					weight = occ_object.Weight(pole_u_direction+1,pole_v_direction+1)
					control_polygon_coordinates[i,:] = [pnt.X(), pnt.Y(), pnt.Z()]
					i += 1

			# pushing the control points coordinates to the mesh_points array (used for FFD)
			mesh_points = np.append(mesh_points, control_polygon_coordinates, axis=0)
			control_point_position.append(control_point_position[-1] + n_poles_u*n_poles_v)

			n_faces += 1
			faces_explorer.Next()

		self._control_point_position = control_point_position

		return mesh_points
		

	def write(self, mesh_points, filename):
		"""
		Writes a iges file, called filename, copying all the structures from self.filename but
		the coordinates. mesh_points is a matrix that contains the new coordinates to
		write in the iges file.

		:param numpy.ndarray mesh_points: it is a `n_points`-by-3 matrix containing
			the coordinates of the points of the mesh
		:param string filename: name of the output file.

		.. todo:: DOCS
			- control_point_position
		"""
		self._check_filename_type(filename)
		self._check_extension(filename)
		self._check_infile_instantiation(self.infile)

		self.outfile = filename	

		## init the ouput file writer
		iges_writer = IGESControl_Writer()

		## read in the IGES file
		iges_reader = IGESControl_Reader()
		iges_reader.ReadFile(self.infile)
		iges_reader.TransferRoots()
		shapeIn = iges_reader.Shape()

		## cycle on the faces to update the control points position
		# init some quantities
		explorer = TopExp_Explorer(shapeIn, TopAbs_FACE)
		nbFaces = 0
		controlPointPosition = self._control_point_position

		while explorer.More():
	
			# TODO: togliere tutta questa merda e salvarsi prima il numero di punti e gli occObjs
			face = OCC.TopoDS.topods_Face(explorer.Current())
			converter = BRepBuilderAPI_NurbsConvert(face)
			converter.Perform(face)
			face = converter.Shape()
			face_aux = OCC.TopoDS.topods_Face(face)
			brep_surf = BRep_Tool.Surface(face_aux)
			bspline_surf_handle = geomconvert_SurfaceToBSplineSurface(brep_surf)
			occObj = bspline_surf_handle.GetObject()

			nU = occObj.NbUPoles()
			nV = occObj.NbVPoles()

			i = 0

			for poleU in xrange(nU):
				for poleV in xrange(nV):
					point = mesh_points[i+controlPointPosition[nbFaces],:]
					point_XYZ = gp_XYZ(point[0], point[1], point[2])
					gp_point = gp_Pnt(point_XYZ)
					occObj.SetPole(poleU+1,poleV+1,gp_point)
					i += 1

			## construct the deformed wire for the trimmed surfaces
			wireMaker = BRepBuilderAPI_MakeWire()
			tol = ShapeFix_ShapeTolerance()
			brep = BRepBuilderAPI_MakeFace(occObj.GetHandle(), 1e-4).Face()
			brep_surf = BRep_Tool.Surface(brep)
		
			# cycle on the edges
			edgeExplorer = TopExp_Explorer(face, TopAbs_EDGE)
			while edgeExplorer.More():
				edge = OCC.TopoDS.topods_Edge(edgeExplorer.Current())
				# edge in the (u,v) coordinates
				edgeUV = OCC.BRep.BRep_Tool.CurveOnSurface(edge, face_aux)
				# evaluating the new edge: same (u,v) coordinates, but different (x,y,x) ones
				edgeStar = BRepBuilderAPI_MakeEdge(edgeUV[0], brep_surf)
				edgeStarEdge = edgeStar.Edge()
				tol.SetTolerance(edgeStarEdge, 1e-4)
				wireMaker.Add(edgeStarEdge)
				edgeExplorer.Next()

			#grouping the edges in a wire
			wire = wireMaker.Wire()

			## trimming the surfaces (TODO: check if a surface is actually trimmed)
			brep = BRepBuilderAPI_MakeFace(occObj.GetHandle(), wire, 1e-4).Face()
			iges_writer.AddShape(brep)
			
			print iges_writer

			nbFaces += 1
			explorer.Next()	

		## write out the iges file
		iges_writer.Write(self.outfile)
		

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
		
		
	def plot(self, plot_file=None, save_fig=False):
		"""
		Method to plot an iges file. If `plot_file` is not given it plots `self.infile`.

		:param string plot_file: the iges filename you want to plot.
		:param bool save_fig: a flag to save the figure in png or not. If True the
			plot is not shown.
		"""
		if plot_file is None:
			plot_file = self.infile
		else:
			self._check_filename_type(plot_file)

		## read in the IGES file
		iges_reader = IGESControl_Reader()
		iges_reader.ReadFile(plot_file)
		iges_reader.TransferRoots()
		shape = iges_reader.Shape()

		# Show the plot to the screen
		if not save_fig:
			display, start_display, add_menu, add_function_to_menu = init_display()
			display.FitAll()
			display.DisplayShape(shape, update=True)
			start_display()
		else:
			figure.savefig(plot_file.split('.')[0] + '.png') # da mettere sicuro a posto

