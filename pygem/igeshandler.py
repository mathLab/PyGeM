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
		self._control_point_position = None


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
		mesh_points = np.zeros(shape=(0,3))
		
		while faces_explorer.More():

			# performing some conversions to get the right format (BSplineSurface)
			iges_face = OCC.TopoDS.topods_Face(faces_explorer.Current())
			iges_nurbs_converter = BRepBuilderAPI_NurbsConvert(iges_face)
			iges_nurbs_converter.Perform(iges_face)
			nurbs_face = iges_nurbs_converter.Shape()
			brep_face = BRep_Tool.Surface(OCC.TopoDS.topods_Face(nurbs_face))
			bspline_face = geomconvert_SurfaceToBSplineSurface(brep_face)

			# openCascade object
			occ_face = bspline_face.GetObject()

			# extract the Control Points of each face
			n_poles_u = occ_face.NbUPoles()
			n_poles_v = occ_face.NbVPoles()
			control_polygon_coordinates = np.zeros(shape=(n_poles_u*n_poles_v,3))

			# cycle over the poles to get their coordinates
			i = 0
			for pole_u_direction in xrange(n_poles_u):
				for pole_v_direction in xrange(n_poles_v):
					control_point_coordinates = occ_face.Pole(pole_u_direction+1,pole_v_direction+1)
					weight = occ_face.Weight(pole_u_direction+1,pole_v_direction+1)
					control_polygon_coordinates[i,:] = [control_point_coordinates.X(), control_point_coordinates.Y(), control_point_coordinates.Z()]
					i += 1	

			# pushing the control points coordinates to the mesh_points array (used for FFD)
			mesh_points = np.append(mesh_points, control_polygon_coordinates, axis=0)
			control_point_position.append(control_point_position[-1] + n_poles_u*n_poles_v)

			n_faces += 1
			faces_explorer.Next()
			
		print control_point_position

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
		"""
		self._check_filename_type(filename)
		self._check_extension(filename)
		self._check_infile_instantiation(self.infile)

		self.outfile = filename	

		## init the ouput file writer
		writer = IGESControl_Writer()

		## read in the IGES file
		reader = IGESControl_Reader()
		reader.ReadFile(self.infile)
		reader.TransferRoots()
		shape_read = reader.Shape()

		## cycle on the faces to update the control points position
		# init some quantities
		faces_explorer = TopExp_Explorer(shape_read, TopAbs_FACE)
		n_faces = 0
		control_point_position = self._control_point_position

		while faces_explorer.More():
	
			# similar to the parser method
			iges_face = OCC.TopoDS.topods_Face(faces_explorer.Current())
			iges_nurbs_converter = BRepBuilderAPI_NurbsConvert(iges_face)
			iges_nurbs_converter.Perform(iges_face)
			nurbs_face = iges_nurbs_converter.Shape()
			face_aux = OCC.TopoDS.topods_Face(nurbs_face)
			brep_face = BRep_Tool.Surface(OCC.TopoDS.topods_Face(nurbs_face))
			bspline_face = geomconvert_SurfaceToBSplineSurface(brep_face)
			occ_face = bspline_face.GetObject()

			n_poles_u = occ_face.NbUPoles()
			n_poles_v = occ_face.NbVPoles()

			i = 0
			for pole_u_direction in xrange(n_poles_u):
				for pole_v_direction in xrange(n_poles_v):
					control_point_coordinates = mesh_points[i+control_point_position[n_faces],:]
					point_xyz = gp_XYZ(control_point_coordinates[0], control_point_coordinates[1], control_point_coordinates[2])
					gp_point = gp_Pnt(point_xyz)
					occ_face.SetPole(pole_u_direction+1,pole_v_direction+1,gp_point)
					i += 1

			## construct the deformed wire for the trimmed surfaces
			wireMaker = BRepBuilderAPI_MakeWire()
			tol = ShapeFix_ShapeTolerance()
			brep = BRepBuilderAPI_MakeFace(occ_face.GetHandle(), 1e-4).Face()
			brep_face = BRep_Tool.Surface(brep)
		
			# cycle on the edges
			edge_explorer = TopExp_Explorer(nurbs_face, TopAbs_EDGE)
			while edge_explorer.More():
				edge = OCC.TopoDS.topods_Edge(edge_explorer.Current())
				# edge in the (u,v) coordinates
				edge_uv_coordinates = OCC.BRep.BRep_Tool.CurveOnSurface(edge, face_aux)
				# evaluating the new edge: same (u,v) coordinates, but different (x,y,x) ones
				edge_phis_coordinates_aux = BRepBuilderAPI_MakeEdge(edge_uv_coordinates[0], brep_face)
				edge_phis_coordinates = edge_phis_coordinates_aux.Edge()
				tol.SetTolerance(edge_phis_coordinates, 1e-4)
				wireMaker.Add(edge_phis_coordinates)
				edge_explorer.Next()

			#grouping the edges in a wire
			wire = wireMaker.Wire()

			## trimming the surfaces
			brep_surf = BRepBuilderAPI_MakeFace(occ_face.GetHandle(), wire).Face()
			writer.AddShape(brep_surf)
			
			#print writer

			n_faces += 1
			faces_explorer.Next()	

		## write out the iges file
		writer.Write(self.outfile)
		
		
	def plot(self, plot_file=None, save_fig=False):
		"""
		Method to plot an iges file. If `plot_file` is not given it plots `self.infile`.

		:param string plot_file: the iges filename you want to plot.
		:param bool save_fig: a flag to save the figure in png or not. If True the
			plot is not shown.
			
		.. warning::
			It does not work well up to now
		"""
		if plot_file is None:
			plot_file = self.infile
		else:
			self._check_filename_type(plot_file)

		## read in the IGES file
		reader = IGESControl_Reader()
		reader.ReadFile(plot_file)
		reader.TransferRoots()
		shape = reader.Shape()
		
		display, start_display, add_menu, add_function_to_menu = init_display()
		display.FitAll()
		display.DisplayShape(shape, update=True)
		
		# Show the plot to the screen
		if not save_fig:
			start_display()
		else:
			display.View.Dump(plot_file.split('.')[0] + '.ppm')

