"""
Derived module from filehandler.py to handle iges and igs files.
"""
import os
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from OCC.IGESControl import (IGESControl_Reader, IGESControl_Writer)
from OCC.BRep import (BRep_Tool, BRep_Builder)
from OCC.BRepBuilderAPI import (BRepBuilderAPI_NurbsConvert, BRepBuilderAPI_MakeWire)
from OCC.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace)
from OCC.GeomConvert import geomconvert_SurfaceToBSplineSurface
import OCC.TopoDS
from OCC.TopAbs import (TopAbs_FACE, TopAbs_EDGE)
from OCC.TopExp import TopExp_Explorer
from OCC.gp import (gp_Pnt, gp_XYZ)
from OCC.Display.SimpleGui import init_display
from OCC.ShapeFix import ShapeFix_ShapeTolerance
from OCC.StlAPI import StlAPI_Writer
from stl import mesh
import pygem.filehandler as fh


class IgesHandler(fh.FileHandler):
	"""
	Iges file handler class

	:cvar string infile: name of the input file to be processed.
	:cvar string outfile: name of the output file where to write in.
	:cvar list extension: list of extensions of the input/output files.
		It is equal to ['.iges', '.igs'].
	:cvar list control_point_position: index of the first NURBS control point (or pole)
		of each face of the iges file.
	:cvar float tolerance: tolerance for the construction of the faces and wires 
		in the write function. Default value is 1e-6.
		
	.. warning::

			- For non trivial geometries it could be necessary to increase the tolerance.
			  Linking edges into a single wire and then trimming the surface with the wire
			  can be hard for the software, especially when the starting CAD has not been 
			  made for analysis but for design purposes.
	"""
	def __init__(self):
		super(IgesHandler, self).__init__()
		self.extension = ['.iges', '.igs']
		self._control_point_position = None
		self.tolerance = 1e-6


	def parse(self, filename):
		"""
		Method to parse the file `filename`. It returns a matrix with all the coordinates.

		:param string filename: name of the input file.

		:return: mesh_points: it is a `n_points`-by-3 matrix containing the coordinates of
			the points of the mesh
		:rtype: numpy.ndarray

		"""
		self._check_filename_type(filename)
		self._check_extension(filename)

		self.infile = filename

		# read in the IGES file
		reader = IGESControl_Reader()
		reader.ReadFile(self.infile)
		reader.TransferRoots()
		shape = reader.Shape()

		# cycle on the faces to get the control points
		# init some quantities
		n_faces = 0
		control_point_position = [0]
		faces_explorer = TopExp_Explorer(shape, TopAbs_FACE)
		mesh_points = np.zeros(shape=(0, 3))

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
			control_polygon_coordinates = np.zeros(shape=(n_poles_u * n_poles_v, 3))

			# cycle over the poles to get their coordinates
			i = 0
			for pole_u_direction in xrange(n_poles_u):
				for pole_v_direction in xrange(n_poles_v):
					control_point_coordinates = occ_face.Pole(pole_u_direction+1, pole_v_direction+1)
					control_polygon_coordinates[i, :] = [control_point_coordinates.X(), \
						control_point_coordinates.Y(), control_point_coordinates.Z()]
					i += 1

			# pushing the control points coordinates to the mesh_points array (used for FFD)
			mesh_points = np.append(mesh_points, control_polygon_coordinates, axis=0)
			control_point_position.append(control_point_position[-1] + n_poles_u * n_poles_v)

			n_faces += 1
			faces_explorer.Next()

		self._control_point_position = control_point_position

		return mesh_points


	def write(self, mesh_points, filename, tolerance=None):
		"""
		Writes a iges file, called filename, copying all the structures from self.filename but
		the coordinates. mesh_points is a matrix that contains the new coordinates to
		write in the iges file.

		:param numpy.ndarray mesh_points: it is a `n_points`-by-3 matrix containing
			the coordinates of the points of the mesh
		:param string filename: name of the output file.
		:param float tolerance: tolerance for the construction of the faces and wires
			in the write function. If not given it uses `self.tolerance`.
		"""
		self._check_filename_type(filename)
		self._check_extension(filename)
		self._check_infile_instantiation(self.infile)

		self.outfile = filename
		
		if tolerance is not None:
			self.tolerance = tolerance

		# init the ouput file writer
		writer = IGESControl_Writer()

		# read in the IGES file
		reader = IGESControl_Reader()
		reader.ReadFile(self.infile)
		reader.TransferRoots()
		shape_read = reader.Shape()

		# cycle on the faces to update the control points position
		# init some quantities
		faces_explorer = TopExp_Explorer(shape_read, TopAbs_FACE)
		n_faces = 0
		control_point_position = self._control_point_position

		compound_builder = BRep_Builder()
		compound = OCC.TopoDS.TopoDS_Compound()
		compound_builder.MakeCompound(compound)

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
					control_point_coordinates = mesh_points[i+control_point_position[n_faces], :]
					point_xyz = gp_XYZ(control_point_coordinates[0], control_point_coordinates[1], \
						control_point_coordinates[2])
					gp_point = gp_Pnt(point_xyz)
					occ_face.SetPole(pole_u_direction+1, pole_v_direction+1, gp_point)
					i += 1

			# construct the deformed wire for the trimmed surfaces
			wire_maker = BRepBuilderAPI_MakeWire()
			tol = ShapeFix_ShapeTolerance()
			brep = BRepBuilderAPI_MakeFace(occ_face.GetHandle(), self.tolerance).Face()
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
				tol.SetTolerance(edge_phis_coordinates, self.tolerance)
				wire_maker.Add(edge_phis_coordinates)
				edge_explorer.Next()

			# grouping the edges in a wire
			wire = wire_maker.Wire()

			# trimming the surfaces
			brep_surf = BRepBuilderAPI_MakeFace(occ_face.GetHandle(), wire).Shape()
			compound_builder.Add(compound, brep_surf)
			n_faces += 1
			faces_explorer.Next()

		writer.AddShape(compound)

		writer.Write(self.outfile)


	def plot(self, plot_file=None, save_fig=False):
		"""
		Method to plot an iges file. If `plot_file` is not given it plots `self.infile`.

		:param string plot_file: the iges filename you want to plot.
		:param bool save_fig: a flag to save the figure in png or not. If True the
			plot is not shown.
			
		:return: figure: matlplotlib structure for the figure of the chosen geometry
		:rtype: matplotlib.pyplot.figure
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

		stl_writer = StlAPI_Writer()
		# Do not switch SetASCIIMode() from False to True.
		stl_writer.SetASCIIMode(False)
		stl_writer.Write(shape, 'aux_figure.stl')

		# Create a new plot
		figure = pyplot.figure()
		axes = mplot3d.Axes3D(figure)

		# Load the STL files and add the vectors to the plot
		stl_mesh = mesh.Mesh.from_file('aux_figure.stl')
		os.remove('aux_figure.stl')
		axes.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors/1000))
		
		## Get the limits of the axis and center the geometry
		max_dim = np.array([np.max(stl_mesh.vectors[:,:,0])/1000, \
						np.max(stl_mesh.vectors[:,:,1])/1000, \
						np.max(stl_mesh.vectors[:,:,2])/1000])
		min_dim = np.array([np.min(stl_mesh.vectors[:,:,0])/1000, \
						np.min(stl_mesh.vectors[:,:,1])/1000, \
						np.min(stl_mesh.vectors[:,:,2])/1000])
		
		max_lenght = np.max(max_dim - min_dim)
		axes.set_xlim(-.6*max_lenght + (max_dim[0]+min_dim[0])/2, .6*max_lenght + (max_dim[0]+min_dim[0])/2)
		axes.set_ylim(-.6*max_lenght + (max_dim[1]+min_dim[1])/2, .6*max_lenght + (max_dim[1]+min_dim[1])/2)
		axes.set_zlim(-.6*max_lenght + (max_dim[2]+min_dim[2])/2, .6*max_lenght + (max_dim[2]+min_dim[2])/2)

		# Show the plot to the screen
		if not save_fig:
			pyplot.show()
		else:
			figure.savefig(plot_file.split('.')[0] + '.png')
			
		return figure


	def show(self, show_file=None):
		"""
		Method to show an iges file. If `show_file` is not given it plots `self.infile`.

		:param string show_file: the iges filename you want to show.
		"""
		if show_file is None:
			show_file = self.infile
		else:
			self._check_filename_type(show_file)

		# read in the IGES file
		reader = IGESControl_Reader()
		reader.ReadFile(show_file)
		reader.TransferRoots()
		shape = reader.Shape()

		display, start_display, __, __ = init_display()
		display.FitAll()
		display.DisplayShape(shape, update=True)

		# Show the plot to the screen
		start_display()

