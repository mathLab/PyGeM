"""
Derived module from filehandler.py to handle iges and igs files.
"""

from OCC.IGESControl import (IGESControl_Reader, IGESControl_Writer)

from pygem.nurbshandler import NurbsHandler


class IgesHandler(NurbsHandler):
	"""
	Iges file handler class

	:cvar string infile: name of the input file to be processed.
	:cvar string outfile: name of the output file where to write in.
	:cvar list EXTENSIONS: list of extensions of the input/output files.
		It is equal to ['.iges', '.igs'].
	:cvar list control_point_position: index of the first NURBS control point (or pole)
		of each face of the iges file.
	:cvar float tolerance: tolerance for the construction of the faces and wires 
		in the write function. Default value is 1e-6.
	:cvar TopoDS_Shape shape: shape meant for modification.
		
	.. warning::

			- For non trivial geometries it could be necessary to increase the tolerance.
			  Linking edges into a single wire and then trimming the surface with the wire
			  can be hard for the software, especially when the starting CAD has not been 
			  made for analysis but for design purposes.
	"""
	EXTENSIONS = ['.iges', '.igs']

	def __init__(self):
		super(IgesHandler, self).__init__()

	@classmethod
	def load_shape_from_file(cls, filename):
		"""
		This class method loads a shape from the file `filename`.

		:param string filename: name of the input file.
			It should have proper extension (.iges or .igs)

		:return: shape: loaded shape
		:rtype: TopoDS_Shape
		"""
		cls._check_filename_type(filename)
		cls._check_extension(filename)
		reader = IGESControl_Reader()
		reader.ReadFile(filename)
		reader.TransferRoots()
		shape = reader.Shape()
		return shape

	@classmethod
	def write_shape_to_file(cls, shape, filename):
		"""
		This class method saves the `shape` to the file `filename`.

		:param: TopoDS_Shape shape: loaded shape
		:param string filename: name of the input file.
			It should have proper extension (.iges or .igs)
		"""
		cls._check_filename_type(filename)
		cls._check_extension(filename)
		writer = IGESControl_Writer()
		writer.AddShape(shape)
		writer.Write(filename)
