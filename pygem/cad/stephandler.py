"""
Derived module from nurbshandler.py to handle step and stp files.
"""
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_Reader
from OCC.Core.STEPControl import STEPControl_AsIs
from pygem.cad import NurbsHandler


class StepHandler(NurbsHandler):
    """
    Step file handler class

    :cvar string infile: name of the input file to be processed.
    :cvar string outfile: name of the output file where to write in.
    :cvar list extensions: list of extensions of the input/output files.
        It is equal to ['.step', '.stp'].
    :cvar list control_point_position: index of the first NURBS control
        point (or pole) of each face of the iges file.
    :cvar float tolerance: tolerance for the construction of the faces and wires
        in the write function. Default value is 1e-6.
    :cvar TopoDS_Shape shape: shape meant for modification.

    .. warning::

        - For non trivial geometries it could be necessary to increase the
          tolerance. Linking edges into a single wire and then trimming the
          surface with the wire can be hard for the software, especially when
          the starting CAD has not been made for analysis but for design
          purposes.
    """

    def __init__(self):
        super(StepHandler, self).__init__()
        self._control_point_position = None
        self.extensions = ['.step', '.stp']

    def load_shape_from_file(self, filename):
        """
        This method loads a shape from the file `filename`.

        :param string filename: name of the input file.
            It should have proper extension (.step or .stp)

        :return: shape: loaded shape
        :rtype: TopoDS_Shape
        """
        self._check_filename_type(filename)
        self._check_extension(filename)
        reader = STEPControl_Reader()
        return_reader = reader.ReadFile(filename)
        # check status
        if return_reader == IFSelect_RetDone:
            return_transfer = reader.TransferRoots()
            if return_transfer:
                # load all shapes in one
                shape = reader.OneShape()
                return shape
            else:
                raise RuntimeError("Shapes not loaded.")
        else:
            raise RuntimeError("Cannot read the file.")

    def write_shape_to_file(self, shape, filename):
        """
        This method saves the `shape` to the file `filename`.

        :param: TopoDS_Shape shape: loaded shape
        :param string filename: name of the input file.
            It should have proper extension (.step or .stp)
        """
        self._check_filename_type(filename)
        self._check_extension(filename)
        step_writer = STEPControl_Writer()
        # Changes write schema to STEP standard AP203
        # It is considered the most secure standard for STEP.
        # *According to PythonOCC documentation (http://www.pythonocc.org/)
        Interface_Static_SetCVal("write.step.schema", "AP203")
        step_writer.Transfer(shape, STEPControl_AsIs)
        step_writer.Write(filename)
