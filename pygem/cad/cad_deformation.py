"""
Module for generic deformation for CAD file.
"""
import os
import numpy as np
from itertools import product
from OCC.Core.TopoDS import (TopoDS_Shape, topods_Wire, TopoDS_Compound,
                             topods_Face, topods_Edge, TopoDS_Face, TopoDS_Wire)
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import (TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE)
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeFace,
                                     BRepBuilderAPI_MakeWire,
                                     BRepBuilderAPI_MakeEdge,
                                     BRepBuilderAPI_NurbsConvert)
from OCC.Core.BRep import BRep_Tool, BRep_Tool_Curve
from OCC.Core.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCC.Core.GeomConvert import (geomconvert_SurfaceToBSplineSurface,
                                  geomconvert_CurveToBSplineCurve,
                                  GeomConvert_CompCurveToBSplineCurve)
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepTools import breptools_OuterWire
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.STEPControl import (STEPControl_Writer, STEPControl_Reader,
                                  STEPControl_AsIs)
from OCC.Core.IGESControl import (IGESControl_Writer, IGESControl_Reader,
                                  IGESControl_Controller_Init)


class CADDeformation():
    """
    Base class for applting deformation to CAD geometries.
    
    :param int u_knots_to_add: the number of knots to add to the NURBS surfaces
        along `u` direction before the deformation. This parameter is useful
        whenever the gradient of the imposed deformation present spatial scales
        that are smaller than the local distance among the original poles of
        the surface/curve. Enriching the poles will allow for a more accurate
        application of the deformation, and might also reduce possible
        mismatches between bordering faces. On the orther hand, it might result
        in higher computational cost and bigger output files. Default is 0.
    :param int v_knots_to_add: the number of knots to add to the NURBS surfaces
        along `v` direction before the deformation. This parameter is useful
        whenever the gradient of the imposed deformation present spatial scales
        that are smaller than the local distance among the original poles of
        the surface/curve. Enriching the poles will allow for a more accurate
        application of the deformation, and might also reduce possible
        mismatches between bordering faces. On the orther hand, it might result
        in higher computational cost and bigger output files.  Default is 0.
    :param int t_knots_to_add: the number of knots to add to the NURBS curves
        before the deformation. This parameter is useful whenever the gradient
        of the imposed deformation present spatial scales that are smaller than
        the local distance among the original poles of the surface/curve.
        Enriching the poles will allow for a more accurate application of the
        deformation, and might also reduce possible mismatches between
        bordering faces. On the orther hand, it might result in higher
        computational cost and bigger output files. Default is 0.
    :param float tolerance: the tolerance involved in several internal
        operations of the procedure (joining wires in a single curve before
        deformation and placing new poles on curves and surfaces). Change the
        default value only if the input file scale is significantly different
        form mm, making some of the aforementioned operations fail. Default is
        0.0001.
        
    :cvar int u_knots_to_add: the number of knots to add to the NURBS surfaces
        along `u` direction before the deformation.   
    :cvar int v_knots_to_add: the number of knots to add to the NURBS surfaces
        along `v` direction before the deformation.
    :cvar int t_knots_to_add: the number of knots to add to the NURBS curves
        before the deformation.
    :cvar float tolerance: the tolerance involved in several internal
        operations of the procedure (joining wires in a single curve before
        deformation and placing new poles on curves and surfaces).
    """
    def __init__(self,
                 u_knots_to_add=0,
                 v_knots_to_add=0,
                 t_knots_to_add=0,
                 tolerance=1e-4):
        self.u_knots_to_add = u_knots_to_add
        self.v_knots_to_add = v_knots_to_add
        self.t_knots_to_add = t_knots_to_add
        self.tolerance = tolerance

    @staticmethod
    def read_shape(filename):
        """
    	Static method to load the `topoDS_Shape` from a file.
    	Supported extensions are: ".iges", ".step".
    	
    	:param str filename: the name of the file containing the geometry.
    	
    	Example:
    	   
    	   >>> from pygem.cad import CADDeformation as CAD
    	   >>> shape = CAD.read_shape('example.iges')
    	"""

        file_extension = os.path.splitext(filename)[1]

        av_readers = {
            '.step': STEPControl_Reader,
            '.iges': IGESControl_Reader,
        }

        reader_class = av_readers.get(file_extension)
        if reader_class is None:
            raise ValueError('Unable to open the input file')
        reader = reader_class()

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

    @staticmethod
    def write_shape(filename, shape):
        """
    	Static method to save a `topoDS_Shape` object to a file.
    	Supported extensions are: ".iges", ".step".
    	
    	:param str filename: the name of the file where the shape will be saved.
    	
    	Example:
    	   
    	   >>> from pygem.cad import CADDeformation as CAD
    	   >>> CAD.read_shape('example.step', my_shape)
    	"""
        def write_iges(filename, shape):
            """ IGES writer """
            IGESControl_Controller_Init()
            writer = IGESControl_Writer()
            writer.AddShape(shape)
            writer.Write(filename)

        def write_step(filename, shape):
            """ STEP writer """
            step_writer = STEPControl_Writer()
            # Changes write schema to STEP standard AP203
            # It is considered the most secure standard for STEP.
            # *According to PythonOCC documentation (http://www.pythonocc.org/)
            Interface_Static_SetCVal("write.step.schema", "AP203")
            Interface_Static_SetCVal('write.surfacecurve.mode','0')
            step_writer.Transfer(shape, STEPControl_AsIs)
            step_writer.Write(filename)

        file_extension = os.path.splitext(filename)[1]

        av_writers = {
            '.step': write_step,
            '.iges': write_iges,
        }

        writer = av_writers.get(file_extension)
        if writer is None:
            raise ValueError('Unable to open the output file')
        writer(filename, shape)

    def _bspline_surface_from_face(self, face):
        """
        Private method that takes a TopoDS_Face and transforms it into a
        Bspline_Surface.
        
    	:param TopoDS_Face face: the TopoDS_Face to be converted
        :rtype: Geom_BSplineSurface
        """
        if not isinstance(face, TopoDS_Face):
            raise TypeError("face must be a TopoDS_Face")
        # TopoDS_Face converted to Nurbs
        nurbs_face = topods_Face(BRepBuilderAPI_NurbsConvert(face).Shape())
        # GeomSurface obtained from Nurbs face
        surface = BRep_Tool.Surface(nurbs_face)
        # surface is now further converted to a bspline surface
        bspline_surface = geomconvert_SurfaceToBSplineSurface(surface)
        return bspline_surface

    def _bspline_curve_from_wire(self, wire):
        """
        Private method that takes a TopoDS_Wire and transforms it into a
        Bspline_Curve.

        :param TopoDS_Wire wire: the TopoDS_Face to be converted
        :rtype: Geom_BSplineSurface
        """
        if not isinstance(wire, TopoDS_Wire):
            raise TypeError("wire must be a TopoDS_Wire")

        # joining all the wire edges in a single curve here
        # composite curve builder (can only join Bspline curves)
        composite_curve_builder = GeomConvert_CompCurveToBSplineCurve()

        # iterator to edges in the TopoDS_Wire
        edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)
        while edge_explorer.More():
            # getting the edge from the iterator
            edge = topods_Edge(edge_explorer.Current())

            # edge can be joined only if it is not degenerated (zero length)
            if BRep_Tool.Degenerated(edge):
                edge_explorer.Next()
                continue

            # the edge must be converted to Nurbs edge
            nurbs_converter = BRepBuilderAPI_NurbsConvert(edge)
            nurbs_converter.Perform(edge)
            nurbs_edge = topods_Edge(nurbs_converter.Shape())

            # here we extract the underlying curve from the Nurbs edge
            nurbs_curve = BRep_Tool_Curve(nurbs_edge)[0]

            # we convert the Nurbs curve to Bspline curve
            bspline_curve = geomconvert_CurveToBSplineCurve(nurbs_curve)

            # we can now add the Bspline curve to the composite wire curve
            composite_curve_builder.Add(bspline_curve, self.tolerance)
            edge_explorer.Next()

        # GeomCurve obtained by the builder after edges are joined
        comp_curve = composite_curve_builder.BSplineCurve()
        return comp_curve

    def _enrich_curve_knots(self, bsp_curve):
        """
        Private method that adds `self.t_knots_to_add` poles to the passed
        curve.
        
        :param Geom_BSplineCurve bsp_curve: the NURBS curve to enrich
        """
        if not isinstance(bsp_curve, Geom_BSplineCurve):
            raise TypeError("bsp_curve must be a Geom_BSplineCurve")
        # number of knots is enriched here, if required, to
        # enhance precision
        # start parameter of composite curve
        first_param = bsp_curve.FirstParameter()
        # end parameter of composite curve
        last_param = bsp_curve.LastParameter()
        for i in range(self.t_knots_to_add):
            bsp_curve.InsertKnot(first_param+ \
                           i*(last_param-first_param)/self.t_knots_to_add, 1, \
                           self.tolerance)

    def _enrich_surface_knots(self, bsp_surface):
        """
        Private method that adds `self.u_knots_to_add` and `self.v_knots_to_add`
        knots to the input surface `bsp_surface`, in u and v direction respectively.

        :param Geom_BSplineCurve bsp_surface: the NURBS surface to enrich
        """
        if not isinstance(bsp_surface, Geom_BSplineSurface):
            raise TypeError("bsp_surface must be a Geom_BSplineSurface")
        # we will add the prescribed amount of nodes
        # both along u and v parametric directions

        # bounds (in surface parametric space) of the surface
        bounds = bsp_surface.Bounds()

        for i in range(self.u_knots_to_add):
            bsp_surface.InsertUKnot(bounds[0]+ \
                i*(bounds[1]-bounds[0])/self.u_knots_to_add, 1, self.tolerance)
        for i in range(self.v_knots_to_add):
            bsp_surface.InsertVKnot(bounds[2]+ \
                i*(bounds[3]-bounds[2])/self.v_knots_to_add, 1, self.tolerance)

    def _pole_get_components(self, pole):
        """ Extract component from gp_Pnt """
        return pole.X(), pole.Y(), pole.Z()

    def _pole_set_components(self, components):
        """ Return a gp_Pnt with the passed components """
        assert len(components) == 3
        return gp_Pnt(*components)

    def _deform_bspline_curve(self, bsp_curve):
        """
        Private method that deforms the control points of `bsp_curve` using the
        inherited method. All the changes are performed in place.

        :param Geom_Bspline_Curve bsp_curve: the curve to deform
        """
        if not isinstance(bsp_curve, Geom_BSplineCurve):
            raise TypeError("bsp_curve must be a Geom_BSplineCurve")

        # we first extract the poles of the curve poles number
        n_poles = bsp_curve.NbPoles()
        # array containing the poles coordinates
        poles_coordinates = np.zeros(shape=(n_poles, 3))

        # cycle over the poles to get their coordinates
        for pole_id in range(n_poles):
            # gp_Pnt corresponding to the pole
            pole = bsp_curve.Pole(pole_id + 1)
            poles_coordinates[pole_id, :] = self._pole_get_components(pole)

        # the new poles positions are computed through FFD
        new_pts = super().__call__(poles_coordinates)

        # the Bspline curve is now looped again to
        # set the poles positions  to new_points
        for pole_id in range(n_poles):
            new_pole = self._pole_set_components(new_pts[pole_id, :])
            bsp_curve.SetPole(pole_id + 1, new_pole)

    def _deform_bspline_surface(self, bsp_surface):
        """
        Private method that deforms the control points of `surface` using the
        inherited method. All the changes are performed in place.

        :param Geom_Bspline_Surface bsp_surface: the surface to deform
        """
        if not isinstance(bsp_surface, Geom_BSplineSurface):
            raise TypeError("bsp_surface must be a Geom_BSplineSurface")
        # we first extract the poles of the curve
        # number of poles in u direction
        n_poles_u = bsp_surface.NbUPoles()
        # number of poles in v direction
        n_poles_v = bsp_surface.NbVPoles()
        # array which will contain the coordinates of the poles
        poles_coordinates = np.zeros(shape=(n_poles_u * n_poles_v, 3))

        # cycle over the poles to get their coordinates
        pole_ids = product(range(n_poles_u), range(n_poles_v))
        for pole_id, (u, v) in enumerate(pole_ids):
            pole = bsp_surface.Pole(u + 1, v + 1)
            poles_coordinates[pole_id, :] = self._pole_get_components(pole)

        # the new poles positions are computed through FFD
        new_pts = super().__call__(poles_coordinates)

        # the surface is now looped again to set the new poles positions
        pole_ids = product(range(n_poles_u), range(n_poles_v))
        for pole_id, (u, v) in enumerate(pole_ids):
            new_pole = self._pole_set_components(new_pts[pole_id, :])
            bsp_surface.SetPole(u + 1, v + 1, new_pole)

    def __call__(self, obj, dst=None):
        """
        This method performs the deformation on the CAD geometry. If `obj` is a
        TopoDS_Shape, the method returns a TopoDS_Shape containing the deformed
        geometry. If `obj` is a filename, the method deforms the geometry
        contained in the file and writes the deformed shape to `dst` (which has
        to be set).
        
        :param obj: the input geometry.
        :type obj: str or TopoDS_Shape
        :param str dst: if `obj` is a string containing the input filename,
            `dst` refers to the file where the deformed geometry is saved.
        """
        # Manage input
        if isinstance(obj, str):  # if a input filename is passed
            if dst is None:
                raise ValueError(
                    'Source file is provided, but no destination specified')
            shape = self.read_shape(obj)
        elif isinstance(obj, TopoDS_Shape):
            shape = obj
        # Maybe do we need to handle also Compound?
        else:
            raise TypeError

        #create compound to store modified faces
        compound_builder = BRep_Builder()
        compound = TopoDS_Compound()
        compound_builder.MakeCompound(compound)

        # cycle on the faces to get the control points

        # iterator to faces (TopoDS_Shape) contained in the shape
        faces_explorer = TopExp_Explorer(shape, TopAbs_FACE)

        while faces_explorer.More():
            # performing some conversions to get the right
            # format (BSplineSurface)
            # TopoDS_Face obtained from iterator
            face = topods_Face(faces_explorer.Current())
            # performing some conversions to get the right
            # format (BSplineSurface)
            bspline_surface = self._bspline_surface_from_face(face)

            # add the required amount of poles in u and v directions
            self._enrich_surface_knots(bspline_surface)

            # deform the Bspline surface through FFD
            self._deform_bspline_surface(bspline_surface)

            # through moving the control points, we now changed the SURFACE
            # underlying FACE we are processing. we now need to obtain the
            # curves (actually, the WIRES) that define the bounds of the
            # surface and TRIM the surface with them, to obtain the new FACE

            #we now start really looping on the wires
            #we will create a single curve joining all the edges in the wire
            # the curve must be a bspline curve so we need to make conversions
            # through all the way

            # list that will contain the (single) outer wire of the face
            outer_wires = []
            # list that will contain all the inner wires (holes) of the face
            inner_wires = []
            # iterator to loop over TopoDS_Wire in the original (undeformed)
            # face
            wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
            while wire_explorer.More():
                # wire obtained from the iterator
                wire = topods_Wire(wire_explorer.Current())

                # getting a bpline curve joining all the edges of the wire
                composite_curve = self._bspline_curve_from_wire(wire)

                # adding all the required knots to the Bspline curve
                self._enrich_curve_knots(composite_curve)

                # deforming the Bspline curve through FFD
                self._deform_bspline_curve(composite_curve)

                # the GeomCurve corresponding to the whole edge has now
                # been deformed. Now we must make it become an proper
                # wire

                # list of shapes (needed by the wire generator)
                shapes_list = TopTools_ListOfShape()

                # edge (to be converted to wire) obtained from the modified
                # Bspline curve
                modified_composite_edge = \
                    BRepBuilderAPI_MakeEdge(composite_curve).Edge()
                # modified edge is added to shapes_list
                shapes_list.Append(modified_composite_edge)

                # wire builder
                wire_maker = BRepBuilderAPI_MakeWire()
                wire_maker.Add(shapes_list)
                # deformed wire is finally obtained
                modified_wire = wire_maker.Wire()

                # now, the wire can be outer or inner. we store the outer
                # and (possible) inner ones in different lists
                # this is because we first need to trim the surface
                # using the outer wire, and then we can trim it
                # with the wires corresponding to all the holes.
                # the wire order is important, in the trimming process
                if wire == breptools_OuterWire(face):
                    outer_wires.append(modified_wire)
                else:
                    inner_wires.append(modified_wire)
                wire_explorer.Next()

            # so once we finished looping on all the wires to modify them,
            # we first use the only outer one to trim the surface
            # face builder object
            face_maker = BRepBuilderAPI_MakeFace(bspline_surface,
                                                 outer_wires[0])

            # and then add all other inner wires for the holes
            for inner_wire in inner_wires:
                face_maker.Add(inner_wire)

            # finally, we get our trimmed face with all its holes
            trimmed_modified_face = face_maker.Face()

            # trimmed_modified_face is added to the modified faces compound
            compound_builder.Add(compound, trimmed_modified_face)

            # and move to the next face
            faces_explorer.Next()

        ## END SURFACES #################################################

        if isinstance(dst, str):  # if a input filename is passed
            # save the shape exactly to the filename, aka `dst`
            self.write_shape(dst, compound)
        else:
            return compound
