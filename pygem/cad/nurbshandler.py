"""
Derived module from filehandler.py to handle iges/igs and step/stp files.
Implements all methods for parsing an object and applying FFD.
File handling operations (reading/writing) must be implemented
in derived classes.
"""
import os
import numpy as np
from OCC.Core.BRep import BRep_Tool, BRep_Builder, BRep_Tool_Curve
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepAlgo import brepalgo_IsValid
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_NurbsConvert, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Sewing)
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_FindContigousEdges
from OCC.Display.SimpleGui import init_display
from OCC.Core.GeomConvert import (geomconvert_SurfaceToBSplineSurface,
                             geomconvert_CurveToBSplineCurve)
from OCC.Core.gp import gp_Pnt, gp_XYZ
from OCC.Core.Precision import precision_Confusion
from OCC.Core.ShapeAnalysis import ShapeAnalysis_WireOrder
from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance, ShapeFix_Shell
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.TopAbs import (TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE, TopAbs_FORWARD,
                        TopAbs_SHELL)
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopoDS import (topods_Face, TopoDS_Compound, topods_Shell, topods_Edge,
                        topods_Wire, topods, TopoDS_Shape)
from matplotlib import pyplot
from mpl_toolkits import mplot3d
from stl import mesh
import pygem.filehandler as fh


class NurbsHandler(fh.FileHandler):
    """
    Nurbs file handler base class

    :cvar str infile: name of the input file to be processed.
    :cvar str outfile: name of the output file where to write in.
    :cvar list control_point_position: index of the first NURBS
        control point (or pole) of each face of the files.
    :cvar TopoDS_Shape shape: shape meant for modification.
    :cvar float tolerance: tolerance for the construction of the faces
        and wires in the write function. Default value is 1e-6.

    .. warning::

        For non trivial geometries it could be necessary to increase the
        tolerance. Linking edges into a single wire and then trimming the
        surface with the wire can be hard for the software, especially when the
        starting CAD has not been made for analysis but for design purposes.
    """

    def __init__(self):
        super(NurbsHandler, self).__init__()
        self._control_point_position = None
        self.tolerance = 1e-6
        self.shape = None
        self.check_topo = 0

    def _check_infile_instantiation(self):
        """
        This private method checks if `self.infile` and `self.shape` are
        instantiated. If not it means that nobody called the parse method
        and at least one of them is None. If the check fails it raises a
        RuntimeError.
        """
        if not self.shape or not self.infile:
            raise RuntimeError(
                'You can not write a file without having parsed one.')

    def load_shape_from_file(self, filename):
        """
        Abstract method to load a specific file as a shape.

        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError('Subclass must implement abstract method'
                                  '{}.load_shape_from_file'.format(
                                      self.__class__.__name__))

    def parse(self, filename):
        """
        Method to parse the file `filename`. It returns a matrix with all
        the coordinates.

        :param string filename: name of the input file.

        :return: mesh_points: it is a `n_points`-by-3 matrix containing
            the coordinates of the points of the mesh
        :rtype: numpy.ndarray

        """
        self.infile = filename
        self.shape = self.load_shape_from_file(filename)

        # cycle on the faces to get the control points
        # init some quantities
        n_faces = 0
        control_point_position = [0]
        faces_explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
        mesh_points = np.zeros(shape=(0, 3))

        while faces_explorer.More():
            # performing some conversions to get the right format (BSplineSurface)
            face = topods_Face(faces_explorer.Current())
            nurbs_converter = BRepBuilderAPI_NurbsConvert(face)
            nurbs_converter.Perform(face)
            nurbs_face = nurbs_converter.Shape()
            brep_face = BRep_Tool.Surface(topods_Face(nurbs_face))
            bspline_face = geomconvert_SurfaceToBSplineSurface(brep_face)

            # openCascade object
            occ_face = bspline_face

            # extract the Control Points of each face
            n_poles_u = occ_face.NbUPoles()
            n_poles_v = occ_face.NbVPoles()
            control_polygon_coordinates = np.zeros(
                shape=(n_poles_u * n_poles_v, 3))

            # cycle over the poles to get their coordinates
            i = 0
            for pole_u_direction in range(n_poles_u):
                for pole_v_direction in range(n_poles_v):
                    control_point_coordinates = occ_face.Pole(
                        pole_u_direction + 1, pole_v_direction + 1)
                    control_polygon_coordinates[i, :] = [
                        control_point_coordinates.X(),
                        control_point_coordinates.Y(),
                        control_point_coordinates.Z()
                    ]
                    i += 1
            # pushing the control points coordinates to the mesh_points array
            # (used for FFD)
            mesh_points = np.append(
                mesh_points, control_polygon_coordinates, axis=0)
            control_point_position.append(
                control_point_position[-1] + n_poles_u * n_poles_v)

            n_faces += 1
            faces_explorer.Next()
        self._control_point_position = control_point_position
        return mesh_points

    def write(self, mesh_points, filename, tolerance=None):
        """
        Writes a output file, called `filename`, copying all the structures
        from self.filename but the coordinates. `mesh_points` is a matrix
        that contains the new coordinates to write in the output file.

        :param numpy.ndarray mesh_points: it is a *n_points*-by-3 matrix
            containing the coordinates of the points of the mesh.
        :param str filename: name of the output file.
        :param float tolerance: tolerance for the construction of the faces
            and wires in the write function. If not given it uses
            `self.tolerance`.
        """
        self._check_filename_type(filename)
        self._check_extension(filename)
        self._check_infile_instantiation()

        self.outfile = filename

        if tolerance is not None:
            self.tolerance = tolerance

        # cycle on the faces to update the control points position
        # init some quantities
        faces_explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
        n_faces = 0
        control_point_position = self._control_point_position

        compound_builder = BRep_Builder()
        compound = TopoDS_Compound()
        compound_builder.MakeCompound(compound)

        while faces_explorer.More():
            # similar to the parser method
            face = topods_Face(faces_explorer.Current())
            nurbs_converter = BRepBuilderAPI_NurbsConvert(face)
            nurbs_converter.Perform(face)
            nurbs_face = nurbs_converter.Shape()
            face_aux = topods_Face(nurbs_face)
            brep_face = BRep_Tool.Surface(topods_Face(nurbs_face))
            bspline_face = geomconvert_SurfaceToBSplineSurface(brep_face)
            occ_face = bspline_face

            n_poles_u = occ_face.NbUPoles()
            n_poles_v = occ_face.NbVPoles()

            i = 0
            for pole_u_direction in range(n_poles_u):
                for pole_v_direction in range(n_poles_v):
                    control_point_coordinates = mesh_points[
                        i + control_point_position[n_faces], :]
                    point_xyz = gp_XYZ(*control_point_coordinates)

                    gp_point = gp_Pnt(point_xyz)
                    occ_face.SetPole(pole_u_direction + 1, pole_v_direction + 1,
                                     gp_point)
                    i += 1

            # construct the deformed wire for the trimmed surfaces
            wire_maker = BRepBuilderAPI_MakeWire()
            tol = ShapeFix_ShapeTolerance()
            brep = BRepBuilderAPI_MakeFace(occ_face,
                                           self.tolerance).Face()
            brep_face = BRep_Tool.Surface(brep)

            # cycle on the edges
            edge_explorer = TopExp_Explorer(nurbs_face, TopAbs_EDGE)
            while edge_explorer.More():
                edge = topods_Edge(edge_explorer.Current())
                # edge in the (u,v) coordinates
                edge_uv_coordinates = BRep_Tool.CurveOnSurface(edge, face_aux)
                # evaluating the new edge: same (u,v) coordinates, but
                # different (x,y,x) ones
                edge_phis_coordinates_aux = BRepBuilderAPI_MakeEdge(
                    edge_uv_coordinates[0], brep_face)
                edge_phis_coordinates = edge_phis_coordinates_aux.Edge()
                tol.SetTolerance(edge_phis_coordinates, self.tolerance)
                wire_maker.Add(edge_phis_coordinates)
                edge_explorer.Next()

            # grouping the edges in a wire
            wire = wire_maker.Wire()

            # trimming the surfaces
            brep_surf = BRepBuilderAPI_MakeFace(occ_face,
                                                wire).Shape()
            compound_builder.Add(compound, brep_surf)
            n_faces += 1
            faces_explorer.Next()
        self.write_shape_to_file(compound, self.outfile)

    def check_topology(self):
        """
        Method to check the topology of imported geometry; it sets
        *self.check_topo* as:

        - 0 if 1 solid = 1 shell = n faces
        - 1 if 1 solid = 0 shell = n free faces
        - 2 if 1 solid = n shell = n faces (1 shell = 1 face)

        :return: {0, 1, 2}
        :rtype: int
        """
        # read shells and faces
        shells_explorer = TopExp_Explorer(self.shape, TopAbs_SHELL)
        n_shells = 0
        while shells_explorer.More():
            n_shells += 1
            shells_explorer.Next()

        faces_explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
        n_faces = 0
        while faces_explorer.More():
            n_faces += 1
            faces_explorer.Next()

        # print("##############################################\n"
        #       "Model statistics -- Nb Shells: {0} Faces: {1} \n"
        #       "----------------------------------------------\n".format(
        #           n_shells, n_faces))

        if n_shells == 0:
            self.check_topo = 1
        elif n_shells == n_faces:
            self.check_topo = 2
        else:
            self.check_topo = 0

    @staticmethod
    def parse_face(topo_face):
        """
        Method to parse a single `Face` (a single patch nurbs surface).
        It returns a matrix with all the coordinates of control points of the
        `Face` and a second list with all the control points related to the
        `Edges` of the `Face.`

        :param Face topo_face: the input Face.

        :return: control points of the `Face`, control points related to
            `Edges`.
        :rtype: tuple(numpy.ndarray, list)

        """
        # get some Face - Edge - Vertex data map information
        mesh_points_edge = []
        face_exp_wire = TopExp_Explorer(topo_face, TopAbs_WIRE)
        # loop on wires per face
        while face_exp_wire.More():
            twire = topods_Wire(face_exp_wire.Current())
            wire_exp_edge = TopExp_Explorer(twire, TopAbs_EDGE)
            # loop on edges per wire
            while wire_exp_edge.More():
                edge = topods_Edge(wire_exp_edge.Current())
                bspline_converter = BRepBuilderAPI_NurbsConvert(edge)
                bspline_converter.Perform(edge)
                bspline_tshape_edge = bspline_converter.Shape()
                h_geom_edge = BRep_Tool_Curve(
                    topods_Edge(bspline_tshape_edge))[0]
                h_bspline_edge = geomconvert_CurveToBSplineCurve(h_geom_edge)
                bspline_geom_edge = h_bspline_edge

                nb_poles = bspline_geom_edge.NbPoles()

                # Edge geometric properties
                edge_ctrlpts = TColgp_Array1OfPnt(1, nb_poles)
                bspline_geom_edge.Poles(edge_ctrlpts)

                points_single_edge = np.zeros((0, 3))
                for i in range(1, nb_poles + 1):
                    ctrlpt = edge_ctrlpts.Value(i)
                    ctrlpt_position = np.array(
                        [[ctrlpt.Coord(1),
                          ctrlpt.Coord(2),
                          ctrlpt.Coord(3)]])
                    points_single_edge = np.append(
                        points_single_edge, ctrlpt_position, axis=0)

                mesh_points_edge.append(points_single_edge)

                wire_exp_edge.Next()

            face_exp_wire.Next()
        # extract mesh points (control points) on Face
        mesh_points_face = np.zeros((0, 3))
        # convert Face to Geom B-spline Face
        nurbs_converter = BRepBuilderAPI_NurbsConvert(topo_face)
        nurbs_converter.Perform(topo_face)
        nurbs_face = nurbs_converter.Shape()
        h_geomsurface = BRep_Tool.Surface(topods.Face(nurbs_face))
        h_bsurface = geomconvert_SurfaceToBSplineSurface(h_geomsurface)
        bsurface = h_bsurface

        # get access to control points (poles)
        nb_u = bsurface.NbUPoles()
        nb_v = bsurface.NbVPoles()
        ctrlpts = TColgp_Array2OfPnt(1, nb_u, 1, nb_v)
        bsurface.Poles(ctrlpts)

        for indice_u_direction in range(1, nb_u + 1):
            for indice_v_direction in range(1, nb_v + 1):
                ctrlpt = ctrlpts.Value(indice_u_direction, indice_v_direction)
                ctrlpt_position = np.array(
                    [[ctrlpt.Coord(1),
                      ctrlpt.Coord(2),
                      ctrlpt.Coord(3)]])
                mesh_points_face = np.append(
                    mesh_points_face, ctrlpt_position, axis=0)

        return mesh_points_face, mesh_points_edge

    def parse_shape(self, filename):
        """
        Method to parse a Shape with multiple objects (1 compound = multi-shells
        and 1 shell = multi-faces)
        It returns a list of matrix with all the coordinates of control points
        of each Face and a second list with all the control points related to
        Edges of each Face.

        :param str filename: the input filename.

        :return: list of (mesh_points: `n_points`-by-3 matrix containing
        the coordinates of the control points of the Face (surface),
                 edge_points: it is a list of numpy.narray)
        :rtype: a list of shells

        """
        self.infile = filename
        self.shape = self.load_shape_from_file(filename)

        self.check_topology()

        # parse and get control points
        l_shells = []  # an empty list of shells
        n_shells = 0

        if self.check_topo == 0:

            shells_explorer = TopExp_Explorer(self.shape, TopAbs_SHELL)

            # cycle on shells
            while shells_explorer.More():
                topo_shell = topods.Shell(shells_explorer.Current())
                shell_faces_explorer = TopExp_Explorer(topo_shell, TopAbs_FACE)
                l_faces = []  # an empty list of faces per shell

                # cycle on faces
                while shell_faces_explorer.More():
                    topo_face = topods.Face(shell_faces_explorer.Current())
                    mesh_point, edge_point = self.parse_face(topo_face)
                    l_faces.append((mesh_point, edge_point))
                    shell_faces_explorer.Next()

                l_shells.append(l_faces)
                n_shells += 1
                shells_explorer.Next()

        else:
            # cycle only on faces
            shell_faces_explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
            l_faces = []  # an empty list of faces per shell

            while shell_faces_explorer.More():
                topo_face = topods.Face(shell_faces_explorer.Current())
                mesh_point, edge_point = self.parse_face(topo_face)
                l_faces.append((mesh_point, edge_point))
                shell_faces_explorer.Next()

            l_shells.append(l_faces)
            n_shells += 1

        return l_shells

    @staticmethod
    def write_edge(points_edge, topo_edge):
        """
        Method to recreate an Edge associated to a geometric curve
        after the modification of its points.
        :param points_edge: the deformed points array.
        :param topo_edge: the Edge to be modified
        :return: Edge (Shape)

        :rtype: TopoDS_Edge

        """
        # convert Edge to Geom B-spline Curve
        nurbs_converter = BRepBuilderAPI_NurbsConvert(topo_edge)
        nurbs_converter.Perform(topo_edge)
        nurbs_curve = nurbs_converter.Shape()
        topo_curve = topods_Edge(nurbs_curve)
        h_geomcurve = BRep_Tool.Curve(topo_curve)[0]
        h_bcurve = geomconvert_CurveToBSplineCurve(h_geomcurve)
        bspline_edge_curve = h_bcurve

        # Edge geometric properties
        nb_cpt = bspline_edge_curve.NbPoles()
        # check consistency
        if points_edge.shape[0] != nb_cpt:
            raise ValueError("Input control points do not have not have the "
                             "same number as the geometric edge!")

        else:
            for i in range(1, nb_cpt + 1):
                cpt = points_edge[i - 1]
                bspline_edge_curve.SetPole(i, gp_Pnt(cpt[0], cpt[1], cpt[2]))

        new_edge = BRepBuilderAPI_MakeEdge(bspline_edge_curve)

        return new_edge.Edge()

    def write_face(self, points_face, list_points_edge, topo_face, toledge):
        """
        Method to recreate a Face associated to a geometric surface
        after the modification of Face points. It returns a TopoDS_Face.

        :param points_face: the new face points array.
        :param list_points_edge: new edge points
        :param topo_face: the face to be modified
        :param toledge: tolerance on the surface creation after modification
        :return: TopoDS_Face (Shape)

        :rtype: TopoDS_Shape

        """

        # convert Face to Geom B-spline Surface
        nurbs_converter = BRepBuilderAPI_NurbsConvert(topo_face)
        nurbs_converter.Perform(topo_face)
        nurbs_face = nurbs_converter.Shape()
        topo_nurbsface = topods.Face(nurbs_face)
        h_geomsurface = BRep_Tool.Surface(topo_nurbsface)
        h_bsurface = geomconvert_SurfaceToBSplineSurface(h_geomsurface)
        bsurface = h_bsurface

        nb_u = bsurface.NbUPoles()
        nb_v = bsurface.NbVPoles()
        # check consistency
        if points_face.shape[0] != nb_u * nb_v:
            raise ValueError("Input control points do not have not have the "
                             "same number as the geometric face!")

        # cycle on the face points
        indice_cpt = 0
        for iu in range(1, nb_u + 1):
            for iv in range(1, nb_v + 1):
                cpt = points_face[indice_cpt]
                bsurface.SetPole(iu, iv, gp_Pnt(cpt[0], cpt[1], cpt[2]))
                indice_cpt += 1

        # create modified new face
        new_bspline_tface = BRepBuilderAPI_MakeFace()
        toler = precision_Confusion()
        new_bspline_tface.Init(bsurface, False, toler)

        # cycle on the wires
        face_wires_explorer = TopExp_Explorer(
            topo_nurbsface.Oriented(TopAbs_FORWARD), TopAbs_WIRE)
        ind_edge_total = 0

        while face_wires_explorer.More():
            # get old wire
            twire = topods_Wire(face_wires_explorer.Current())

            # cycle on the edges
            ind_edge = 0
            wire_explorer_edge = TopExp_Explorer(
                twire.Oriented(TopAbs_FORWARD), TopAbs_EDGE)
            # check edges order on the wire
            mode3d = True
            tolerance_edges = toledge

            wire_order = ShapeAnalysis_WireOrder(mode3d, tolerance_edges)
            # an edge list
            deformed_edges = []
            # cycle on the edges
            while wire_explorer_edge.More():
                tedge = topods_Edge(wire_explorer_edge.Current())
                new_bspline_tedge = self.write_edge(
                    list_points_edge[ind_edge_total], tedge)

                deformed_edges.append(new_bspline_tedge)
                analyzer = topexp()
                vfirst = analyzer.FirstVertex(new_bspline_tedge)
                vlast = analyzer.LastVertex(new_bspline_tedge)
                pt1 = BRep_Tool.Pnt(vfirst)
                pt2 = BRep_Tool.Pnt(vlast)

                wire_order.Add(pt1.XYZ(), pt2.XYZ())

                ind_edge += 1
                ind_edge_total += 1
                wire_explorer_edge.Next()

            # grouping the edges in a wire, then in the face
            # check edges order and connectivity within the wire
            wire_order.Perform()
            # new wire to be created
            stol = ShapeFix_ShapeTolerance()
            new_bspline_twire = BRepBuilderAPI_MakeWire()
            for order_i in range(1, wire_order.NbEdges() + 1):
                deformed_edge_i = wire_order.Ordered(order_i)
                if deformed_edge_i > 0:
                    # insert the deformed edge to the new wire
                    new_edge_toadd = deformed_edges[deformed_edge_i - 1]
                    stol.SetTolerance(new_edge_toadd, toledge)
                    new_bspline_twire.Add(new_edge_toadd)
                    if new_bspline_twire.Error() != 0:
                        stol.SetTolerance(new_edge_toadd, toledge * 10.0)
                        new_bspline_twire.Add(new_edge_toadd)
                else:
                    deformed_edge_revers = deformed_edges[
                        np.abs(deformed_edge_i) - 1]
                    stol.SetTolerance(deformed_edge_revers, toledge)
                    new_bspline_twire.Add(deformed_edge_revers)
                    if new_bspline_twire.Error() != 0:
                        stol.SetTolerance(deformed_edge_revers, toledge * 10.0)
                        new_bspline_twire.Add(deformed_edge_revers)
            # add new wire to the Face
            new_bspline_tface.Add(new_bspline_twire.Wire())
            face_wires_explorer.Next()

        return topods.Face(new_bspline_tface.Face())

    @staticmethod
    def combine_faces(compshape, sew_tolerance):
        """
        Method to combine faces in a shell by adding connectivity and continuity
        :param compshape: TopoDS_Shape
        :param sew_tolerance: tolerance for sewing
        :return: Topo_Shell
        """

        offsew = BRepOffsetAPI_FindContigousEdges(sew_tolerance)
        sew = BRepBuilderAPI_Sewing(sew_tolerance)

        face_explorers = TopExp_Explorer(compshape, TopAbs_FACE)
        n_faces = 0
        # cycle on Faces
        while face_explorers.More():
            tface = topods.Face(face_explorers.Current())
            sew.Add(tface)
            offsew.Add(tface)
            n_faces += 1
            face_explorers.Next()

        offsew.Perform()
        offsew.Dump()
        sew.Perform()
        shell = sew.SewedShape()
        sew.Dump()

        shell = topods.Shell(shell)
        shell_fixer = ShapeFix_Shell()
        shell_fixer.FixFaceOrientation(shell)
        """
        if shell_fixer.Perform():
            print("{} shells fixed! ".format(shell_fixer.NbShells()))
        else:
            print "Shells not fixed! "

        new_shell = shell_fixer.Shell()

        if brepalgo_IsValid(new_shell):
            print "Shell valid! "
        else:
            print "Shell failed! "
        """
        return new_shell

    def write_shape(self, l_shells, filename, tol):
        """
        Method to recreate a TopoDS_Shape associated to a geometric shape
        after the modification of points of each Face. It
        returns a TopoDS_Shape (Shape).

        :param l_shells: the list of shells after initial parsing
        :param filename: the output filename
        :param tol: tolerance on the surface creation after modification
        :return: None

        """
        self.outfile = filename
        # global compound containing multiple shells
        global_compound_builder = BRep_Builder()
        global_comp = TopoDS_Compound()
        global_compound_builder.MakeCompound(global_comp)

        if self.check_topo == 0:
            # cycle on shells (multiple objects)
            shape_shells_explorer = TopExp_Explorer(
                self.shape.Oriented(TopAbs_FORWARD), TopAbs_SHELL)
            ishell = 0

            while shape_shells_explorer.More():
                per_shell = topods_Shell(shape_shells_explorer.Current())
                # a local compound containing a shell
                compound_builder = BRep_Builder()
                comp = TopoDS_Compound()
                compound_builder.MakeCompound(comp)

                # cycle on faces
                faces_explorer = TopExp_Explorer(
                    per_shell.Oriented(TopAbs_FORWARD), TopAbs_FACE)
                iface = 0
                while faces_explorer.More():
                    topoface = topods.Face(faces_explorer.Current())
                    newface = self.write_face(l_shells[ishell][iface][0],
                                              l_shells[ishell][iface][1],
                                              topoface, tol)

                    # add face to compound
                    compound_builder.Add(comp, newface)
                    iface += 1
                    faces_explorer.Next()

                new_shell = self.combine_faces(comp, 0.01)
                itype = TopoDS_Shape.ShapeType(new_shell)
                # add the new shell to the global compound
                global_compound_builder.Add(global_comp, new_shell)

                # TODO
                #print("Shell {0} of type {1} Processed ".format(ishell, itype))
                #print "=============================================="

                ishell += 1
                shape_shells_explorer.Next()

        else:
            # cycle on faces
            # a local compound containing a shell
            compound_builder = BRep_Builder()
            comp = TopoDS_Compound()
            compound_builder.MakeCompound(comp)

            # cycle on faces
            faces_explorer = TopExp_Explorer(
                self.shape.Oriented(TopAbs_FORWARD), TopAbs_FACE)
            iface = 0
            while faces_explorer.More():
                topoface = topods.Face(faces_explorer.Current())
                newface = self.write_face(l_shells[0][iface][0],
                                          l_shells[0][iface][1], topoface, tol)

                # add face to compound
                compound_builder.Add(comp, newface)
                iface += 1
                faces_explorer.Next()

            new_shell = self.combine_faces(comp, 0.01)
            itype = TopoDS_Shape.ShapeType(new_shell)
            # add the new shell to the global compound
            global_compound_builder.Add(global_comp, new_shell)

            # TODO print to logging
            # print("Shell {0} of type {1} Processed ".format(0, itype))
            # print "=============================================="

        self.write_shape_to_file(global_comp, self.outfile)

    def write_shape_to_file(self, shape, filename):
        """
        Abstract method to write the 'shape' to the `filename`.

        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError(\
            "Subclass must implement abstract method " +\
            self.__class__.__name__ + ".write_shape_to_file")

    def plot(self, plot_file=None, save_fig=False):
        """
        Method to plot a file. If `plot_file` is not given it plots
        `self.shape`.

        :param string plot_file: the filename you want to plot.
        :param bool save_fig: a flag to save the figure in png or not.
            If True the plot is not shown.

        :return: figure: matlplotlib structure for the figure of the
            chosen geometry
        :rtype: matplotlib.pyplot.figure
        """
        if plot_file is None:
            shape = self.shape
            plot_file = self.infile
        else:
            shape = self.load_shape_from_file(plot_file)

        stl_writer = StlAPI_Writer()
        # Do not switch SetASCIIMode() from False to True.
        stl_writer.SetASCIIMode(False)

        # Necessary to write to STL [to check]
        stl_mesh = BRepMesh_IncrementalMesh(shape, 0.01)
        stl_mesh.Perform()

        f = stl_writer.Write(shape, 'aux_figure.stl')

        # Create a new plot
        figure = pyplot.figure()
        axes = mplot3d.Axes3D(figure)

        # Load the STL files and add the vectors to the plot
        stl_mesh = mesh.Mesh.from_file('aux_figure.stl')
        os.remove('aux_figure.stl')
        axes.add_collection3d(
            mplot3d.art3d.Poly3DCollection(stl_mesh.vectors / 1000))

        # Get the limits of the axis and center the geometry
        max_dim = np.array([\
            np.max(stl_mesh.vectors[:, :, 0]) / 1000,\
            np.max(stl_mesh.vectors[:, :, 1]) / 1000,\
            np.max(stl_mesh.vectors[:, :, 2]) / 1000])
        min_dim = np.array([\
            np.min(stl_mesh.vectors[:, :, 0]) / 1000,\
            np.min(stl_mesh.vectors[:, :, 1]) / 1000,\
            np.min(stl_mesh.vectors[:, :, 2]) / 1000])

        max_lenght = np.max(max_dim - min_dim)
        axes.set_xlim(\
            -.6 * max_lenght + (max_dim[0] + min_dim[0]) / 2,\
            .6 * max_lenght + (max_dim[0] + min_dim[0]) / 2)
        axes.set_ylim(\
            -.6 * max_lenght + (max_dim[1] + min_dim[1]) / 2,\
            .6 * max_lenght + (max_dim[1] + min_dim[1]) / 2)
        axes.set_zlim(\
            -.6 * max_lenght + (max_dim[2] + min_dim[2]) / 2,\
            .6 * max_lenght + (max_dim[2] + min_dim[2]) / 2)

        # Show the plot to the screen
        if not save_fig:
            pyplot.show()
        else:
            figure.savefig(plot_file.split('.')[0] + '.png')

        return figure

    def show(self, show_file=None):
        """
        Method to show a file. If `show_file` is not given it plots
        `self.shape`.

        :param string show_file: the filename you want to show.
        """
        if show_file is None:
            shape = self.shape
        else:
            shape = self.load_shape_from_file(show_file)

        display, start_display, __, __ = init_display()
        display.FitAll()
        display.DisplayShape(shape, update=True)

        # Show the plot to the screen
        start_display()
