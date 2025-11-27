"""
Module focused on the Inverse Distance Weighting interpolation technique.
The IDW algorithm is an average moving interpolation that is usually applied to
highly variable data. The main idea of this interpolation strategy lies in
fact that it is not desirable to honour local high/low values but rather to look
at a moving average of nearby data points and estimate the local trends.
The node value is calculated by averaging the weighted sum of all the points.
Data points that lie progressively farther from the node inuence much less the
computed value than those lying closer to the node.

:Theoretical Insight:

    This implementation is based on the simplest form of inverse distance
    weighting interpolation, proposed by D. Shepard, A two-dimensional
    interpolation function for irregularly-spaced data, Proceedings of the 23 rd
    ACM National Conference.

    The interpolation value :math:`u` of a given point :math:`\\mathrm{x}`
    from a set of samples :math:`u_k = u(\\mathrm{x}_k)`, with
    :math:`k = 1,2,\\dotsc,\\mathcal{N}`, is given by:

    .. math::
        u(\\mathrm{x}) = \\displaystyle\\sum_{k=1}^\\mathcal{N}
        \\frac{w(\\mathrm{x},\\mathrm{x}_k)}
        {\\displaystyle\\sum_{j=1}^\\mathcal{N} w(\\mathrm{x},\\mathrm{x}_j)}
        u_k

    where, in general, :math:`w(\\mathrm{x}, \\mathrm{x}_i)` represents the
    weighting function:

    .. math::
        w(\\mathrm{x}, \\mathrm{x}_i) = \\| \\mathrm{x} - \\mathrm{x}_i \\|^{-p}

    being :math:`\\| \\mathrm{x} - \\mathrm{x}_i \\|^{-p} \\ge 0` is the
    Euclidean distance between :math:`\\mathrm{x}` and data point
    :math:`\\mathrm{x}_i` and :math:`p` is a power parameter, typically equal to
    2.
"""

import numpy as np
from pygem import IDW as OriginalIDW
from .cad_deformation import CADDeformation

class IDW(CADDeformation, OriginalIDW):
    """
    Class that perform the Inverse Distance Weighting (IDW) on CAD geometries.

    :param int power: the power parameter. The default value is 2.
    :param numpy.ndarray original_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the original
        interpolation control points before the deformation. The default is the
        vertices of the unit cube.
    :param numpy.ndarray deformed_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the
        interpolation control points after the deformation. The default is the
        vertices of the unit cube.
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
        
    :cvar int power: the power parameter. The default value is 2.
    :cvar numpy.ndarray original_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the original
        interpolation control points before the deformation. The default is the
        vertices of the unit cube.
    :cvar numpy.ndarray deformed_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the
        interpolation control points after the deformation. The default is the
        vertices of the unit cube.
    :cvar int u_knots_to_add: the number of knots to add to the NURBS surfaces
        along `u` direction before the deformation. This parameter is useful
        whenever the gradient of the imposed deformation present spatial scales
        that are smaller than the local distance among the original poles of
        the surface/curve. Enriching the poles will allow for a more accurate
        application of the deformation, and might also reduce possible
        mismatches between bordering faces. On the orther hand, it might result
        in higher computational cost and bigger output files. Default is 0.
    :cvar int v_knots_to_add: the number of knots to add to the NURBS surfaces
        along `v` direction before the deformation. This parameter is useful
        whenever the gradient of the imposed deformation present spatial scales
        that are smaller than the local distance among the original poles of
        the surface/curve. Enriching the poles will allow for a more accurate
        application of the deformation, and might also reduce possible
        mismatches between bordering faces. On the orther hand, it might result
        in higher computational cost and bigger output files.  Default is 0.
    :cvar int t_knots_to_add: the number of knots to add to the NURBS curves
        before the deformation. This parameter is useful whenever the gradient
        of the imposed deformation present spatial scales that are smaller than
        the local distance among the original poles of the surface/curve.
        Enriching the poles will allow for a more accurate application of the
        deformation, and might also reduce possible mismatches between
        bordering faces. On the orther hand, it might result in higher
        computational cost and bigger output files. Default is 0.
    :cvar float tolerance: the tolerance involved in several internal
        operations of the procedure (joining wires in a single curve before
        deformation and placing new poles on curves and surfaces). Change the
        default value only if the input file scale is significantly different
        form mm, making some of the aforementioned operations fail. Default is
        0.0001.

    :Example:

        >>> from pygem.cad import IDW
        >>> idw = IDW()
        >>> idw.read_parameters(
        >>>        'tests/test_datasets/parameters_test_idw_iges.prm')
        >>> input_cad_file_name = "input.iges"
        >>> modified_cad_file_name = "output.iges"
        >>> idw(input_cad_file_name, modified_cad_file_name)
    """
    def __init__(self,
                 original_control_points=None,
                 deformed_control_points=None,
                 power=2,
                 u_knots_to_add=0,
                 v_knots_to_add=0,
                 t_knots_to_add=0,
                 tolerance=1e-4):
        OriginalIDW.__init__(self,
                             original_control_points=original_control_points,
                             deformed_control_points=deformed_control_points,
                             power=power)
        CADDeformation.__init__(self,
                                u_knots_to_add=u_knots_to_add,
                                v_knots_to_add=v_knots_to_add,
                                t_knots_to_add=t_knots_to_add,
                                tolerance=tolerance)
