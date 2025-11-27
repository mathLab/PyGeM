"""
Module for custom deformations to CAD geometries.
"""

import numpy as np
from pygem import CustomDeformation as OriginalCustomDeformation
from .cad_deformation import CADDeformation

class CustomDeformation(CADDeformation, OriginalCustomDeformation):
    """
    Class to perform a custom deformation to the CAD geometries.

    :param callable func: the function definying the deformation of the input
        points. This function should take as input: *i*) a 2D array of shape
        (*n_points*, *3*) in which the points are arranged by row, or *ii*) an
        iterable object with 3 components. In this last case, computation of
        deformation is not vectorizedand the overall cost may become heavy.
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
         
    :Example:

        >>> from pygem.cad import CustomDeformation
        >>> def move(x):
        >>>     return x + x**2
        >>> deform = CustomDeformation(move)
        >>> deform('original_shape.iges', dst='new_shape.iges')
    """
    def __init__(self,
                 func,
                 u_knots_to_add=0,
                 v_knots_to_add=0,
                 t_knots_to_add=0,
                 tolerance=1e-4):
        OriginalCustomDeformation.__init__(self, func)
        CADDeformation.__init__(self, 
                                u_knots_to_add=u_knots_to_add, 
                                v_knots_to_add=v_knots_to_add, 
                                t_knots_to_add=t_knots_to_add, 
                                tolerance=tolerance)
