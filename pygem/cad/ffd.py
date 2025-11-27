"""
Utilities for performing Free Form Deformation (FFD) to CAD geometries.

:Theoretical Insight:

    Free Form Deformation is a technique for the efficient, smooth and accurate
    geometrical parametrization. It has been proposed the first time in
    *Sederberg, Thomas W., and Scott R. Parry. "Free-form deformation of solid
    geometric models." ACM SIGGRAPH computer graphics 20.4 (1986): 151-160*. It
    consists in three different step:

    - Mapping the physical domain to the reference one with map
      :math:`\\boldsymbol{\\psi}`.  In the code it is named *transformation*.

    - Moving some control points to deform the lattice with :math:`\\hat{T}`.
      The movement of the control points is basically the weight (or
      displacement) :math:`\\boldsymbol{\\mu}` we set in the *parameters file*.

    - Mapping back to the physical domain with map
      :math:`\\boldsymbol{\\psi}^{-1}`.  In the code it is named
      *inverse_transformation*.

    FFD map (:math:`T`) is the composition of the three maps, that is

    .. math:: T(\\cdot, \\boldsymbol{\\mu}) = (\\Psi^{-1} \\circ \\hat{T} \\circ
            \\Psi) (\\cdot, \\boldsymbol{\\mu})

    In this way, every point inside the FFD box changes it position according to

    .. math:: \\boldsymbol{P} = \\boldsymbol{\\psi}^{-1} \\left( \\sum_{l=0}^L
            \\sum_{m=0}^M \\sum_{n=0}^N
            \\mathsf{b}_{lmn}(\\boldsymbol{\\psi}(\\boldsymbol{P}_0))
            \\boldsymbol{\\mu}_{lmn} \\right)

    where :math:`\\mathsf{b}_{lmn}` are Bernstein polynomials.  We improve the
    traditional version by allowing a rotation of the FFD lattice in order to
    give more flexibility to the tool.

    You can try to add more shapes to the lattice to allow more and more
    involved transformations.

"""

import numpy as np
from pygem import FFD as OriginalFFD
from .cad_deformation import CADDeformation

class FFD(CADDeformation, OriginalFFD):
    """
    Class that handles the Free Form Deformation on CAD geometries.

    :param list n_control_points: number of control points in the x, y, and z
        direction. If not provided it is set to [2, 2, 2].
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

    :cvar numpy.ndarray box_length: dimension of the FFD bounding box, in the
        x, y and z direction (local coordinate system).
    :cvar numpy.ndarray box_origin: the x, y and z coordinates of the origin of
        the FFD bounding box.
    :cvar numpy.ndarray rot_angle: rotation angle around x, y and z axis of the
        FFD bounding box.
    :cvar numpy.ndarray n_control_points: the number of control points in the
        x, y, and z direction.
    :cvar numpy.ndarray array_mu_x: collects the displacements (weights) along
        x, normalized with the box length x.
    :cvar numpy.ndarray array_mu_y: collects the displacements (weights) along
        y, normalized with the box length y.
    :cvar numpy.ndarray array_mu_z: collects the displacements (weights) along
        z, normalized with the box length z.
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

        >>> from pygem.cad import FFD
        >>> from pygem.cad import CADDeformation
        >>> ffd = FFD()
        >>> ffd.read_parameters('parameters_test_ffd_iges.prm')
        >>> input_cad_file_name = "input.iges"
        >>> modified_cad_file_name = "output.iges"
        >>> ffd(input_cad_file_name, modified_cad_file_name)
        >>> # is equivalent to
        >>> new_shape = ffd(CADDeformation.read_shape(input_cad_file_name))
    """
    def __init__(self,
                 n_control_points=None,
                 u_knots_to_add=0,
                 v_knots_to_add=0,
                 t_knots_to_add=0,
                 tolerance=1e-4):
        OriginalFFD.__init__(self,
                             n_control_points=n_control_points)
        CADDeformation.__init__(self, 
                                u_knots_to_add=u_knots_to_add, 
                                v_knots_to_add=v_knots_to_add, 
                                t_knots_to_add=t_knots_to_add, 
                                tolerance=tolerance)
