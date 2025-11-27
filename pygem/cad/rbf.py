"""
Module focused on the implementation of the Radial Basis Functions interpolation
technique.  This technique is still based on the use of a set of parameters, the
so-called control points, as for FFD, but RBF is interpolatory. Another
important key point of RBF strategy relies in the way we can locate the control
points: in fact, instead of FFD where control points need to be placed inside a
regular lattice, with RBF we hano no more limitations. So we have the
possibility to perform localized control points refiniments.
The module is analogous to the freeform one.

:Theoretical Insight:

    As reference please consult M.D. Buhmann, Radial Basis Functions, volume 12
    of Cambridge monographs on applied and computational mathematics. Cambridge
    University Press, UK, 2003.  This implementation follows D. Forti and G.
    Rozza, Efficient geometrical parametrization techniques of interfaces for
    reduced order modelling: application to fluid-structure interaction coupling
    problems, International Journal of Computational Fluid Dynamics.

    RBF shape parametrization technique is based on the definition of a map,
    :math:`\\mathcal{M}(\\boldsymbol{x}) : \\mathbb{R}^n \\rightarrow
    \\mathbb{R}^n`, that allows the possibility of transferring data across
    non-matching grids and facing the dynamic mesh handling. The map introduced
    is defines as follows

    .. math::
        \\mathcal{M}(\\boldsymbol{x}) = p(\\boldsymbol{x}) + 
        \\sum_{i=1}^{\\mathcal{N}_C} \\gamma_i
        \\varphi(\\| \\boldsymbol{x} - \\boldsymbol{x_{C_i}} \\|)

    where :math:`p(\\boldsymbol{x})` is a low_degree polynomial term,
    :math:`\\gamma_i` is the weight, corresponding to the a-priori selected
    :math:`\\mathcal{N}_C` control points, associated to the :math:`i`-th basis
    function, and :math:`\\varphi(\\| \\boldsymbol{x} - \\boldsymbol{x_{C_i}}
    \\|)` a radial function based on the Euclidean distance between the control
    points position :math:`\\boldsymbol{x_{C_i}}` and :math:`\\boldsymbol{x}`.
    A radial basis function, generally, is a real-valued function whose value
    depends only on the distance from the origin, so that
    :math:`\\varphi(\\boldsymbol{x}) = \\tilde{\\varphi}(\\| \\boldsymbol{x}
    \\|)`.

    The matrix version of the formula above is:

    .. math::
        \\mathcal{M}(\\boldsymbol{x}) = \\boldsymbol{c} +
        \\boldsymbol{Q}\\boldsymbol{x} +
        \\boldsymbol{W^T}\\boldsymbol{d}(\\boldsymbol{x})

    The idea is that after the computation of the weights and the polynomial
    terms from the coordinates of the control points before and after the
    deformation, we can deform all the points of the mesh accordingly.  Among
    the most common used radial basis functions for modelling 2D and 3D shapes,
    we consider Gaussian splines, Multi-quadratic biharmonic splines, Inverted
    multi-quadratic biharmonic splines, Thin-plate splines, Beckert and
    Wendland :math:`C^2` basis and Polyharmonic splines all defined and
    implemented below.
"""

import numpy as np
from pygem import RBF as OriginalRBF
from .cad_deformation import CADDeformation

class RBF(CADDeformation, OriginalRBF):
    """
    Class that handles the Radial Basis Functions interpolation on CAD
    geometries.
    
    :param numpy.ndarray original_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the original
        interpolation control points before the deformation. The default is the
        vertices of the unit cube.
    :param numpy.ndarray deformed_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the
        interpolation control points after the deformation. The default is the
        vertices of the unit cube.
    :param func: the basis function to use in the transformation. Several basis
        function are already implemented and they are available through the
        :class:`~pygem.rbf_factory.RBFFactory` by passing the name of the right
        function (see class documentation for the updated list of basis
        function).  A callable object can be passed as basis function. Default
        is 'gaussian_spline'.
    :param float radius: the scaling parameter r that affects the shape of the
        basis functions.  For details see the class
        :class:`~pygem.radialbasis.RBF`. The default value is 0.5.
    :param dict extra_parameter: the additional parameters that may be passed to
    	the kernel function. Default is None.
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
        
    :cvar numpy.ndarray weights: the matrix formed by the weights corresponding
        to the a-priori selected N control points, associated to the basis
        functions and c and Q terms that describe the polynomial of order one
        p(x) = c + Qx.  The shape is (n_control_points+1+3)-by-3. It is computed
        internally.
    :cvar numpy.ndarray original_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the original
        interpolation control points before the deformation.
    :cvar numpy.ndarray deformed_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the
        interpolation control points after the deformation.
    :cvar callable basis: the basis functions to use in the
        transformation.
    :cvar float radius: the scaling parameter that affects the shape of the
        basis functions.
    :cvar dict extra_parameter: the additional parameters that may be passed to
    	the kernel function.
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

        >>> from pygem.cad import RBF
        >>> rbf = RBF()
        >>> rbf.read_parameters(
        >>>        'tests/test_datasets/parameters_test_ffd_iges.prm')
        >>> input_cad_file_name = "input.iges"
        >>> modified_cad_file_name = "output.iges"
        >>> rbf(input_cad_file_name, modified_cad_file_name)
    """
    def __init__(self,
                 original_control_points=None,
                 deformed_control_points=None,
                 func='gaussian_spline',
                 radius=0.5,
                 extra_parameter=None,
                 u_knots_to_add=0,
                 v_knots_to_add=0,
                 t_knots_to_add=0,
                 tolerance=1e-4):
        OriginalRBF.__init__(self, 
                             original_control_points=original_control_points, 
                             deformed_control_points=deformed_control_points,
                             func=func,
                             radius=radius,
                             extra_parameter=extra_parameter)
        CADDeformation.__init__(self,
                                u_knots_to_add=u_knots_to_add,
                                v_knots_to_add=v_knots_to_add,
                                t_knots_to_add=t_knots_to_add,
                                tolerance=tolerance)
