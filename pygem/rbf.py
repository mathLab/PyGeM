"""
Module focused on the implementation of the Radial Basis Functions interpolation
technique.  This technique is still based on the use of a set of parameters, the
so-called control points, as for FFD, but RBF is interpolatory. Another
important key point of RBF strategy relies in the way we can locate the control
points: in fact, instead of FFD where control points need to be placed inside a
regular lattice, with RBF we have no more limitations. So we have the
possibility to perform localized control points refinements.
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
import os
import numpy as np
try:
    import configparser as configparser
except ImportError:
    import ConfigParser as configparser


from scipy.spatial.distance import cdist

from .deformation import Deformation
from .rbf_factory import RBFFactory

import matplotlib.pyplot as plt


class RBF(Deformation):
    """
    Class that handles the Radial Basis Functions interpolation on the mesh
    points.

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
        :py:class:`~pygem.rbf.RBF` by passing the name of the right
        function (see class documentation for the updated list of basis
        function).  A callable object can be passed as basis function.
    :param float radius: the scaling parameter r that affects the shape of the
        basis functions.  For details see the class
        :class:`RBF`. The default value is 0.5.
    :param dict extra_parameter: the additional parameters that may be passed to
    	the kernel function. Default is None.
        
    :cvar numpy.ndarray weights: the matrix formed by the weights corresponding
        to the a-priori selected N control points, associated to the basis
        functions and c and Q terms that describe the polynomial of order one
        p(x) = c + Qx.  The shape is (*n_control_points+1+3*, *3*). It is
        computed internally.
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
    :cvar dict extra: the additional parameters that may be passed to the
        kernel function.
        
    :Example:

        >>> from pygem import RBF
        >>> import numpy as np
        >>> rbf = RBF(func='gaussian_spline')
        >>> xv = np.linspace(0, 1, 20)
        >>> yv = np.linspace(0, 1, 20)
        >>> zv = np.linspace(0, 1, 20)
        >>> z, y, x = np.meshgrid(zv, yv, xv)
        >>> mesh = np.array([x.ravel(), y.ravel(), z.ravel()])
        >>> deformed_mesh = rbf(mesh)
    """
    def __init__(self,
                 original_control_points=None,
                 deformed_control_points=None,
                 func='gaussian_spline',
                 radius=0.5,
                 extra_parameter=None):

        self.basis = func
        self.radius = radius

        if original_control_points is None:
            self.original_control_points = np.array([[0., 0., 0.], [0., 0., 1.],
                                                     [0., 1., 0.], [1., 0., 0.],
                                                     [0., 1., 1.], [1., 0., 1.],
                                                     [1., 1., 0.], [1., 1.,
                                                                    1.]])
        else:
            self.original_control_points = original_control_points

        if deformed_control_points is None:
            self.deformed_control_points = np.array([[0., 0., 0.], [0., 0., 1.],
                                                     [0., 1., 0.], [1., 0., 0.],
                                                     [0., 1., 1.], [1., 0., 1.],
                                                     [1., 1., 0.], [1., 1.,
                                                                    1.]])
        else:
            self.deformed_control_points = deformed_control_points

        self.extra = extra_parameter if extra_parameter else dict()

        self.weights = self._get_weights(self.original_control_points,
                                         self.deformed_control_points)


    @property
    def n_control_points(self):
        """
        Total number of control points.

        :rtype: int
        """
        return self.original_control_points.shape[0]

    @property
    def basis(self):
        """
        The kernel to use in the deformation.

        :getter: Returns the callable kernel
        :setter: Sets the kernel. It is possible to pass the name of the
            function (check the list of all implemented functions in the
            `pygem.rbf_factory.RBFFactory` class) or directly the callable
            function.
        :type: callable
        """
        return self.__basis

    @basis.setter
    def basis(self, func):
        if callable(func):
            self.__basis = func
        elif isinstance(func, str):
            self.__basis = RBFFactory(func)
        else:
            raise TypeError('`func` is not valid.')

    def _get_weights(self, X, Y):
        """
        This private method, given the original control points and the deformed
        ones, returns the matrix with the weights and the polynomial terms, that
        is :math:`W`, :math:`c^T` and :math:`Q^T`. The shape is
        (*n_control_points+1+3*, *3*).

        :param numpy.ndarray X: it is an n_control_points-by-3 array with the
            coordinates of the original interpolation control points before the
            deformation.
        :param numpy.ndarray Y: it is an n_control_points-by-3 array with the
            coordinates of the interpolation control points after the
            deformation.

        :return: weights: the 2D array with the weights and the polynomial terms.
        :rtype: numpy.ndarray
        """
        npts, dim = X.shape
        H = np.zeros((npts + 3 + 1, npts + 3 + 1))
        H[:npts, :npts] = self.basis(cdist(X, X), self.radius, **self.extra)
        H[npts, :npts] = 1.0
        H[:npts, npts] = 1.0
        H[:npts, -3:] = X
        H[-3:, :npts] = X.T

        rhs = np.zeros((npts + 3 + 1, dim))
        rhs[:npts, :] = Y
        weights = np.linalg.solve(H, rhs)
        return weights

    def read_parameters(self, filename='parameters_rbf.prm'):
        """
        Reads in the parameters file and fill the self structure.

        :param string filename: parameters file to be read in. Default value is
            parameters_rbf.prm.
        """
        if not isinstance(filename, str):
            raise TypeError('filename must be a string')

        # Checks if the parameters file exists. If not it writes the default
        # class into filename.  It consists in the vetices of a cube of side one
        # with a vertex in (0, 0, 0) and opposite one in (1, 1, 1).
        if not os.path.isfile(filename):
            self.write_parameters(filename)
            return

        config = configparser.RawConfigParser()
        config.read(filename)

        rbf_settings = dict(config.items('Radial Basis Functions'))
        
        self.basis = rbf_settings.pop('basis function')
        self.radius = float(rbf_settings.pop('radius'))
        self.extra = {k: eval(v) for k, v in rbf_settings.items()}

        ctrl_points = config.get('Control points', 'original control points')
        lines = ctrl_points.split('\n')
        self.original_control_points = np.array(
            list(map(lambda x: x.split(), lines)), dtype=float)

        mod_points = config.get('Control points', 'deformed control points')
        lines = mod_points.split('\n')
        self.deformed_control_points = np.array(
            list(map(lambda x: x.split(), lines)), dtype=float)

        if len(lines) != self.n_control_points:
            raise TypeError("The number of control points must be equal both in"
                            "the 'original control points' and in the 'deformed"
                            "control points' section of the parameters file"
                            "({0!s})".format(filename))

    def write_parameters(self, filename='parameters_rbf.prm'):
        """
        This method writes a parameters file (.prm) called `filename` and fills
        it with all the parameters class members. Default value is
        parameters_rbf.prm.

        :param string filename: parameters file to be written out.
        """
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")

        output_string = ""
        output_string += '\n[Radial Basis Functions]\n'
        output_string += '# This section describes the radial basis functions'
        output_string += ' shape.\n'

        output_string += '\n# basis funtion is the name of the basis functions'
        output_string += ' to use in the transformation. The functions\n'
        output_string += '# implemented so far are: gaussian_spline,'
        output_string += ' multi_quadratic_biharmonic_spline,\n'
        output_string += '# inv_multi_quadratic_biharmonic_spline,'
        output_string += ' thin_plate_spline, beckert_wendland_c2_basis,'
        output_string += ' polyharmonic_spline.\n'
        output_string += '# For a comprehensive list with details see the'
        output_string += ' class RBF.\n'
        output_string += 'basis function: {}\n'.format('gaussian_spline')

        output_string += '\n# radius is the scaling parameter r that affects'
        output_string += ' the shape of the basis functions. See the'
        output_string += ' documentation\n'
        output_string += '# of the class RBF for details.\n'
        output_string += 'radius: {}\n'.format(str(self.radius))

        output_string += '\n\n[Control points]\n'
        output_string += '# This section describes the RBF control points.\n'

        output_string += '\n# original control points collects the coordinates'
        output_string += ' of the interpolation control points before the'
        output_string += ' deformation.\n'

        output_string += 'original control points:'
        offset = 1
        for i in range(0, self.n_control_points):
            output_string += offset * ' ' + str(
                self.original_control_points[i][0]) + '   ' + str(
                    self.original_control_points[i][1]) + '   ' + str(
                        self.original_control_points[i][2]) + '\n'
            offset = 25

        output_string += '\n# deformed control points collects the coordinates'
        output_string += ' of the interpolation control points after the'
        output_string += ' deformation.\n'

        output_string += 'deformed control points:'
        offset = 1
        for i in range(0, self.n_control_points):
            output_string += offset * ' ' + str(
                self.deformed_control_points[i][0]) + '   ' + str(
                    self.deformed_control_points[i][1]) + '   ' + str(
                        self.deformed_control_points[i][2]) + '\n'
            offset = 25

        with open(filename, 'w') as f:
            f.write(output_string)

    def __str__(self):
        """
        This method prints all the RBF parameters on the screen. Its purpose is
        for debugging.
        """
        string = ''
        string += 'basis function = {}\n'.format(self.basis)
        string += 'radius = {}\n'.format(self.radius)
        string += 'extra_parameter = {}\n'.format(self.extra)
        string += '\noriginal control points =\n'
        string += '{}\n'.format(self.original_control_points)
        string += '\ndeformed control points =\n'
        string += '{}\n'.format(self.deformed_control_points)
        return string

    def plot_points(self, filename=None):
        """
        Method to plot the control points. It is possible to save the resulting
        figure.

        :param str filename: if None the figure is shown, otherwise it is saved
            on the specified `filename`. Default is None.
        """
        fig = plt.figure(1)
        axes = fig.add_subplot(111, projection='3d')
        orig = axes.scatter(self.original_control_points[:, 0],
                            self.original_control_points[:, 1],
                            self.original_control_points[:, 2],
                            c='blue',
                            marker='o')
        defor = axes.scatter(self.deformed_control_points[:, 0],
                             self.deformed_control_points[:, 1],
                             self.deformed_control_points[:, 2],
                             c='red',
                             marker='x')

        axes.set_xlabel('X axis')
        axes.set_ylabel('Y axis')
        axes.set_zlabel('Z axis')

        plt.legend((orig, defor), ('Original', 'Deformed'),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=2,
                   fontsize=10)

        # Show the plot to the screen
        if filename is None:
            plt.show()
        else:
            fig.savefig(filename)

    def compute_weights(self):
        """
        This method compute the weights according to the
        `original_control_points` and `deformed_control_points` arrays.
        """
        self.weights = self._get_weights(self.original_control_points,
                                         self.deformed_control_points)

    def __call__(self, src_pts):
        """
        This method performs the deformation of the mesh points. After the
        execution it sets `self.modified_mesh_points`.
        """
        self.compute_weights()

        H = np.zeros((src_pts.shape[0], self.n_control_points + 3 + 1))
        H[:, :self.n_control_points] = self.basis(
            cdist(src_pts, self.original_control_points), 
            self.radius,
            **self.extra)
        H[:, self.n_control_points] = 1.0
        H[:, -3:] = src_pts
        return np.asarray(np.dot(H, self.weights))
