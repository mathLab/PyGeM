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
import os
import numpy as np
try:
    import configparser as configparser
except ImportError:
    import ConfigParser as configparser

from scipy.spatial.distance import cdist
from .deformation import Deformation


class IDW(Deformation):
    """
    Class that perform the Inverse Distance Weighting (IDW).

    :param int power: the power parameter. The default value is 2.
    :param numpy.ndarray original_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the original
        interpolation control points before the deformation. The default is the
        vertices of the unit cube.
    :param numpy.ndarray deformed_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the
        interpolation control points after the deformation. The default is the
        vertices of the unit cube.
        
    :cvar int power: the power parameter. The default value is 2.
    :cvar numpy.ndarray original_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the original
        interpolation control points before the deformation. The default is the
        vertices of the unit cube.
    :cvar numpy.ndarray deformed_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the
        interpolation control points after the deformation. The default is the
        vertices of the unit cube.

    :Example:

    >>> from pygem import IDW
    >>> import numpy as np
    >>> nx, ny, nz = (20, 20, 20)
    >>> mesh = np.zeros((nx * ny * nz, 3))
    >>> xv = np.linspace(0, 1, nx)
    >>> yv = np.linspace(0, 1, ny)
    >>> zv = np.linspace(0, 1, nz)
    >>> z, y, x = np.meshgrid(zv, yv, xv)
    >>> mesh_points = np.array([x.ravel(), y.ravel(), z.ravel()])
    >>> idw = IDW()
    >>> idw.read_parameters('tests/test_datasets/parameters_idw_cube.prm')
    >>> new_mesh_points = idw(mesh_points.T)
    """
    def __init__(self,
                 original_control_points=None,
                 deformed_control_points=None,
                 power=2):

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

        self.power = power

    def __call__(self, src_pts):
        """
        This method performs the deformation of the mesh points. After the
        execution it sets `self.modified_mesh_points`.
        """
        def distance(u, v):
            """ Norm of u - v """
            return np.linalg.norm(u - v, ord=self.power)

        # Compute displacement of the control points
        displ = self.deformed_control_points - self.original_control_points

        # Compute the distance between the mesh points and the control points
        dist = cdist(src_pts, self.original_control_points, distance)

        # Weights are set as the reciprocal of the distance if the distance is
        # not zero, otherwise 1.0 where distance is zero.
        weights = np.zeros(dist.shape)
        for i, d in enumerate(dist):
            weights[i] = 1. / d if d.all() else np.where(d == 0.0, 1.0, 0.0)

        offset = np.array([
            np.sum(displ * wi[:, np.newaxis] / np.sum(wi), axis=0)
            for wi in weights
        ])

        return src_pts + offset

    def read_parameters(self, filename):
        """
        Reads in the parameters file and fill the self structure.

        :param string filename: parameters file to be read in.
        """
        if not isinstance(filename, str):
            raise TypeError('filename must be a string')

        if not os.path.isfile(filename):
            raise IOError('filename does not exist')

        config = configparser.RawConfigParser()
        config.read(filename)

        self.power = config.getint('Inverse Distance Weighting', 'power')

        ctrl_points = config.get('Control points', 'original control points')
        self.original_control_points = np.array(
            [line.split() for line in ctrl_points.split('\n')], dtype=float)

        defo_points = config.get('Control points', 'deformed control points')
        self.deformed_control_points = np.array(
            [line.split() for line in defo_points.split('\n')], dtype=float)

    def write_parameters(self, filename):
        """
        This method writes a parameters file (.prm) called `filename` and fills
        it with all the parameters class members.

        :param string filename: parameters file to be written out.
        """
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")

        output_string = ""
        output_string += "\n[Inverse Distance Weighting]\n"
        output_string += "# This section describes the settings of idw.\n\n"
        output_string += "# the power parameter\n"
        output_string += "power = {}\n".format(self.power)

        output_string += "\n\n[Control points]\n"
        output_string += "# This section describes the IDW control points.\n\n"
        output_string += "# original control points collects the coordinates\n"
        output_string += "# of the interpolation control points before the\n"
        output_string += "# deformation.\n"

        output_string += "original control points: "
        output_string += (
            '   '.join(map(str, self.original_control_points[0])) + "\n")
        for points in self.original_control_points[1:]:
            output_string += 25 * ' ' + '   '.join(map(str, points)) + "\n"
        output_string += "\n"
        output_string += "# deformed control points collects the coordinates\n"
        output_string += "# of the interpolation control points after the\n"
        output_string += "# deformation.\n"
        output_string += "deformed control points: "
        output_string += (
            '   '.join(map(str, self.original_control_points[0])) + "\n")
        for points in self.deformed_control_points[1:]:
            output_string += 25 * ' ' + '   '.join(map(str, points)) + "\n"

        with open(filename, 'w') as f:
            f.write(output_string)

    def __str__(self):
        """
        This method prints all the IDW parameters on the screen. Its purpose is
        for debugging.
        """
        string = ''
        string += 'p = {}\n'.format(self.power)
        string += '\noriginal_control_points =\n'
        string += '{}\n'.format(self.original_control_points)
        string += '\ndeformed_control_points =\n'
        string += '{}\n'.format(self.deformed_control_points)
        return string
