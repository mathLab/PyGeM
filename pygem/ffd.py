"""
Utilities for performing Free Form Deformation (FFD).

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
try:
    import configparser as configparser
except ImportError:
    import ConfigParser as configparser
import os
import copy
import numpy as np
from scipy import special

from pygem import Deformation
from pygem.utils import fit_affine_transformation, angles2matrix


class FFD(Deformation):
    """
    Class that handles the Free Form Deformation on the mesh points.

    :param list n_control_points: number of control points in the x, y, and z
        direction. Default is [2, 2, 2].
        
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

    :Example:

        >>> from pygem import FFD
        >>> import numpy as np
        >>> ffd = FFD()
        >>> ffd.read_parameters(
        >>>        'tests/test_datasets/parameters_test_ffd_sphere.prm')
        >>> original_mesh_points = np.load(
        >>>         'tests/test_datasets/meshpoints_sphere_orig.npy')
        >>> new_mesh_points = ffd(original_mesh_points)
    """
    reference_frame = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def __init__(self, n_control_points=None):
        self.conversion_unit = 1. # TODO: unused at the moment

        self.box_length = np.array([1., 1., 1.])
        self.box_origin = np.array([0., 0., 0.])
        self.rot_angle = np.array([0., 0., 0.])

        self.array_mu_x = None
        self.array_mu_y = None
        self.array_mu_z = None

        if n_control_points is None:
            n_control_points = [2, 2, 2]
        self.n_control_points = n_control_points

    @property
    def n_control_points(self):
        """
        The number of control points in X, Y and Z directions

        :rtype: numpy.ndarray
        """
        return self._n_control_points

    @n_control_points.setter
    def n_control_points(self, npts):
        self._n_control_points = np.array(npts)
        self.array_mu_x = np.zeros(self.n_control_points)
        self.array_mu_y = np.zeros(self.n_control_points)
        self.array_mu_z = np.zeros(self.n_control_points)

    @property
    def psi(self):
        """
        Return the function that map the physical domain to the reference
        domain.

        :rtype: callable
        """
        physical_frame = self.position_vertices - self.box_origin
        return fit_affine_transformation(physical_frame, self.reference_frame)

    @property
    def inverse_psi(self):
        """
        Return the function that map the reference domain to the physical
        domain.

        :rtype: callable
        """
        physical_frame = self.position_vertices - self.box_origin
        return fit_affine_transformation(self.reference_frame, physical_frame)

    @property
    def T(self):
        """
        Return the function that deforms the points within the unit cube.

        :rtype: callable
        """
        def T_mapping(points):
            (n_rows, n_cols) = points.shape
            (dim_n_mu, dim_m_mu, dim_t_mu) = self.array_mu_x.shape

            # Initialization. In order to exploit the contiguity in memory the
            # following are transposed
            bernstein_x = np.zeros((dim_n_mu, n_rows))
            bernstein_y = np.zeros((dim_m_mu, n_rows))
            bernstein_z = np.zeros((dim_t_mu, n_rows))
            shift_points = np.zeros((n_cols, n_rows))

            # TODO check no-loop implementation
            #bernstein_x = (
            #    np.power(mesh_points[:, 0][:, None], range(dim_n_mu)) *
            #    np.power(1 - mesh_points[:, 0][:, None], range(dim_n_mu-1, -1, -1)) *
            #    special.binom(np.array([dim_n_mu-1]*dim_n_mu), np.arange(dim_n_mu))
            #)
            for i in range(0, dim_n_mu):
                aux1 = np.power((1 - points[:, 0]), dim_n_mu - 1 - i)
                aux2 = np.power(points[:, 0], i)
                bernstein_x[i, :] = (special.binom(dim_n_mu - 1, i) *
                                     np.multiply(aux1, aux2))

            for i in range(0, dim_m_mu):
                aux1 = np.power((1 - points[:, 1]), dim_m_mu - 1 - i)
                aux2 = np.power(points[:, 1], i)
                bernstein_y[i, :] = special.binom(dim_m_mu - 1,
                                                  i) * np.multiply(aux1, aux2)

            for i in range(0, dim_t_mu):
                aux1 = np.power((1 - points[:, 2]), dim_t_mu - 1 - i)
                aux2 = np.power(points[:, 2], i)
                bernstein_z[i, :] = special.binom(dim_t_mu - 1,
                                                  i) * np.multiply(aux1, aux2)

            aux_x = 0.
            aux_y = 0.
            aux_z = 0.

            for j in range(0, dim_m_mu):
                for k in range(0, dim_t_mu):
                    bernstein_yz = np.multiply(bernstein_y[j, :],
                                               bernstein_z[k, :])
                    for i in range(0, dim_n_mu):
                        aux = np.multiply(bernstein_x[i, :], bernstein_yz)
                        aux_x += aux * self.array_mu_x[i, j, k]
                        aux_y += aux * self.array_mu_y[i, j, k]
                        aux_z += aux * self.array_mu_z[i, j, k]

            shift_points[0, :] += aux_x
            shift_points[1, :] += aux_y
            shift_points[2, :] += aux_z
            return shift_points.T + points

        return T_mapping

    @property
    def rotation_matrix(self):
        """
        The rotation matrix (according to rot_angle_x, rot_angle_y,
        rot_angle_z).

        :rtype: numpy.ndarray
        """
        return angles2matrix(np.radians(self.rot_angle[2]),
                             np.radians(self.rot_angle[1]),
                             np.radians(self.rot_angle[0]))

    @property
    def position_vertices(self):
        """
        The position of the vertices of the FFD bounding box.

        :rtype: numpy.ndarray
        """
        return self.box_origin + np.vstack([
            np.zeros((1, 3)),
            self.rotation_matrix.dot(np.diag(self.box_length)).T
        ])

    def reset_weights(self):
        """
        Set transformation parameters to arrays of zeros.
        """
        self.array_mu_x.fill(0.0)
        self.array_mu_y.fill(0.0)
        self.array_mu_z.fill(0.0)

    def read_parameters(self, filename='parameters.prm'):
        """
        Reads in the parameters file and fill the self structure.

        :param string filename: parameters file to be read in.
        """
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")

        # Checks if the parameters file exists. If not it writes the default
        # class into filename.
        if not os.path.isfile(filename):
            self.write_parameters(filename)
            return

        config = configparser.RawConfigParser()
        config.read(filename)

        self.n_control_points[0] = config.getint('Box info',
                                                 'n control points x')
        self.n_control_points[1] = config.getint('Box info',
                                                 'n control points y')
        self.n_control_points[2] = config.getint('Box info',
                                                 'n control points z')

        self.box_length[0] = config.getfloat('Box info', 'box length x')
        self.box_length[1] = config.getfloat('Box info', 'box length y')
        self.box_length[2] = config.getfloat('Box info', 'box length z')

        self.box_origin[0] = config.getfloat('Box info', 'box origin x')
        self.box_origin[1] = config.getfloat('Box info', 'box origin y')
        self.box_origin[2] = config.getfloat('Box info', 'box origin z')

        self.rot_angle[0] = config.getfloat('Box info', 'rotation angle x')
        self.rot_angle[1] = config.getfloat('Box info', 'rotation angle y')
        self.rot_angle[2] = config.getfloat('Box info', 'rotation angle z')

        self.array_mu_x = np.zeros(self.n_control_points)
        self.array_mu_y = np.zeros(self.n_control_points)
        self.array_mu_z = np.zeros(self.n_control_points)

        mux = config.get('Parameters weights', 'parameter x')
        muy = config.get('Parameters weights', 'parameter y')
        muz = config.get('Parameters weights', 'parameter z')

        for line in mux.split('\n'):
            values = np.array(line.split())
            self.array_mu_x[tuple(map(int, values[0:3]))] = float(values[3])

        for line in muy.split('\n'):
            values = line.split()
            self.array_mu_y[tuple(map(int, values[0:3]))] = float(values[3])

        for line in muz.split('\n'):
            values = line.split()
            self.array_mu_z[tuple(map(int, values[0:3]))] = float(values[3])

    def write_parameters(self, filename='parameters.prm'):
        """
        This method writes a parameters file (.prm) called `filename` and fills
        it with all the parameters class members.

        :param string filename: parameters file to be written out.
        """
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")

        output_string = ""
        output_string += '\n[Box info]\n'
        output_string += '# This section collects all the properties of the'
        output_string += ' FFD bounding box.\n'

        output_string += '\n# n control points indicates the number of control'
        output_string += ' points in each direction (x, y, z).\n'
        output_string += '# For example, to create a 2 x 3 x 2 grid, use the'
        output_string += ' following: n control points: 2, 3, 2\n'
        output_string += 'n control points x: ' + str(
            self.n_control_points[0]) + '\n'
        output_string += 'n control points y: ' + str(
            self.n_control_points[1]) + '\n'
        output_string += 'n control points z: ' + str(
            self.n_control_points[2]) + '\n'

        output_string += '\n# box length indicates the length of the FFD '
        output_string += 'bounding box along the three canonical directions '
        output_string += '(x, y, z).\n'

        output_string += '# It uses the local coordinate system.\n'
        output_string += '# For example to create a 2 x 1.5 x 3 meters box '
        output_string += 'use the following: box length: 2.0, 1.5, 3.0\n'

        output_string += 'box length x: ' + str(self.box_length[0]) + '\n'
        output_string += 'box length y: ' + str(self.box_length[1]) + '\n'
        output_string += 'box length z: ' + str(self.box_length[2]) + '\n'

        output_string += '\n# box origin indicates the x, y, and z coordinates '
        output_string += 'of the origin of the FFD bounding box. That is '
        output_string += 'center of\n'

        output_string += '# rotation of the bounding box. It corresponds to '
        output_string += 'the point coordinates with position [0][0][0].\n'

        output_string += '# See section "Parameters weights" for more '
        output_string += 'details.\n'
        output_string += '# For example, if the origin is equal to 0., 0., 0., '
        output_string += 'use the following: box origin: 0., 0., 0.\n'

        output_string += 'box origin x: ' + str(self.box_origin[0]) + '\n'
        output_string += 'box origin y: ' + str(self.box_origin[1]) + '\n'
        output_string += 'box origin z: ' + str(self.box_origin[2]) + '\n'

        output_string += '\n# rotation angle indicates the rotation angle '
        output_string += 'around the x, y, and z axis of the FFD bounding box '
        output_string += 'in degrees.\n'

        output_string += '# The rotation is done with respect to the box '
        output_string += 'origin.\n'
        output_string += '# For example, to rotate the box by 2 deg along '
        output_string += 'the z '
        output_string += 'direction, use the following: rotation angle: '
        output_string += '0., 0., 2.\n'

        output_string += 'rotation angle x: ' + str(self.rot_angle[0]) + '\n'
        output_string += 'rotation angle y: ' + str(self.rot_angle[1]) + '\n'
        output_string += 'rotation angle z: ' + str(self.rot_angle[2]) + '\n'

        output_string += '\n\n[Parameters weights]\n'
        output_string += '# This section describes the weights of the FFD '
        output_string += 'control points.\n'

        output_string += '# We adopt the following convention:\n'
        output_string += '# For example with a 2x2x2 grid of control points we '
        output_string += 'have to fill a 2x2x2 matrix of weights.\n'

        output_string += '# If a weight is equal to zero you can discard the '
        output_string += 'line since the default is zero.\n'

        output_string += '#\n'
        output_string += '# | x index | y index | z index | weight |\n'
        output_string += '#  --------------------------------------\n'
        output_string += '# |    0    |    0    |    0    |  1.0   |\n'
        output_string += '# |    0    |    1    |    1    |  0.0   | --> you '
        output_string += 'can erase this line without effects\n'
        output_string += '# |    0    |    1    |    0    | -2.1   |\n'
        output_string += '# |    0    |    0    |    1    |  3.4   |\n'

        output_string += '\n# parameter x collects the displacements along x, '
        output_string += 'normalized with the box length x.'

        output_string += '\nparameter x:'
        offset = 1
        for i in range(0, self.n_control_points[0]):
            for j in range(0, self.n_control_points[1]):
                for k in range(0, self.n_control_points[2]):
                    output_string += offset * ' ' + str(i) + '   ' + str(
                        j) + '   ' + str(k) + '   ' + str(
                            self.array_mu_x[i][j][k]) + '\n'
                    offset = 13

        output_string += '\n# parameter y collects the displacements along y, '
        output_string += 'normalized with the box length y.'

        output_string += '\nparameter y:'
        offset = 1
        for i in range(0, self.n_control_points[0]):
            for j in range(0, self.n_control_points[1]):
                for k in range(0, self.n_control_points[2]):
                    output_string += offset * ' ' + str(i) + '   ' + str(
                        j) + '   ' + str(k) + '   ' + str(
                            self.array_mu_y[i][j][k]) + '\n'
                    offset = 13

        output_string += '\n# parameter z collects the displacements along z, '
        output_string += 'normalized with the box length z.'

        output_string += '\nparameter z:'
        offset = 1
        for i in range(0, self.n_control_points[0]):
            for j in range(0, self.n_control_points[1]):
                for k in range(0, self.n_control_points[2]):
                    output_string += offset * ' ' + str(i) + '   ' + str(
                        j) + '   ' + str(k) + '   ' + str(
                            self.array_mu_z[i][j][k]) + '\n'
                    offset = 13

        with open(filename, 'w') as f:
            f.write(output_string)

    def __str__(self):
        """
        This method prints all the FFD parameters on the screen. Its purpose is
        for debugging.
        """
        string = ""
        string += 'conversion_unit = {}\n'.format(self.conversion_unit)
        string += 'n_control_points = {}\n\n'.format(self.n_control_points)
        string += 'box_length = {}\n'.format(self.box_length)
        string += 'box_origin = {}\n'.format(self.box_origin)
        string += 'rot_angle  = {}\n'.format(self.rot_angle)
        string += '\narray_mu_x =\n{}\n'.format(self.array_mu_x)
        string += '\narray_mu_y =\n{}\n'.format(self.array_mu_y)
        string += '\narray_mu_z =\n{}\n'.format(self.array_mu_z)
        string += '\nrotation_matrix = \n{}\n'.format(self.rotation_matrix)
        string += '\nposition_vertices = {}\n'.format(self.position_vertices)
        return string

    def control_points(self, deformed=True):
        """
        Method that returns the FFD control points. If the `deformed` flag is
        set to True the method returns the deformed lattice, otherwise it
        returns the original undeformed lattice.

        :param bool deformed: flag to select the original or modified FFD
            control lattice. The default is True.
        :return: the FFD control points (by row).
        :rtype: numpy.ndarray
        """
        x = np.linspace(0, self.box_length[0], self.n_control_points[0])
        y = np.linspace(0, self.box_length[1], self.n_control_points[1])
        z = np.linspace(0, self.box_length[2], self.n_control_points[2])

        y_coords, x_coords, z_coords = np.meshgrid(y, x, z)

        box_points = np.array(
            [x_coords.ravel(),
             y_coords.ravel(),
             z_coords.ravel()])

        if deformed:
            box_points += np.array([
                self.array_mu_x.ravel() * self.box_length[0],
                self.array_mu_y.ravel() * self.box_length[1],
                self.array_mu_z.ravel() * self.box_length[2]
            ])

        n_rows = box_points.shape[1]

        box_points = np.dot(self.rotation_matrix, box_points) + np.transpose(
            np.tile(self.box_origin, (n_rows, 1)))

        return box_points.T

    def reflect(self, axis=0, in_place=True):
        """
        Reflect the lattice of control points along the direction defined
        by `axis`. In particular the origin point of the lattice is preserved.
        So, for instance, the reflection along x, is made with respect to the
        face of the lattice in the yz plane that is opposite to the origin.
        Same for the other directions. Only the weights (mu) along the chosen
        axis are reflected, while the others are preserved. The symmetry plane
        can not present deformations along the chosen axis.
        After the refletcion there will be 2n-1 control points along `axis`,
        witha doubled box length.

        :param int axis: axis along which the reflection is performed.
            Default is 0. Possible values are 0, 1, or 2, corresponding
            to x, y, and z respectively.
        :param bool in_place: if True, the object attributes are modified in
            place; if False, a new object is return with the reflected lattice.
            Default is True.
        :return: a new object with the same parameters and the reflected
            lattice if `in_place` is False, otherwise NoneType.
 
        :Example:

            >>> ffd.reflect(axis=0, in_place=True) # irreversible
            >>> # or ...
            >>> refle_ffd = ffd.reflect(axis=0, in_place=False)
        """
        # check axis value
        if axis not in (0, 1, 2):
            raise ValueError(
                "The axis has to be 0, 1, or 2. Current value {}.".format(axis))

        # check that the plane of symmetry is undeformed
        if (axis == 0 and np.count_nonzero(self.array_mu_x[-1, :, :]) != 0) or (
                axis == 1 and np.count_nonzero(self.array_mu_y[:, -1, :]) != 0
        ) or (axis == 2 and np.count_nonzero(self.array_mu_z[:, :, -1]) != 0):
            raise RuntimeError(
                "If you want to reflect the FFD bounding box along axis " + \
                "{} you can not diplace the control ".format(axis) + \
                "points in the symmetry plane along that axis."
                )

        if in_place is False:
            self = copy.deepcopy(self)

        # double the control points in the given axis -1 (the symmetry plane)
        self.n_control_points[axis] = 2 * self.n_control_points[axis] - 1
        # double the box length
        self.box_length[axis] *= 2

        # we have to reflect the dispacements only along the correct axis
        reflection = np.ones(3)
        reflection[axis] = -1

        # we select all the indeces but the ones in the plane of symmetry
        indeces = [slice(None), slice(None), slice(None)]  # = [:, :, :]
        indeces[axis] = slice(1, None)  # = [1:]
        indeces = tuple(indeces)

        # we append along the given axis all the displacements reflected
        # and in the reverse order
        self.array_mu_x = np.append(self.array_mu_x,
                                    reflection[0] *
                                    np.flip(self.array_mu_x, axis)[indeces],
                                    axis=axis)
        self.array_mu_y = np.append(self.array_mu_y,
                                    reflection[1] *
                                    np.flip(self.array_mu_y, axis)[indeces],
                                    axis=axis)
        self.array_mu_z = np.append(self.array_mu_z,
                                    reflection[2] *
                                    np.flip(self.array_mu_z, axis)[indeces],
                                    axis=axis)
        if in_place is False:
            return self



    def __call__(self, src_pts):
        """
        This method performs the FFD to `src_pts` and return the deformed
        points.
        
        :param numpy.ndarray src_pts: the array of dimensions (*n_points*, *3*)
            containing the points to deform. The points have to be arranged by
            row.
        :return: the deformed points
        :rtype: numpy.ndarray (with shape = (*n_points*, *3*))
        """
        def is_inside(pts, boundaries):
            """ 
            Check is `pts` is inside the ranges provided by `boundaries`.
            """
            return np.all(np.logical_and(pts >= boundaries[0],
                                         pts <= boundaries[1]),
                          axis=1)

        # map to the reference domain
        src_reference_frame_pts = self.psi(src_pts - self.box_origin)

        # apply deformation for all the pts in the unit cube
        index_pts_inside = is_inside(src_reference_frame_pts,
                                     np.array([[0., 0., 0.], [1., 1., 1.]]))
        shifted_reference_frame_pts = self.T(
            src_reference_frame_pts[index_pts_inside])

        # map to the physical domain
        shifted_pts = self.inverse_psi(
            shifted_reference_frame_pts) + self.box_origin

        dst_pts = src_pts.copy()
        dst_pts[index_pts_inside] = shifted_pts
        return dst_pts
