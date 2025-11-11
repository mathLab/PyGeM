from pygem.cffd import CFFD
import numpy as np


class VFFD(CFFD):
    """
    Class that handles the Volumetric Free Form Deformation on the mesh points.

    :param list n_control_points: number of control points in the x, y, and z
        direction. Default is [2, 2, 2].
    :param string mode: it can be ``affine`` or ``triaffine``. The first option is for the F that are affine in all the coordinates of the points.
        The second one is for functions that are F in the coordinates of the points. The first option implies the second, but is optimal for that class of functions.
    :cvar numpy.ndarray box_length: dimension of the FFD bounding box, in the
        x, y and z direction (local coordinate system).
    :cvar numpy.ndarray box_origin: the x, y and z coordinates of the origin of
        the FFD bounding box.
    :cvar numpy.ndarray n_control_points: the number of control points in the
        x, y, and z direction.
    :cvar numpy.ndarray array_mu_x: collects the displacements (weights) along
        x, normalized with the box length x.
    :cvar numpy.ndarray array_mu_y: collects the displacements (weights) along
        y, normalized with the box length y.
    :cvar numpy.ndarray array_mu_z: collects the displacements (weights) along
        z, normalized with the box length z.
    :cvar callable fun: it defines the F of the constraint F(x)=c. Default is the constant 1 function.
    :cvar numpy.ndarray fixval: it defines the c of the constraint F(x)=c. Default is 1.
    :cvar numpy.ndarray ffd_mask: a boolean tensor that tells to the class
        which control points can be moved, and in what direction, to enforce the constraint.
        The tensor has shape (n_x,n_y,n_z,3), where the last dimension indicates movement
        on x,y,z respectively. Default is all true.
    :cvar numpy.ndarray fun_mask: a boolean tensor that tells to the class
        on which axis which constraint depends on. The tensor has shape (n_cons,3), where the last dimension indicates dependency on
        on x,y,z respectively. Default is all true. It used only in the triaffine mode.

    :Example:

        >>> from pygem import VFFD
        >>> import numpy as np
        >>> import meshio
        >>> mesh = meshio.read('tests/test_datasets/test_sphere_cffd.stl')
        >>> original_mesh_points = mesh.points
        >>> triangles = mesh.cells_dict["triangle"]
        >>> b = np.random.rand(1)
        >>> vffd = VFFD(triangles, b, [2, 2, 2])
        >>> vffd.read_parameters('tests/test_datasets/parameters_test_cffd.prm')
        >>> vffd.adjust_control_points(original_mesh_points)
        >>> new_mesh_points = vffd(original_mesh_points)
        >>> assert np.isclose(np.linalg.norm(vffd.fun(new_mesh_points) - b), np.array([0.]), atol=1e-07)

    """

    def __init__(self, triangles, fixval, n_control_points=None, ffd_mask=None):
        super().__init__(fixval, None, n_control_points, ffd_mask, None)

        self.triangles = triangles

        def volume_inn(x):
            return _volume(x, self.triangles)

        self.fun = volume_inn
        self.fixval = fixval
        self.fun_mask = np.array([[True, True, True]])


def _volume(x, triangles):
    x = x.reshape(-1, 3)
    mesh = x[triangles]
    return np.array([np.sum(np.linalg.det(mesh))])
