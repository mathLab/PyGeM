"""
Utilities for performing Barycenter Free Form Deformation (BFFD).

:Theoretical Insight:
    It performs Free Form Deformation while trying to enforce the barycenter to be a certain value specified by the user.
    The constraint is enforced exactly (up to numerical errors).
    For details see the mother and the grandmother classes.

"""

from pygem.cffd import CFFD
import numpy as np


class BFFD(CFFD):
    '''
    Class that handles the Barycenter Free Form Deformation on the mesh points.
 
    :param list n_control_points: number of control points in the x, y, and z
        direction. Default is [2, 2, 2].
        
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
    :cvar callable fun: it defines the F of the constraint F(x)=c. Default is the constant 1 function.
    :cvar numpy.ndarray fixval: it defines the c of the constraint F(x)=c. Default is 1.
    :cvar numpy.ndarray mask: a boolean tensor that tells to the class 
        which control points can be moved, and in what direction, to enforce the constraint. 
        The tensor has shape (n_x,n_y,n_z,3), where the last dimension indicates movement
        on x,y,z respectively. Default is all true.
    :cvar numpy.ndarray weight_matrix: a symmetric positive definite weigth matrix. 
        It must be of row and column size the number of trues in the mask.
        It weights the movemement of the control points which have a true flag in the mask.
        Default is identity.

    :Example:

        >>> from pygem import BFFD
        >>> import numpy as np
        >>> b=np.random.rand(3)
        >>> bffd = BFFD([2,2,2],b)
        >>> bffd.read_parameters('tests/test_datasets/parameters_test_ffd_sphere.prm')
        >>> original_mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
        >>> bffd.adjust_control_points(original_mesh_points[:-4])
        >>> assert np.isclose(np.linalg.norm(bffd.fun(bffd.ffd(original_mesh_points[:-4]))-b),np.array([0.]))
        >>> new_mesh_points = bffd.ffd(original_mesh_points)
    '''
    def __init__(self,
                 n_control_points=None,
                 fixval=None,
                 weight_matrix=None,
                 mask=None):
        super().__init__(n_control_points, None, fixval, weight_matrix, mask)

        def linfun(x):
            return np.mean(x.reshape(-1, 3), axis=0)

        self.fun = linfun
