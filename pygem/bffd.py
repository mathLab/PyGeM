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
    :cvar callable linconstraint: it defines the F of the constraint F(x)=c.
    :cvar numpy.ndarray valconstraint: it defines the c of the constraint F(x)=c.
    :cvar list indices: it defines the indices of the control points 
        that are moved to enforce the constraint. The control index is obtained by doing:
        all_indices=np.arange(n_x*n_y*n_z*3).reshape(n_x,n_y,n_z,3).tolist().
    :cvar numpy.ndarray M: a SDP weigth matrix. It must be of size len(indices) x len(indices).

    :Example:

        >>> from pygem import BFFD
        >>> import numpy as np
        >>> bffd = BFFD()
        >>> bffd.read_parameters('tests/test_datasets/parameters_test_ffd_sphere.prm')
        >>> original_mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
        >>> b=bffd.linconstraint(original_mesh_points)
        >>> bffd.valconstraint=b
        >>> bffd.indices=np.arange(np.prod(bffd.n_control_points)*3).tolist()
        >>> bffd.M=np.eye(len(bffd.indices))
        >>> new_mesh_points = bffd(original_mesh_points)
        >>> assert np.isclose(np.linalg.norm(bffd.linconstraint(new_mesh_points)-b),np.array([0.]))
    '''
    def __init__(self, n_control_points=None):
        super().__init__(n_control_points)
        def linfun(x):
            return np.mean(x.reshape(-1,3),axis=0)

        self.linconstraint=linfun

    def __call__(self, src_pts):
        return super().__call__(src_pts)

if __name__ == "__main__":
        from pygem import BFFD
        import numpy as np
        bffd = BFFD()
        bffd.read_parameters('tests/test_datasets/parameters_test_ffd_sphere.prm')
        original_mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
        b=bffd.linconstraint(original_mesh_points)
        bffd.valconstraint=b
        bffd.indices=np.arange(np.prod(bffd.n_control_points)*3).tolist()
        bffd.M=np.eye(len(bffd.indices))
        new_mesh_points = bffd(original_mesh_points)
        assert np.isclose(np.linalg.norm(bffd.linconstraint(new_mesh_points)-b),np.array([0.]))
