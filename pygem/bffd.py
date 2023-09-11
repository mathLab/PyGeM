from pygem.cffd import CFFD
import numpy as np

"""
Utilities for performing Barycenter Free Form Deformation (BFFD).

:Theoretical Insight:
    It performs Free Form Deformation while trying to enforce the barycenter to be a certain value specified by the user.
    The constraint is enforced exactly (up to numerical errors).
    For details see the mother and the grandmother clases.

 
"""



class BFFD(CFFD):
    '''
    Class that handles the Barycenter Free Form Deformation on the mesh points.
 
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
