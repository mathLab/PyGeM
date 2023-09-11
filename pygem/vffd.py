from pygem.cffd import CFFD
import numpy as np
from copy import deepcopy

"""
Utilities for performing Volume Free Form Deformation (BFFD).

:Theoretical Insight:
    It performs Free Form Deformation while trying to enforce the volume to be a certain value specified by the user.
    The constraint is enforced exactly (up to numerical errors).
    For details see the mother and the grandmother clases.

"""


class VFFD(CFFD):
    '''
    Class that handles the Volumetric Free Form Deformation on the mesh points.
 
    cvar np.ndarray vweight: specifies the weight of every step of the volume enforcement.

    :Example:

        >>> from pygem import VFFD
        >>> import numpy as np
        >>> import meshio
        >>> mesh = meshio.read('tests/test_datasets/test_sphere.stl')  
        >>> original_mesh_points=mesh.points
        >>> triangles = mesh.cells_dict["triangle"]
        >>> vffd = VFFD(triangles,[2,2,2])
        >>> vffd.read_parameters('tests/test_datasets/parameters_test_ffd_sphere.prm')
        >>> b=vffd.linconstraint(original_mesh_points)
        >>> vffd.valconstraint=np.array([b])
        >>> vffd.indices=np.arange(np.prod(vffd.n_control_points)*3).tolist()
        >>> vffd.M=np.eye(len(vffd.indices))
        >>> new_mesh_points = vffd(original_mesh_points)
        >>> assert np.isclose(np.linalg.norm(vffd.linconstraint(new_mesh_points)-b),np.array([0.]))

    '''
    def __init__(self,triangles ,n_control_points=None):
        super().__init__(n_control_points)
        self.triangles=triangles
        self.vweight=[1/3,1/3,1/3]
        def volume(x):
            x=x.reshape(-1,3)
            mesh=x[self.triangles]
            return np.sum(np.linalg.det(mesh)) 
        self.linconstraint=volume


    def __call__(self,src_pts):
        self.vweight=np.abs(self.vweight)/np.sum(np.abs(self.vweight))
        indices_bak=deepcopy(self.indices)
        self.indices=np.array(self.indices)
        indices_x=self.indices[self.indices%3==0].tolist()
        indices_y=self.indices[self.indices%3==1].tolist()
        indices_z=self.indices[self.indices%3==2].tolist()
        indexes=[indices_x,indices_y,indices_z]
        diffvolume=self.valconstraint-self.linconstraint(self.ffd(src_pts))
        for i in range(3):
            self.indices=indexes[i]
            self.M=np.eye(len(self.indices))
            self.valconstraint=self.linconstraint(self.ffd(src_pts))+self.vweight[i]*(diffvolume)
            _=super().__call__(src_pts)
        tmp=super().__call__(src_pts)
        self.indices=indices_bak
        return tmp