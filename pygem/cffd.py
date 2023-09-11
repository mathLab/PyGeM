from pygem.ffd import FFD
import numpy as np

"""
Utilities for performing Constrained Free Form Deformation (CFFD).

:Theoretical Insight:
    It performs Free Form Deformation while trying to enforce a costraint of the form F(x)=c. 
    The constraint is enforced exactly (up to numerical errors) if and only if the function is linear.
    For details on Free Form Deformation see the mother class.

 
"""


class CFFD(FFD):
    """
    Class that handles the Constrained Free Form Deformation on the mesh points.

    :cvar callable linconstraint: it defines the F of the constraint F(x)=c.
    
    :cvar numpy.ndarray valconstraint: it defines the c of the constraint F(x)=c.
    :param list indices: it defines the indices of the control points 
        that are moved to enforce the constraint. The control index is obtained by doing:
        all_indices=np.arange(n_x*n_y*n_z*3).reshape(n_x,n_y,n_z,3).tolist().
    :cvar numpy.ndarray M: a SDP weigth matrix. It must be of size len(indices) x len(indices).

    
    :Example:

        >>> from pygem import CFFD
        >>> import numpy as np
        >>> cffd = CFFD()
        >>> cffd.read_parameters('tests/test_datasets/parameters_test_ffd_sphere.prm')
        >>> original_mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
        >>> A=np.random.rand(3,original_mesh_points.reshape(-1).shape[0])
        >>> def fun(x):
        >>>     x=x.reshape(-1)
        >>>     return A@x
        >>> b=fun(original_mesh_points)
        >>> cffd.linconstraint=fun
        >>> cffd.valconstraint=b
        >>> cffd.indices=np.arange(np.prod(cffd.n_control_points)*3).tolist()
        >>> cffd.M=np.eye(len(cffd.indices))
        >>> new_mesh_points = cffd(original_mesh_points)
        >>> assert np.isclose(np.linalg.norm(fun(new_mesh_points)-b),np.array([0.]))

    """
    def __init__(self, n_control_points=None):
        super().__init__(n_control_points)
        self.linconstraint=None 
        self.valconstraint=None
        self.indices=None
        self.M=None

    def __call__(self, src_pts):
        saved_parameters=self._save_parameters()
        A,b=self._compute_linear_map(src_pts,saved_parameters.copy())
        d=A@saved_parameters[self.indices]+b
        deltax=np.linalg.inv(self.M)@A.T@np.linalg.inv((A@np.linalg.inv(self.M)@A.T))@(self.valconstraint-d)
        saved_parameters[self.indices]=saved_parameters[self.indices]+deltax
        self._load_parameters(saved_parameters)
        return self.ffd(src_pts)

    def ffd(self,src_pts):
        '''
            Performs Classic Free Form Deformation.
    
            :param np.ndarray src_pts
            :return the deformed points
            :rtype: numpy.ndarray
        '''
        return super().__call__(src_pts)


    def _save_parameters(self):
        '''
            Saves the FFD control points in an array of shape [n_x,ny,nz,3]
            :return the FFD control points in an array of shape [n_x,ny,nz,3]
            :rtype: numpy.ndarray
        '''
        tmp=np.zeros([*self.n_control_points,3])
        tmp[:,:,:,0]=self.array_mu_x
        tmp[:,:,:,1]=self.array_mu_y
        tmp[:,:,:,2]=self.array_mu_z
        return tmp.reshape(-1)
    
    def _load_parameters(self,tmp):
        '''
            Loads the FFD control points from an array of shape [n_x,ny,nz,3]
            :param np.ndarray tmp
            :rtype: None
        '''
        tmp=tmp.reshape(*self.n_control_points,3)
        self.array_mu_x=tmp[:,:,:,0]
        self.array_mu_y=tmp[:,:,:,1]
        self.array_mu_z=tmp[:,:,:,2]


# I see that a similar function already exists in pygem.utils, but it does not work for inputs and outputs of different dimensions
    def _compute_linear_map(self,src_pts,saved_parameters):
        '''
        Computes the coefficient and the intercept of the linear map from the control points to the output
        
        :param np.ndarray src_pts
        :param np.ndarray saved_parameters
        :return a tuple containing the coefficient and the intercept
        :rtype tuple(np.ndarray,np.ndarray)

        '''
        
        
        saved_parameters_bak=saved_parameters.copy()
        n_indices=len(self.indices)
        inputs=np.zeros([n_indices+1,n_indices+1])
        outputs=np.zeros([n_indices+1,self.valconstraint.shape[0]])
        tmp=saved_parameters_bak[self.indices].reshape(1,-1)
        inputs[0]=np.hstack([tmp, np.ones((tmp.shape[0], 1))])
        saved_parameters[self.indices]=tmp
        self._load_parameters(saved_parameters)
        def_pts=super().__call__(src_pts)
        outputs[0]=self.linconstraint(def_pts)
        for i in range(1,n_indices+1):
            tmp=np.eye(n_indices)[i%n_indices]
            tmp=tmp.reshape(1,-1)
            inputs[i]=np.hstack([tmp, np.ones((tmp.shape[0], 1))])
            saved_parameters[self.indices]=tmp
            self._load_parameters(saved_parameters)
            def_pts=super().__call__(src_pts)
            outputs[i]=self.linconstraint(def_pts)
        self._load_parameters(saved_parameters_bak)
        sol=np.linalg.lstsq(inputs,outputs,rcond=None)
        A=sol[0].T[:,:-1]
        b=sol[0].T[:,-1]
        self._load_parameters(saved_parameters_bak)
        return A,b