"""
Utilities for performing Constrained Free Form Deformation (CFFD).

:Theoretical Insight:
    
    It performs Free Form Deformation while trying to enforce a costraint of the form F(x)=c. 
    The constraint is enforced exactly (up to numerical errors) if and only if the function is linear.
    For details on Free Form Deformation see the mother class.

"""

from pygem.ffd import FFD
import numpy as np


class CFFD(FFD):
    """
    Class that handles the Constrained Free Form Deformation on the mesh points.

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

        >>> from pygem import CFFD
        >>> import numpy as np
        >>> original_mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
        >>> A=np.random.rand(3,original_mesh_points[:-4].reshape(-1).shape[0])
        >>> fun=lambda x: A@x.reshape(-1)
        >>> b=np.random.rand(3)
        >>> cffd = CFFD([2,2,2],fun,b)
        >>> cffd.read_parameters('tests/test_datasets/parameters_test_ffd_sphere.prm')
        >>> cffd.adjust_control_points(original_mesh_points[:-4])
        >>> assert np.isclose(np.linalg.norm(fun(cffd.ffd(original_mesh_points[:-4]))-b),np.array([0.]),atol=1e-06)
        >>> new_mesh_points = cffd.ffd(original_mesh_points)
    """
    def __init__(self,
                 n_control_points=None,
                 fun=None,
                 fixval=None,
                 weight_matrix=None,
                 mask=None):
        super().__init__(n_control_points)

        if mask is None:
            self.mask = np.full((*self.n_control_points, 3), True, dtype=bool)
        else:
            self.mask = mask

        if fixval is None:
            self.fixval = np.array([1.])
        else:
            self.fixval = fixval

        if fun is None:
            self.fun = lambda x: self.fixval

        else:
            self.fun = fun

        if weight_matrix is None:
            self.weight_matrix = np.eye(np.sum(self.mask.astype(int)))

    def adjust_control_points(self, src_pts):
        '''
        Adjust the FFD control points such that fun(ffd(src_pts))=fixval
            
        :param np.ndarray src_pts: the points whose deformation we want to be 
            constrained.
        :rtype: None.
        '''

        saved_parameters = self._save_parameters()
        indices = np.arange(np.prod(self.n_control_points) *
                            3)[self.mask.reshape(-1)]
        A, b = self._compute_linear_map(src_pts, saved_parameters.copy(),
                                        indices)
        d = A @ saved_parameters[indices] + b
        invM = np.linalg.inv(self.weight_matrix)
        deltax = np.linalg.multi_dot([
            invM, A.T,
            np.linalg.inv(np.linalg.multi_dot([A, invM, A.T])),
            (self.fixval - d)
        ])
        saved_parameters[indices] = saved_parameters[indices] + deltax
        self._load_parameters(saved_parameters)

    def ffd(self, src_pts):
        '''
        Performs Classic Free Form Deformation.
    
        :param np.ndarray src_pts: the points to deform.
        :return: the deformed points.
        :rtype: numpy.ndarray
        '''
        return super().__call__(src_pts)

    def _save_parameters(self):
        '''
        Saves the FFD control points in an array of shape [n_x,ny,nz,3].

        :return: the FFD control points in an array of shape [n_x,ny,nz,3].
        :rtype: numpy.ndarray
        '''
        tmp = np.zeros([*self.n_control_points, 3])
        tmp[:, :, :, 0] = self.array_mu_x
        tmp[:, :, :, 1] = self.array_mu_y
        tmp[:, :, :, 2] = self.array_mu_z
        return tmp.reshape(-1)

    def _load_parameters(self, tmp):
        '''
        Loads the FFD control points from an array of shape [n_x,ny,nz,3].

        :param np.ndarray tmp: the array of FFD control points.
        :rtype: None
        '''
        tmp = tmp.reshape(*self.n_control_points, 3)
        self.array_mu_x = tmp[:, :, :, 0]
        self.array_mu_y = tmp[:, :, :, 1]
        self.array_mu_z = tmp[:, :, :, 2]

    def read_parameters(self, filename='parameters.prm'):
        super().read_parameters(filename)
        self.mask = np.full((*self.n_control_points, 3), True, dtype=bool)
        self.weight_matrix = np.eye(np.sum(self.mask.astype(int)))


# I see that a similar function already exists in pygem.utils, but it does not work for inputs and outputs of different dimensions

    def _compute_linear_map(self, src_pts, saved_parameters, indices):
        '''
        Computes the coefficient and the intercept of the linear map from the control points to the output.
        
        :param np.ndarray src_pts: the points to deform.
        :param np.ndarray saved_parameters: the array of FFD control points.
        :return: a tuple containing the coefficient and the intercept.
        :rtype: tuple(np.ndarray,np.ndarray)
        '''
        n_indices = len(indices)
        inputs = np.zeros([n_indices + 1, n_indices + 1])
        outputs = np.zeros([n_indices + 1, self.fixval.shape[0]])
        np.random.seed(0)
        for i in range(n_indices +
                       1):  ##now we generate the interpolation points
            tmp = np.random.rand(1, n_indices)
            tmp = tmp.reshape(1, -1)
            inputs[i] = np.hstack([tmp, np.ones(
                (tmp.shape[0], 1))])  #dependent variable
            saved_parameters[indices] = tmp
            self._load_parameters(
                saved_parameters
            )  #loading the depent variable as a control point
            def_pts = super().__call__(
                src_pts)  #computing the deformation with the dependent variable
            outputs[i] = self.fun(def_pts)  #computing the independent variable
        sol = np.linalg.lstsq(inputs, outputs,
                              rcond=None)  #computation of the linear map
        A = sol[0].T[:, :-1]  #coefficient
        b = sol[0].T[:, -1]  #intercept
        return A, b
