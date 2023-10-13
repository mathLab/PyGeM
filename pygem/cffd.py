"""
Utilities for performing Constrained Free Form Deformation (CFFD).

:Theoretical Insight:
    
    It performs Free Form Deformation while trying to enforce a costraint of the form F(x)=c. 
    The constraint is enforced exactly (up to numerical errors) if and only if the function is linear.
    For details on Free Form Deformation see the mother class.

"""

from pygem.ffd import FFD
import numpy as np
from scipy.optimize import LinearConstraint, differential_evolution


class CFFD(FFD):
    """
    Class that handles the Constrained Free Form Deformation on the mesh points.

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
        >>> from pygem import CFFD
        >>> import numpy as np
        >>> original_mesh_points = np.random.rand(100, 3)
        >>> A = np.random.rand(3, original_mesh_points[:-4].reshape(-1).shape[0])
        >>> fun = lambda x: A @ x.reshape(-1)
        >>> b = np.random.rand(3)
        >>> cffd = CFFD(b, fun, [2, 2, 2])
        >>> cffd.read_parameters('tests/test_datasets/parameters_test_cffd.prm')
        >>> cffd.adjust_control_points(original_mesh_points[:-4])
        >>> assert np.isclose(np.linalg.norm(fun(cffd.ffd(original_mesh_points[:-4])) - b), np.array([0.]), atol = 1e-06)
        >>> new_mesh_points = cffd.ffd(original_mesh_points)

    """
    def __init__(self,
                 fixval,
                 fun,
                 n_control_points=None,
                 ffd_mask=None,
                 fun_mask=None):
        super().__init__(n_control_points)

        if ffd_mask is None:
            self.ffd_mask = np.full((*self.n_control_points, 3),
                                    True,
                                    dtype=bool)
        else:
            self.ffd_mask = ffd_mask

        self.num_cons = len(fixval)
        self.fun = fun
        self.fixval = fixval
        if fun_mask is None:
            self.fun_mask = np.full((self.num_cons, 3), True, dtype=bool)
        else:
            self.fun_mask = fun_mask

    def adjust_control_points(self, src_pts):
        '''
        Adjust the FFD control points such that fun(ffd(src_pts))=fixval
            
        :param np.ndarray src_pts: the points whose deformation we want to be 
            constrained.
        :rtype: None.
        '''
        hyper_param = self.fun_mask.copy().astype(float)
        hyper_param = hyper_param / np.sum(hyper_param, axis=1)
        mask_bak = self.ffd_mask.copy()
        fixval_bak = self.fixval.copy()
        diffvolume = self.fixval - self.fun(self.ffd(src_pts))
        for i in range(3):
            self.ffd_mask = np.full((*self.n_control_points, 3),
                                    False,
                                    dtype=bool)
            self.ffd_mask[:, :, :, i] = mask_bak[:, :, :, i].copy()
            self.fixval = self.fun(
                self.ffd(src_pts)) + hyper_param[:, i] * (diffvolume)
            saved_parameters = self._save_parameters()
            indices = np.arange(np.prod(self.n_control_points) *
                                3)[self.ffd_mask.reshape(-1)]
            A, b = self._compute_linear_map(src_pts, saved_parameters.copy(),
                                            indices)
            A = A[self.fun_mask[:, i].reshape(-1), :]
            b = b[self.fun_mask[:, i].reshape(-1)]
            d = A @ saved_parameters[indices] + b
            fixval = self.fixval[self.fun_mask[:, i].reshape(-1)]
            deltax = np.linalg.multi_dot([
                A.T,
                np.linalg.inv(np.linalg.multi_dot([A, A.T])), (fixval - d)
            ])
            saved_parameters[indices] = saved_parameters[indices] + deltax
            self._load_parameters(saved_parameters)
        self.ffd_mask = mask_bak.copy()
        self.fixval = fixval_bak.copy()

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


class BFFD(CFFD):
    '''
    Class that handles the Barycenter Free Form Deformation on the mesh points.
 
    :param list n_control_points: number of control points in the x, y, and z
        direction. Default is [2, 2, 2].
        
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
    :cvar numpy.ndarray mask: a boolean tensor that tells to the class 
        which control points can be moved, and in what direction, to enforce the constraint. 
        The tensor has shape (n_x,n_y,n_z,3), where the last dimension indicates movement
        on x,y,z respectively. Default is all true.

    :Example:

        >>> from pygem import BFFD
        >>> b = np.random.rand(3)
        >>> bffd = BFFD(b, [2, 2, 2])
        >>> bffd.read_parameters('tests/test_datasets/parameters_test_cffd')
        >>> original_mesh_points = np.random.rand(100, 3)
        >>> bffd.adjust_control_points(original_mesh_points[:-4])
        >>> assert np.isclose(np.linalg.norm(bffd.fun(bffd.ffd(original_mesh_points[:-4])) - b), np.array([0.]))
        new_mesh_points = bffd.ffd(original_mesh_points)
    '''

    def __init__(self, fixval=None, n_control_points=None, ffd_mask=None):
        super().__init__(fixval, None, n_control_points, ffd_mask, None)

        def linfun(x):
            return np.mean(x.reshape(-1, 3), axis=0)

        self.fun = linfun
        self.fixval = fixval
        self.fun_mask = np.array([[True, False, False], [False, True, False],
                                  [False, False, True]])


class VFFD(CFFD):
    '''
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

    '''
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


