"""
Module for a custom deformation.
"""
import numpy as np

from pygem import Deformation

class CustomDeformation(Deformation):
    """
    Class to perform a custom deformation to the mesh points.

    :param callable func: the function definying the deformation of the input
        points. This function should take as input: *i*) a 2D array of shape
        (*n_points*, *3*) in which the points are arranged by row, or *ii*) an
        iterable object with 3 components. In this last case, computation of
        deformation is not vectorized and the overall cost may become heavy.

    :Example:

        >>> from pygem import CustomDeformation
        >>> import numpy as np
        >>> def move(x):
        >>>     return x + x**2
        >>> deform = CustomDeformation(move)
        >>> original_mesh_points = np.load(
        >>>         'tests/test_datasets/meshpoints_sphere_orig.npy')
        >>> new_mesh_points = deform(original_mesh_points)
        >>> # Deformation with non-vectorized function
        >>> def move(x):
        >>>     x0, x1, x2 = x
        >>>     return [x0**2, x1, x2]
        >>> deform = CustomDeformation(move)
        >>> new_mesh_points = deform(original_mesh_points)
    """

    def __init__(self, func):
        self.__func = func

    def __call__(self, src_pts):
        """
        This method performs the deformation on the input points.

        :param numpy.ndarray src_pts: the array of dimensions (*n_points*, *3*)
            containing the points to deform. The points have to be arranged by
            row.
        :return: the deformed points
        :rtype: numpy.ndarray (with shape = (*n_points*, *3*))
        """
        try:
            return self.__func(src_pts)
        except:
            return np.array([self.__func(pt) for pt in src_pts])
