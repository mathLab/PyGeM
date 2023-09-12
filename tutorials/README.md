# Tutorials

In this folder we collect several useful tutorials in order to understand the principles and the potential of **PyGeM**. Please read the following table for details about the tutorials.


| Name  | Description   | PyGeM used classes | input geometries  |
|-------|---------------|--------------------|-------------------|
| Tutorial1&#160;[[.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial1/tutorial-1-ffd.ipynb),&#160;[.py](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial1/tutorial-1-ffd.py),&#160;[.html](http://mathlab.github.io/PyGeM/tutorial-1-ffd.html)]| free-form deformation to morph a spherical mesh | `pygem.FFD`  | `numpy.ndarray`  |
| Tutorial2&#160;[[.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial2/tutorial-2-iges.ipynb),&#160;[.py](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial2/tutorial-2-iges.py),&#160;[.html](http://mathlab.github.io/PyGeM/tutorial-2-iges.html)] | free-form deformation to morph a cylinder | `pygem.cad.FFD`  | IGES file |
| Tutorial3&#160;[[.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial3/tutorial-3-rbf.ipynb),&#160;[.py](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial3/tutorial-3-rbf.py),&#160;[.html](http://mathlab.github.io/PyGeM/tutorial-3-rbf.html)] | radial basis function to morph a cubic mesh | `pygem.RBF`  | `numpy.ndarray` |
| Tutorial4&#160;[[.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial4/tutorial-4-idw.ipynb),&#160;[.py](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial4/tutorial-4-idw.py),&#160;[.html](http://mathlab.github.io/PyGeM/tutorial-4-idw.html)] | inverse distance weighting to deform a cubic mesh | `pygem.IDW`  | `numpy.ndarray` |
| Tutorial5&#160;[[.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial5/tutorial-5-file.ipynb),&#160;[.py](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial5/tutorial-5-file.py),&#160;[.html](http://mathlab.github.io/PyGeM/tutorial-5-file.html)] | free-form deformation to deform an object contained to file | `pygem.FFD`  | `.vtp` file, `.stl` file |
| Tutorial6&#160;[[.ipynb](https://github.com/fAndreuzzi/PyGeM/blob/master/tutorials/tutorial6/tutorial-6-ffd-rbf.ipynb),&#160;[.py](https://github.com/fAndreuzzi/PyGeM/blob/master/tutorials/tutorial6/tutorial-6-ffd-rbf.py),&#160;[.html](http://mathlab.github.io/PyGeM/tutorial-6-ffd-rbf.html)] | interpolation of an OpenFOAM mesh after a deformation | `pygem.FFD/RBF`  | OpenFOAM |
| Tutorial8&#160;[[.ipynb](https://github.com/gpadula/PyGeM/blob/master/tutorials/tutorial6/tutorial-8-cffd.ipynb),&#160;[.py](https://github.com/gpadula/PyGeM/blob/master/tutorials/tutorial8/tutorial-8-cffd.py),&#160;[.html](http://mathlab.github.io/PyGeM/tutorial-8-cffd.html)] | constrained free form deformation | `pygem.CFFD/BFFD/VFFD`  | `.stl` file |


# Old version
Below all the tutorials for the previous release. We will convert them in order to make compatible with the latest version of **PyGeM**; meanwhile we still publish these _old_ examples since they may help the users in some application.

#### [Tutorial 3](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial-3-unv.ipynb)
Here it is possible to understand how to deform a unv file with a prescribed continuity using the free form deformation.

#### [Tutorial 6](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial-6-k.ipynb) [[.py]](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial-6-k.py)
This tutorial shows how to deform a LS-Dyna k file with a prescribed continuity using the free-form deformation.

#### [Tutorial 7](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial-7-mdpa.ipynb) [[.py]](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial-7-mdpa.py)
This tutorial shows how to deform a Kratos Multiphysics mdpa file with a prescribed continuity using the free-form deformation.

#### More to come...
We plan to add more tutorials but the time is often against us. If you want to contribute with a notebook on a feature not covered yet we will be very happy and give you support on editing!
