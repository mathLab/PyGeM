# Tutorials

In this folder we collect several useful tutorials to understand the
principles and the potential of **PyGeM**. Please read the following table
for details about the tutorials.

| Name | Description & Links | PyGeM used classes | Input geometries |
| ---- | ------------------- | ------------------ | ---------------- |
| Tutorial 1 | Free-form deformation to morph a spherical mesh. Links: [.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial1/tutorial-1-ffd.ipynb), [.py](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial1/tutorial-1-ffd.py), [.html](http://mathlab.github.io/PyGeM/tutorial-1-ffd.html) | `pygem.FFD` | `numpy.ndarray` |
| Tutorial 2 | Free-form deformation to morph a cylinder. Links: [.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial2/tutorial-2-iges.ipynb), [.py](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial2/tutorial-2-iges.py), [.html](http://mathlab.github.io/PyGeM/tutorial-2-iges.html) | `pygem.cad.FFD` | IGES file |
| Tutorial 3 | Radial basis function to morph a cubic mesh. Links: [.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial3/tutorial-3-rbf.ipynb), [.py](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial3/tutorial-3-rbf.py), [.html](http://mathlab.github.io/PyGeM/tutorial-3-rbf.html) | `pygem.RBF` | `numpy.ndarray` |
| Tutorial 4 | Inverse distance weighting to deform a cubic mesh. Links: [.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial4/tutorial-4-idw.ipynb), [.py](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial4/tutorial-4-idw.py), [.html](http://mathlab.github.io/PyGeM/tutorial-4-idw.html) | `pygem.IDW` | `numpy.ndarray` |
| Tutorial 5 | Free-form deformation to deform an object in a file. Links: [.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial5/tutorial-5-file.ipynb), [.py](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial5/tutorial-5-file.py), [.html](http://mathlab.github.io/PyGeM/tutorial-5-file.html) | `pygem.FFD` | `.vtp` file, `.stl` file |
| Tutorial 6 | Interpolation of an OpenFOAM mesh after a deformation. Links: [.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial6/tutorial-6-ffd-rbf.ipynb), [.py](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial6/tutorial-6-ffd-rbf.py), [.html](http://mathlab.github.io/PyGeM/tutorial-6-ffd-rbf.html) | `pygem.FFD/RBF` | OpenFOAM |
| Tutorial 7 | Constrained free-form deformation. Links: [.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial7/tutorial-7-cffd.ipynb), [.py](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial7/tutorial-7-cffd.py), [.html](http://mathlab.github.io/PyGeM/tutorial-7-cffd.html) | `pygem.CFFD/BFFD/VFFD` | `.stl` file |

## Old Version Tutorials

Below are the tutorials from previous releases. We will update them for the
latest version of **PyGeM**. Meanwhile, they may still be useful for users.

### Tutorial 3: Deforming a UNV File

[.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial-3-unv.ipynb)
This tutorial shows how to deform a UNV file with prescribed continuity using
Free Form Deformation.

### Tutorial 6: Deforming LS-Dyna K File

[.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial-6-k.ipynb),
[.py](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial-6-k.py)
This tutorial demonstrates how to deform an LS-Dyna K file with prescribed
continuity using Free Form Deformation.

### Tutorial 7: Deforming Kratos MDPA File

[.ipynb](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial-7-mdpa.ipynb),
[.py](https://github.com/mathLab/PyGeM/blob/master/tutorials/tutorial-7-mdpa.py)
This tutorial demonstrates how to deform a Kratos Multiphysics MDPA file with
prescribed continuity using Free Form Deformation.

### More to Come

We plan to add more tutorials. If you want to contribute a notebook for a
feature not yet covered, we will be happy to help and support you in editing!

