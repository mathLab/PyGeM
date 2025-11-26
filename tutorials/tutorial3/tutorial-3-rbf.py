#!/usr/bin/env python
# coding: utf-8

# # PyGeM
# ## Tutorial 3: Radial Basis Functions interpolation technique on a cube

# In this tutorial we will show how to use the Radial Basis Functions interpolation technique to deform a cube.
#
# First of all we import the required **PyGeM** class, we import numpy and matplotlib

import platform

# In[1]:
import sys

print(f"Python Version: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"System: {platform.system()} {platform.release()}")

try:
    import pygem

    print(f"PyGeM version: {pygem.__version__}")
except ImportError:
    print("PyGeM not found. Installing...")
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", ".[tut]"]
    )
    import pygem

    print(f"PyGeM version: {pygem.__version__}")

import numpy as np

np.random.seed(42)
import matplotlib.pyplot as plt

from pygem import RBF

# Using RBF, we can control the deformation by arranging some control points around the object to deform, then moving these latter to induce the morphing. Within **PyGeM**, the setting of such parameters can be done by parsing an input text file or manually touching the `RBF` attributes.
#
# Let's try togheter by using an input file: the first step is the creation of the new object. After this, we can use the `read_parameters` to set the parameters.

# In[2]:
rbf = RBF()
rbf.read_parameters(filename="../tests/test_datasets/parameters_rbf_cube.prm")


# The following is the parameters file for this particular case. The Radial Basis Functions section describes the basis functions to use. Here we use Gaussian splines with the distance parameter equal to 0.5 (see the documentation of the [RBF](http://mathlab.github.io/PyGeM/rbf.html) class for more details). As control points we consider the 8 vertices of the cube (the first one is not exactly the vertex), and we move 3 of them. In the Control points section there are all the coordinates of the control points.

# In[3]:
import subprocess

subprocess.run(["cat", "../tests/test_datasets/parameters_rbf_cube.prm"])


# Here we create a $10 \times10 \times 10$ lattice to mimic a cube.

# In[4]:
nx, ny, nz = (10, 10, 10)
mesh = np.zeros((nx * ny * nz, 3))

xv = np.linspace(0, 1, nx)
yv = np.linspace(0, 1, ny)
zv = np.linspace(0, 1, nz)
z, y, x = np.meshgrid(zv, yv, xv)

mesh = np.array([x.ravel(), y.ravel(), z.ravel()])
mesh = mesh.T


# Now we plot the points to see what we are doing.

# In[5]:
fig = plt.figure(1)
ax = fig.add_subplot(111, projection="3d")
ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], c="blue", marker="o")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
plt.show()


# We can also plot the original control points and the deformed ones.

# In[6]:
rbf.plot_points()


# Finally we perform the RBF interpolation using the `RBF` class.

# In[7]:
new_mesh = rbf(mesh)


# We can plot the new points in order to see the deformation. Try different basis functions and radius to better fit your specific problem.

# In[8]:
fig = plt.figure(2)
ax = fig.add_subplot(111, projection="3d")
ax.scatter(new_mesh[:, 0], new_mesh[:, 1], new_mesh[:, 2], c="red", marker="o")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
plt.show()
