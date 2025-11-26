#!/usr/bin/env python
# coding: utf-8

import platform
import sys
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# System info
logging.info(f"Python Version: {sys.version}")
logging.info(f"Platform: {sys.platform}")
logging.info(f"System: {platform.system()} {platform.release()}")

# Import PyGeM
try:
    import pygem
except ImportError:
    raise ImportError(
        "PyGeM not found. Please install it before running this tutorial.\n"
        "For example, run: pip install -e '.[tut]' in your environment."
    )

logging.info(f"PyGeM version: {pygem.__version__}")

np.random.seed(42)

import matplotlib.pyplot as plt

from pygem.cffd import CFFD

x = 0.5 * np.random.rand(100, 3) + 0.25
ax = plt.axes(projection="3d")
ax.plot3D(x[:, 0], x[:, 1], x[:, 2], "o")

from pygem.ffd import FFD

ffd = FFD([8, 8, 1])
np.random.seed(0)
ffd.array_mu_x = ffd.array_mu_x + 0.5 * np.random.rand(*ffd.array_mu_x.shape)
ffd.array_mu_y = ffd.array_mu_x + 0.5 * np.random.rand(*ffd.array_mu_x.shape)
x_def = ffd(x)
x_def = x_def
ax = plt.axes(projection="3d")
ax.plot3D(x_def[:, 0], x_def[:, 1], x_def[:, 2], "o")


def custom_linear_constraint(x):
    x = x[:, :-1]  # removing z component
    return np.mean(np.sum(x, axis=1))


print(
    "The custom linear function on the non deformed points is",
    custom_linear_constraint(x),
)
print(
    "The custom linear function on the classic FFD deformed points is",
    custom_linear_constraint(x_def),
)


ffd = CFFD(np.array([1.0]), custom_linear_constraint, [3, 3, 1])
np.random.seed(0)
ffd.array_mu_x = ffd.array_mu_x + 0.5 * np.random.rand(*ffd.array_mu_x.shape)
ffd.array_mu_y = ffd.array_mu_x + 0.5 * np.random.rand(*ffd.array_mu_x.shape)
ffd.adjust_control_points(x)
x_def = ffd(x)
ax = plt.axes(projection="3d")
ax.plot3D(x_def[:, 0], x_def[:, 1], x_def[:, 2], "o")
print(
    "The custom linear function on the constrained FFD deformed points is",
    custom_linear_constraint(x_def),
)

from pygem.bffd import BFFD


def mesh_points(num_pts=2000):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    return np.array(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    ).T


mesh = mesh_points()
ffd = BFFD(np.array([0.0, 0.0, 0.0]), [2, 2, 2])
ffd.array_mu_x[1, 1, 1] = 2
ffd.array_mu_z[1, 1, 1] = 0
ffd.adjust_control_points(mesh)
mesh_def = ffd(mesh)
print(np.mean(mesh_def, axis=0))
ax = plt.figure(figsize=(8, 8)).add_subplot(111, projection="3d")
ax.scatter(*mesh_def.T)
ax.scatter(*ffd.control_points().T, s=50, c="red")
plt.show()

try:
    import meshio
except ImportError:
    raise ImportError(
        "meshio not found. Please install it before running this tutorial.\n"
        "For example, run: pip install meshio"
    )


mesh = meshio.read("tests/test_datasets/Stanford_Bunny.stl")
points = mesh.points
faces = mesh.cells_dict["triangle"]
points = points - np.min(points) + 0.1
points = points / np.max(points)
points = 0.95 * points
points[:, 1] = points[:, 1] - np.min(points[:, 1])
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot_trisurf(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    triangles=faces,
    cmap=plt.cm.Spectral,
)

from pygem.vffd import VFFD, _volume

initvolume = _volume(points, faces)
vffd = VFFD(faces, np.array(initvolume), [2, 2, 2])
np.random.seed(0)
vffd.array_mu_x = vffd.array_mu_x + 0.5 * np.random.rand(2, 2, 2)
vffd.array_mu_y = vffd.array_mu_y + 0.5 * np.random.rand(2, 2, 2)
vffd.array_mu_z = vffd.array_mu_z + 0.5 * np.random.rand(2, 2, 2)
vffd.adjust_control_points(points)
mesh_def = vffd(points)
mesh_def = mesh_def.reshape(points.shape)
print(
    "Percentage difference from the original mesh is ",
    np.linalg.norm(mesh_def - points) / np.linalg.norm(points) * 100,
)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot_trisurf(
    mesh_def[:, 0],
    mesh_def[:, 1],
    mesh_def[:, 2],
    triangles=faces,
    cmap=plt.cm.Spectral,
)
