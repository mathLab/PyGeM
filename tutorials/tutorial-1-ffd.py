import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt

from pygem import FFD


def mesh_points(num_pts=2000):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    return np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).T


mesh = mesh_points()
plt.figure(figsize=(8, 8)).add_subplot(111, projection='3d').scatter(*mesh.T)
plt.show()

ffd = FFD([2, 2, 2])
print(ffd)

print('Movements of point[{}, {}, {}] along x: {}'.format(1, 1, 1, ffd.array_mu_x[1, 1, 1]))
print('Movements of point[{}, {}, {}] along z: {}'.format(1, 1, 1, ffd.array_mu_z[1, 1, 1]))

ffd.array_mu_x[1, 1, 1] = 2
ffd.array_mu_z[1, 1, 1] = 0.8
print()
print('Movements of point[{}, {}, {}] along x: {}'.format(1, 1, 1, ffd.array_mu_x[1, 1, 1]))
print('Movements of point[{}, {}, {}] along z: {}'.format(1, 1, 1, ffd.array_mu_z[1, 1, 1]))

new_mesh = ffd(mesh)
print(type(new_mesh), new_mesh.shape)


ax = plt.figure(figsize=(8, 8)).add_subplot(111, projection='3d')
ax.scatter(*new_mesh.T)
ax.scatter(*ffd.control_points().T, s=50, c='red')
plt.show()
