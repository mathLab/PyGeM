from pygem.cffd import CFFD
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
x=0.5*np.random.rand(100,2)+0.25
plt.plot(x[:,0],x[:,1],'o')
x=np.concatenate((x,0.5*np.ones((x.shape[0],1))),axis=1)
from pygem.ffd import FFD
ffd=FFD([8,8,1])
np.random.seed(0)
ffd.array_mu_x=ffd.array_mu_x+0.5*np.random.rand(*ffd.array_mu_x.shape)
ffd.array_mu_y=ffd.array_mu_x+0.5*np.random.rand(*ffd.array_mu_x.shape)
x_def=ffd(x)
x_def=x_def
plt.plot(x_def[:,0],x_def[:,1],'o')
def custom_linear_constraint(x):
    x=x[:,:-1] #removing z component
    return np.mean(np.sum(x,axis=1))
print("The custom linear function on the non deformed points is", custom_linear_constraint(x))
print("The custom linear function on the classic FFD deformed points is", custom_linear_constraint(x_def))
from pygem.cffd import CFFD
ffd=CFFD([3,3,1],custom_linear_constraint,np.array([1.]))
np.random.seed(0)
ffd.array_mu_x=ffd.array_mu_x+0.5*np.random.rand(*ffd.array_mu_x.shape)
ffd.array_mu_y=ffd.array_mu_x+0.5*np.random.rand(*ffd.array_mu_x.shape)
ffd.mask[:,:,:,-1]=False #removing z masks
ffd.weight_matrix=np.eye(np.sum(ffd.mask))#no weighting
ffd.adjust_control_points(x)
x_def=ffd(x)
plt.plot(x_def[:,0],x_def[:,1],'o')
print("The custom linear function on the constrained FFD deformed points is", custom_linear_constraint(x_def))
from pygem.bffd import BFFD
def mesh_points(num_pts = 2000):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    return np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).T
mesh = mesh_points()
ffd = BFFD([2, 2, 2],np.array([0.,0.,0.]))
ffd.array_mu_x[1, 1, 1] = 2
ffd.array_mu_z[1, 1, 1] = 0
ffd.adjust_control_points(mesh)
mesh_def=ffd(mesh)
print(np.mean(mesh_def,axis=0))
ax = plt.figure(figsize=(8,8)).add_subplot(111, projection='3d')
ax.scatter(*mesh_def.T)
ax.scatter(*ffd.control_points().T, s=50, c='red')
plt.show()
import meshio
import numpy as np
mesh=meshio.read("../../tests/test_datasets/Stanford_Bunny.stl")
points=mesh.points
faces=mesh.cells_dict["triangle"]
points=points-np.min(points)+0.1
points=points/np.max(points)
points=0.95*points
points[:,1]=points[:,1]-np.min(points[:,1])
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=faces, cmap=plt.cm.Spectral)
from pygem.vffd import VFFD
vffd=VFFD(faces,[2,2,2])
initvolume=vffd.fun(points)
vffd.fixval=np.array([initvolume])
vffd.vweight=np.array([0,1,0])
np.random.seed(0)
vffd.array_mu_x=vffd.array_mu_x+0.5*np.random.rand(2,2,2)
vffd.array_mu_y=vffd.array_mu_y+0.5*np.random.rand(2,2,2)
vffd.array_mu_z=vffd.array_mu_z+0.5*np.random.rand(2,2,2)
vffd.adjust_control_points(points)
mesh_def=vffd(points)
mesh_def=mesh_def.reshape(points.shape)
print("Percentage difference from the original mesh is ", np.linalg.norm(mesh_def-points)/np.linalg.norm(points)*100)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_trisurf(mesh_def[:,0], mesh_def[:,1], mesh_def[:,2], triangles=faces, cmap=plt.cm.Spectral)


