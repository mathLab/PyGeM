import pygem as pg
from pygem.mdpahandler import MdpaHandler
import vedo
import meshio

ffd = pg.FFD()
ffd.read_parameters('../tests/test_datasets/parameters_test_ffd_pipe_unv_C0.prm')

mdpa_handler = MdpaHandler()
mesh_points = mdpa_handler.parse('../tests/test_datasets/test_pipe.mdpa')

UndeformedMDPA = meshio.read('../tests/test_datasets/test_pipe.mdpa')
meshio.write('test_pipe_undeformed_mdpa.vtk', UndeformedMDPA)
UndeformedVTK = vedo.load('test_pipe_undeformed_mdpa.vtk')
vedo.settings.embedWindow(backend='k3d', verbose=True)
vedo.show(UndeformedVTK, viewup="z", resetcam=True)

new_mesh_points = ffd(mesh_points)

DeformedMDPA = meshio.read('test_pipe_mod_C0.mdpa')
meshio.write('test_pipe_deformed_mdpa.vtk', DeformedMDPA)
DeformedVTK = vedo.load('test_pipe_deformed_mdpa.vtk')
vedo.settings.embedWindow(backend='k3d', verbose=True)
vedo.show(DeformedVTK, viewup="z", resetcam=True)

ffd = pg.FFD()
ffd.read_parameters('../tests/test_datasets/parameters_test_ffd_pipe_unv_C1.prm')
new_mesh_points = ffd(mesh_points)
mdpa_handler.write(new_mesh_points, 'test_pipe_mod_C1.mdpa')

mesh = meshio.read('test_pipe_mod_C1.mdpa')
meshio.write('test_pipe_mod_C1_mdpa.vtk', mesh)

DeformedMDPA = meshio.read('test_pipe_mod_C1.mdpa')
meshio.write('test_pipe_deformed_mdpa.vtk', DeformedMDPA)
DeformedVTK = vedo.load('test_pipe_deformed_mdpa.vtk')
vedo.settings.embedWindow(backend='k3d', verbose=True)
vedo.show(DeformedVTK, viewup="z", resetcam=True)
