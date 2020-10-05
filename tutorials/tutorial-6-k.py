import pygem as pg
from pygem.khandler import KHandler
import vedo
import meshio

ffd = pg.FFD()
ffd.read_parameters('../tests/test_datasets/parameters_test_ffd_pipe_unv_C0.prm')

%cat ../tests/test_datasets/parameters_test_ffd_pipe_unv_C0.prm

handler = KHandler()
mesh_points = handler.parse('../tests/test_datasets/test_pipe.k')

new_mesh_points = ffd(mesh_points)

%cat ../tests/test_datasets/parameters_test_ffd_pipe_unv_C1.prm

ffd = pg.FFD()
ffd.read_parameters('../tests/test_datasets/parameters_test_ffd_pipe_unv_C1.prm')

new_mesh_points = ffd(mesh_points)

handler.write(new_mesh_points, 'test_pipe_mod_C1.k')