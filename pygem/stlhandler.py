"""
Derived module from filehandler.py to handle STereoLithography files.
"""
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from stl import mesh, Mode
import pygem.filehandler as fh


class StlHandler(fh.FileHandler):
	"""
	STereoLithography file handler class

	:cvar string infile: name of the input file to be processed.
	:cvar string outfile: name of the output file where to write in.
	:cvar string extension: extension of the input/output files. It is equal to '.stl'.
	"""
	def __init__(self):
		super(StlHandler, self).__init__()
		self.extension = '.stl'


	def parse(self, filename):
		"""
		Method to parse the `filename`. It returns a matrix with all the coordinates.

		:param string filename: name of the input file.
		
		:return: mesh_points: it is a `n_points`-by-3 matrix containing the coordinates of
			the points of the mesh
		:rtype: numpy.ndarray

		.. todo::

			- specify when it works
		"""
		self._check_filename_type(filename)
		self._check_extension(filename)

		self.infile = filename

		stl_mesh = mesh.Mesh.from_file(self.infile)
		mesh_points = np.array([stl_mesh.x.ravel(), stl_mesh.y.ravel(), stl_mesh.z.ravel()])
		mesh_points = mesh_points.T

		return mesh_points


	def write(self, mesh_points, filename, write_bin=False):
		"""
		Writes a stl file, called filename, copying all the lines from self.filename but
		the coordinates. mesh_points is a matrix that contains the new coordinates to
		write in the stl file.

		:param numpy.ndarray mesh_points: it is a `n_points`-by-3 matrix containing
			the coordinates of the points of the mesh.
		:param string filename: name of the output file.
		:param boolean write_bin: flag to write in the binary format. Default is False.
		"""
		self._check_filename_type(filename)
		self._check_extension(filename)
		self._check_infile_instantiation(self.infile)

		self.outfile = filename

		n_vertices = mesh_points.shape[0]
		# number of triplets of vertices
		n_triplets = n_vertices/3
		data = np.zeros(n_triplets, dtype=mesh.Mesh.dtype)
		stl_mesh = mesh.Mesh(data, remove_empty_areas=False)

		for i in range(0, n_triplets):
			for j in range(0, 3):
				data['vectors'][i][j] = mesh_points[3*i + j]

		if not write_bin:
			stl_mesh.save(self.outfile, mode=Mode.ASCII, update_normals=True)
		else:
			stl_mesh.save(self.outfile, update_normals=True)


	def plot(self, plot_file=None, save_fig=False):
		"""
		Method to plot an stl file. If `plot_file` is not given it plots `self.infile`.

		:param string plot_file: the stl filename you want to plot.
		:param bool save_fig: a flag to save the figure in png or not. If True the
			plot is not shown. The default value is False.
			
		:return: figure: matlplotlib structure for the figure of the chosen geometry
		:rtype: matplotlib.pyplot.figure
		"""
		if plot_file is None:
			plot_file = self.infile
		else:
			self._check_filename_type(plot_file)

		# Create a new plot
		figure = pyplot.figure()
		axes = mplot3d.Axes3D(figure)

		# Load the STL files and add the vectors to the plot
		stl_mesh = mesh.Mesh.from_file(plot_file)
		axes.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors))

		# Get the limits of the axis and center the geometry
		max_dim = np.array([np.max(stl_mesh.vectors[:,:,0]), \
						np.max(stl_mesh.vectors[:,:,1]), \
						np.max(stl_mesh.vectors[:,:,2])])
		min_dim = np.array([np.min(stl_mesh.vectors[:,:,0]), \
						np.min(stl_mesh.vectors[:,:,1]), \
						np.min(stl_mesh.vectors[:,:,2])])
		
		max_lenght = np.max(max_dim - min_dim)
		axes.set_xlim(-.6*max_lenght + (max_dim[0]+min_dim[0])/2, .6*max_lenght + (max_dim[0]+min_dim[0])/2)
		axes.set_ylim(-.6*max_lenght + (max_dim[1]+min_dim[1])/2, .6*max_lenght + (max_dim[1]+min_dim[1])/2)
		axes.set_zlim(-.6*max_lenght + (max_dim[2]+min_dim[2])/2, .6*max_lenght + (max_dim[2]+min_dim[2])/2)

		# Show the plot to the screen
		if not save_fig:
			pyplot.show()
		else:
			figure.savefig(plot_file.split('.')[0] + '.png')
			
		return figure
		
