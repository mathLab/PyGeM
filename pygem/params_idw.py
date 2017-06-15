import numpy as np
import os
try:
    import configparser as configparser
except ImportError:
    import ConfigParser as configparser

class IDWParameters(object):
	"""
	Class that handles the Inverse Distance Weighting parameters in terms of
	control points.

	:cvar int power: the power parameter. The default value is 2.
	:cvar numpy.ndarray original_control_points: it is an
		`n_control_points`-by-3 array with the coordinates of the original
		interpolation control points before the deformation. The default is the
		vertices of the unit cube.
	:cvar numpy.ndarray deformed_control_points: it is an
		`n_control_points`-by-3 array with the coordinates of the interpolation
		control points after the deformation. The default is the vertices of
		the unit cube.
	"""
	def __init__(self):
		self.power = 2
		self.original_control_points = np.array([
				[0., 0., 0.], 
				[0., 0., 1.],
				[0., 1., 0.],
				[1., 0., 0.],
				[0., 1., 1.],
				[1., 0., 1.],
				[1., 1., 0.],
				[1., 1., 1.]
		])
		self.deformed_control_points = np.array([
				[0., 0., 0.],
				[0., 0., 1.],
				[0., 1., 0.],
				[1., 0., 0.],
				[0., 1., 1.],
				[1., 0., 1.],
				[1., 1., 0.],
				[1., 1., 1.]
		])


	def read_parameters(self, filename):
		"""
		Reads in the parameters file and fill the self structure.

		:param string filename: parameters file to be read in.
		"""
		if not isinstance(filename, str):
			raise TypeError('filename must be a string')

		if not os.path.isfile(filename):
			raise IOError('filename does not exist')

		config = configparser.RawConfigParser()
		config.read(filename)

		self.power = config.getint('Inverse Distance Weighting', 'power')

		ctrl_points = config.get('Control points', 'original control points')
		self.original_control_points = np.array(
				[line.split() for line in ctrl_points.split('\n')],
				dtype=float
		)

		defo_points = config.get('Control points', 'deformed control points')
		self.deformed_control_points = np.array(
				[line.split() for line in defo_points.split('\n')],
				dtype=float
		)


	def write_parameters(self, filename):
		"""
		This method writes a parameters file (.prm) called `filename` and fills
		it with all the parameters class members.

		:param string filename: parameters file to be written out.
		"""
		if not isinstance(filename, str):
			raise TypeError("filename must be a string")

		output_string = ""
		output_string += "\n[Inverse Distance Weighting]\n"
		output_string += "# This section describes the settings of idw.\n\n"
		output_string += "# the power parameter\n"
		output_string += "power = {}\n".format(self.power)
		
		output_string += "\n\n[Control points]\n"
		output_string += "# This section describes the IDW control points.\n\n"
		output_string += "# original control points collects the coordinates\n"
		output_string += "# of the interpolation control points before the\n"
		output_string += "# deformation.\n"

		output_string += "original control points: "
		output_string += (
			'   '.join(map(str, self.original_control_points[0])) + "\n"
		)
		for points in self.original_control_points[1:]:
			output_string += 25 * ' ' + '   '.join(map(str, points)) + "\n"
		output_string += "\n"	
		output_string += "# deformed control points collects the coordinates\n"
		output_string += "# of the interpolation control points after the\n"
		output_string += "# deformation.\n"
		output_string += "deformed control points: "
		output_string += (
			'   '.join(map(str, self.original_control_points[0])) + "\n"
		)
		for points in self.deformed_control_points[1:]:
			output_string += 25 * ' ' + '   '.join(map(str, points)) + "\n"
		
		with open(filename, 'w') as f:
			f.write(output_string)
		

	def __str__(self):
		"""
		This method prints all the IDW parameters on the screen. Its purpose is
		for debugging.
		"""
		string = ''
		string += 'p = {}\n'.format(self.power)
		string += '\noriginal_control_points =\n'
		string += '{}\n'.format(self.original_control_points)
		string += '\ndeformed_control_points =\n'
		string += '{}\n'.format(self.deformed_control_points)
		return string
