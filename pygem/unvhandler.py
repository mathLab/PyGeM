"""
Derived module from filehandler.py to handle Universal (unv) files.

.. warning::
    This module will be deprecated in next releases. Follow updates on
    https://github.com/mathLab for news about file handling. 
"""
import numpy as np
import pygem.filehandler as fh
import warnings
warnings.warn("This module will be deprecated in next releases", DeprecationWarning)


class UnvHandler(fh.FileHandler):
    """
    Universal file handler class

    :cvar string infile: name of the input file to be processed.
    :cvar string outfile: name of the output file where to write in.
    :cvar list extensions: extensions of the input/output files.
        It is equal to ['.unv'].
    """

    def __init__(self):
        super(UnvHandler, self).__init__()
        self.extensions = ['.unv']

    def parse(self, filename):
        """
        Method to parse the file `filename`. It returns a matrix with
        all the coordinates. It reads only the section 2411 of the unv
        files and it assumes there are only triangles.

        :param string filename: name of the input file.

        :return: mesh_points: it is a `n_points`-by-3 matrix containing
            the coordinates of the points of the mesh.
        :rtype: numpy.ndarray
        """
        self._check_filename_type(filename)
        self._check_extension(filename)

        self.infile = filename

        index = -9
        mesh_points = []
        with open(self.infile, 'r') as input_file:
            for num, line in enumerate(input_file):
                if line.startswith('  2411'):
                    index = num
                if num == index + 2:
                    if line.startswith('    -1'):
                        break
                    else:
                        line = line.replace('D', 'E')
                        l = []
                        for t in line.split():
                            try:
                                l.append(float(t))
                            except ValueError:
                                pass
                        mesh_points.append(l)
                        index = num
            mesh_points = np.array(mesh_points)

        return mesh_points

    def write(self, mesh_points, filename):
        """
        Writes a unv file, called filename, copying all the lines from
        `self.filename` but the coordinates. mesh_points is a matrix
        that contains the new coordinates to write in the unv file.

        :param numpy.ndarray mesh_points: it is a `n_points`-by-3 matrix
            containing the coordinates of the points of the mesh
        :param string filename: name of the output file.
        """
        self._check_filename_type(filename)
        self._check_extension(filename)
        self._check_infile_instantiation()

        self.outfile = filename

        index = -9
        i = 0
        with open(self.outfile, 'w') as output_file:
            with open(self.infile, 'r') as input_file:
                for num, line in enumerate(input_file):
                    if line.startswith('  2411'):
                        index = num
                    if num == index + 2:
                        if line.startswith('    -1'):
                            index = -9
                            output_file.write(line)
                        else:
                            for j in range(0, 3):
                                output_file.write(3 * ' ' + '{:.16E}'.format(
                                    mesh_points[i][j]))
                            output_file.write('\n')
                            i += 1
                            index = num
                    else:
                        output_file.write(line)
