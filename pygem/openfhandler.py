"""
Derived module from filehandler.py to handle OpenFOAM files.

.. warning::
    This module will be deprecated in next releases. Follow updates on
    https://github.com/mathLab for news about file handling. 
"""
import numpy as np
import pygem.filehandler as fh
import warnings
warnings.warn("This module will be deprecated in next releases", DeprecationWarning)


class OpenFoamHandler(fh.FileHandler):
    """
    OpenFOAM mesh file handler class.

    :cvar string infile: name of the input file to be processed.
    :cvar string outfile: name of the output file where to write in.
    :cvar list extensions: extensions of the input/output files. It
        is equal to [''] since openFOAM files do not have extension.
    """

    def __init__(self):
        super(OpenFoamHandler, self).__init__()
        self.extensions = ['']

    def parse(self, filename):
        """
        Method to parse the `filename`. It returns a matrix with all
        the coordinates.

        :param string filename: name of the input file.
        
        :return: mesh_points: it is a `n_points`-by-3 matrix containing
            the coordinates of the points of the mesh
        :rtype: numpy.ndarray

        .. todo::

            - specify when it works
        """
        self._check_filename_type(filename)
        self._check_extension(filename)

        self.infile = filename

        nrow = 0
        i = 0
        with open(self.infile, 'r') as input_file:
            for line in input_file:
                nrow += 1
                if nrow == 19:
                    n_points = int(line)
                    mesh_points = np.zeros(shape=(n_points, 3))
                if 20 < nrow < 21 + n_points:
                    line = line[line.index("(") + 1:line.rindex(")")]
                    j = 0
                    for number in line.split():
                        mesh_points[i][j] = float(number)
                        j += 1
                    i += 1

        return mesh_points

    def write(self, mesh_points, filename):
        """
        Writes a openFOAM file, called filename, copying all the
        lines from self.filename but the coordinates. mesh_points
        is a matrix that contains the new coordinates to write in
        the openFOAM file.

        :param numpy.ndarray mesh_points: it is a `n_points`-by-3
            matrix containing the coordinates of the points of the mesh.
        :param string filename: name of the output file.

        .. todo:: DOCS
        """
        self._check_filename_type(filename)
        self._check_extension(filename)
        self._check_infile_instantiation()

        self.outfile = filename

        n_points = mesh_points.shape[0]
        nrow = 0
        i = 0
        with open(self.infile, 'r') as input_file, open(self.outfile,
                                                        'w') as output_file:
            for line in input_file:
                nrow += 1
                if 20 < nrow < 21 + n_points:
                    output_file.write('(' + str(mesh_points[i][0]) + ' ' + str(
                        mesh_points[i][1]) + ' ' + str(mesh_points[i][2]) + ')')
                    output_file.write('\n')
                    i += 1
                else:
                    output_file.write(line)
