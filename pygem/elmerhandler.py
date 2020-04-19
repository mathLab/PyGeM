"""
Derived module from filehandler.py to handle ElmerFEM files.
"""
import numpy as np
import pygem.filehandler as fh


class ElmerHandler(fh.FileHandler):
    """
    Elmer mesh file handler class.

    :cvar string infile: name of the input file to be processed.
    :cvar string outfile: name of the output file where to write in.
    :cvar list extensions: extensions of the input/output files. It
        is equal to ['.node'] since elmer files do not have extension.
    """

    def __init__(self):
        super(ElmerHandler, self).__init__()
        self.extensions = ['.nodes']

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

        i = 0
        n_points = 0
        with open(self.infile, 'r') as input_file:
            for line in input_file:
                n_points += 1
            mesh_points = np.zeros(shape=(n_points, 3))

        with open(self.infile, 'r') as input_file:
            
            i = 0
            for line in input_file:
                numbers = line.split() #[n1 p x y z] -> [x y z]
                del numbers[0:2] 

                j = 0
                for number in numbers:

                    mesh_points[i][j] = float(number)
                    j += 1
                i += 1

        return mesh_points

    def write(self, mesh_points, filename):
        """
        Writes a elmer file, called filename, copying all the
        lines from self.filename but the coordinates. mesh_points
        is a matrix that contains the new coordinates to write in
        the elmer file.

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
        i = 0
        with open(self.infile, 'r') as input_file, open(self.outfile,
                                                        'w') as output_file:
            for line in input_file:
                numbers = line.split() #[n1 p x y z]

                output_file.write(numbers[0] + ' ' +numbers[1] + ' ' \
                    + str(mesh_points[i][0]) + ' ' + str(mesh_points[i][1]) \
                    + ' ' + str(mesh_points[i][2]))
                i += 1

                if i != n_points:
                    output_file.write('\n')
            