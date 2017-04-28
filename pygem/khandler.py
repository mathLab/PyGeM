"""
Derived module from filehandler.py to handle LS-DYNA keyword (.k) files.
"""
import numpy as np
import pygem.filehandler as fh


class KHandler(fh.FileHandler):
    """
    LS-Dyna keyword file handler class

    :cvar string infile: name of the input file to be processed.
    :cvar string outfile: name of the output file where to write in.
    :cvar list extensions: extensions of the input/output files. It is equal
            to '.k'.
    """
    def __init__(self):
        super(KHandler, self).__init__()
        self.extensions = ['.k']

    def parse(self, filename):
        """
        Method to parse the file `filename`. It returns a matrix with all the
        coordinates. It reads only the section *NODE of the k files.

        :param string filename: name of the input file.

        :return: mesh_points: it is a `n_points`-by-3 matrix containing the
                coordinates of the points of the mesh.
        :rtype: numpy.ndarray
        """
        self._check_filename_type(filename)
        self._check_extension(filename)
        self.infile = filename
        index = -9
        mesh_points = []
        with open(self.infile, 'r') as input_file:
            for num, line in enumerate(input_file):
                if line.startswith('*NODE'):
                    index = num
                if num == index + 1:
                    if line.startswith('$'):
                        index = num
                    elif line.startswith('*'):
                        index = -9
                    else:
                        l = []
                        l.append(float(line[8:24]))
                        l.append(float(line[24:40]))
                        l.append(float(line[40:56]))
                        mesh_points.append(l)
                        index = num
            mesh_points = np.array(mesh_points)
        return mesh_points

    def write(self, mesh_points, filename):
        """
        Writes a .k file, called filename, copying all the lines from
        self.filename but the coordinates. mesh_points is a matrix that
        contains the new coordinates to write in the .k file.

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
                    if line.startswith('*NODE'):
                        index = num
                    if num == index + 1:
                        if line.startswith('$'):
                            index = num
                        elif line.startswith('*'):
                            index = -9
                        else:
                            for j in range(0, 3):
                                line = line[:8+16*(j)] + '{:16.10f}'.format(mesh_points[i][j]) + line[8+16*(j+1):]
                            i += 1
                            index = num
                    output_file.write(line)
