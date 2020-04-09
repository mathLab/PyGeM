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
        import re
        self._check_filename_type(filename)
        self._check_extension(filename)
        self.infile = filename

        mesh_points = []
        node_indicator = False

        with open(self.infile, 'r') as input_file:
            for num, line in enumerate(input_file):
                expression = re.compile(r'(.+?)(?:,|$)')
                expression = expression.findall(line)

                if line.startswith('*NODE'):
                    node_indicator = True
                    continue
                if line.startswith('*ELEMENT'):
                    break

                if node_indicator == False:
                    pass
                else:
                    if len(expression) == 1:
                        expression = re.findall(r'\S+', expression[0])
                    l = []
                    l.append(float(expression[1]))
                    l.append(float(expression[2]))
                    l.append(float(expression[3]))
                    mesh_points.append(l)

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
        import re

        i = 0
        node_indicator = False

        with open(self.outfile, 'w') as output_file:
            with open(self.infile, 'r') as input_file:
                for num, line in enumerate(input_file):
                    get_num = re.findall(r'[-+]?[0-9]*\.?[0-9]+', line)

                    if line.startswith('*ELEMENT'):
                        node_indicator = False

                    if node_indicator == False:
                        output_file.write(line)
                    else:
                        line = get_num[0] + ", " + str(mesh_points[i][0]) + ", " + str(mesh_points[i][1]) + ", " + str(
                            mesh_points[i][2]) + "\n"
                        output_file.write(line)
                        i += 1

                    if line.startswith('*NODE'):
                        node_indicator = True





