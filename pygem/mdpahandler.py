"""Derived module from khandler.py to handle Kratos mesh (.mdpa) files."""

import numpy as np

import pygem.filehandler as fh


class MdpaHandler(fh.FileHandler):
    """Kratos mesh file handler class.

    :cvar string infile: name of the input file to be processed.
    :cvar string outfile: name of the output file where to write in.
    :cvar list extensions: extensions of the input/output files. It is
        equal to '.mdpa'.
    """

    def __init__(self):
        super().__init__()
        self.extensions = [".mdpa"]

    def parse(self, filename):
        """Method to parse the file `filename`. It returns a matrix with all
        the coordinates. It reads only the section after "Begin Nodes" of the
        mdpa files.

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
        with open(self.infile, "r", encoding="utf-8") as input_file:
            for num, line in enumerate(input_file):
                if line.startswith("Begin Nodes"):
                    index = num
                if num == index + 1:
                    if line.startswith("End Nodes"):
                        break
                    line = line.replace("D", "E")
                    coordinates_list = []
                    for token in line.split()[1:]:
                        try:
                            coordinates_list.append(float(token))
                        except ValueError:
                            pass
                    mesh_points.append(coordinates_list)
                    index = num
            mesh_points = np.array(mesh_points)
        return mesh_points

    def write(self, mesh_points, filename):
        """Writes a .mdpa file, called filename, copying all the lines from
        self.filename but the coordinates. mesh_points is a matrix that
        contains the new coordinates to write in the .mdpa file.

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
        with open(self.outfile, "w", encoding="utf-8") as output_file:
            with open(self.infile, "r", encoding="utf-8") as input_file:
                for num, line in enumerate(input_file):
                    if line.startswith("Begin Nodes"):
                        index = num
                    if num == index + 1:
                        if line.startswith("End Nodes"):
                            index = -9
                        else:
                            line = (
                                f" {i + 1:6d} {mesh_points[i][0]:23.16E} "
                                f"{mesh_points[i][1]:23.16E} "
                                f"{mesh_points[i][2]:23.16E}\n"
                            )
                            i += 1
                            index = num
                    output_file.write(line)
