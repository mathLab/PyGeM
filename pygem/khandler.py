"""
Derived module from filehandler.py to handle LS-DYNA keyword (.k) files.
"""
import numpy as np
import pygem.filehandler as fh
import re


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

        mesh_points = []
        node_indicator = False

        with open(self.infile, 'r') as input_file:
            for num, line in enumerate(input_file):

                # Regex to find all Node elements
                expression = re.compile(r'(.+?)(?:,|$)')
                expression = expression.findall(line)

                # Discount any header lines
                if line.startswith("$"):
                    continue

                # Find the the start of the nodes section and continue if true
                if line.startswith('*NODE'):
                    node_indicator = True
                    continue

                # Find the start of the elements and break if found.
                if line.startswith('*ELEMENT'):
                    break

                # If the node is found then iterate through and append the nodes list points to mesh_points
                if not node_indicator:
                    pass
                else:
                    if len(expression) == 1:
                        expression = re.findall(r'\S+', expression[0])
                    l = [float(expression[1]), float(expression[2]), float(expression[3])]
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

        i = 0
        node_indicator = False

        with open(self.outfile, 'w') as output_file:
            with open(self.infile, 'r') as input_file:
                for num, line in enumerate(input_file):
                    get_num = re.findall(r'[-+]?[0-9]*\.?[0-9]+', line)

                    # Write header files
                    if line.startswith('$'):
                        output_file.write(line)
                        continue

                    # Change the node indicator if you find the elements section
                    if line.startswith('*ELEMENT'):
                        node_indicator = False

                    # Change the nodes indicator if you find the nodes section
                    if line.startswith('*NODE'):
                        node_indicator = True
                        output_file.write(line)
                        continue

                    # If in the nodes section append the mesh points otherwise copy the data from parsed file
                    if not node_indicator:
                        output_file.write(line)
                    else:

                        # Split lines to find decimeter
                        split_line = line.split(" ")

                        # Format the data into correct format
                        data = [int(get_num[0]), '{:.10f}'.format(float(mesh_points[i][0])),
                                '{:.10f}'.format(float(mesh_points[i][1])),
                                '{:.10f}'.format(float(mesh_points[i][2]))]

                        comma_seperator = False
                        pointer = 0

                        # Enumerate through the line and change the relevent information retaining the delimetered value
                        for index, value in enumerate(split_line):

                            # Only read the none space values
                            if value:

                                # Format if delimited by commas
                                if value[len(value) - 1] == ",":
                                    comma_seperator = True
                                    new_str = value.replace(value[:-1], str(data[pointer]))
                                    split_line[index] = new_str

                                # Else format the data delimited by spaces
                                else:
                                    new_str = value.replace(value, str(data[pointer]))
                                    split_line[index] = new_str
                                    if float(data[pointer]) < 0 and comma_seperator == False:
                                        del split_line[index - 1]

                                pointer += 1
                            else:
                                pass

                        # Format the split string back into normal string and write
                        original_str = ""
                        for j in split_line:
                            original_str += j + " "
                        original_str = original_str[:-1]

                        output_file.write(original_str + "\n")

                        i += 1
