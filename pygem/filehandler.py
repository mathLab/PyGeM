"""
Base module with the base class for reading and writing different files.

.. warning::
    This module will be deprecated in next releases. Follow updates on
    https://github.com/mathLab for news about file handling. 
"""
import os
import warnings
warnings.warn("This module will be deprecated in next releases", DeprecationWarning)


class FileHandler(object):
    """
    A base class for file handling.

    :cvar string infile: name of the input file to be processed.
    :cvar string outfile: name of the output file where to write in.
    :cvar list extensions: extensions of the input/output files. It is specific
        for each subclass.
    """

    def __init__(self):
        self.infile = None
        self.outfile = None
        self.extensions = []

    def parse(self, *args):
        """
        Abstract method to parse a specific file.

        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError(
            "Subclass must implement abstract method " \
         + self.__class__.__name__ + ".parse")

    def write(self, *args):
        """
        Abstract method to write a specific file.

        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError(
            "Subclass must implement abstract method " \
        + self.__class__.__name__ + ".write")

    def _check_extension(self, filename):
        """
        This private class method checks if the given `filename` has the proper
        `extension` set in the child class. If not it raises a ValueError.

        :param string filename: file to check.
        """
        __, file_ext = os.path.splitext(filename)
        if file_ext not in self.extensions:
            raise ValueError(
                'The input file does not have the proper extension. \
                It is {0!s}, instead of {1!s}.'.format(file_ext,
                                                       self.extensions))

    @staticmethod
    def _check_filename_type(filename):
        """
        This private static method checks if `filename` is a string. If not it
        raises a TypeError.

        :param string filename: file to check.
        """
        if not isinstance(filename, str):
            raise TypeError(
                'The given filename ({0!s}) must be a string'.format(filename))

    def _check_infile_instantiation(self):
        """
        This private method checks if `self.infile` is instantiated. If not
        it means that nobody called the parse method and `self.infile` is None.
        If the check fails it raises a RuntimeError.

        """
        if not self.infile:
            raise RuntimeError(
                "You can not write a file without having parsed one.")
