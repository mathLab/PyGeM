from unittest import TestCase

import numpy as np

import pygem.filehandler as fh


class TestFilehandler(TestCase):
    def test_base_class_infile(self):
        file_handler = fh.FileHandler()
        self.assertIsNone(file_handler.infile)

    def test_base_class_outfile(self):
        file_handler = fh.FileHandler()
        self.assertIsNone(file_handler.outfile)

    def test_base_class_extension(self):
        file_handler = fh.FileHandler()
        self.assertListEqual(file_handler.extensions, [])

    def test_base_class_parse(self):
        file_handler = fh.FileHandler()
        with self.assertRaises(NotImplementedError):
            file_handler.parse('input')

    def test_base_class_write(self):
        file_handler = fh.FileHandler()
        mesh_points = np.zeros((3, 3))
        with self.assertRaises(NotImplementedError):
            file_handler.write(mesh_points, 'output')
