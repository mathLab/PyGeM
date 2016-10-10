from unittest import TestCase
import unittest
import pygem.filehandler as fh
import numpy as np
import filecmp
import os


class TestFilehandler(TestCase):
	def test_base_class_infile(self):
		file_handler = fh.FileHandler()
		assert file_handler.infile == None

	def test_base_class_outfile(self):
		file_handler = fh.FileHandler()
		assert file_handler.outfile == None

	def test_base_class_extension(self):
		file_handler = fh.FileHandler()
		assert file_handler.extension == None

	def test_base_class_parse(self):
		file_handler = fh.FileHandler()
		with self.assertRaises(NotImplementedError):
			file_handler.parse('input')

	def test_base_class_write(self):
		file_handler = fh.FileHandler()
		mesh_points = np.zeros((3, 3))
		with self.assertRaises(NotImplementedError):
			file_handler.write(mesh_points, 'output')
