from unittest import TestCase
import unittest
from pygem.cad.nurbshandler import NurbsHandler


class TestNurbsHandler(TestCase):
    def test_nurbs_instantiation(self):
        nurbs_handler = NurbsHandler()

    def test_nurbs_default_infile_member(self):
        nurbs_handler = NurbsHandler()
        self.assertIsNone(nurbs_handler.infile)

    def test_nurbs_default_shape_member(self):
        nurbs_handler = NurbsHandler()
        self.assertIsNone(nurbs_handler.shape)

    def test_nurbs_default_control_point_position_member(self):
        nurbs_handler = NurbsHandler()
        self.assertIsNone(nurbs_handler._control_point_position)

    def test_nurbs_default_outfile_member(self):
        nurbs_handler = NurbsHandler()
        self.assertIsNone(nurbs_handler.outfile)

    def test_nurbs_default_tolerance(self):
        nurbs_handler = NurbsHandler()
        self.assertAlmostEqual(nurbs_handler.tolerance, 1e-6)

    def test_nurbs_default_extension_member(self):
        nurbs_handler = NurbsHandler()
        self.assertListEqual(nurbs_handler.extensions, [])

    def test_nurbs_load_shape_from_file_raises(self):
        nurbs_handler = NurbsHandler()
        with self.assertRaises(NotImplementedError):
            nurbs_handler.load_shape_from_file(None)

    def test_nurbs_write_shape_to_file_raises(self):
        nurbs_handler = NurbsHandler()
        with self.assertRaises(NotImplementedError):
            nurbs_handler.write_shape_to_file(None, None)

    def test_nurbs_check_infile_instantiation_no_shape(self):
        nurbs_handler = NurbsHandler()
        nurbs_handler.infile = "something"
        with self.assertRaises(RuntimeError):
            nurbs_handler._check_infile_instantiation()

    def test_nurbs_check_infile_instantiation_no_infile(self):
        nurbs_handler = NurbsHandler()
        nurbs_handler.shape = True
        with self.assertRaises(RuntimeError):
            nurbs_handler._check_infile_instantiation()

    def test_nurbs_check_infile_instantiation_shape_infile_wrong(self):
        nurbs_handler = NurbsHandler()
        with self.assertRaises(RuntimeError):
            nurbs_handler._check_infile_instantiation()

    def test_nurbs_check_infile_instantiation_correct(self):
        nurbs_handler = NurbsHandler()
        nurbs_handler.shape = True
        nurbs_handler.infile = "something"
        try:
            nurbs_handler._check_infile_instantiation()
        except RuntimeError:
            self.fail(
                "Handler was instantiated correctly, yet an error was raised.")
