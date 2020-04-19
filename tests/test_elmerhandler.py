from unittest import TestCase
import unittest
import pygem.elmerhandler as elh
import numpy as np
import filecmp
import os


class TestElmerHandler(TestCase):
    def test_elmer_instantiation(self):
        elmer_handler = elh.ElmerHandler()
       
    def test_elmer_default_infile_member(self):
        elmer_handler = elh.ElmerHandler()
        self.assertIsNone(elmer_handler.infile)

    def test_elmer_default_outfile_member(self):
        elmer_handler = elh.ElmerHandler()
        self.assertIsNone(elmer_handler.outfile)

    def test_elmer_default_extension_member(self):
        elmer_handler = elh.ElmerHandler()
        self.assertListEqual(elmer_handler.extensions, ['.nodes'])

    def test_elmer_parse_failing_filename_type(self):
        elmer_handler = elh.ElmerHandler()
        with self.assertRaises(TypeError):
            mesh_points = elmer_handler.parse(.2)

    def test_elmer_parse_failing_check_extension(self):
        elmer_handler = elh.ElmerHandler()
        with self.assertRaises(ValueError):
            mesh_points = elmer_handler.parse(
                'tests/test_datasets/test_square.iges')

    def test_elmer_parse_infile(self):
        elmer_handler = elh.ElmerHandler()
        mesh_points = elmer_handler.parse(
            'tests/test_datasets/test_elmer.nodes')
        self.assertEqual(elmer_handler.infile,
                         'tests/test_datasets/test_elmer.nodes')

    def test_elmer_parse_shape(self):
        elmer_handler = elh.ElmerHandler()
        mesh_points = elmer_handler.parse(
            'tests/test_datasets/test_elmer.nodes')
        self.assertTupleEqual(mesh_points.shape, (240, 3))

    def test_elmer_parse_coords_1(self):
        elmer_handler = elh.ElmerHandler()
        mesh_points = elmer_handler.parse(
            'tests/test_datasets/test_elmer.nodes')
        np.testing.assert_almost_equal(mesh_points[33][0], 2.94650796191)

    def test_open_foam_parse_coords_2(self):
        elmer_handler = elh.ElmerHandler()
        mesh_points = elmer_handler.parse(
            'tests/test_datasets/test_elmer.nodes')
        np.testing.assert_almost_equal(mesh_points[149][1], 2)

    def test_elmer_parse_coords_3(self):
        elmer_handler = elh.ElmerHandler()
        mesh_points = elmer_handler.parse(
            'tests/test_datasets/test_elmer.nodes')
        np.testing.assert_almost_equal(mesh_points[239][2], .0)

    def test_elmer_parse_coords_4(self):
        elmer_handler = elh.ElmerHandler()
        mesh_points = elmer_handler.parse(
            'tests/test_datasets/test_elmer.nodes')
        np.testing.assert_almost_equal(mesh_points[0][0], 0.0)

    def test_elmer_parse_coords_5(self):
        elmer_handler = elh.ElmerHandler()
        mesh_points = elmer_handler.parse(
            'tests/test_datasets/test_elmer.nodes')
        np.testing.assert_almost_equal(mesh_points[-1][1], 2)

    def test_elmer_write_failing_filename_type(self):
        elmer_handler = elh.ElmerHandler()
        mesh_points = elmer_handler.parse(
            'tests/test_datasets/test_elmer.nodes')
        with self.assertRaises(TypeError):
            elmer_handler.write(mesh_points, -1.)

    def test_elmer_write_failing_check_extension(self):
        elmer_handler = elh.ElmerHandler()
        mesh_points = elmer_handler.parse(
            'tests/test_datasets/test_elmer.nodes')
        with self.assertRaises(ValueError):
            elmer_handler.write(mesh_points,
                                    'tests/test_datasets/test_square.iges')

    def test_elmer_write_failing_infile_instantiation(self):
        elmer_handler = elh.ElmerHandler()
        mesh_points = np.zeros((40, 3))
        with self.assertRaises(RuntimeError):
            elmer_handler.write(mesh_points,
                                    'tests/test_datasets/test_elmer_out.nodes')

    def test_elmer_write_outfile(self):
        elmer_handler = elh.ElmerHandler()
        mesh_points = elmer_handler.parse(
            'tests/test_datasets/test_elmer.nodes')
        outfilename = 'tests/test_datasets/test_elmer_out.nodes'
        elmer_handler.write(mesh_points, outfilename)
        self.assertEqual(elmer_handler.outfile, outfilename)
        self.addCleanup(os.remove, outfilename)

    def test_elmer_write_comparison(self):
        elmer_handler = elh.ElmerHandler()
        mesh_points = elmer_handler.parse(
            'tests/test_datasets/test_elmer.nodes')
        mesh_points[0] = [0.1, 1.1, 0.1]
        mesh_points[1] = [0.1, 1.2, 0.1]
        mesh_points[2] = [0.1, 1.6, 0.1]
        mesh_points[149] = [11.7910193185, 2.0, 0.1]
        mesh_points[150] = [12.858303628, 0.0, 0.1]
        mesh_points[151] = [12.858303628, 0.125, 0.1]
        mesh_points[-3] = [26.2, 1.6, 0.1]
        mesh_points[-2] = [26.2, 2.01666666667, 0.1]
        mesh_points[-1] = [26.2, 2.1, 0.1]
        
        outfilename = 'tests/test_datasets/test_elmer_out.nodes'
        outfilename_expected = 'tests/test_datasets/test_elmer_out_true.nodes'

        elmer_handler.write(mesh_points, outfilename)
        self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
        self.addCleanup(os.remove, outfilename)
