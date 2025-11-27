from unittest import TestCase
import pygem.mdpahandler as uh
import numpy as np
import filecmp
import os


class TestMdpaHandler(TestCase):
    def test_mdpa_instantiation(self):
        mdpa_handler = uh.MdpaHandler()

    def test_mdpa_default_infile_member(self):
        mdpa_handler = uh.MdpaHandler()
        self.assertIsNone(mdpa_handler.infile)

    def test_mdpa_default_outfile_member(self):
        mdpa_handler = uh.MdpaHandler()
        self.assertIsNone(mdpa_handler.outfile)

    def test_mdpa_default_extension_member(self):
        mdpa_handler = uh.MdpaHandler()
        self.assertListEqual(mdpa_handler.extensions, ['.mdpa'])

    def test_mdpa_parse_failing_filename_type(self):
        mdpa_handler = uh.MdpaHandler()
        with self.assertRaises(TypeError):
            mesh_points = mdpa_handler.parse(5.2)

    def test_mdpa_parse_failing_check_extension(self):
        mdpa_handler = uh.MdpaHandler()
        with self.assertRaises(ValueError):
            mesh_points = mdpa_handler.parse(
                'tests/test_datasets/test_square.iges')

    def test_mdpa_parse_infile(self):
        mdpa_handler = uh.MdpaHandler()
        mesh_points = mdpa_handler.parse('tests/test_datasets/test_square.mdpa')
        self.assertEqual(mdpa_handler.infile, 'tests/test_datasets/test_square.mdpa')

    def test_mdpa_parse_shape(self):
        mdpa_handler = uh.MdpaHandler()
        mesh_points = mdpa_handler.parse('tests/test_datasets/test_square.mdpa')
        self.assertTupleEqual(mesh_points.shape, (256, 3))

    def test_mdpa_parse_coords_1(self):
        mdpa_handler = uh.MdpaHandler()
        mesh_points = mdpa_handler.parse('tests/test_datasets/test_square.mdpa')
        np.testing.assert_almost_equal(mesh_points[190][0], 1.0)

    def test_mdpa_parse_coords_2(self):
        mdpa_handler = uh.MdpaHandler()
        mesh_points = mdpa_handler.parse('tests/test_datasets/test_square.mdpa')
        np.testing.assert_almost_equal(mesh_points[72][1], 0.4)

    def test_mdpa_parse_coords_3(self):
        mdpa_handler = uh.MdpaHandler()
        mesh_points = mdpa_handler.parse('tests/test_datasets/test_square.mdpa')
        np.testing.assert_almost_equal(mesh_points[100][2], 0.0)

    def test_mdpa_parse_coords_4(self):
        mdpa_handler = uh.MdpaHandler()
        mesh_points = mdpa_handler.parse('tests/test_datasets/test_square.mdpa')
        np.testing.assert_almost_equal(mesh_points[0][0], 0.0)

    def test_mdpa_parse_coords_5(self):
        mdpa_handler = uh.MdpaHandler()
        mesh_points = mdpa_handler.parse('tests/test_datasets/test_square.mdpa')
        np.testing.assert_almost_equal(mesh_points[-1][2], 0.0)

    def test_mdpa_write_failing_filename_type(self):
        mdpa_handler = uh.MdpaHandler()
        mesh_points = mdpa_handler.parse('tests/test_datasets/test_square.mdpa')
        with self.assertRaises(TypeError):
            mdpa_handler.write(mesh_points, -2)

    def test_mdpa_write_failing_check_extension(self):
        mdpa_handler = uh.MdpaHandler()
        mesh_points = mdpa_handler.parse('tests/test_datasets/test_square.mdpa')
        with self.assertRaises(ValueError):
            mdpa_handler.write(mesh_points, 'tests/test_datasets/test_square.iges')

    def test_mdpa_write_failing_infile_instantiation(self):
        mdpa_handler = uh.MdpaHandler()
        mesh_points = np.zeros((20, 3))
        with self.assertRaises(RuntimeError):
            mdpa_handler.write(mesh_points,
                            'tests/test_datasets/test_square_out.mdpa')

    def test_mdpa_write_outfile(self):
        infilename = 'tests/test_datasets/test_square.mdpa'
        outfilename = 'tests/test_datasets/test_square_out.mdpa'
        mdpa_handler = uh.MdpaHandler()
        mesh_points = mdpa_handler.parse(infilename)
        mdpa_handler.write(mesh_points, outfilename)
        self.assertEqual(mdpa_handler.outfile, outfilename)
        #self.addCleanup(os.remove, outfilename)

    def test_mdpa_write_comparison_1(self):
        infilename = 'tests/test_datasets/test_square.mdpa'
        outfilename = 'tests/test_datasets/test_square_out.mdpa'
        outfilename_expected = 'tests/test_datasets/test_square.mdpa'
        mdpa_handler = uh.MdpaHandler()
        mesh_points = mdpa_handler.parse(infilename)
        mdpa_handler.write(mesh_points, outfilename)
        self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
        self.addCleanup(os.remove, outfilename)

    def test_mdpa_write_comparison_2(self):
        infilename = 'tests/test_datasets/test_square.mdpa'
        outfilename = 'tests/test_datasets/test_square_out.mdpa'
        outfilename_expected = 'tests/test_datasets/test_square_out_true.mdpa'
        mdpa_handler = uh.MdpaHandler()
        mesh_points = mdpa_handler.parse(infilename)
        mesh_points[0][0] = 0.0
        mesh_points[5][1] = 1.0
        mesh_points[9][2] = 0.0
        mesh_points[44][0] = 0.0
        mesh_points[122][1] = 0.2
        mesh_points[255][2] = 0.0
        mdpa_handler.write(mesh_points, outfilename)
        self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
        self.addCleanup(os.remove, outfilename)
