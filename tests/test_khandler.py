from unittest import TestCase
import pygem.khandler as uh
import numpy as np
import filecmp
import os


class TestKHandler(TestCase):
    def test_k_instantiation(self):
        k_handler = uh.KHandler()

    def test_k_default_infile_member(self):
        k_handler = uh.KHandler()
        self.assertIsNone(k_handler.infile)

    def test_k_default_outfile_member(self):
        k_handler = uh.KHandler()
        self.assertIsNone(k_handler.outfile)

    def test_k_default_extension_member(self):
        k_handler = uh.KHandler()
        self.assertListEqual(k_handler.extensions, ['.k'])

    def test_k_parse_failing_filename_type(self):
        k_handler = uh.KHandler()
        with self.assertRaises(TypeError):
            mesh_points = k_handler.parse(5.2)

    def test_k_parse_failing_check_extension(self):
        k_handler = uh.KHandler()
        with self.assertRaises(ValueError):
            mesh_points = k_handler.parse(
                'tests/test_datasets/test_square.iges')

    def test_k_parse_infile(self):
        k_handler = uh.KHandler()
        mesh_points = k_handler.parse('tests/test_datasets/test_square.k')
        self.assertEqual(k_handler.infile, 'tests/test_datasets/test_square.k')

    def test_k_parse_shape(self):
        k_handler = uh.KHandler()
        mesh_points = k_handler.parse('tests/test_datasets/test_square.k')
        self.assertTupleEqual(mesh_points.shape, (256, 3))

    def test_k_parse_coords_1(self):
        k_handler = uh.KHandler()
        mesh_points = k_handler.parse('tests/test_datasets/test_square.k')
        np.testing.assert_almost_equal(mesh_points[33][0], 1.0)

    def test_k_parse_coords_2(self):
        k_handler = uh.KHandler()
        mesh_points = k_handler.parse('tests/test_datasets/test_square.k')
        np.testing.assert_almost_equal(mesh_points[178][1], 0.4)

    def test_k_parse_coords_3(self):
        k_handler = uh.KHandler()
        mesh_points = k_handler.parse('tests/test_datasets/test_square.k')
        np.testing.assert_almost_equal(mesh_points[100][2], 0.0)

    def test_k_parse_coords_4(self):
        k_handler = uh.KHandler()
        mesh_points = k_handler.parse('tests/test_datasets/test_square.k')
        np.testing.assert_almost_equal(mesh_points[0][0], 0.0)

    def test_k_parse_coords_5(self):
        k_handler = uh.KHandler()
        mesh_points = k_handler.parse('tests/test_datasets/test_square.k')
        np.testing.assert_almost_equal(mesh_points[-1][2], 0.0)

    def test_k_write_failing_filename_type(self):
        k_handler = uh.KHandler()
        mesh_points = k_handler.parse('tests/test_datasets/test_square.k')
        with self.assertRaises(TypeError):
            k_handler.write(mesh_points, -2)

    def test_k_write_failing_check_extension(self):
        k_handler = uh.KHandler()
        mesh_points = k_handler.parse('tests/test_datasets/test_square.k')
        with self.assertRaises(ValueError):
            k_handler.write(mesh_points, 'tests/test_datasets/test_square.iges')

    def test_k_write_failing_infile_instantiation(self):
        k_handler = uh.KHandler()
        mesh_points = np.zeros((20, 3))
        with self.assertRaises(RuntimeError):
            k_handler.write(mesh_points,
                            'tests/test_datasets/test_square_out.k')

    def test_k_write_outfile(self):
        k_handler = uh.KHandler()
        mesh_points = k_handler.parse('tests/test_datasets/test_square.k')
        outfilename = 'tests/test_datasets/test_square_out.k'
        k_handler.write(mesh_points, outfilename)
        self.assertEqual(k_handler.outfile, outfilename)
        self.addCleanup(os.remove, outfilename)

    def test_k_write_comparison_1(self):
        k_handler = uh.KHandler()
        mesh_points = k_handler.parse('tests/test_datasets/test_square.k')
        outfilename = 'tests/test_datasets/test_square_out.k'
        outfilename_expected = 'tests/test_datasets/test_square.k'
        k_handler.write(mesh_points, outfilename)
        self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
        self.addCleanup(os.remove, outfilename)

    def test_k_write_comparison_2(self):
        k_handler = uh.KHandler()
        mesh_points = k_handler.parse('tests/test_datasets/test_square.k')
        mesh_points[0][0] = 2.2
        mesh_points[5][1] = 4.3
        mesh_points[9][2] = 0.5
        mesh_points[45][0] = 7.2
        mesh_points[132][1] = -1.2
        mesh_points[255][2] = -3.6
        outfilename = 'tests/test_datasets/test_square_out.k'
        outfilename_expected = 'tests/test_datasets/test_square_out_true.k'
        k_handler.write(mesh_points, outfilename)
        self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
        self.addCleanup(os.remove, outfilename)

    def test_comma_seperated_parse(self):
        k_handler = uh.KHandler()
        mesh_points = k_handler.parse('tests/test_datasets/test_square_comma.k')
        np.testing.assert_almost_equal([mesh_points[0][0], mesh_points[0][1], mesh_points[0][2]],
                                       [-0.0500000007, -0.0250000004, -0.0250000004])

    def test_comma_seperated_write(self):
        k_handler = uh.KHandler()
        mesh_points = k_handler.parse('tests/test_datasets/test_square_comma.k')
        mesh_points[0][0] = 2.2
        mesh_points[5][1] = 4.3
        mesh_points[9][2] = 0.5
        mesh_points[45][0] = 7.2
        mesh_points[132][1] = -1.2
        mesh_points[255][2] = -3.6
        outfilename = 'tests/test_datasets/test_square_comma_out.k'
        outfilename_expected = 'tests/test_datasets/test_square_comma_out_true.k'
        k_handler.write(mesh_points, outfilename)
        self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
        self.addCleanup(os.remove, outfilename)



