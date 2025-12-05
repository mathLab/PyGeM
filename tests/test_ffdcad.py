import filecmp
import os
from unittest import TestCase

import numpy as np
import re
from OCC.Core.TopoDS import TopoDS_Shape

from pygem.cad import FFD
from pygem.cad import CADDeformation


class TestFFDCAD(TestCase):

    def extract_floats(self, lines):
        """
        Extract all numeric values from IGES file content.

        This helper function parses a list of IGES file lines and returns
        a flattened numpy array of all floating-point numbers, ignoring
        line breaks, empty lines, and non-numeric text.

        IGES files often wrap data lines differently on different platforms.
        By flattening all numeric values into a single array, this function enables reliable numerical comparison of IGES files regardless of
        line wrapping or minor formatting differences.
        """
        all_values = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split(',')[:-1]
            for p in parts:
                try:
                    all_values.append(float(p))
                except ValueError:
                    pass
        return np.asarray(all_values)

    def test_ffd_iges_pipe_mod_through_files(self):
        ffd = FFD(None,30,30,30,1e-4)
        ffd.read_parameters(
            filename='tests/test_datasets/parameters_test_ffd_iges.prm')
        ffd('tests/test_datasets/test_pipe.iges', 'test_pipe_result.iges')
        with open('test_pipe_result.iges', "r") as created, \
             open('tests/test_datasets/test_pipe_out_true.iges', "r") as reference:
             ref_data = self.extract_floats(reference.readlines()[5:])
             cre_data = self.extract_floats(created.readlines()[5:])
             np.testing.assert_array_almost_equal(cre_data, ref_data, decimal=6)
        self.addCleanup(os.remove, 'test_pipe_result.iges')

    def test_ffd_iges_pipe_mod_through_topods_shape(self):
        filename = 'tests/test_datasets/test_pipe_hollow.iges'
        orig_shape = CADDeformation.read_shape(filename)
        ffd = FFD(None,30,30,30,1e-4)
        ffd.read_parameters(
            filename='tests/test_datasets/parameters_test_ffd_iges.prm')
        mod_shape = ffd(orig_shape)
        CADDeformation.write_shape('test_pipe_hollow_result.iges', mod_shape)
        with open('test_pipe_hollow_result.iges', "r") as created, \
             open('tests/test_datasets/test_pipe_hollow_out_true.iges', "r") as reference:
             ref_data = self.extract_floats(reference.readlines()[5:])
             cre_data = self.extract_floats(created.readlines()[5:])
             np.testing.assert_array_almost_equal(cre_data, ref_data, decimal=6)
        self.addCleanup(os.remove, 'test_pipe_hollow_result.iges')

    """
    def test_ffd_step_pipe_mod_through_files(self):
        ffd = FFD(None,30,30,30,1e-4)
        ffd.read_parameters(
            filename='tests/test_datasets/parameters_test_ffd_iges.prm')
        ffd('tests/test_datasets/test_pipe.step', 'test_pipe_result.step')
        with open('test_pipe_result.step', "r") as created, \
             open('tests/test_datasets/test_pipe_out_true.step', "r") as reference:
             ref = reference.readlines()[92:]
             cre = created.readlines()[92:]
             self.assertEqual(len(ref),len(cre))
             for i in range(len(cre)):
                 ref_ = np.asarray(re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", ref[i]),dtype=float)
                 cre_ = np.asarray(re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", cre[i]),dtype=float)
                 np.testing.assert_array_almost_equal(cre_, ref_, decimal=6)
        self.addCleanup(os.remove, 'test_pipe_result.step')
    """