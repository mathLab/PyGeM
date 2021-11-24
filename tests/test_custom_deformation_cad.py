from unittest import TestCase

import numpy as np
from pygem.cad import CustomDeformation
from pygem.cad import CADDeformation


class TestCustomDeformation(TestCase):

    def test_class_members_func(self):
        def move(x):
            return x + x**2
        deform = CustomDeformation(move)

    def test_customdeform_cad_type(self):
        def move(x):
            x0, x1, x2 = x
            return [x0**2, x1, x2]
        filename = 'tests/test_datasets/test_pipe_hollow.iges'
        orig_shape = CADDeformation.read_shape(filename)
        deform = CustomDeformation(move)
        new_shape = deform(orig_shape)
        assert type(new_shape) == type(orig_shape)
