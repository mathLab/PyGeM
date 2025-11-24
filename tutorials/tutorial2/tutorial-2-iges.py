#!/usr/bin/env python
# coding: utf-8

# # PyGeM

# ## Tutorial 2: Free Form Deformation on a cylinder in CAD file format

import platform
import sys

print(f"Python Version: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"System: {platform.system()} {platform.release()}")

try:
    import pygem

    print(f"PyGeM version: {pygem.__version__}")
except ImportError:
    print(f"PyGeM not found. Installing...")
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", ".[tut]"]
    )
    import pygem

    print(f"PyGeM version: {pygem.__version__}")

import numpy as np

np.random.seed(42)

# In this tutorial we show again an application of _free form deformation_ method, now to a CAD file. These files, that are often adopted to model complex geometries, require an additional pre- and post-processing of the surfaces to perform the deformation.
#
# The **CAD** submodule of **PyGeM** takes care of the deformation to all CAD files (.step, .iges, etc.), so first of all we import from the submodule the `FFD` class.

from pygem.cad import FFD

# Since the visualisation of CAD files may be tricky (depending by the operating system, the graphical front-end, ...), we avoid for this tutorial any graphical renderer, letting to the reader the possibility to implement by himself the needed plotting routines.
#
# The `FFD` class in the **CAD** module shares the same interface with the original `FFD` class (for discrete geometries). We can simply parse a parameter file to set everything like we want (remember you can do the same directly setting the object attributes).

ffd = FFD()
ffd.read_parameters("../tests/test_datasets/parameters_test_ffd_iges.prm")
print(ffd)


# Almost already completed! We now specify the input file (the one which contains the shape to deform) and the output file: these are the two input argument to pass to the object in order to perform the deformation.

input_cad_file_name = "../tests/test_datasets/test_pipe.iges"
modified_cad_file_name = "test_pipe_deformed.iges"
ffd(input_cad_file_name, modified_cad_file_name)


# The output file is created and the deformed shape is stored into it. We skip any visual check because of the **CAD** format file, so as final proof we simply show the differences, lines by lines, between the input and the output. Even if we can't be sure about the correctness of the results, in this way we ensure the outcome is different from the original inpuit.

import subprocess

subprocess.run(
    [
        "diff",
        "-y",
        "../tests/test_datasets/test_pipe.iges",
        "test_pipe_deformed.iges",
    ]
)
