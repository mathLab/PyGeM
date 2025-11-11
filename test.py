import matplotlib

matplotlib.use("agg")

import nose

test_defaults = [
    "tests/test_ffd.py",
    "tests/test_idw.py",
    "tests/test_rbf.py",
    #'tests/test_khandler.py',
    #'tests/test_mdpahandler.py',
    #'tests/test_openfhandler.py',
    #'tests/test_stlhandler.py',
    #'tests/test_unvhandler.py',
    #'tests/test_vtkhandler.py',
]


test_cad = [
    "tests/test_ffdcad.py",
]

default_argv = ["--tests"] + test_defaults
cad_argv = ["--tests"] + test_cad

return_value = 0  # Success

try:
    import pygem.cad

    return_value = 1 if nose.run(argv=cad_argv) is False else 0
except:
    print("module OCC not found, skip tests for CAD")

return_value = 1 if nose.run(argv=default_argv) is False else 0

import sys

sys.exit(int(return_value))
