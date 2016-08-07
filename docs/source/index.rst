Welcome to PyGeM's documentation!
===================================================

.. image:: _static/logo_PyGeM.png
   :height: 150px
   :width: 150 px
   :align: right

Python utilities for geometrical morphing.


Description
^^^^^^^^^^^^

PyGeM is a python library using Free Form Deformation and Radial Basis Functions to parametrize and morph complex geometries. It is ideally suited for actual industrial problems, since it allows to handle:

- Computer Aided Design files (in .iges and .stl formats)
- Mesh files (in .unv and OpenFOAM formats)
- Output files (in .vtk format)

By now, it has been used with meshes with up to 14 milions of cells. Try with more and more complicated input files!


Guide
^^^^^

.. toctree::
   :maxdepth: 3

   code
   contact
   contributing
   LICENSE


Tutorials
^^^^^^^^^^

We made some tutorial examples:

- `Tutorial 1 <tutorial1stl.html>`_ shows how to deal with stl files and FFD
- `Turorial 2 <tutorial2iges.html>`_ shows how to deal with iges files and FFD
- `Turorial 3 <tutorial3unv.html>`_ shows how to deal with unv files and how to set the desired continuity to the geometry using FFD
- `Turorial 4 <tutorial4rbf.html>`_ shows how to perform RBF interpolation



Indices and tables
^^^^^^^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

