"""
PyGeM setup.py
"""
from setuptools import setup, find_packages

meta = {}
with open("pygem/meta.py") as fp:
    exec(fp.read(), meta)

# Package meta-data.
NAME = meta['__title__']
DESCRIPTION = 'Python Geometrical Morphing.'
URL = 'https://github.com/mathLab/PyGeM'
MAIL = meta['__mail__']
AUTHOR = meta['__author__']
VERSION = meta['__version__']
KEYWORDS = 'dimension_reduction mathematics ffd morphing iges stl vtk openfoam'

REQUIRED = [
    'future', 'numpy', 'scipy',	'matplotlib',
]

EXTRAS = {
    'docs': ['Sphinx>=1.4', 'sphinx_rtd_theme'],
    'test': ['pytest', 'pytest-cov'],
}

LDESCRIPTION = (
    "PyGeM is a python package using Free Form Deformation, Radial Basis "
    "Functions and Inverse Distance Weighting to parametrize and morph "
    "complex geometries. It is ideally suited for actual industrial problems, "
    "since it allows to handle:\n"
    "1) Computer Aided Design files (in .iges, .step, and .stl formats) Mesh "
    "files (in .unv and OpenFOAM formats)\n"
    "2) Output files (in .vtk format)\n"
    "3) LS-Dyna Keyword files (.k format).\n"
    "\n"
    "By now, it has been used with meshes with up to 14 milions of cells. Try "
    "with more and more complicated input files! See the Examples section "
    "below and the Tutorials to have an idea of the potential of this package."
)


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LDESCRIPTION,
    author=AUTHOR,
    author_email=MAIL,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords=KEYWORDS,
    url=URL,
    license='MIT',
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    zip_safe=False,
)
