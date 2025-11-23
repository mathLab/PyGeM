"""PyGeM init."""

from .deformation import Deformation
from .ffd import FFD
from .cffd import CFFD
from .rbf import RBF
from .idw import IDW
from .rbf_factory import RBFFactory
from .custom_deformation import CustomDeformation
from .bffd import BFFD
from .vffd import VFFD

from .meta import (
    __project__,
    __title__,
    __author__,
    __copyright__,
    __license__,
    __version__,
    __mail__,
    __maintainer__,
    __status__,
)

__all__ = [
    "Deformation",
    "FFD",
    "CFFD",
    "RBF",
    "IDW",
    "RBFFactory",
    "CustomDeformation",
    "BFFD",
    "VFFD",
    "deformation",
    "ffd",
    "rbf",
    "idw",
    "rbf_factory",
    "custom_deformation",
    "cffd",
    "bffd",
    "vffd",
    "__project__",
    "__title__",
    "__author__",
    "__copyright__",
    "__license__",
    "__version__",
    "__mail__",
    "__maintainer__",
    "__status__",
]
