"""
PyGeM init
"""
from .deformation import Deformation
from .ffd import FFD
from .rbf import RBF
from .idw import IDW
from .rbf_factory import RBFFactory
from .custom_deformation import CustomDeformation

def get_current_year():
    """ Return current year """
    from datetime import datetime
    return datetime.now().year

__project__ = 'PyGeM'
__title__ = "pygem"
__author__ = "Marco Tezzele, Nicola Demo"
__copyright__ = "Copyright 2017-{}, PyGeM contributors".format(get_current_year())
__license__ = "MIT"
__version__ = "2.0.3"
__mail__ = 'marcotez@gmail.com, demo.nicola@gmail.com'
__maintainer__ = __author__
__status__ = "Stable"
