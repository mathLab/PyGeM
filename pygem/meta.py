__all__ = [
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


def get_current_year():
    """ Return current year """
    from datetime import datetime

    return datetime.now().year


__project__ = "PyGeM"
__title__ = "pygem"
__author__ = "Marco Tezzele, Nicola Demo"
__copyright__ = "Copyright 2017-{}, PyGeM contributors".format(
    get_current_year()
)
__license__ = "MIT"
__version__ = "2.0.0"
__mail__ = "marcotez@gmail.com, demo.nicola@gmail.com"
__maintainer__ = __author__
__status__ = "Stable"
