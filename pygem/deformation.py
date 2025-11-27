"""
Module for the abstract Deformation class
"""

from abc import ABC, abstractmethod
 
class Deformation(ABC):
    """
    Abstract class for generic deformation.
    This class should be inherited for the development of new deformation
    techniques.
    """
 
    @abstractmethod
    def __init__(self, value):
        pass

    @abstractmethod
    def __call__(self, src):
        pass
