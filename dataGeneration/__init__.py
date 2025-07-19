"""
Data Generation Package

This package contains utilities for generating and processing equation datasets.
"""

# Import key functions and classes from utils module
try:
    from .utils import (
        convert_equation,
        read_and_transform
    )
    
    __all__ = [
        'convert_equation',
        'read_and_transform'
    ]
    
except ImportError:
    # Handle case where utils module might not be available
    __all__ = []

__version__ = "1.0.0"
__author__ = "Li Zheng"
__description__ = "Data generation utilities for equation processing" 