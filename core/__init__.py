"""
Core module for Smart Traffic Violation Detection System
"""

from .detector import ViolationDetector
from .ocr import NumberPlateOCR
from .utils import ImageProcessor

__all__ = ['ViolationDetector', 'NumberPlateOCR', 'ImageProcessor']
