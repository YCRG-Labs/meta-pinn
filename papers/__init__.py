"""
Papers module for publication-ready output generation.

This module contains tools for generating publication-quality plots, tables,
and reports for research papers and presentations.
"""

__version__ = "0.1.0"
__author__ = "ML Research Team"

# Import publication tools
from .plot_generator import PaperPlotGenerator
from .table_generator import LaTeXTableGenerator
from .report_generator import ReportGenerator

__all__ = [
    "PaperPlotGenerator",
    "LaTeXTableGenerator", 
    "ReportGenerator",
]