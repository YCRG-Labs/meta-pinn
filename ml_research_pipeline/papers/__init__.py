"""
Publication-ready output generation module.

This module provides tools for generating publication-quality plots, tables,
and reports for ML research papers.
"""

from .plot_generator import PaperPlotGenerator
from .table_generator import LaTeXTableGenerator
from .report_generator import ReportGenerator

__all__ = [
    'PaperPlotGenerator',
    'LaTeXTableGenerator',
    'ReportGenerator'
]