"""
Publication-ready output generation module.

This module provides tools for generating publication-quality plots, tables,
and reports for ML research papers.
"""

from .plot_generator import PaperPlotGenerator
from .report_generator import ReportGenerator
from .table_generator import LaTeXTableGenerator

__all__ = ["PaperPlotGenerator", "LaTeXTableGenerator", "ReportGenerator"]
