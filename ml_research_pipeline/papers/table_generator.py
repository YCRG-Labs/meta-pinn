"""
LaTeX table generation for ML research papers.

This module provides the LaTeXTableGenerator class for creating publication-ready
LaTeX tables with proper formatting, statistical notation, and significance indicators.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class LaTeXTableGenerator:
    """
    Generates publication-quality LaTeX tables with statistical formatting.

    This class provides methods for creating various types of tables commonly
    used in ML research papers, with proper LaTeX formatting and statistical
    notation suitable for academic publications.
    """

    def __init__(
        self,
        precision: int = 3,
        use_booktabs: bool = True,
        table_position: str = "htbp",
        font_size: str = "small",
    ):
        """
        Initialize the LaTeX table generator.

        Args:
            precision: Number of decimal places for numerical values
            use_booktabs: Whether to use booktabs package for better formatting
            table_position: LaTeX table position specifier
            font_size: Font size for tables (tiny, scriptsize, footnotesize, small, normalsize)
        """
        self.precision = precision
        self.use_booktabs = use_booktabs
        self.table_position = table_position
        self.font_size = font_size

        # Statistical significance symbols
        self.significance_symbols = {
            "p < 0.001": "***",
            "p < 0.01": "**",
            "p < 0.05": "*",
            "p >= 0.05": "",
        }

    def create_method_comparison_table(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        method_names: Optional[Dict[str, str]] = None,
        significance_data: Optional[Dict[str, Dict[str, float]]] = None,
        caption: str = "Method comparison results",
        label: str = "tab:method_comparison",
    ) -> str:
        """
        Create a LaTeX table comparing different methods across multiple metrics.

        Args:
            results: Dictionary mapping method names to metric dictionaries
            metrics: List of metrics to include in the table
            method_names: Optional mapping from internal names to display names
            significance_data: Statistical significance test results
            caption: Table caption
            label: LaTeX label for referencing

        Returns:
            LaTeX table string
        """
        if not results:
            raise ValueError("Results dictionary cannot be empty")

        # Prepare method names
        methods = list(results.keys())
        if method_names is None:
            method_names = {method: method for method in methods}

        # Validate metrics exist in results
        for method in methods:
            for metric in metrics:
                if f"{metric}_mean" not in results[method]:
                    raise KeyError(
                        f"Missing metric '{metric}_mean' for method '{method}'"
                    )
                if f"{metric}_std" not in results[method]:
                    raise KeyError(
                        f"Missing metric '{metric}_std' for method '{method}'"
                    )

        # Start building table
        lines = []
        lines.append("\\begin{table}[" + self.table_position + "]")
        lines.append("\\centering")
        lines.append("\\" + self.font_size)

        # Table header
        num_cols = len(metrics) + 1  # +1 for method name column
        col_spec = "l" + "c" * len(metrics)

        if self.use_booktabs:
            lines.append("\\begin{tabular}{" + col_spec + "}")
            lines.append("\\toprule")
        else:
            lines.append("\\begin{tabular}{|" + "|".join(col_spec) + "|}")
            lines.append("\\hline")

        # Column headers
        header_row = "Method"
        for metric in metrics:
            formatted_metric = self._format_metric_name(metric)
            header_row += " & " + formatted_metric
        header_row += " \\\\"
        lines.append(header_row)

        if self.use_booktabs:
            lines.append("\\midrule")
        else:
            lines.append("\\hline")

        # Data rows
        for method in methods:
            display_name = method_names.get(method, method)
            row = self._escape_latex(display_name)

            for metric in metrics:
                mean_val = results[method][f"{metric}_mean"]
                std_val = results[method][f"{metric}_std"]

                # Format value with uncertainty
                formatted_value = self._format_value_with_uncertainty(mean_val, std_val)

                # Add significance indicator if available
                if significance_data and method in significance_data:
                    if metric in significance_data[method]:
                        p_value = significance_data[method][metric]
                        sig_symbol = self._get_significance_symbol(p_value)
                        formatted_value += sig_symbol

                row += " & " + formatted_value

            row += " \\\\"
            lines.append(row)

        # Table footer
        if self.use_booktabs:
            lines.append("\\bottomrule")
        else:
            lines.append("\\hline")

        lines.append("\\end{tabular}")

        # Caption and label
        lines.append("\\caption{" + caption + "}")
        lines.append("\\label{" + label + "}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def create_statistical_summary_table(
        self,
        data: Dict[str, Dict[str, Any]],
        statistics: List[str] = ["mean", "std", "min", "max"],
        caption: str = "Statistical summary",
        label: str = "tab:statistical_summary",
    ) -> str:
        """
        Create a LaTeX table with statistical summaries.

        Args:
            data: Dictionary mapping variable names to statistics
            statistics: List of statistics to include
            caption: Table caption
            label: LaTeX label for referencing

        Returns:
            LaTeX table string
        """
        if not data:
            raise ValueError("Data dictionary cannot be empty")

        lines = []
        lines.append("\\begin{table}[" + self.table_position + "]")
        lines.append("\\centering")
        lines.append("\\" + self.font_size)

        # Table structure
        num_cols = len(statistics) + 1
        col_spec = "l" + "c" * len(statistics)

        if self.use_booktabs:
            lines.append("\\begin{tabular}{" + col_spec + "}")
            lines.append("\\toprule")
        else:
            lines.append("\\begin{tabular}{|" + "|".join(col_spec) + "|}")
            lines.append("\\hline")

        # Header
        header_row = "Variable"
        for stat in statistics:
            header_row += " & " + stat.capitalize()
        header_row += " \\\\"
        lines.append(header_row)

        if self.use_booktabs:
            lines.append("\\midrule")
        else:
            lines.append("\\hline")

        # Data rows
        for variable, stats in data.items():
            row = self._escape_latex(variable)

            for stat in statistics:
                if stat in stats:
                    value = stats[stat]
                    formatted_value = self._format_number(value)
                else:
                    formatted_value = "---"

                row += " & " + formatted_value

            row += " \\\\"
            lines.append(row)

        # Footer
        if self.use_booktabs:
            lines.append("\\bottomrule")
        else:
            lines.append("\\hline")

        lines.append("\\end{tabular}")
        lines.append("\\caption{" + caption + "}")
        lines.append("\\label{" + label + "}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def create_hyperparameter_table(
        self,
        hyperparameters: Dict[str, Dict[str, Any]],
        caption: str = "Hyperparameter settings",
        label: str = "tab:hyperparameters",
    ) -> str:
        """
        Create a LaTeX table for hyperparameter settings.

        Args:
            hyperparameters: Dictionary mapping method names to hyperparameter dictionaries
            caption: Table caption
            label: LaTeX label for referencing

        Returns:
            LaTeX table string
        """
        if not hyperparameters:
            raise ValueError("Hyperparameters dictionary cannot be empty")

        # Get all unique hyperparameter names
        all_params = set()
        for method_params in hyperparameters.values():
            all_params.update(method_params.keys())
        all_params = sorted(list(all_params))

        lines = []
        lines.append("\\begin{table}[" + self.table_position + "]")
        lines.append("\\centering")
        lines.append("\\" + self.font_size)

        # Table structure
        methods = list(hyperparameters.keys())
        num_cols = len(methods) + 1
        col_spec = "l" + "c" * len(methods)

        if self.use_booktabs:
            lines.append("\\begin{tabular}{" + col_spec + "}")
            lines.append("\\toprule")
        else:
            lines.append("\\begin{tabular}{|" + "|".join(col_spec) + "|}")
            lines.append("\\hline")

        # Header
        header_row = "Parameter"
        for method in methods:
            header_row += " & " + self._escape_latex(method)
        header_row += " \\\\"
        lines.append(header_row)

        if self.use_booktabs:
            lines.append("\\midrule")
        else:
            lines.append("\\hline")

        # Data rows
        for param in all_params:
            row = self._escape_latex(param)

            for method in methods:
                if param in hyperparameters[method]:
                    value = hyperparameters[method][param]
                    formatted_value = self._format_hyperparameter_value(value)
                else:
                    formatted_value = "---"

                row += " & " + formatted_value

            row += " \\\\"
            lines.append(row)

        # Footer
        if self.use_booktabs:
            lines.append("\\bottomrule")
        else:
            lines.append("\\hline")

        lines.append("\\end{tabular}")
        lines.append("\\caption{" + caption + "}")
        lines.append("\\label{" + label + "}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def create_correlation_matrix_table(
        self,
        correlation_matrix: np.ndarray,
        variable_names: List[str],
        caption: str = "Correlation matrix",
        label: str = "tab:correlation_matrix",
    ) -> str:
        """
        Create a LaTeX table for a correlation matrix.

        Args:
            correlation_matrix: Square correlation matrix
            variable_names: Names of variables corresponding to matrix rows/columns
            caption: Table caption
            label: LaTeX label for referencing

        Returns:
            LaTeX table string
        """
        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square")

        if len(variable_names) != correlation_matrix.shape[0]:
            raise ValueError("Number of variable names must match matrix dimensions")

        lines = []
        lines.append("\\begin{table}[" + self.table_position + "]")
        lines.append("\\centering")
        lines.append("\\" + self.font_size)

        # Table structure
        n_vars = len(variable_names)
        col_spec = "l" + "c" * n_vars

        if self.use_booktabs:
            lines.append("\\begin{tabular}{" + col_spec + "}")
            lines.append("\\toprule")
        else:
            lines.append("\\begin{tabular}{|" + "|".join(col_spec) + "|}")
            lines.append("\\hline")

        # Header
        header_row = ""
        for var_name in variable_names:
            header_row += " & " + self._escape_latex(var_name)
        header_row += " \\\\"
        lines.append(header_row)

        if self.use_booktabs:
            lines.append("\\midrule")
        else:
            lines.append("\\hline")

        # Data rows
        for i, var_name in enumerate(variable_names):
            row = self._escape_latex(var_name)

            for j in range(n_vars):
                if i == j:
                    # Diagonal elements
                    formatted_value = "1.00"
                elif i > j:
                    # Lower triangle
                    corr_val = correlation_matrix[i, j]
                    formatted_value = self._format_number(corr_val)
                else:
                    # Upper triangle (leave empty for cleaner look)
                    formatted_value = ""

                row += " & " + formatted_value

            row += " \\\\"
            lines.append(row)

        # Footer
        if self.use_booktabs:
            lines.append("\\bottomrule")
        else:
            lines.append("\\hline")

        lines.append("\\end{tabular}")
        lines.append("\\caption{" + caption + "}")
        lines.append("\\label{" + label + "}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def _format_metric_name(self, metric_name: str) -> str:
        """Format metric name for display in table headers."""
        # Convert snake_case to Title Case
        formatted = metric_name.replace("_", " ").title()

        # Handle common abbreviations
        replacements = {
            "Mse": "MSE",
            "Mae": "MAE",
            "Rmse": "RMSE",
            "R2": "$R^2$",
            "Pde": "PDE",
            "Pinn": "PINN",
        }

        for old, new in replacements.items():
            formatted = formatted.replace(old, new)

        return formatted

    def _format_value_with_uncertainty(self, mean: float, std: float) -> str:
        """Format a value with its uncertainty (mean ± std)."""
        mean_str = self._format_number(mean)
        std_str = self._format_number(std)
        return f"{mean_str} $\\pm$ {std_str}"

    def _format_number(self, value: Union[int, float]) -> str:
        """Format a number with appropriate precision."""
        if isinstance(value, int):
            return str(value)

        if abs(value) < 1e-10:
            if self.precision > 0:
                return f"0.{'0' * self.precision}"
            else:
                return "0"

        # Use scientific notation for very small or very large numbers
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.{self.precision-1}e}"
        else:
            return f"{value:.{self.precision}f}"

    def _format_hyperparameter_value(self, value: Any) -> str:
        """Format hyperparameter values for display."""
        if isinstance(value, bool):
            return "True" if value else "False"
        elif isinstance(value, str):
            return self._escape_latex(value)
        elif isinstance(value, (int, float)):
            return self._format_number(value)
        elif isinstance(value, (list, tuple)):
            formatted_items = [
                self._format_hyperparameter_value(item) for item in value
            ]
            return "[" + ", ".join(formatted_items) + "]"
        else:
            return self._escape_latex(str(value))

    def _get_significance_symbol(self, p_value: float) -> str:
        """Get significance symbol based on p-value."""
        if p_value < 0.001:
            return "$^{***}$"
        elif p_value < 0.01:
            return "$^{**}$"
        elif p_value < 0.05:
            return "$^{*}$"
        else:
            return ""

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters in text."""
        # Handle backslash first to avoid double escaping
        escaped_text = text.replace("\\", "\\textbackslash{}")

        # Dictionary of other characters to escape
        escape_chars = {
            "&": "\\&",
            "%": "\\%",
            "$": "\\$",
            "#": "\\#",
            "^": "\\textasciicircum{}",
            "_": "\\_",
            "{": "\\{",
            "}": "\\}",
            "~": "\\textasciitilde{}",
        }

        for char, escaped in escape_chars.items():
            escaped_text = escaped_text.replace(char, escaped)

        return escaped_text

    def save_table_to_file(
        self, table_latex: str, filepath: Path, include_packages: bool = True
    ) -> None:
        """
        Save LaTeX table to file with optional package includes.

        Args:
            table_latex: LaTeX table string
            filepath: Path to save the file
            include_packages: Whether to include necessary LaTeX packages
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        content = []

        if include_packages:
            content.extend(
                [
                    "% Required packages for this table:",
                    "% \\usepackage{booktabs}  % for better table formatting",
                    "% \\usepackage{array}     % for advanced column formatting",
                    "% \\usepackage{multirow}  % for multi-row cells",
                    "",
                ]
            )

        content.append(table_latex)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

    def create_multi_table_document(
        self,
        tables: Dict[str, str],
        title: str = "Research Results Tables",
        author: str = "Generated by LaTeXTableGenerator",
    ) -> str:
        """
        Create a complete LaTeX document with multiple tables.

        Args:
            tables: Dictionary mapping table names to LaTeX table strings
            title: Document title
            author: Document author

        Returns:
            Complete LaTeX document string
        """
        lines = []

        # Document preamble
        lines.extend(
            [
                "\\documentclass{article}",
                "\\usepackage[utf8]{inputenc}",
                "\\usepackage{booktabs}",
                "\\usepackage{array}",
                "\\usepackage{multirow}",
                "\\usepackage{geometry}",
                "\\geometry{margin=1in}",
                "",
                f"\\title{{{title}}}",
                f"\\author{{{author}}}",
                "\\date{\\today}",
                "",
                "\\begin{document}",
                "",
                "\\maketitle",
                "",
            ]
        )

        # Add tables
        for table_name, table_latex in tables.items():
            lines.extend(
                [f"\\section{{{table_name}}}", "", table_latex, "", "\\clearpage", ""]
            )

        # Document end
        lines.append("\\end{document}")

        return "\n".join(lines)
