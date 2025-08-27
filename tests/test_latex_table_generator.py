"""
Unit tests for LaTeXTableGenerator class.

Tests LaTeX table generation functionality, formatting accuracy, and syntax correctness
for publication-quality tables.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import re

from ml_research_pipeline.papers.table_generator import LaTeXTableGenerator


class TestLaTeXTableGenerator:
    """Test suite for LaTeXTableGenerator class."""
    
    @pytest.fixture
    def table_generator(self):
        """Create a LaTeXTableGenerator instance for testing."""
        return LaTeXTableGenerator()
    
    @pytest.fixture
    def sample_results_data(self):
        """Sample experimental results data for testing."""
        return {
            'Meta-PINN': {
                'accuracy_mean': 0.95,
                'accuracy_std': 0.02,
                'speed_mean': 10.5,
                'speed_std': 1.2,
                'loss_mean': 0.001,
                'loss_std': 0.0002
            },
            'Standard PINN': {
                'accuracy_mean': 0.87,
                'accuracy_std': 0.04,
                'speed_mean': 25.3,
                'speed_std': 3.1,
                'loss_mean': 0.005,
                'loss_std': 0.001
            },
            'Transfer Learning': {
                'accuracy_mean': 0.91,
                'accuracy_std': 0.03,
                'speed_mean': 15.8,
                'speed_std': 2.0,
                'loss_mean': 0.003,
                'loss_std': 0.0005
            }
        }
    
    @pytest.fixture
    def sample_significance_data(self):
        """Sample statistical significance data for testing."""
        return {
            'Meta-PINN': {
                'accuracy': 0.0001,  # p < 0.001 for ***
                'speed': 0.005,      # p < 0.01 for **
                'loss': 0.0001
            },
            'Standard PINN': {
                'accuracy': 0.1,
                'speed': 0.05,
                'loss': 0.2
            },
            'Transfer Learning': {
                'accuracy': 0.02,
                'speed': 0.03,
                'loss': 0.01
            }
        }
    
    def test_initialization(self):
        """Test LaTeXTableGenerator initialization."""
        generator = LaTeXTableGenerator(
            precision=4,
            use_booktabs=False,
            table_position='h',
            font_size='footnotesize'
        )
        
        assert generator.precision == 4
        assert generator.use_booktabs == False
        assert generator.table_position == 'h'
        assert generator.font_size == 'footnotesize'
        
        # Check significance symbols are set
        assert '***' in generator.significance_symbols.values()
    
    def test_method_comparison_table_creation(self, table_generator, sample_results_data):
        """Test creation of method comparison tables."""
        metrics = ['accuracy', 'speed']
        
        latex_table = table_generator.create_method_comparison_table(
            results=sample_results_data,
            metrics=metrics,
            caption="Test Method Comparison",
            label="tab:test_comparison"
        )
        
        assert isinstance(latex_table, str)
        
        # Check LaTeX structure
        assert '\\begin{table}' in latex_table
        assert '\\end{table}' in latex_table
        assert '\\begin{tabular}' in latex_table
        assert '\\end{tabular}' in latex_table
        
        # Check caption and label
        assert 'Test Method Comparison' in latex_table
        assert 'tab:test_comparison' in latex_table
        
        # Check method names appear
        for method in sample_results_data.keys():
            assert method in latex_table
        
        # Check metric headers
        assert 'Accuracy' in latex_table
        assert 'Speed' in latex_table
        
        # Check booktabs formatting (default)
        assert '\\toprule' in latex_table
        assert '\\midrule' in latex_table
        assert '\\bottomrule' in latex_table
    
    def test_method_comparison_table_with_significance(self, table_generator, sample_results_data, sample_significance_data):
        """Test method comparison table with significance indicators."""
        metrics = ['accuracy', 'speed']
        
        latex_table = table_generator.create_method_comparison_table(
            results=sample_results_data,
            metrics=metrics,
            significance_data=sample_significance_data
        )
        
        # Check significance symbols are included
        assert '$^{***}$' in latex_table  # p < 0.001 (for p=0.0001)
        assert '$^{**}$' in latex_table   # p < 0.01
        assert '$^{*}$' in latex_table    # p < 0.05
    
    def test_method_comparison_table_custom_names(self, table_generator, sample_results_data):
        """Test method comparison table with custom method names."""
        metrics = ['accuracy']
        method_names = {
            'Meta-PINN': 'Meta-Learning PINN',
            'Standard PINN': 'Baseline PINN',
            'Transfer Learning': 'Transfer Learning PINN'
        }
        
        latex_table = table_generator.create_method_comparison_table(
            results=sample_results_data,
            metrics=metrics,
            method_names=method_names
        )
        
        # Check custom names appear
        for display_name in method_names.values():
            assert display_name in latex_table
    
    def test_method_comparison_table_without_booktabs(self, sample_results_data):
        """Test method comparison table without booktabs formatting."""
        generator = LaTeXTableGenerator(use_booktabs=False)
        metrics = ['accuracy']
        
        latex_table = generator.create_method_comparison_table(
            results=sample_results_data,
            metrics=metrics
        )
        
        # Check traditional table formatting
        assert '\\hline' in latex_table
        assert '\\toprule' not in latex_table
        assert '\\midrule' not in latex_table
        assert '\\bottomrule' not in latex_table
    
    def test_statistical_summary_table_creation(self, table_generator):
        """Test creation of statistical summary tables."""
        data = {
            'Parameter A': {
                'mean': 0.85,
                'std': 0.12,
                'min': 0.45,
                'max': 0.98
            },
            'Parameter B': {
                'mean': 15.3,
                'std': 2.8,
                'min': 8.2,
                'max': 22.1
            }
        }
        
        latex_table = table_generator.create_statistical_summary_table(
            data=data,
            caption="Statistical Summary Test",
            label="tab:stats_test"
        )
        
        assert isinstance(latex_table, str)
        
        # Check structure
        assert '\\begin{table}' in latex_table
        assert '\\end{table}' in latex_table
        
        # Check data appears
        assert 'Parameter A' in latex_table
        assert 'Parameter B' in latex_table
        
        # Check statistics headers
        assert 'Mean' in latex_table
        assert 'Std' in latex_table
        assert 'Min' in latex_table
        assert 'Max' in latex_table
    
    def test_hyperparameter_table_creation(self, table_generator):
        """Test creation of hyperparameter tables."""
        hyperparameters = {
            'Meta-PINN': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'hidden_layers': [64, 128, 64],
                'use_dropout': True
            },
            'Standard PINN': {
                'learning_rate': 0.01,
                'batch_size': 64,
                'hidden_layers': [32, 64, 32],
                'use_dropout': False
            }
        }
        
        latex_table = table_generator.create_hyperparameter_table(
            hyperparameters=hyperparameters,
            caption="Hyperparameter Settings",
            label="tab:hyperparams"
        )
        
        assert isinstance(latex_table, str)
        
        # Check structure
        assert '\\begin{table}' in latex_table
        assert '\\end{table}' in latex_table
        
        # Check method names
        assert 'Meta-PINN' in latex_table
        assert 'Standard PINN' in latex_table
        
        # Check parameter names (they should be escaped)
        assert 'learning\\_rate' in latex_table or 'learning\\textbackslash{}_rate' in latex_table
        assert 'batch\\_size' in latex_table or 'batch\\textbackslash{}_size' in latex_table
        
        # Check boolean formatting
        assert 'True' in latex_table
        assert 'False' in latex_table
    
    def test_correlation_matrix_table_creation(self, table_generator):
        """Test creation of correlation matrix tables."""
        correlation_matrix = np.array([
            [1.0, 0.8, -0.3],
            [0.8, 1.0, -0.1],
            [-0.3, -0.1, 1.0]
        ])
        variable_names = ['Variable A', 'Variable B', 'Variable C']
        
        latex_table = table_generator.create_correlation_matrix_table(
            correlation_matrix=correlation_matrix,
            variable_names=variable_names,
            caption="Correlation Matrix",
            label="tab:correlation"
        )
        
        assert isinstance(latex_table, str)
        
        # Check structure
        assert '\\begin{table}' in latex_table
        assert '\\end{table}' in latex_table
        
        # Check variable names
        for var_name in variable_names:
            assert var_name in latex_table
        
        # Check diagonal elements (should be 1.00)
        assert '1.00' in latex_table
        
        # Check correlation values
        assert '0.800' in latex_table or '0.8' in latex_table
    
    def test_format_metric_name(self, table_generator):
        """Test metric name formatting."""
        assert table_generator._format_metric_name('accuracy') == 'Accuracy'
        assert table_generator._format_metric_name('mse_loss') == 'MSE Loss'
        assert table_generator._format_metric_name('r2_score') == '$R^2$ Score'
        assert table_generator._format_metric_name('pinn_residual') == 'PINN Residual'
    
    def test_format_value_with_uncertainty(self, table_generator):
        """Test value formatting with uncertainty."""
        formatted = table_generator._format_value_with_uncertainty(0.95, 0.02)
        assert '$\\pm$' in formatted
        assert '0.950' in formatted
        assert '0.020' in formatted
    
    def test_format_number(self, table_generator):
        """Test number formatting."""
        # Regular numbers
        assert table_generator._format_number(0.123) == '0.123'
        assert table_generator._format_number(123) == '123'
        
        # Very small numbers (scientific notation)
        formatted_small = table_generator._format_number(1e-5)
        assert 'e' in formatted_small
        
        # Very large numbers (scientific notation)
        formatted_large = table_generator._format_number(1e6)
        assert 'e' in formatted_large
        
        # Zero
        assert table_generator._format_number(0.0) == '0.000'
    
    def test_format_hyperparameter_value(self, table_generator):
        """Test hyperparameter value formatting."""
        # Boolean
        assert table_generator._format_hyperparameter_value(True) == 'True'
        assert table_generator._format_hyperparameter_value(False) == 'False'
        
        # String
        assert table_generator._format_hyperparameter_value('relu') == 'relu'
        
        # Number
        assert table_generator._format_hyperparameter_value(0.001) == '0.001'
        
        # List
        formatted_list = table_generator._format_hyperparameter_value([64, 128, 64])
        assert '[64, 128, 64]' in formatted_list
    
    def test_get_significance_symbol(self, table_generator):
        """Test significance symbol generation."""
        assert table_generator._get_significance_symbol(0.0001) == '$^{***}$'
        assert table_generator._get_significance_symbol(0.005) == '$^{**}$'
        assert table_generator._get_significance_symbol(0.03) == '$^{*}$'
        assert table_generator._get_significance_symbol(0.1) == ''
    
    def test_escape_latex(self, table_generator):
        """Test LaTeX character escaping."""
        # Test common special characters
        assert table_generator._escape_latex('test & data') == 'test \\& data'
        assert table_generator._escape_latex('50% accuracy') == '50\\% accuracy'
        assert table_generator._escape_latex('$100 cost') == '\\$100 cost'
        assert table_generator._escape_latex('test_variable') == 'test\\_variable'
        assert table_generator._escape_latex('x^2 + y') == 'x\\textasciicircum\\{\\}2 + y'
        # Test backslash escaping
        assert table_generator._escape_latex('test\\data') == 'test\\textbackslash\\{\\}data'
    
    def test_save_table_to_file(self, table_generator, sample_results_data):
        """Test saving tables to files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / 'test_table.tex'
            
            latex_table = table_generator.create_method_comparison_table(
                results=sample_results_data,
                metrics=['accuracy']
            )
            
            table_generator.save_table_to_file(
                table_latex=latex_table,
                filepath=temp_path,
                include_packages=True
            )
            
            assert temp_path.exists()
            
            # Check file content
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert '\\usepackage{booktabs}' in content
            assert latex_table in content
    
    def test_create_multi_table_document(self, table_generator, sample_results_data):
        """Test creation of multi-table documents."""
        table1 = table_generator.create_method_comparison_table(
            results=sample_results_data,
            metrics=['accuracy']
        )
        
        table2 = table_generator.create_statistical_summary_table(
            data={'Param A': {'mean': 0.5, 'std': 0.1}}
        )
        
        tables = {
            'Method Comparison': table1,
            'Statistical Summary': table2
        }
        
        document = table_generator.create_multi_table_document(
            tables=tables,
            title="Test Document",
            author="Test Author"
        )
        
        assert isinstance(document, str)
        
        # Check document structure
        assert '\\documentclass{article}' in document
        assert '\\begin{document}' in document
        assert '\\end{document}' in document
        
        # Check title and author
        assert 'Test Document' in document
        assert 'Test Author' in document
        
        # Check sections
        assert '\\section{Method Comparison}' in document
        assert '\\section{Statistical Summary}' in document
        
        # Check tables are included
        assert table1 in document
        assert table2 in document
    
    def test_error_handling_empty_results(self, table_generator):
        """Test error handling with empty results."""
        with pytest.raises(ValueError, match="Results dictionary cannot be empty"):
            table_generator.create_method_comparison_table(
                results={},
                metrics=['accuracy']
            )
    
    def test_error_handling_missing_metrics(self, table_generator):
        """Test error handling with missing metrics."""
        invalid_results = {
            'Method1': {'other_metric_mean': 0.5, 'other_metric_std': 0.1}
        }
        
        with pytest.raises(KeyError, match="Missing metric 'accuracy_mean'"):
            table_generator.create_method_comparison_table(
                results=invalid_results,
                metrics=['accuracy']
            )
    
    def test_error_handling_invalid_correlation_matrix(self, table_generator):
        """Test error handling with invalid correlation matrix."""
        # Non-square matrix
        with pytest.raises(ValueError, match="Correlation matrix must be square"):
            table_generator.create_correlation_matrix_table(
                correlation_matrix=np.array([[1, 0], [0, 1], [1, 0]]),
                variable_names=['A', 'B']
            )
        
        # Mismatched variable names
        with pytest.raises(ValueError, match="Number of variable names must match matrix dimensions"):
            table_generator.create_correlation_matrix_table(
                correlation_matrix=np.array([[1, 0], [0, 1]]),
                variable_names=['A', 'B', 'C']
            )
    
    def test_latex_syntax_correctness(self, table_generator, sample_results_data):
        """Test that generated LaTeX has correct syntax."""
        latex_table = table_generator.create_method_comparison_table(
            results=sample_results_data,
            metrics=['accuracy', 'speed']
        )
        
        # Check balanced braces
        open_braces = latex_table.count('{')
        close_braces = latex_table.count('}')
        assert open_braces == close_braces
        
        # Check balanced table environments
        begin_table = latex_table.count('\\begin{table}')
        end_table = latex_table.count('\\end{table}')
        assert begin_table == end_table == 1
        
        begin_tabular = latex_table.count('\\begin{tabular}')
        end_tabular = latex_table.count('\\end{tabular}')
        assert begin_tabular == end_tabular == 1
        
        # Check proper line endings
        lines = latex_table.split('\n')
        table_lines = [line for line in lines if line.strip() and not line.strip().startswith('%')]
        
        # Most table content lines should end with \\ or be environment commands
        content_lines = [line for line in table_lines 
                        if not any(cmd in line for cmd in ['\\begin', '\\end', '\\caption', '\\label', '\\centering', '\\toprule', '\\midrule', '\\bottomrule', '\\hline', '\\small', '\\footnotesize', '\\tiny', '\\scriptsize', '\\normalsize'])]
        
        for line in content_lines:
            if line.strip():
                assert line.strip().endswith('\\\\'), f"Line should end with \\\\: {line}"
    
    @pytest.mark.parametrize("precision,expected_digits", [
        (2, 2),
        (3, 3),
        (4, 4)
    ])
    def test_precision_setting(self, precision, expected_digits, sample_results_data):
        """Test that precision setting affects number formatting."""
        generator = LaTeXTableGenerator(precision=precision)
        
        latex_table = generator.create_method_comparison_table(
            results=sample_results_data,
            metrics=['accuracy']
        )
        
        # Check that numbers have the expected precision
        # Look for decimal numbers in the table
        decimal_pattern = r'\d+\.\d+'
        matches = re.findall(decimal_pattern, latex_table)
        
        if matches:
            # Check at least one number has the expected precision
            for match in matches:
                decimal_part = match.split('.')[1]
                if len(decimal_part) == expected_digits:
                    break
            else:
                # If no exact match found, check if any number has the expected precision
                assert any(len(match.split('.')[1]) == expected_digits for match in matches), \
                    f"Expected precision {expected_digits} not found in numbers: {matches}"


if __name__ == '__main__':
    pytest.main([__file__])