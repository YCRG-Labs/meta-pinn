"""
Tests for result management system.
"""

import pytest
import json
import pickle
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from ml_research_pipeline.utils.result_management import (
    ResultManager,
    ExperimentResult,
    ResultComparison
)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def result_manager(temp_dir):
    """Create result manager."""
    return ResultManager(temp_dir)


@pytest.fixture
def sample_result():
    """Create sample experiment result."""
    return ExperimentResult(
        experiment_id="exp_001",
        name="test_experiment",
        timestamp=datetime.now().isoformat(),
        status="completed",
        config_hash="abc123",
        hyperparameters={"lr": 0.01, "batch_size": 32},
        final_metrics={"accuracy": 0.95, "loss": 0.05},
        best_metrics={"accuracy": 0.97, "loss": 0.03},
        training_history=[
            {"step": 0, "loss": 1.0, "accuracy": 0.5},
            {"step": 1, "loss": 0.8, "accuracy": 0.6},
            {"step": 2, "loss": 0.6, "accuracy": 0.7},
            {"step": 3, "loss": 0.4, "accuracy": 0.8},
            {"step": 4, "loss": 0.2, "accuracy": 0.9}
        ],
        duration_seconds=120.5,
        hardware_info={"gpu_count": 1, "cpu_count": 8},
        reproducibility_info={"validation_performed": True, "overall_status": "PASSED"}
    )


class TestExperimentResult:
    """Test experiment result data structure."""
    
    def test_result_creation(self, sample_result):
        """Test experiment result creation."""
        assert sample_result.experiment_id == "exp_001"
        assert sample_result.name == "test_experiment"
        assert sample_result.status == "completed"
        assert sample_result.final_metrics["accuracy"] == 0.95
        assert len(sample_result.training_history) == 5
    
    def test_result_serialization(self, sample_result):
        """Test result serialization and deserialization."""
        # Test to_dict
        result_dict = sample_result.to_dict()
        
        assert result_dict["experiment_id"] == "exp_001"
        assert result_dict["final_metrics"]["accuracy"] == 0.95
        assert len(result_dict["training_history"]) == 5
        
        # Test from_dict
        restored_result = ExperimentResult.from_dict(result_dict)
        
        assert restored_result.experiment_id == sample_result.experiment_id
        assert restored_result.final_metrics == sample_result.final_metrics
        assert restored_result.training_history == sample_result.training_history


class TestResultManager:
    """Test result manager."""
    
    def test_manager_initialization(self, temp_dir):
        """Test result manager initialization."""
        manager = ResultManager(temp_dir)
        
        assert manager.results_dir == temp_dir
        assert manager.results_db_file == temp_dir / "results_database.json"
        assert manager.comparisons_file == temp_dir / "comparisons.json"
        assert manager.analysis_cache_file == temp_dir / "analysis_cache.json"
        
        # Check initial database structure
        assert "results" in manager.results_db
        assert "metadata" in manager.results_db
    
    def test_store_and_retrieve_result(self, result_manager, sample_result):
        """Test storing and retrieving experiment results."""
        # Store result
        result_manager.store_result(sample_result)
        
        # Check that result was stored in database
        assert sample_result.experiment_id in result_manager.results_db["results"]
        
        # Check that result file was created
        result_file = result_manager.results_dir / f"{sample_result.experiment_id}_result.json"
        assert result_file.exists()
        
        # Retrieve result
        retrieved_result = result_manager.get_result(sample_result.experiment_id)
        
        assert retrieved_result is not None
        assert retrieved_result.experiment_id == sample_result.experiment_id
        assert retrieved_result.final_metrics == sample_result.final_metrics
        assert retrieved_result.convergence_analysis is not None  # Should be computed during storage
    
    def test_store_result_with_large_history(self, result_manager, sample_result):
        """Test storing result with large training history."""
        # Create large training history
        large_history = [{"step": i, "loss": 1.0 - i * 0.01} for i in range(200)]
        sample_result.training_history = large_history
        
        # Store result
        result_manager.store_result(sample_result)
        
        # Check that history was saved separately
        history_file = result_manager.results_dir / f"{sample_result.experiment_id}_history.pkl"
        assert history_file.exists()
        
        # Retrieve result and check history is loaded
        retrieved_result = result_manager.get_result(sample_result.experiment_id)
        assert len(retrieved_result.training_history) == 200
    
    def test_list_results_with_filtering(self, result_manager):
        """Test listing results with various filters."""
        # Create multiple results with different properties
        results = []
        
        for i in range(5):
            result = ExperimentResult(
                experiment_id=f"exp_{i:03d}",
                name=f"experiment_{i}",
                timestamp=(datetime.now() - timedelta(days=i)).isoformat(),
                status="completed" if i % 2 == 0 else "failed",
                config_hash=f"hash_{i}",
                hyperparameters={"lr": 0.01 * (i + 1)},
                final_metrics={"accuracy": 0.9 - i * 0.05},
                best_metrics={"accuracy": 0.95 - i * 0.05},
                training_history=[],
                duration_seconds=100.0 + i * 10,
                hardware_info={},
                reproducibility_info={}
            )
            results.append(result)
            result_manager.store_result(result)
        
        # Test listing all results
        all_results = result_manager.list_results()
        assert len(all_results) == 5
        
        # Test filtering by status
        completed_results = result_manager.list_results(status="completed")
        assert len(completed_results) == 3  # exp_000, exp_002, exp_004
        
        failed_results = result_manager.list_results(status="failed")
        assert len(failed_results) == 2  # exp_001, exp_003
        
        # Test filtering by name pattern
        filtered_results = result_manager.list_results(name_pattern="experiment_1")
        assert len(filtered_results) == 1
        assert "exp_001" in filtered_results
        
        # Test filtering by date range
        start_date = (datetime.now() - timedelta(days=2)).isoformat()
        end_date = datetime.now().isoformat()
        
        recent_results = result_manager.list_results(date_range=(start_date, end_date))
        assert len(recent_results) >= 2  # Should include recent experiments
    
    def test_convergence_analysis(self, result_manager):
        """Test convergence analysis computation."""
        # Create result with clear convergence pattern
        training_history = []
        for i in range(50):
            # Exponential decay with some noise
            loss = 2.0 * np.exp(-i * 0.1) + np.random.normal(0, 0.01)
            accuracy = 1.0 - np.exp(-i * 0.1) + np.random.normal(0, 0.01)
            training_history.append({"step": i, "loss": loss, "accuracy": accuracy})
        
        analysis = result_manager._analyze_convergence(training_history)
        
        assert "total_steps" in analysis
        assert "initial_loss" in analysis
        assert "final_loss" in analysis
        assert "best_loss" in analysis
        assert "loss_reduction" in analysis
        assert "relative_improvement" in analysis
        
        assert analysis["total_steps"] == 50
        assert analysis["initial_loss"] > analysis["final_loss"]  # Should show improvement
        assert analysis["loss_reduction"] > 0  # Should be positive
        
        # Check convergence detection
        if analysis["convergence_step"] is not None:
            assert 0 <= analysis["convergence_step"] <= 49
        
        # Check stability analysis
        if "final_stability" in analysis:
            assert "mean" in analysis["final_stability"]
            assert "std" in analysis["final_stability"]
            assert "coefficient_of_variation" in analysis["final_stability"]
    
    def test_metric_comparison(self, result_manager):
        """Test metric comparison between experiments."""
        experiment_ids = ["exp_001", "exp_002", "exp_003"]
        values = [0.95, 0.90, 0.85]
        
        comparison = result_manager._compare_metric_values(
            experiment_ids, values, "accuracy"
        )
        
        assert comparison["metric_name"] == "accuracy"
        assert comparison["experiment_values"]["exp_001"] == 0.95
        assert comparison["best_experiment"] == "exp_003"  # Assuming lower is better
        assert comparison["worst_experiment"] == "exp_001"
        
        # Check statistics
        stats = comparison["statistics"]
        assert stats["mean"] == np.mean(values)
        assert stats["min"] == np.min(values)
        assert stats["max"] == np.max(values)
        
        # Check relative differences
        rel_diffs = comparison["relative_differences"]
        assert len(rel_diffs) == 3
        assert rel_diffs["exp_003"] == 0.0  # Best experiment should have 0 relative difference
    
    def test_result_comparison(self, result_manager):
        """Test comprehensive result comparison."""
        # Create multiple results for comparison
        results = []
        
        for i in range(3):
            result = ExperimentResult(
                experiment_id=f"comp_exp_{i}",
                name=f"comparison_experiment_{i}",
                timestamp=datetime.now().isoformat(),
                status="completed",
                config_hash=f"comp_hash_{i}",
                hyperparameters={"lr": 0.01 * (i + 1), "batch_size": 32},
                final_metrics={
                    "accuracy": 0.9 + i * 0.02,
                    "loss": 0.1 - i * 0.02,
                    "f1_score": 0.85 + i * 0.03
                },
                best_metrics={
                    "accuracy": 0.92 + i * 0.02,
                    "loss": 0.08 - i * 0.02,
                    "f1_score": 0.87 + i * 0.03
                },
                training_history=[],
                duration_seconds=100.0,
                hardware_info={},
                reproducibility_info={}
            )
            results.append(result)
            result_manager.store_result(result)
        
        # Compare results
        experiment_ids = [r.experiment_id for r in results]
        comparison = result_manager.compare_results(experiment_ids)
        
        assert isinstance(comparison, ResultComparison)
        assert comparison.experiment_ids == experiment_ids
        assert comparison.comparison_type == "group"
        
        # Check metric comparisons
        assert "accuracy" in comparison.metric_comparisons
        assert "loss" in comparison.metric_comparisons
        assert "f1_score" in comparison.metric_comparisons
        
        # Check statistical tests
        assert len(comparison.statistical_tests) > 0
        
        # Check performance ranking
        assert len(comparison.performance_ranking) == 3
        
        # Check summary
        assert "total_experiments" in comparison.summary
        assert "compared_metrics" in comparison.summary
        assert "best_overall_experiment" in comparison.summary
    
    def test_performance_ranking(self, result_manager):
        """Test performance ranking creation."""
        # Create results with clear performance differences
        results = []
        
        # Best performer
        best_result = ExperimentResult(
            experiment_id="best_exp",
            name="best_experiment",
            timestamp=datetime.now().isoformat(),
            status="completed",
            config_hash="best_hash",
            hyperparameters={},
            final_metrics={"accuracy": 0.98, "loss": 0.02},  # Best values
            best_metrics={},
            training_history=[],
            duration_seconds=100.0,
            hardware_info={},
            reproducibility_info={}
        )
        
        # Worst performer
        worst_result = ExperimentResult(
            experiment_id="worst_exp",
            name="worst_experiment",
            timestamp=datetime.now().isoformat(),
            status="completed",
            config_hash="worst_hash",
            hyperparameters={},
            final_metrics={"accuracy": 0.80, "loss": 0.20},  # Worst values
            best_metrics={},
            training_history=[],
            duration_seconds=100.0,
            hardware_info={},
            reproducibility_info={}
        )
        
        results = [best_result, worst_result]
        metrics = ["accuracy", "loss"]
        
        ranking = result_manager._create_performance_ranking(results, metrics)
        
        assert len(ranking) == 2
        
        # Check that best experiment is ranked first (lower score is better)
        best_exp_id, best_score = ranking[0]
        worst_exp_id, worst_score = ranking[1]
        
        assert best_score <= worst_score  # Best should have lower or equal score
    
    def test_statistical_tests(self, result_manager):
        """Test statistical tests on results."""
        # Create results with different metric values
        results = []
        
        for i in range(4):
            result = ExperimentResult(
                experiment_id=f"stat_exp_{i}",
                name=f"statistical_experiment_{i}",
                timestamp=datetime.now().isoformat(),
                status="completed",
                config_hash=f"stat_hash_{i}",
                hyperparameters={},
                final_metrics={"accuracy": 0.85 + i * 0.03},  # Increasing accuracy
                best_metrics={},
                training_history=[],
                duration_seconds=100.0,
                hardware_info={},
                reproducibility_info={}
            )
            results.append(result)
        
        metrics = ["accuracy"]
        statistical_tests = result_manager._perform_statistical_tests(results, metrics)
        
        assert "accuracy" in statistical_tests
        
        accuracy_tests = statistical_tests["accuracy"]
        
        # Check pairwise tests
        assert "pairwise" in accuracy_tests
        pairwise_tests = accuracy_tests["pairwise"]
        
        # Should have pairwise comparisons
        assert len(pairwise_tests) > 0
        
        # Check individual pairwise test
        for comparison_key, test_result in pairwise_tests.items():
            assert "values" in test_result
            assert "difference" in test_result
            assert "relative_difference" in test_result
            assert "effect_size" in test_result
            assert "better_experiment" in test_result
    
    def test_results_report_generation(self, result_manager):
        """Test comprehensive results report generation."""
        # Create multiple results
        for i in range(3):
            result = ExperimentResult(
                experiment_id=f"report_exp_{i}",
                name=f"report_experiment_{i}",
                timestamp=datetime.now().isoformat(),
                status="completed" if i < 2 else "failed",
                config_hash=f"report_hash_{i}",
                hyperparameters={"lr": 0.01 * (i + 1)},
                final_metrics={"accuracy": 0.9 + i * 0.02} if i < 2 else {},
                best_metrics={"accuracy": 0.92 + i * 0.02} if i < 2 else {},
                training_history=[],
                duration_seconds=100.0,
                hardware_info={},
                reproducibility_info={
                    "validation_performed": True,
                    "overall_status": "PASSED" if i == 0 else "FAILED"
                }
            )
            result_manager.store_result(result)
        
        # Generate report
        report = result_manager.generate_results_report(include_comparisons=True)
        
        # Check report structure
        assert "summary" in report
        assert "experiment_details" in report
        assert "performance_analysis" in report
        assert "reproducibility_analysis" in report
        assert "comparison_analysis" in report
        
        # Check summary
        summary = report["summary"]
        assert summary["total_experiments"] == 3
        assert summary["completed_experiments"] == 2
        assert summary["failed_experiments"] == 1
        assert summary["success_rate"] == 2/3
        
        # Check performance analysis
        performance = report["performance_analysis"]
        assert "accuracy" in performance
        
        # Check reproducibility analysis
        repro = report["reproducibility_analysis"]
        assert repro["reproducible_experiments"] == 1  # Only first experiment passed
        assert repro["validation_performed"] == 3  # All had validation
    
    def test_result_export_json(self, result_manager, sample_result, temp_dir):
        """Test result export in JSON format."""
        # Store sample result
        result_manager.store_result(sample_result)
        
        # Export to JSON
        export_file = temp_dir / "exported_results.json"
        result_manager.export_results(export_file, format="json")
        
        assert export_file.exists()
        
        # Check export content
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        assert "results_db" in export_data
        assert "comparisons" in export_data
        assert "exported_at" in export_data
        
        # Verify result data is included
        assert sample_result.experiment_id in export_data["results_db"]["results"]
    
    def test_result_export_csv(self, result_manager, sample_result, temp_dir):
        """Test result export in CSV format."""
        # Store sample result
        result_manager.store_result(sample_result)
        
        # Export to CSV
        export_file = temp_dir / "exported_results.csv"
        result_manager.export_results(export_file, format="csv")
        
        assert export_file.exists()
        
        # Check CSV content
        df = pd.read_csv(export_file)
        
        assert len(df) == 1
        assert df.iloc[0]["experiment_id"] == sample_result.experiment_id
        assert df.iloc[0]["name"] == sample_result.name
        assert df.iloc[0]["status"] == sample_result.status
        
        # Check that metrics are included
        assert "final_accuracy" in df.columns
        assert "final_loss" in df.columns
        
        # Check that hyperparameters are included
        assert "param_lr" in df.columns
        assert "param_batch_size" in df.columns
    
    def test_result_export_excel(self, result_manager, sample_result, temp_dir):
        """Test result export in Excel format."""
        # Store sample result
        result_manager.store_result(sample_result)
        
        # Export to Excel
        export_file = temp_dir / "exported_results.xlsx"
        result_manager.export_results(export_file, format="excel")
        
        assert export_file.exists()
        
        # Check Excel content
        with pd.ExcelFile(export_file) as xls:
            # Check that Results sheet exists
            assert "Results" in xls.sheet_names
            
            results_df = pd.read_excel(xls, sheet_name="Results")
            assert len(results_df) == 1
            assert results_df.iloc[0]["experiment_id"] == sample_result.experiment_id
            
            # Check that Hyperparameters sheet exists
            if "Hyperparameters" in xls.sheet_names:
                hyperparams_df = pd.read_excel(xls, sheet_name="Hyperparameters")
                assert len(hyperparams_df) == 2  # lr and batch_size
    
    def test_unsupported_export_format(self, result_manager, temp_dir):
        """Test error handling for unsupported export format."""
        export_file = temp_dir / "exported_results.xml"
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            result_manager.export_results(export_file, format="xml")
    
    def test_database_persistence(self, temp_dir, sample_result):
        """Test database persistence across manager instances."""
        # Create first manager and store result
        manager1 = ResultManager(temp_dir)
        manager1.store_result(sample_result)
        
        # Create second manager (should load existing data)
        manager2 = ResultManager(temp_dir)
        
        # Should be able to retrieve result stored by first manager
        retrieved_result = manager2.get_result(sample_result.experiment_id)
        
        assert retrieved_result is not None
        assert retrieved_result.experiment_id == sample_result.experiment_id
        assert retrieved_result.final_metrics == sample_result.final_metrics
    
    def test_comparison_caching(self, result_manager):
        """Test comparison result caching."""
        # Create two results
        result1 = ExperimentResult(
            experiment_id="cache_exp_1",
            name="cache_experiment_1",
            timestamp=datetime.now().isoformat(),
            status="completed",
            config_hash="cache_hash_1",
            hyperparameters={},
            final_metrics={"accuracy": 0.9},
            best_metrics={},
            training_history=[],
            duration_seconds=100.0,
            hardware_info={},
            reproducibility_info={}
        )
        
        result2 = ExperimentResult(
            experiment_id="cache_exp_2",
            name="cache_experiment_2",
            timestamp=datetime.now().isoformat(),
            status="completed",
            config_hash="cache_hash_2",
            hyperparameters={},
            final_metrics={"accuracy": 0.85},
            best_metrics={},
            training_history=[],
            duration_seconds=100.0,
            hardware_info={},
            reproducibility_info={}
        )
        
        result_manager.store_result(result1)
        result_manager.store_result(result2)
        
        # First comparison (should be computed and cached)
        comparison1 = result_manager.compare_results([result1.experiment_id, result2.experiment_id])
        
        # Check that comparison was cached
        assert len(result_manager.comparisons) > 0
        
        # Second comparison (should use cache)
        comparison2 = result_manager.compare_results([result1.experiment_id, result2.experiment_id])
        
        # Results should be identical
        assert comparison1.similarity_score == comparison2.similarity_score
        assert comparison1.timestamp == comparison2.timestamp  # Should be same cached result


class TestResultComparison:
    """Test result comparison data structure."""
    
    def test_comparison_creation(self):
        """Test comparison object creation and serialization."""
        comparison = ResultComparison(
            experiment_ids=["exp1", "exp2"],
            comparison_type="pairwise",
            metric_comparisons={"accuracy": {"exp1": 0.9, "exp2": 0.85}},
            statistical_tests={"accuracy": {"t_test": {"p_value": 0.05}}},
            performance_ranking=[("exp1", 0.1), ("exp2", 0.2)],
            summary={"best_experiment": "exp1"},
            timestamp="2024-01-01T00:00:00"
        )
        
        # Test serialization
        comparison_dict = comparison.to_dict()
        
        assert comparison_dict["experiment_ids"] == ["exp1", "exp2"]
        assert comparison_dict["comparison_type"] == "pairwise"
        assert "metric_comparisons" in comparison_dict
        assert "statistical_tests" in comparison_dict
        assert "performance_ranking" in comparison_dict
        assert "summary" in comparison_dict
    
    def test_comparison_with_significance_matrix(self):
        """Test comparison with significance matrix."""
        significance_matrix = np.array([[0.0, 0.3], [0.3, 0.0]])
        
        comparison = ResultComparison(
            experiment_ids=["exp1", "exp2"],
            comparison_type="pairwise",
            metric_comparisons={},
            statistical_tests={},
            performance_ranking=[],
            significance_matrix=significance_matrix,
            summary={},
            timestamp="2024-01-01T00:00:00"
        )
        
        # Test serialization with numpy array
        comparison_dict = comparison.to_dict()
        
        assert "significance_matrix" in comparison_dict
        assert isinstance(comparison_dict["significance_matrix"], list)
        assert comparison_dict["significance_matrix"] == [[0.0, 0.3], [0.3, 0.0]]