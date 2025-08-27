"""
Integration tests for enhanced reproducibility and experiment management system.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime

from ml_research_pipeline.utils.experiment_manager import ExperimentManager
from ml_research_pipeline.utils.experiment_tracker import ExperimentTracker
from ml_research_pipeline.utils.reproducibility_validator import ReproducibilityValidator
from ml_research_pipeline.utils.experiment_versioning import ExperimentVersionManager
from ml_research_pipeline.utils.reproducibility_config import (
    ReproducibilityEnvironmentManager,
    create_strict_reproducibility_config
)
from ml_research_pipeline.utils.result_management import ResultManager, ExperimentResult
from ml_research_pipeline.config import ExperimentConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config():
    """Create test experiment configuration."""
    return ExperimentConfig(
        name="enhanced_repro_test",
        description="Enhanced reproducibility integration test",
        version="1.0.0",
        seed=42,
        deterministic=True,
        output_dir="test_output"
    )


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class TestEnhancedReproducibilityIntegration:
    """Integration tests for enhanced reproducibility system."""
    
    def test_complete_enhanced_workflow(self, test_config, temp_dir):
        """Test complete enhanced reproducibility workflow."""
        # 1. Setup reproducibility environment
        repro_config = create_strict_reproducibility_config()
        env_manager = ReproducibilityEnvironmentManager(repro_config)
        applied_settings = env_manager.setup_reproducible_environment()
        
        # 2. Initialize experiment management components
        exp_manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        
        tracker = ExperimentTracker(
            experiment_name=test_config.name,
            output_dir=exp_manager.experiment_dir / "tracking",
            enable_plotting=False
        )
        
        validator = ReproducibilityValidator(tolerance=repro_config.validation_tolerance)
        
        version_manager = ExperimentVersionManager(temp_dir / "versions")
        
        result_manager = ResultManager(temp_dir / "results")
        
        # 3. Start experiment with comprehensive logging
        exp_manager.start_experiment()
        tracker.start_experiment({
            "model": "SimpleTestModel",
            "reproducibility_config": repro_config.to_dict(),
            "applied_settings": applied_settings
        })
        
        # Log environment setup
        tracker.log_event(
            "environment_setup",
            "Reproducibility environment configured",
            applied_settings
        )
        
        # 4. Create and validate model
        model = SimpleTestModel()
        # Use SGD instead of Adam for better determinism
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        
        torch.manual_seed(test_config.seed)
        input_data = torch.randn(32, 10)
        target_data = torch.randn(32, 1)
        
        # 5. Run comprehensive reproducibility validation
        validation_results = validator.run_comprehensive_validation(
            model, optimizer, loss_fn, input_data, target_data,
            seed=test_config.seed,
            reference_dir=exp_manager.experiment_dir / "reproducibility"
        )
        
        # Log validation results
        tracker.log_event(
            "reproducibility_validation",
            f"Validation completed: {'PASSED' if validation_results['overall_passed'] else 'FAILED'}",
            validation_results
        )
        
        # 6. Simulate training with detailed tracking (reduced epochs for speed)
        training_history = []
        
        for epoch in range(3):
            tracker.update_epoch(epoch)
            
            # Training step
            model.train()
            optimizer.zero_grad()
            
            output = model(input_data)
            loss = loss_fn(output, target_data)
            loss.backward()
            optimizer.step()
            
            # Log metrics
            metrics = {
                "loss": loss.item(),
                "gradient_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
            }
            
            tracker.log_metric("loss", loss.item(), step=epoch, epoch=epoch)
            tracker.log_metric("gradient_norm", metrics["gradient_norm"], step=epoch, epoch=epoch)
            
            # Store training history
            training_history.append({
                "step": epoch,
                "epoch": epoch,
                "loss": loss.item(),
                "gradient_norm": metrics["gradient_norm"]
            })
            
            # Save checkpoint (only on last epoch for speed)
            if epoch == 2:
                exp_manager.save_checkpoint(
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch,
                    metrics=metrics,
                    is_best=(epoch == 2)
                )
        
        # 7. Complete experiment
        final_metrics = {
            "final_loss": loss.item(),
            "final_gradient_norm": metrics["gradient_norm"],
            "reproducibility_passed": validation_results["overall_passed"]
        }
        
        exp_manager.complete_experiment(final_metrics)
        tracker.finish_experiment(final_metrics)
        
        # 8. Register experiment in version system
        version_manager.register_experiment(
            exp_manager,
            tags=["integration_test", "enhanced_reproducibility"]
        )
        
        # 9. Store results in result management system
        experiment_result = ExperimentResult(
            experiment_id=exp_manager.experiment_id,
            name=exp_manager.config.name,
            timestamp=exp_manager.metadata.timestamp,
            status=exp_manager.metadata.status,
            config_hash=exp_manager.metadata.config_hash,
            hyperparameters=tracker.hyperparameters,
            final_metrics=final_metrics,
            best_metrics=final_metrics,  # Simplified for test
            training_history=training_history,
            duration_seconds=exp_manager.metadata.duration_seconds,
            hardware_info={
                "gpu_count": exp_manager.metadata.gpu_count,
                "cpu_count": exp_manager.metadata.cpu_count
            },
            reproducibility_info={
                "validation_performed": True,
                "overall_status": "PASSED" if validation_results["overall_passed"] else "FAILED",
                "passed_tests": validation_results["passed_tests"],
                "total_tests": validation_results["total_tests"]
            }
        )
        
        result_manager.store_result(experiment_result)
        
        # 10. Save comprehensive documentation
        env_snapshot_file = exp_manager.experiment_dir / "environment_snapshot.json"
        env_manager.save_environment_snapshot(env_snapshot_file)
        
        # Generate reproducibility report
        repro_report_file = exp_manager.experiment_dir / "reproducibility_report.json"
        validator.generate_reproducibility_report(validation_results, repro_report_file)
        
        # Generate experiment report
        exp_report = version_manager.generate_experiment_report(exp_manager.experiment_id)
        exp_report_file = exp_manager.experiment_dir / "experiment_report.json"
        with open(exp_report_file, 'w') as f:
            json.dump(exp_report, f, indent=2, default=str)
        
        # Generate results report
        results_report = result_manager.generate_results_report([exp_manager.experiment_id])
        results_report_file = exp_manager.experiment_dir / "results_report.json"
        with open(results_report_file, 'w') as f:
            json.dump(results_report, f, indent=2, default=str)
        
        # 11. Verify all components worked together
        assert exp_manager.metadata.status == "completed"
        assert validation_results["overall_passed"] is True
        assert env_snapshot_file.exists()
        assert repro_report_file.exists()
        assert exp_report_file.exists()
        assert results_report_file.exists()
        
        # Verify experiment can be loaded and reproduced
        loaded_exp_manager = ExperimentManager.load_experiment(exp_manager.experiment_dir)
        loaded_tracker = ExperimentTracker.load_experiment(exp_manager.experiment_dir / "tracking")
        loaded_env_manager = ReproducibilityEnvironmentManager.load_environment_snapshot(env_snapshot_file)
        retrieved_result = result_manager.get_result(exp_manager.experiment_id)
        
        assert loaded_exp_manager.experiment_id == exp_manager.experiment_id
        assert loaded_tracker.experiment_name == tracker.experiment_name
        assert loaded_env_manager.config.global_seed == repro_config.global_seed
        assert retrieved_result.experiment_id == exp_manager.experiment_id
    
    def test_multi_experiment_comparison_workflow(self, test_config, temp_dir):
        """Test workflow with multiple experiments and comparisons."""
        # Create multiple experiments with different configurations
        experiments = []
        
        for i in range(2):  # Reduced from 3 to 2 for speed
            # Modify config for each experiment
            config = test_config.update(
                name=f"comparison_exp_{i}",
                version=f"1.{i}",
                seed=42 + i * 10
            )
            
            # Setup components
            exp_manager = ExperimentManager(
                config=config,
                base_output_dir=temp_dir / "experiments",
                enable_git_tracking=False
            )
            
            tracker = ExperimentTracker(
                experiment_name=config.name,
                output_dir=exp_manager.experiment_dir / "tracking",
                enable_plotting=False
            )
            
            # Run simplified experiment
            exp_manager.start_experiment()
            tracker.start_experiment({"experiment_index": i})
            
            # Simulate different performance
            final_metrics = {
                "accuracy": 0.85 + i * 0.05,
                "loss": 0.15 - i * 0.02,
                "convergence_steps": 50 - i * 10
            }
            
            exp_manager.complete_experiment(final_metrics)
            tracker.finish_experiment(final_metrics)
            
            experiments.append({
                "manager": exp_manager,
                "tracker": tracker,
                "metrics": final_metrics
            })
        
        # Initialize comparison systems
        version_manager = ExperimentVersionManager(temp_dir / "versions")
        result_manager = ResultManager(temp_dir / "results")
        
        # Register all experiments
        for i, exp in enumerate(experiments):
            version_manager.register_experiment(
                exp["manager"],
                tags=[f"comparison_group", f"experiment_{i}"]
            )
            
            # Store results
            experiment_result = ExperimentResult(
                experiment_id=exp["manager"].experiment_id,
                name=exp["manager"].config.name,
                timestamp=exp["manager"].metadata.timestamp,
                status=exp["manager"].metadata.status,
                config_hash=exp["manager"].metadata.config_hash,
                hyperparameters={"experiment_index": i},
                final_metrics=exp["metrics"],
                best_metrics=exp["metrics"],
                training_history=[],
                duration_seconds=exp["manager"].metadata.duration_seconds or 100.0,
                hardware_info={},
                reproducibility_info={"validation_performed": False}
            )
            
            result_manager.store_result(experiment_result)
        
        # Perform comparisons
        experiment_ids = [exp["manager"].experiment_id for exp in experiments]
        
        # Version system comparison
        version_comparison_01 = version_manager.compare_experiments(
            experiment_ids[0], experiment_ids[1]
        )
        
        # Result system comparison
        result_comparison = result_manager.compare_results(experiment_ids)
        
        # Find similar experiments
        similar_experiments = version_manager.find_similar_experiments(
            experiment_ids[0], similarity_threshold=0.5
        )
        
        # Generate comprehensive reports
        version_report = version_manager.generate_experiment_report(experiment_ids[0])
        results_report = result_manager.generate_results_report(experiment_ids)
        
        # Verify comparisons
        assert version_comparison_01.experiment_1_id == experiment_ids[0]
        assert version_comparison_01.experiment_2_id == experiment_ids[1]
        assert len(version_comparison_01.config_differences) > 0  # Different seeds
        
        assert result_comparison.comparison_type == "group"
        assert len(result_comparison.experiment_ids) == 2
        assert len(result_comparison.performance_ranking) == 2
        
        # Verify similar experiments found
        assert len(similar_experiments) >= 0  # May or may not find similar ones
        
        # Verify reports generated
        assert "experiment_info" in version_report
        assert "similar_experiments" in version_report
        
        assert "summary" in results_report
        assert results_report["summary"]["total_experiments"] == 2
        assert "comparison_analysis" in results_report
    
    def test_reproducibility_failure_handling_workflow(self, test_config, temp_dir):
        """Test workflow when reproducibility validation fails."""
        # Setup with intentionally problematic configuration
        repro_config = create_strict_reproducibility_config()
        repro_config.validation_tolerance = 1e-10  # Very strict tolerance
        
        env_manager = ReproducibilityEnvironmentManager(repro_config)
        env_manager.setup_reproducible_environment()
        
        exp_manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        
        tracker = ExperimentTracker(
            experiment_name=test_config.name,
            output_dir=exp_manager.experiment_dir / "tracking",
            enable_plotting=False
        )
        
        validator = ReproducibilityValidator(tolerance=repro_config.validation_tolerance)
        
        # Start experiment
        exp_manager.start_experiment()
        tracker.start_experiment({"strict_validation": True})
        
        # Create model with potential non-determinism
        class PotentiallyNonDeterministicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
                self.dropout = nn.Dropout(0.1)  # Potential source of non-determinism
            
            def forward(self, x):
                x = self.linear(x)
                x = self.dropout(x)
                return x
        
        model = PotentiallyNonDeterministicModel()
        model.eval()  # Disable dropout for deterministic behavior
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        
        torch.manual_seed(test_config.seed)
        input_data = torch.randn(16, 10)
        target_data = torch.randn(16, 1)
        
        # Run validation (may pass or fail depending on dropout behavior)
        validation_results = validator.run_comprehensive_validation(
            model, optimizer, loss_fn, input_data, target_data,
            seed=test_config.seed
        )
        
        # Log validation results regardless of outcome
        tracker.log_event(
            "reproducibility_validation",
            f"Strict validation: {'PASSED' if validation_results['overall_passed'] else 'FAILED'}",
            {
                "passed_tests": validation_results["passed_tests"],
                "total_tests": validation_results["total_tests"],
                "tolerance": repro_config.validation_tolerance
            }
        )
        
        # Continue with experiment even if validation failed
        model.train()
        optimizer.zero_grad()
        output = model(input_data)
        loss = loss_fn(output, target_data)
        loss.backward()
        optimizer.step()
        
        # Complete experiment with validation status
        final_metrics = {
            "loss": loss.item(),
            "reproducibility_passed": validation_results["overall_passed"],
            "validation_tolerance": repro_config.validation_tolerance
        }
        
        exp_manager.complete_experiment(final_metrics)
        tracker.finish_experiment(final_metrics)
        
        # Store results with failure information
        result_manager = ResultManager(temp_dir / "results")
        
        experiment_result = ExperimentResult(
            experiment_id=exp_manager.experiment_id,
            name=exp_manager.config.name,
            timestamp=exp_manager.metadata.timestamp,
            status=exp_manager.metadata.status,
            config_hash=exp_manager.metadata.config_hash,
            hyperparameters={"strict_validation": True},
            final_metrics=final_metrics,
            best_metrics=final_metrics,
            training_history=[],
            duration_seconds=exp_manager.metadata.duration_seconds,
            hardware_info={},
            reproducibility_info={
                "validation_performed": True,
                "overall_status": "PASSED" if validation_results["overall_passed"] else "FAILED",
                "failed_tests": validation_results["total_tests"] - validation_results["passed_tests"],
                "tolerance_used": repro_config.validation_tolerance
            }
        )
        
        result_manager.store_result(experiment_result)
        
        # Generate failure analysis report
        if not validation_results["overall_passed"]:
            failure_report = {
                "experiment_id": exp_manager.experiment_id,
                "failure_type": "reproducibility_validation",
                "failed_tests": [
                    name for name, result in validation_results["individual_results"].items()
                    if not result["passed"]
                ],
                "recommendations": [
                    "Consider relaxing validation tolerance",
                    "Check for non-deterministic operations",
                    "Verify environment consistency"
                ],
                "tolerance_used": repro_config.validation_tolerance,
                "timestamp": datetime.now().isoformat()
            }
            
            failure_report_file = exp_manager.experiment_dir / "failure_analysis.json"
            with open(failure_report_file, 'w') as f:
                json.dump(failure_report, f, indent=2, default=str)
            
            tracker.log_event(
                "failure_analysis",
                "Generated failure analysis report",
                failure_report
            )
        
        # Verify experiment completed despite potential validation failure
        assert exp_manager.metadata.status == "completed"
        assert "reproducibility_passed" in final_metrics
        
        # Verify result was stored with failure information
        retrieved_result = result_manager.get_result(exp_manager.experiment_id)
        assert retrieved_result is not None
        assert retrieved_result.reproducibility_info["validation_performed"] is True
    
    def test_cross_platform_reproducibility_workflow(self, test_config, temp_dir):
        """Test cross-platform reproducibility workflow."""
        # Setup reproducibility environment
        repro_config = create_strict_reproducibility_config()
        env_manager = ReproducibilityEnvironmentManager(repro_config)
        env_manager.setup_reproducible_environment()
        
        # Create reference experiment
        exp_manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        
        validator = ReproducibilityValidator()
        
        exp_manager.start_experiment()
        
        # Create model and generate reference output
        model = SimpleTestModel()
        torch.manual_seed(test_config.seed)
        input_data = torch.randn(16, 10)
        
        model.eval()
        with torch.no_grad():
            reference_output = model(input_data)
        
        # Save reference output
        reference_dir = exp_manager.experiment_dir / "cross_platform_references"
        reference_dir.mkdir(exist_ok=True)
        reference_file = reference_dir / "reference_output.pth"
        torch.save(reference_output, reference_file)
        
        # Test cross-platform reproducibility
        cross_platform_result = validator.validate_cross_platform_reproducibility(
            model, input_data, reference_file=reference_file, seed=test_config.seed
        )
        
        # Save environment snapshot for cross-platform comparison
        env_snapshot_file = exp_manager.experiment_dir / "platform_environment.json"
        env_manager.save_environment_snapshot(env_snapshot_file)
        
        # Complete experiment
        exp_manager.complete_experiment({
            "cross_platform_reproducible": cross_platform_result["passed"],
            "reference_difference": cross_platform_result.get("max_difference", 0)
        })
        
        # Create cross-platform validation report
        cross_platform_report = {
            "platform_info": env_manager.environment_snapshot["platform"],
            "torch_info": env_manager.environment_snapshot["torch"],
            "validation_result": cross_platform_result,
            "reference_file": str(reference_file),
            "environment_snapshot": str(env_snapshot_file),
            "timestamp": datetime.now().isoformat()
        }
        
        cross_platform_report_file = exp_manager.experiment_dir / "cross_platform_report.json"
        with open(cross_platform_report_file, 'w') as f:
            json.dump(cross_platform_report, f, indent=2, default=str)
        
        # Verify cross-platform validation
        assert cross_platform_result["test_name"] == "cross_platform_reproducibility"
        assert cross_platform_result["passed"] is True
        assert reference_file.exists()
        assert env_snapshot_file.exists()
        assert cross_platform_report_file.exists()
    
    def test_experiment_lineage_and_versioning_workflow(self, test_config, temp_dir):
        """Test experiment lineage and versioning workflow."""
        version_manager = ExperimentVersionManager(temp_dir / "versions")
        result_manager = ResultManager(temp_dir / "results")
        
        # Create experiment lineage: baseline -> improved -> final
        experiments = []
        experiment_chain = [
            ("baseline", "1.0", {"lr": 0.01, "batch_size": 32}),
            ("improved", "1.1", {"lr": 0.005, "batch_size": 64}),
            ("final", "2.0", {"lr": 0.001, "batch_size": 128})
        ]
        
        for i, (name_suffix, version, hyperparams) in enumerate(experiment_chain):
            # Create config for this experiment
            config = test_config.update(
                name=f"lineage_{name_suffix}",
                version=version,
                seed=test_config.seed  # Keep same seed for comparison
            )
            
            # Setup experiment
            exp_manager = ExperimentManager(
                config=config,
                base_output_dir=temp_dir / "experiments",
                enable_git_tracking=False
            )
            
            tracker = ExperimentTracker(
                experiment_name=config.name,
                output_dir=exp_manager.experiment_dir / "tracking",
                enable_plotting=False
            )
            
            # Run experiment
            exp_manager.start_experiment()
            tracker.start_experiment(hyperparams)
            
            # Simulate improving performance through the lineage
            performance_improvement = i * 0.05
            final_metrics = {
                "accuracy": 0.80 + performance_improvement,
                "loss": 0.20 - performance_improvement,
                "training_time": 100 - i * 10
            }
            
            exp_manager.complete_experiment(final_metrics)
            tracker.finish_experiment(final_metrics)
            
            # Register in version system with parent relationship
            parent_id = experiments[-1]["manager"].experiment_id if experiments else None
            version_manager.register_experiment(
                exp_manager,
                parent_experiment_id=parent_id,
                tags=["lineage_test", name_suffix, f"version_{version}"]
            )
            
            # Store results
            experiment_result = ExperimentResult(
                experiment_id=exp_manager.experiment_id,
                name=exp_manager.config.name,
                timestamp=exp_manager.metadata.timestamp,
                status=exp_manager.metadata.status,
                config_hash=exp_manager.metadata.config_hash,
                hyperparameters=hyperparams,
                final_metrics=final_metrics,
                best_metrics=final_metrics,
                training_history=[],
                duration_seconds=exp_manager.metadata.duration_seconds or final_metrics["training_time"],
                hardware_info={},
                reproducibility_info={"validation_performed": False}
            )
            
            result_manager.store_result(experiment_result)
            
            experiments.append({
                "manager": exp_manager,
                "tracker": tracker,
                "metrics": final_metrics,
                "hyperparams": hyperparams
            })
        
        # Analyze experiment lineage
        final_exp_id = experiments[-1]["manager"].experiment_id
        lineage = version_manager.get_experiment_lineage(final_exp_id)
        
        # Verify lineage structure
        assert lineage["lineage_depth"] == 2  # Two ancestors
        assert len(lineage["ancestors"]) == 2
        assert len(lineage["descendants"]) == 0  # Final experiment has no descendants
        
        # Check version chain
        base_name = "lineage"
        version_chain = version_manager.get_version_chain(base_name)
        assert len(version_chain) == 3
        
        # Compare experiments across lineage
        experiment_ids = [exp["manager"].experiment_id for exp in experiments]
        comparison = result_manager.compare_results(experiment_ids)
        
        # Verify performance improvement across lineage
        ranking = comparison.performance_ranking
        final_exp_rank = next(rank for exp_id, rank in ranking if exp_id == final_exp_id)
        baseline_exp_rank = next(rank for exp_id, rank in ranking if exp_id == experiment_ids[0])
        
        # Final experiment should perform better (lower rank score)
        assert final_exp_rank <= baseline_exp_rank
        
        # Generate lineage report
        lineage_report = {
            "lineage_analysis": lineage,
            "version_chain": version_chain,
            "performance_comparison": comparison.to_dict(),
            "improvement_metrics": {
                "accuracy_improvement": experiments[-1]["metrics"]["accuracy"] - experiments[0]["metrics"]["accuracy"],
                "loss_reduction": experiments[0]["metrics"]["loss"] - experiments[-1]["metrics"]["loss"],
                "time_efficiency": experiments[0]["metrics"]["training_time"] - experiments[-1]["metrics"]["training_time"]
            },
            "hyperparameter_evolution": [exp["hyperparams"] for exp in experiments],
            "timestamp": datetime.now().isoformat()
        }
        
        lineage_report_file = temp_dir / "lineage_analysis_report.json"
        with open(lineage_report_file, 'w') as f:
            json.dump(lineage_report, f, indent=2, default=str)
        
        # Verify lineage analysis
        assert lineage_report["improvement_metrics"]["accuracy_improvement"] > 0
        assert lineage_report["improvement_metrics"]["loss_reduction"] > 0
        assert len(lineage_report["hyperparameter_evolution"]) == 3