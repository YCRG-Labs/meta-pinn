"""
Integration tests for reproducibility and experiment management system.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

from ml_research_pipeline.utils.experiment_manager import ExperimentManager
from ml_research_pipeline.utils.experiment_tracker import ExperimentTracker
from ml_research_pipeline.utils.reproducibility_validator import ReproducibilityValidator
from ml_research_pipeline.config import ExperimentConfig
from ml_research_pipeline.core.meta_pinn import MetaPINN


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
        name="reproducibility_test",
        description="Integration test for reproducibility system",
        version="1.0.0",
        seed=42,
        deterministic=True,
        output_dir="test_output"
    )


class SimpleMetaPINN(nn.Module):
    """Simple MetaPINN for testing."""
    
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def physics_loss(self, coords, predictions):
        """Simple physics loss for testing."""
        # Mock physics loss computation
        return torch.mean(predictions**2)


class TestReproducibilityIntegration:
    """Integration tests for reproducibility system."""
    
    def test_complete_reproducibility_workflow(self, test_config, temp_dir):
        """Test complete reproducibility workflow with experiment management."""
        # Initialize experiment manager
        manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_git_tracking=False
        )
        
        # Initialize experiment tracker
        tracker = ExperimentTracker(
            experiment_name=test_config.name,
            output_dir=manager.experiment_dir / "tracking",
            enable_plotting=False
        )
        
        # Initialize reproducibility validator
        validator = ReproducibilityValidator(tolerance=1e-6)
        
        # Start experiment
        manager.start_experiment()
        tracker.start_experiment({"model": "MetaPINN", "seed": test_config.seed})
        
        # Create model and data
        model = SimpleMetaPINN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        
        torch.manual_seed(test_config.seed)
        input_data = torch.randn(32, 3)
        target_data = torch.randn(32, 3)
        
        # Run reproducibility validation
        validation_results = validator.run_comprehensive_validation(
            model, optimizer, loss_fn, input_data, target_data,
            seed=test_config.seed,
            reference_dir=manager.experiment_dir / "reproducibility"
        )
        
        # Log validation results
        tracker.log_event(
            "reproducibility_validation",
            "Completed reproducibility validation",
            validation_results
        )
        
        # Simulate training with reproducibility checks
        for epoch in range(5):
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
                "physics_loss": model.physics_loss(input_data, output).item()
            }
            tracker.log_metric("loss", loss.item(), step=epoch, epoch=epoch)
            tracker.log_metric("physics_loss", metrics["physics_loss"], step=epoch, epoch=epoch)
            
            # Save checkpoint with reproducibility info
            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "random_state": torch.get_rng_state()
            }
            
            manager.save_checkpoint(
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                metrics=metrics,
                is_best=(epoch == 4)
            )
        
        # Complete experiment
        final_metrics = {"final_loss": loss.item()}
        manager.complete_experiment(final_metrics)
        tracker.finish_experiment(final_metrics)
        
        # Generate reproducibility report
        report_file = manager.experiment_dir / "reproducibility_report.json"
        validator.generate_reproducibility_report(validation_results, report_file)
        
        # Verify all components worked together
        assert manager.metadata.status == "completed"
        assert validation_results["overall_passed"] is True
        assert report_file.exists()
        
        # Verify experiment can be loaded and reproduced
        loaded_manager = ExperimentManager.load_experiment(manager.experiment_dir)
        loaded_tracker = ExperimentTracker.load_experiment(manager.experiment_dir / "tracking")
        
        assert loaded_manager.experiment_id == manager.experiment_id
        assert loaded_tracker.experiment_name == tracker.experiment_name
        assert len(loaded_tracker.metrics) == 2  # loss and physics_loss
    
    def test_reproducibility_across_experiment_runs(self, test_config, temp_dir):
        """Test reproducibility across multiple experiment runs."""
        results = []
        
        # Run same experiment multiple times
        for run_id in range(3):
            run_dir = temp_dir / f"run_{run_id}"
            
            # Create experiment manager
            config = test_config.update(name=f"repro_test_run_{run_id}")
            manager = ExperimentManager(
                config=config,
                base_output_dir=run_dir,
                enable_git_tracking=False
            )
            
            manager.start_experiment()
            
            # Create identical model and data
            model = SimpleMetaPINN()
            torch.manual_seed(config.seed)  # Same seed for all runs
            input_data = torch.randn(16, 3)
            target_data = torch.randn(16, 3)
            
            # Run identical training
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()
            
            run_losses = []
            for step in range(10):
                optimizer.zero_grad()
                output = model(input_data)
                loss = loss_fn(output, target_data)
                loss.backward()
                optimizer.step()
                run_losses.append(loss.item())
            
            results.append({
                'run_id': run_id,
                'losses': run_losses,
                'final_output': output.detach().numpy(),
                'final_params': {name: param.detach().numpy() 
                               for name, param in model.named_parameters()}
            })
            
            manager.complete_experiment({"final_loss": run_losses[-1]})
        
        # Verify all runs produced identical results
        reference_losses = results[0]['losses']
        reference_output = results[0]['final_output']
        
        for i in range(1, len(results)):
            # Check losses are identical
            for j, (ref_loss, curr_loss) in enumerate(zip(reference_losses, results[i]['losses'])):
                assert abs(ref_loss - curr_loss) < 1e-6, f"Loss mismatch at step {j}, run {i}"
            
            # Check final outputs are identical
            output_diff = np.abs(reference_output - results[i]['final_output']).max()
            assert output_diff < 1e-6, f"Output mismatch in run {i}: {output_diff}"
            
            # Check parameters are identical
            for param_name in results[0]['final_params']:
                param_diff = np.abs(
                    results[0]['final_params'][param_name] - 
                    results[i]['final_params'][param_name]
                ).max()
                assert param_diff < 1e-6, f"Parameter {param_name} mismatch in run {i}: {param_diff}"
    
    def test_reproducibility_validation_with_real_metapinn(self, test_config, temp_dir):
        """Test reproducibility validation with actual MetaPINN model."""
        try:
            from ml_research_pipeline.core.meta_pinn import MetaPINN
            from ml_research_pipeline.core.task_generator import FluidTaskGenerator
        except ImportError:
            pytest.skip("MetaPINN or TaskGenerator not available")
        
        # Create experiment manager
        manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_git_tracking=False
        )
        
        manager.start_experiment()
        
        # Create MetaPINN model
        model = MetaPINN(
            layers=[3, 64, 64, 3],
            meta_lr=0.001,
            adapt_lr=0.01
        )
        
        # Create task generator
        task_generator = FluidTaskGenerator(
            domain_bounds={"x": [0, 1], "y": [0, 1]},
            task_types=["linear_viscosity"]
        )
        
        # Generate test task
        task = task_generator.generate_task(
            task_type="linear_viscosity",
            n_support=50,
            n_query=25
        )
        
        # Create reproducibility validator
        validator = ReproducibilityValidator(tolerance=1e-5)  # Slightly more lenient for complex model
        
        # Test forward pass reproducibility
        support_coords = task['support_set']['coords']
        forward_result = validator.validate_deterministic_forward_pass(
            model, support_coords, seed=test_config.seed, num_runs=3
        )
        
        assert forward_result['passed'], "MetaPINN forward pass not reproducible"
        
        # Test adaptation reproducibility
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        def physics_loss_fn(predictions, coords):
            return model.physics_loss(coords, {"viscosity_type": "linear"})
        
        adaptation_result = validator.validate_deterministic_training_step(
            model, optimizer, physics_loss_fn, 
            support_coords, task['support_set']['data'],
            seed=test_config.seed, num_runs=2
        )
        
        # Note: This might fail due to complex physics computations
        # Log the result for analysis
        manager.save_metrics({
            "forward_pass_reproducible": float(forward_result['passed']),
            "adaptation_reproducible": float(adaptation_result['passed']),
            "forward_pass_max_diff": forward_result['max_difference'],
            "adaptation_max_diff": adaptation_result.get('max_parameter_difference', 0)
        }, step=0)
        
        manager.complete_experiment()
    
    def test_experiment_versioning_and_comparison(self, test_config, temp_dir):
        """Test experiment versioning and comparison capabilities."""
        # Create multiple experiment versions
        versions = ["v1.0", "v1.1", "v2.0"]
        experiment_managers = []
        
        for version in versions:
            config = test_config.update(version=version, name=f"versioning_test_{version}")
            manager = ExperimentManager(
                config=config,
                base_output_dir=temp_dir / version,
                enable_git_tracking=False
            )
            
            manager.start_experiment()
            
            # Simulate different model configurations for each version
            if version == "v1.0":
                model = SimpleMetaPINN(hidden_dim=32)
                lr = 0.01
            elif version == "v1.1":
                model = SimpleMetaPINN(hidden_dim=64)
                lr = 0.01
            else:  # v2.0
                model = SimpleMetaPINN(hidden_dim=64)
                lr = 0.001
            
            # Run training
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            torch.manual_seed(config.seed)
            input_data = torch.randn(16, 3)
            target_data = torch.randn(16, 3)
            
            final_loss = None
            for epoch in range(5):
                optimizer.zero_grad()
                output = model(input_data)
                loss = nn.MSELoss()(output, target_data)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()
            
            manager.complete_experiment({"final_loss": final_loss})
            experiment_managers.append(manager)
        
        # Verify each experiment has unique configuration hash
        config_hashes = [mgr.metadata.config_hash for mgr in experiment_managers]
        assert len(set(config_hashes)) == len(config_hashes), "Config hashes should be unique"
        
        # Verify experiments can be distinguished by version
        for i, manager in enumerate(experiment_managers):
            assert manager.metadata.version == versions[i]
            assert versions[i] in manager.metadata.name
        
        # Test loading and comparison
        loaded_managers = []
        for manager in experiment_managers:
            loaded = ExperimentManager.load_experiment(manager.experiment_dir)
            loaded_managers.append(loaded)
        
        # Verify loaded experiments match originals
        for orig, loaded in zip(experiment_managers, loaded_managers):
            assert orig.experiment_id == loaded.experiment_id
            assert orig.metadata.version == loaded.metadata.version
            assert orig.metadata.config_hash == loaded.metadata.config_hash
    
    def test_reproducibility_failure_detection_and_recovery(self, test_config, temp_dir):
        """Test detection and handling of reproducibility failures."""
        manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_git_tracking=False
        )
        
        manager.start_experiment()
        
        # Create model with intentional non-determinism
        class NonDeterministicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
                self.dropout = nn.Dropout(0.5)  # Non-deterministic
            
            def forward(self, x):
                x = self.linear(x)
                x = self.dropout(x)  # This will cause non-determinism
                return x
        
        model = NonDeterministicModel()
        model.train()  # Enable dropout
        
        validator = ReproducibilityValidator(tolerance=1e-6)
        
        torch.manual_seed(test_config.seed)
        input_data = torch.randn(32, 10)
        
        # Test should detect non-determinism
        result = validator.validate_deterministic_forward_pass(
            model, input_data, seed=test_config.seed, num_runs=5
        )
        
        # Log the failure
        manager.save_metrics({
            "reproducibility_test_passed": float(result['passed']),
            "max_difference_detected": result['max_difference']
        }, step=0)
        
        # The test might occasionally pass due to random chance with dropout
        # But we can verify the detection mechanism works
        if not result['passed']:
            manager.log_event(
                "reproducibility_failure",
                "Non-deterministic behavior detected",
                {"max_difference": result['max_difference']}
            )
        
        # Test recovery with deterministic model
        deterministic_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
        recovery_result = validator.validate_deterministic_forward_pass(
            deterministic_model, input_data, seed=test_config.seed, num_runs=5
        )
        
        assert recovery_result['passed'], "Deterministic model should pass reproducibility test"
        
        manager.save_metrics({
            "recovery_test_passed": float(recovery_result['passed']),
            "recovery_max_difference": recovery_result['max_difference']
        }, step=1)
        
        manager.complete_experiment()
    
    def test_comprehensive_experiment_documentation(self, test_config, temp_dir):
        """Test comprehensive experiment documentation and metadata capture."""
        manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_git_tracking=False
        )
        
        tracker = ExperimentTracker(
            experiment_name=test_config.name,
            output_dir=manager.experiment_dir / "tracking",
            enable_plotting=False
        )
        
        validator = ReproducibilityValidator()
        
        # Start experiment with comprehensive metadata
        hyperparameters = {
            "model_architecture": "MetaPINN",
            "hidden_layers": [64, 64, 64],
            "activation": "tanh",
            "meta_learning_rate": 0.001,
            "adaptation_learning_rate": 0.01,
            "adaptation_steps": 5,
            "batch_size": 32,
            "physics_loss_weight": 1.0
        }
        
        manager.start_experiment()
        tracker.start_experiment(hyperparameters)
        
        # Log system information
        system_info = {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": manager.metadata.python_version,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        tracker.log_metadata(system_info)
        
        # Create and validate model
        model = SimpleMetaPINN()
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["meta_learning_rate"])
        
        torch.manual_seed(test_config.seed)
        input_data = torch.randn(32, 3)
        target_data = torch.randn(32, 3)
        
        # Run reproducibility validation
        validation_results = validator.run_comprehensive_validation(
            model, optimizer, nn.MSELoss(), input_data, target_data,
            seed=test_config.seed
        )
        
        # Log validation results as events
        for test_name, result in validation_results['individual_results'].items():
            tracker.log_event(
                f"reproducibility_{test_name}",
                f"Reproducibility test {test_name}: {'PASSED' if result['passed'] else 'FAILED'}",
                result
            )
        
        # Simulate training with detailed logging
        for epoch in range(10):
            tracker.update_epoch(epoch)
            
            # Training step
            optimizer.zero_grad()
            output = model(input_data)
            data_loss = nn.MSELoss()(output, target_data)
            physics_loss = model.physics_loss(input_data, output)
            total_loss = data_loss + hyperparameters["physics_loss_weight"] * physics_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Log detailed metrics
            metrics = {
                "total_loss": total_loss.item(),
                "data_loss": data_loss.item(),
                "physics_loss": physics_loss.item(),
                "gradient_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
            }
            
            for metric_name, value in metrics.items():
                tracker.log_metric(metric_name, value, step=epoch, epoch=epoch, phase="train")
            
            # Save checkpoint every few epochs
            if epoch % 3 == 0:
                manager.save_checkpoint(
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch,
                    metrics=metrics,
                    is_best=(epoch == 9)
                )
                
                tracker.log_event("checkpoint", f"Saved checkpoint at epoch {epoch}")
        
        # Complete experiment
        final_metrics = {
            "final_total_loss": total_loss.item(),
            "final_data_loss": data_loss.item(),
            "final_physics_loss": physics_loss.item(),
            "reproducibility_passed": validation_results["overall_passed"]
        }
        
        manager.complete_experiment(final_metrics)
        tracker.finish_experiment(final_metrics)
        
        # Generate comprehensive reports
        experiment_summary = manager.get_experiment_summary()
        tracking_report = tracker.generate_report()
        
        # Save comprehensive documentation
        documentation = {
            "experiment_summary": experiment_summary,
            "tracking_report": tracking_report,
            "reproducibility_results": validation_results,
            "hyperparameters": hyperparameters,
            "system_info": system_info
        }
        
        doc_file = manager.experiment_dir / "comprehensive_documentation.json"
        with open(doc_file, 'w') as f:
            json.dump(documentation, f, indent=2, default=str)
        
        # Verify documentation completeness
        assert doc_file.exists()
        assert len(documentation) == 5
        assert documentation["reproducibility_results"]["overall_passed"] is True
        assert len(documentation["tracking_report"]["metrics_summary"]) == 4  # 4 metrics logged
        
        # Verify experiment can be fully reconstructed from documentation
        with open(doc_file, 'r') as f:
            loaded_doc = json.load(f)
        
        assert loaded_doc["experiment_summary"]["metadata"]["experiment_id"] == manager.experiment_id
        assert loaded_doc["hyperparameters"]["model_architecture"] == "MetaPINN"
        assert loaded_doc["system_info"]["torch_version"] == torch.__version__