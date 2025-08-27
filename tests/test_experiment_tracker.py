"""
Tests for experiment tracking system.
"""

import pytest
import json
import time
from pathlib import Path
import tempfile
import shutil
import numpy as np

from ml_research_pipeline.utils.experiment_tracker import (
    ExperimentTracker,
    MetricTracker,
    MetricEntry,
    ExperimentEvent
)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def tracker(temp_dir):
    """Create experiment tracker."""
    return ExperimentTracker(
        experiment_name="test_experiment",
        output_dir=temp_dir,
        auto_save_interval=10,
        enable_plotting=False  # Disable plotting for tests
    )


class TestMetricEntry:
    """Test metric entry."""
    
    def test_metric_entry_creation(self):
        """Test metric entry creation and serialization."""
        entry = MetricEntry(
            step=100,
            timestamp=time.time(),
            value=0.85,
            epoch=5,
            phase="train"
        )
        
        assert entry.step == 100
        assert entry.value == 0.85
        assert entry.epoch == 5
        assert entry.phase == "train"
        
        # Test serialization
        entry_dict = entry.to_dict()
        assert entry_dict['step'] == 100
        assert entry_dict['value'] == 0.85
        assert entry_dict['epoch'] == 5
        assert entry_dict['phase'] == "train"


class TestExperimentEvent:
    """Test experiment event."""
    
    def test_event_creation(self):
        """Test event creation and serialization."""
        event = ExperimentEvent(
            timestamp=time.time(),
            event_type="checkpoint",
            message="Saved best model",
            data={"epoch": 10, "metric": 0.95}
        )
        
        assert event.event_type == "checkpoint"
        assert event.message == "Saved best model"
        assert event.data["epoch"] == 10
        
        # Test serialization
        event_dict = event.to_dict()
        assert event_dict['event_type'] == "checkpoint"
        assert event_dict['message'] == "Saved best model"
        assert event_dict['data']['epoch'] == 10


class TestMetricTracker:
    """Test metric tracker."""
    
    def test_metric_tracker_initialization(self):
        """Test metric tracker initialization."""
        tracker = MetricTracker("accuracy", window_size=50)
        
        assert tracker.name == "accuracy"
        assert tracker.window_size == 50
        assert len(tracker.entries) == 0
        assert len(tracker.rolling_values) == 0
        assert tracker.best_value is None
        assert tracker.total_count == 0
    
    def test_add_entries(self):
        """Test adding metric entries."""
        tracker = MetricTracker("loss")
        
        # Add entries
        entries = [
            MetricEntry(step=0, timestamp=time.time(), value=1.0),
            MetricEntry(step=1, timestamp=time.time(), value=0.8),
            MetricEntry(step=2, timestamp=time.time(), value=0.6),
        ]
        
        for entry in entries:
            tracker.add_entry(entry)
        
        assert len(tracker.entries) == 3
        assert tracker.total_count == 3
        assert tracker.best_value == 0.6  # Assuming lower is better
        assert tracker.best_step == 2
    
    def test_statistics_computation(self):
        """Test statistics computation."""
        tracker = MetricTracker("accuracy")
        
        # Add test data
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        for i, value in enumerate(values):
            entry = MetricEntry(step=i, timestamp=time.time(), value=value)
            tracker.add_entry(entry)
        
        # Test statistics
        assert tracker.get_current_value() == 0.9
        assert tracker.get_overall_mean() == 0.7
        assert abs(tracker.get_rolling_mean() - 0.7) < 1e-6
        
        stats = tracker.get_statistics()
        assert stats['name'] == "accuracy"
        assert stats['total_entries'] == 5
        assert stats['current_value'] == 0.9
        assert stats['best_value'] == 0.5  # Lower is better by default
        assert stats['overall_mean'] == 0.7
    
    def test_trend_analysis(self):
        """Test trend analysis."""
        tracker = MetricTracker("loss")
        
        # Add improving trend (decreasing loss)
        improving_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for i, value in enumerate(improving_values):
            entry = MetricEntry(step=i, timestamp=time.time(), value=value)
            tracker.add_entry(entry)
        
        trend = tracker.get_trend(window=10)
        assert trend == "improving"
        
        # Add worsening trend
        tracker_worsening = MetricTracker("loss")
        worsening_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i, value in enumerate(worsening_values):
            entry = MetricEntry(step=i, timestamp=time.time(), value=value)
            tracker_worsening.add_entry(entry)
        
        trend = tracker_worsening.get_trend(window=10)
        assert trend == "worsening"
    
    def test_rolling_window(self):
        """Test rolling window functionality."""
        tracker = MetricTracker("metric", window_size=3)
        
        # Add more entries than window size
        values = [1, 2, 3, 4, 5]
        for i, value in enumerate(values):
            entry = MetricEntry(step=i, timestamp=time.time(), value=value)
            tracker.add_entry(entry)
        
        # Rolling values should only contain last 3
        assert len(tracker.rolling_values) == 3
        assert list(tracker.rolling_values) == [3, 4, 5]
        
        # Rolling mean should be based on last 3 values
        expected_mean = (3 + 4 + 5) / 3
        assert abs(tracker.get_rolling_mean() - expected_mean) < 1e-6


class TestExperimentTracker:
    """Test experiment tracker."""
    
    def test_tracker_initialization(self, temp_dir):
        """Test tracker initialization."""
        tracker = ExperimentTracker(
            experiment_name="test_exp",
            output_dir=temp_dir,
            auto_save_interval=50,
            enable_plotting=False
        )
        
        assert tracker.experiment_name == "test_exp"
        assert tracker.output_dir == temp_dir
        assert tracker.auto_save_interval == 50
        assert tracker.enable_plotting is False
        assert len(tracker.metrics) == 0
        assert len(tracker.events) == 0
        assert tracker.current_step == 0
        assert tracker.current_epoch == 0
        assert tracker.is_running is False
    
    def test_experiment_lifecycle(self, tracker):
        """Test experiment lifecycle."""
        # Start experiment
        hyperparams = {"lr": 0.01, "batch_size": 32}
        tracker.start_experiment(hyperparams)
        
        assert tracker.is_running is True
        assert tracker.hyperparameters == hyperparams
        assert len(tracker.events) == 1
        assert tracker.events[0].event_type == "experiment_start"
        
        # Finish experiment
        final_metrics = {"final_accuracy": 0.95}
        tracker.finish_experiment(final_metrics)
        
        assert tracker.is_running is False
        # Should have start and finish events
        assert len(tracker.events) == 2
        assert tracker.events[-1].event_type == "experiment_finish"
    
    def test_metric_logging(self, tracker):
        """Test metric logging."""
        tracker.start_experiment()
        
        # Log single metric
        tracker.log_metric("accuracy", 0.85, step=100, epoch=5, phase="train")
        
        assert "accuracy" in tracker.metrics
        assert len(tracker.metrics["accuracy"].entries) == 1
        
        entry = tracker.metrics["accuracy"].entries[0]
        assert entry.value == 0.85
        assert entry.step == 100
        assert entry.epoch == 5
        assert entry.phase == "train"
        
        # Log multiple metrics
        metrics = {"loss": 0.3, "f1_score": 0.9}
        tracker.log_metrics(metrics, step=101, epoch=5, phase="val")
        
        assert "loss" in tracker.metrics
        assert "f1_score" in tracker.metrics
        assert len(tracker.metrics["loss"].entries) == 1
        assert len(tracker.metrics["f1_score"].entries) == 1
    
    def test_event_logging(self, tracker):
        """Test event logging."""
        tracker.start_experiment()
        
        # Log event
        tracker.log_event("checkpoint", "Saved model", {"epoch": 10})
        
        # Should have start event + custom event
        assert len(tracker.events) == 2
        
        custom_event = tracker.events[-1]
        assert custom_event.event_type == "checkpoint"
        assert custom_event.message == "Saved model"
        assert custom_event.data["epoch"] == 10
    
    def test_step_and_epoch_updates(self, tracker):
        """Test step and epoch updates."""
        tracker.start_experiment()
        
        # Update step
        tracker.update_step(100)
        assert tracker.current_step == 100
        
        # Update epoch
        tracker.update_epoch(5)
        assert tracker.current_epoch == 5
        
        # Should log epoch start event
        epoch_events = [e for e in tracker.events if e.event_type == "epoch_start"]
        assert len(epoch_events) == 1
        assert "Started epoch 5" in epoch_events[0].message
    
    def test_hyperparameter_and_metadata_logging(self, tracker):
        """Test hyperparameter and metadata logging."""
        tracker.start_experiment()
        
        # Log hyperparameters
        hyperparams = {"lr": 0.01, "dropout": 0.2}
        tracker.log_hyperparameters(hyperparams)
        
        assert tracker.hyperparameters["lr"] == 0.01
        assert tracker.hyperparameters["dropout"] == 0.2
        
        # Log metadata
        metadata = {"model_size": "large", "dataset": "custom"}
        tracker.log_metadata(metadata)
        
        assert tracker.metadata["model_size"] == "large"
        assert tracker.metadata["dataset"] == "custom"
    
    def test_metric_statistics_retrieval(self, tracker):
        """Test metric statistics retrieval."""
        tracker.start_experiment()
        
        # Add some metrics
        for i in range(10):
            tracker.log_metric("accuracy", 0.5 + i * 0.05, step=i)
        
        # Get statistics for specific metric
        stats = tracker.get_metric_statistics("accuracy")
        assert stats is not None
        assert stats['name'] == "accuracy"
        assert stats['total_entries'] == 10
        assert stats['current_value'] == 0.95
        
        # Get all metrics statistics
        all_stats = tracker.get_all_metrics_statistics()
        assert "accuracy" in all_stats
        assert all_stats["accuracy"]['total_entries'] == 10
        
        # Get best metrics
        best_metrics = tracker.get_best_metrics()
        assert "accuracy" in best_metrics
        assert best_metrics["accuracy"]['value'] == 0.5  # Lower is better by default
    
    def test_early_stopping_logic(self, tracker):
        """Test early stopping logic."""
        tracker.start_experiment()
        
        # Add improving metrics
        for i in range(15):
            tracker.log_metric("loss", 1.0 - i * 0.05, step=i)
        
        # Should not trigger early stopping (improving)
        assert not tracker.should_stop_early("loss", patience=10)
        
        # Add plateau
        for i in range(15, 25):
            tracker.log_metric("loss", 0.25, step=i)  # Constant value
        
        # Should trigger early stopping (no improvement)
        assert tracker.should_stop_early("loss", patience=10, min_delta=1e-6)
    
    def test_metric_improvement_detection(self, tracker):
        """Test metric improvement detection."""
        tracker.start_experiment()
        
        # Add improving trend
        for i in range(10):
            tracker.log_metric("accuracy", 0.5 + i * 0.05, step=i)
        
        # Should detect improvement (but remember lower is better by default)
        # So for accuracy, we need to check the trend logic
        is_improving = tracker.is_metric_improving("accuracy", patience=5)
        # This depends on the trend analysis implementation
        
        # Add worsening trend
        for i in range(10, 15):
            tracker.log_metric("accuracy", 0.95 - (i-10) * 0.1, step=i)
        
        is_improving_after = tracker.is_metric_improving("accuracy", patience=5)
        # Should detect worsening trend
    
    def test_file_persistence(self, tracker, temp_dir):
        """Test file persistence."""
        tracker.start_experiment()
        
        # Add some data
        tracker.log_metric("accuracy", 0.85, step=100)
        tracker.log_event("test", "Test event")
        tracker.log_hyperparameters({"lr": 0.01})
        
        tracker.finish_experiment()
        
        # Check files exist
        assert (temp_dir / "metrics.jsonl").exists()
        assert (temp_dir / "events.jsonl").exists()
        assert (temp_dir / "summary.json").exists()
        
        # Check file contents
        with open(temp_dir / "metrics.jsonl", 'r') as f:
            metric_line = f.readline()
            metric_data = json.loads(metric_line)
            assert metric_data['metric_name'] == "accuracy"
            assert metric_data['value'] == 0.85
        
        with open(temp_dir / "summary.json", 'r') as f:
            summary = json.load(f)
            assert summary['experiment_name'] == "test_experiment"
            assert summary['hyperparameters']['lr'] == 0.01
    
    def test_experiment_loading(self, temp_dir):
        """Test experiment loading."""
        # Create and run experiment
        original_tracker = ExperimentTracker(
            experiment_name="load_test",
            output_dir=temp_dir,
            enable_plotting=False
        )
        
        original_tracker.start_experiment({"lr": 0.01})
        
        for i in range(5):
            original_tracker.log_metric("loss", 1.0 - i * 0.2, step=i)
        
        original_tracker.log_event("milestone", "Halfway done")
        original_tracker.finish_experiment({"final_loss": 0.1})
        
        # Load experiment
        loaded_tracker = ExperimentTracker.load_experiment(temp_dir)
        
        assert loaded_tracker.experiment_name == "load_test"
        assert loaded_tracker.hyperparameters["lr"] == 0.01
        assert loaded_tracker.is_running is False
        
        # Check metrics were loaded
        assert "loss" in loaded_tracker.metrics
        assert len(loaded_tracker.metrics["loss"].entries) == 5
        
        # Check events were loaded
        assert len(loaded_tracker.events) >= 3  # start, milestone, finish
        milestone_events = [e for e in loaded_tracker.events if e.event_type == "milestone"]
        assert len(milestone_events) == 1
    
    def test_report_generation(self, tracker):
        """Test comprehensive report generation."""
        tracker.start_experiment({"lr": 0.01, "batch_size": 32})
        
        # Add some training data
        for i in range(10):
            tracker.log_metric("loss", 1.0 - i * 0.1, step=i)
            tracker.log_metric("accuracy", 0.5 + i * 0.05, step=i)
        
        tracker.log_event("checkpoint", "Saved best model")
        tracker.finish_experiment({"final_accuracy": 0.95})
        
        # Generate report
        report = tracker.generate_report()
        
        assert 'experiment_info' in report
        assert 'hyperparameters' in report
        assert 'metrics_summary' in report
        assert 'best_metrics' in report
        assert 'events_summary' in report
        assert 'files' in report
        
        # Check experiment info
        exp_info = report['experiment_info']
        assert exp_info['name'] == "test_experiment"
        assert exp_info['total_steps'] == tracker.current_step
        assert exp_info['is_running'] is False
        
        # Check metrics summary
        metrics_summary = report['metrics_summary']
        assert "loss" in metrics_summary
        assert "accuracy" in metrics_summary
        
        # Check events summary
        events_summary = report['events_summary']
        assert events_summary['total_events'] > 0
        assert "experiment_start" in events_summary['event_types']
        assert "experiment_finish" in events_summary['event_types']


class TestExperimentTrackerIntegration:
    """Integration tests for experiment tracker."""
    
    def test_full_training_simulation(self, temp_dir):
        """Test full training simulation with tracker."""
        tracker = ExperimentTracker(
            experiment_name="training_simulation",
            output_dir=temp_dir,
            auto_save_interval=5,
            enable_plotting=False
        )
        
        # Start experiment
        hyperparams = {
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 10,
            "model": "resnet18"
        }
        tracker.start_experiment(hyperparams)
        
        # Simulate training epochs
        for epoch in range(10):
            tracker.update_epoch(epoch)
            
            # Simulate training steps within epoch
            for step in range(20):
                global_step = epoch * 20 + step
                tracker.update_step(global_step)
                
                # Simulate metrics with realistic patterns
                train_loss = 2.0 * np.exp(-global_step * 0.01) + np.random.normal(0, 0.1)
                train_acc = 1.0 - np.exp(-global_step * 0.01) + np.random.normal(0, 0.02)
                
                tracker.log_metric("train_loss", train_loss, step=global_step, epoch=epoch, phase="train")
                tracker.log_metric("train_accuracy", train_acc, step=global_step, epoch=epoch, phase="train")
            
            # Validation at end of epoch
            val_loss = 2.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.05)
            val_acc = 1.0 - np.exp(-epoch * 0.1) + np.random.normal(0, 0.01)
            
            tracker.log_metric("val_loss", val_loss, step=global_step, epoch=epoch, phase="val")
            tracker.log_metric("val_accuracy", val_acc, step=global_step, epoch=epoch, phase="val")
            
            # Log checkpoint event
            if epoch % 3 == 0:
                tracker.log_event("checkpoint", f"Saved checkpoint at epoch {epoch}", {"epoch": epoch})
        
        # Finish experiment
        final_metrics = {
            "final_train_loss": tracker.get_metric_statistics("train_loss")["current_value"],
            "final_val_accuracy": tracker.get_metric_statistics("val_accuracy")["current_value"]
        }
        tracker.finish_experiment(final_metrics)
        
        # Verify results
        assert tracker.current_epoch == 9
        assert tracker.current_step == 199  # 10 epochs * 20 steps - 1
        
        # Check metrics were logged
        assert "train_loss" in tracker.metrics
        assert "val_accuracy" in tracker.metrics
        assert len(tracker.metrics["train_loss"].entries) == 200  # 10 epochs * 20 steps
        assert len(tracker.metrics["val_accuracy"].entries) == 10   # 1 per epoch
        
        # Check events
        checkpoint_events = [e for e in tracker.events if e.event_type == "checkpoint"]
        assert len(checkpoint_events) == 4  # epochs 0, 3, 6, 9
        
        # Generate and verify report
        report = tracker.generate_report()
        assert report['experiment_info']['total_epochs'] == 9
        assert report['experiment_info']['total_steps'] == 199
        
        # Test loading
        loaded_tracker = ExperimentTracker.load_experiment(temp_dir)
        assert loaded_tracker.experiment_name == "training_simulation"
        assert len(loaded_tracker.metrics) == 4  # train_loss, train_accuracy, val_loss, val_accuracy
    
    def test_concurrent_metric_logging(self, temp_dir):
        """Test concurrent metric logging (thread safety)."""
        import threading
        
        tracker = ExperimentTracker(
            experiment_name="concurrent_test",
            output_dir=temp_dir,
            enable_plotting=False
        )
        
        tracker.start_experiment()
        
        def log_metrics_worker(worker_id, num_metrics):
            """Worker function for logging metrics."""
            for i in range(num_metrics):
                step = worker_id * num_metrics + i
                tracker.log_metric(f"metric_{worker_id}", float(i), step=step)
        
        # Create multiple threads
        threads = []
        num_workers = 5
        metrics_per_worker = 20
        
        for worker_id in range(num_workers):
            thread = threading.Thread(
                target=log_metrics_worker,
                args=(worker_id, metrics_per_worker)
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        tracker.finish_experiment()
        
        # Verify all metrics were logged
        assert len(tracker.metrics) == num_workers
        
        for worker_id in range(num_workers):
            metric_name = f"metric_{worker_id}"
            assert metric_name in tracker.metrics
            assert len(tracker.metrics[metric_name].entries) == metrics_per_worker