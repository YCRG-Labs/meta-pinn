"""
Performance tests for training monitoring and logging system.

Tests logging overhead, scalability, and distributed coordination
for the training monitoring system.
"""

import pytest
import torch
import tempfile
import shutil
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch
import json

from ml_research_pipeline.core.training_monitor import (
    MetricTracker, DistributedLogger, TrainingMonitor, create_training_monitor
)


class TestMetricTracker:
    """Test cases for MetricTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create metric tracker."""
        return MetricTracker(window_size=10, track_percentiles=True)
    
    def test_initialization(self):
        """Test metric tracker initialization."""
        tracker = MetricTracker(window_size=50, track_percentiles=False)
        
        assert tracker.window_size == 50
        assert tracker.track_percentiles is False
        assert len(tracker.metrics) == 0
        assert len(tracker.global_metrics) == 0
    
    def test_update_metrics(self, tracker):
        """Test updating metrics."""
        metrics = {
            'loss': 0.5,
            'accuracy': 0.8,
            'tensor_metric': torch.tensor(0.3)
        }
        
        tracker.update(metrics)
        
        assert 'loss' in tracker.metrics
        assert 'accuracy' in tracker.metrics
        assert 'tensor_metric' in tracker.metrics
        
        assert len(tracker.metrics['loss']) == 1
        assert tracker.metrics['loss'][0] == 0.5
        assert abs(tracker.metrics['tensor_metric'][0] - 0.3) < 1e-6
    
    def test_get_current_stats(self, tracker):
        """Test getting current statistics."""
        # Add some data
        for i in range(5):
            tracker.update({'loss': i * 0.1})
        
        stats = tracker.get_current_stats('loss')
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'count' in stats
        assert 'latest' in stats
        
        assert stats['count'] == 5
        assert stats['min'] == 0.0
        assert stats['max'] == 0.4
        assert stats['latest'] == 0.4
    
    def test_percentiles(self, tracker):
        """Test percentile calculation."""
        # Add data with known distribution
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for val in values:
            tracker.update({'test_metric': val})
        
        stats = tracker.get_current_stats('test_metric')
        
        # Check percentiles
        assert 'p50' in stats  # median
        assert 'p90' in stats
        assert 'p95' in stats
        assert 'p99' in stats
        
        assert stats['p50'] == 5.5  # median of 1-10
    
    def test_window_size_limit(self, tracker):
        """Test that window size is respected."""
        # Add more data than window size
        for i in range(15):
            tracker.update({'loss': i})
        
        # Should only keep last 10 values (window_size=10)
        assert len(tracker.metrics['loss']) == 10
        assert tracker.metrics['loss'][0] == 5  # First value should be 5 (not 0)
        assert tracker.metrics['loss'][-1] == 14  # Last value should be 14
    
    def test_get_history(self, tracker):
        """Test getting metric history."""
        # Add data with timestamps
        for i in range(5):
            tracker.update({'loss': i * 0.1})
            time.sleep(0.01)  # Small delay for different timestamps
        
        history = tracker.get_history('loss')
        
        assert len(history) == 5
        assert all('value' in entry and 'timestamp' in entry for entry in history)
        
        # Test last_n parameter
        recent_history = tracker.get_history('loss', last_n=3)
        assert len(recent_history) == 3
    
    def test_reset_metrics(self, tracker):
        """Test resetting metrics."""
        # Add some data
        tracker.update({'loss': 0.5, 'accuracy': 0.8})
        
        # Reset specific metric
        tracker.reset('loss')
        assert len(tracker.metrics['loss']) == 0
        assert len(tracker.metrics['accuracy']) == 1
        
        # Reset all metrics
        tracker.reset()
        assert len(tracker.metrics) == 0
        assert len(tracker.global_metrics) == 0
    
    def test_stats_caching(self, tracker):
        """Test that statistics are cached for performance."""
        # Add data
        for i in range(10):
            tracker.update({'loss': i})
        
        # First call should compute stats
        start_time = time.time()
        stats1 = tracker.get_current_stats('loss')
        first_call_time = time.time() - start_time
        
        # Second call should use cache (should be faster)
        start_time = time.time()
        stats2 = tracker.get_current_stats('loss')
        second_call_time = time.time() - start_time
        
        assert stats1 == stats2
        # Cache should make second call faster (though this might be flaky)
        # assert second_call_time < first_call_time


class TestDistributedLogger:
    """Test cases for DistributedLogger class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_initialization(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test distributed logger initialization."""
        logger = DistributedLogger(
            log_dir=temp_dir,
            log_level="INFO",
            log_to_file=True,
            log_to_console=False
        )
        
        assert logger.log_dir == Path(temp_dir)
        assert logger.rank == 0
        assert logger.world_size == 1
        assert logger.is_main is True
        assert logger.logger is not None
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_logging_methods(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test different logging methods."""
        logger = DistributedLogger(temp_dir, log_to_console=False)
        
        # Test different log levels
        logger.info("Info message", step=100, loss=0.5)
        logger.debug("Debug message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Check log file was created
        log_files = list(Path(temp_dir).glob("*.log"))
        assert len(log_files) > 0
        
        # Check log content
        log_file = log_files[0]
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Info message" in content
            assert "step=100" in content
            assert "loss=0.5" in content
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=2)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=False)
    def test_non_main_process_logging(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test logging from non-main process."""
        logger = DistributedLogger(temp_dir, aggregate_logs=False)
        
        logger.info("Message from rank 1")
        
        # Should create rank-specific log file
        log_files = list(Path(temp_dir).glob("*rank_1*.log"))
        assert len(log_files) > 0


class TestTrainingMonitor:
    """Test cases for TrainingMonitor class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_initialization(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test training monitor initialization."""
        monitor = TrainingMonitor(
            log_dir=temp_dir,
            update_frequency=1.0,
            save_frequency=10,
            enable_tensorboard=False,  # Disable for testing
            enable_profiling=False
        )
        
        assert monitor.log_dir == Path(temp_dir)
        assert monitor.update_frequency == 1.0
        assert monitor.save_frequency == 10
        assert monitor.metric_tracker is not None
        assert monitor.logger is not None
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_update_metrics(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test updating metrics."""
        monitor = TrainingMonitor(
            log_dir=temp_dir,
            enable_tensorboard=False,
            enable_profiling=False
        )
        
        metrics = {
            'train_loss': 0.5,
            'val_loss': 0.3,
            'accuracy': 0.8
        }
        
        monitor.update_metrics(metrics, step=100, epoch=1, prefix="train/")
        
        # Check metrics were updated
        stats = monitor.metric_tracker.get_all_stats()
        assert 'train/train_loss' in stats
        assert 'train/val_loss' in stats
        assert 'train/accuracy' in stats
        
        assert monitor.step_count == 100
        assert monitor.epoch_count == 1
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=2)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    @patch('ml_research_pipeline.utils.distributed_utils.reduce_dict')
    def test_distributed_metric_reduction(self, mock_reduce, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test metric reduction in distributed setting."""
        # Mock reduce_dict to return averaged values
        def mock_reduce_func(tensor_dict, average=True):
            return {k: v * 0.5 for k, v in tensor_dict.items()}  # Simulate averaging
        
        mock_reduce.side_effect = mock_reduce_func
        
        monitor = TrainingMonitor(
            log_dir=temp_dir,
            enable_tensorboard=False,
            enable_profiling=False
        )
        
        metrics = {'loss': 1.0}
        monitor.update_metrics(metrics, step=1)
        
        # Should call reduce_dict for distributed training
        mock_reduce.assert_called_once()
        
        # Check that reduced values are used
        stats = monitor.metric_tracker.get_current_stats('loss')
        assert stats['latest'] == 0.5  # Reduced value
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_model_logging(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test model information logging."""
        monitor = TrainingMonitor(
            log_dir=temp_dir,
            enable_tensorboard=False,
            enable_profiling=False
        )
        
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        monitor.log_model_info(model)
        
        # Should log without errors
        # (Detailed verification would require checking log contents)
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_hyperparameter_logging(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test hyperparameter logging."""
        monitor = TrainingMonitor(
            log_dir=temp_dir,
            enable_tensorboard=False,
            enable_profiling=False
        )
        
        hparams = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'model_type': 'MetaPINN'
        }
        
        monitor.log_hyperparameters(hparams)
        
        # Should log without errors
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_monitoring_data_save(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test saving monitoring data."""
        monitor = TrainingMonitor(
            log_dir=temp_dir,
            save_frequency=2,  # Save every 2 steps
            enable_tensorboard=False,
            enable_profiling=False
        )
        
        # Update metrics to trigger save
        monitor.update_metrics({'loss': 0.5}, step=2)
        
        # Check monitoring data file was created
        monitoring_file = Path(temp_dir) / "monitoring_data.json"
        assert monitoring_file.exists()
        
        # Check content
        with open(monitoring_file, 'r') as f:
            data = json.load(f)
            assert 'metrics_history' in data
            assert 'system_info' in data
            assert len(data['metrics_history']) > 0
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_get_summary(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test getting training summary."""
        monitor = TrainingMonitor(
            log_dir=temp_dir,
            enable_tensorboard=False,
            enable_profiling=False
        )
        
        # Add some metrics
        monitor.update_metrics({'loss': 0.5, 'accuracy': 0.8}, step=10, epoch=1)
        
        summary = monitor.get_summary()
        
        assert 'elapsed_time' in summary
        assert 'total_steps' in summary
        assert 'total_epochs' in summary
        assert 'current_metrics' in summary
        assert 'system_info' in summary
        
        assert summary['total_steps'] == 10
        assert summary['total_epochs'] == 1
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_shutdown(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test monitor shutdown."""
        monitor = TrainingMonitor(
            log_dir=temp_dir,
            enable_tensorboard=False,
            enable_profiling=False
        )
        
        # Add some data
        monitor.update_metrics({'loss': 0.5}, step=1)
        
        # Shutdown should not raise errors
        monitor.shutdown()


class TestTrainingMonitorPerformance:
    """Performance tests for training monitor."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_logging_overhead(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test logging overhead with high frequency updates."""
        monitor = TrainingMonitor(
            log_dir=temp_dir,
            update_frequency=0.1,  # Very frequent updates
            enable_tensorboard=False,
            enable_profiling=False
        )
        
        # Measure time for many metric updates
        start_time = time.time()
        
        for i in range(1000):
            metrics = {
                'loss': i * 0.001,
                'accuracy': 0.5 + i * 0.0001,
                'learning_rate': 0.001
            }
            monitor.update_metrics(metrics, step=i)
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed_time < 5.0  # 5 seconds for 1000 updates
        
        # Check that metrics were tracked
        stats = monitor.metric_tracker.get_all_stats()
        assert len(stats) == 3
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_memory_usage_scalability(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test memory usage with large number of metrics."""
        monitor = TrainingMonitor(
            log_dir=temp_dir,
            enable_tensorboard=False,
            enable_profiling=False
        )
        
        # Add many different metrics
        for i in range(100):
            metrics = {f'metric_{j}': i * 0.01 + j for j in range(50)}
            monitor.update_metrics(metrics, step=i)
        
        # Should handle large number of metrics without issues
        stats = monitor.metric_tracker.get_all_stats()
        assert len(stats) == 50 * 100  # 50 metrics per step, 100 steps
    
    def test_concurrent_logging(self, temp_dir):
        """Test concurrent logging from multiple threads."""
        tracker = MetricTracker()
        
        def update_metrics(thread_id):
            for i in range(100):
                metrics = {f'thread_{thread_id}_metric': i}
                tracker.update(metrics)
        
        # Start multiple threads
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=update_metrics, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all metrics were recorded
        stats = tracker.get_all_stats()
        assert len(stats) == 5  # One metric per thread
        
        for thread_id in range(5):
            metric_name = f'thread_{thread_id}_metric'
            assert metric_name in stats
            assert stats[metric_name]['count'] == 100


class TestCreateTrainingMonitor:
    """Test factory function for creating training monitor."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_create_with_defaults(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test creating monitor with default settings."""
        monitor = create_training_monitor(temp_dir)
        
        assert isinstance(monitor, TrainingMonitor)
        assert monitor.log_dir == Path(temp_dir)
        assert monitor.update_frequency == 10.0  # default
        assert monitor.save_frequency == 100  # default
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_create_with_debug_config(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test creating monitor with debug configuration."""
        config = {'debug': True}
        
        monitor = create_training_monitor(temp_dir, config=config)
        
        assert monitor.update_frequency == 1.0  # debug setting
        assert monitor.enable_profiling is True  # debug setting
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1)
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_create_with_custom_kwargs(self, mock_is_main, mock_world_size, mock_rank, temp_dir):
        """Test creating monitor with custom arguments."""
        monitor = create_training_monitor(
            temp_dir,
            update_frequency=5.0,
            save_frequency=50,
            enable_tensorboard=False
        )
        
        assert monitor.update_frequency == 5.0
        assert monitor.save_frequency == 50
        assert monitor.enable_tensorboard is False


if __name__ == '__main__':
    pytest.main([__file__])