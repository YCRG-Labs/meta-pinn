"""
Training monitoring and logging system for distributed meta-learning.

This module provides comprehensive logging, real-time metric visualization,
and distributed logging coordination for meta-learning training.
"""

import json
import logging
import os
import queue
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..utils.distributed_utils import (
    get_rank,
    get_world_size,
    is_main_process,
    reduce_dict,
)


class MetricTracker:
    """
    Tracks and aggregates training metrics over time.

    Supports moving averages, percentiles, and statistical summaries
    for comprehensive metric monitoring.

    Args:
        window_size: Size of moving window for averages
        track_percentiles: Whether to track percentile statistics
        percentiles: List of percentiles to track (e.g., [50, 90, 95])
    """

    def __init__(
        self,
        window_size: int = 100,
        track_percentiles: bool = True,
        percentiles: List[float] = None,
    ):
        self.window_size = window_size
        self.track_percentiles = track_percentiles
        self.percentiles = percentiles or [50, 90, 95, 99]

        # Storage for metrics
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.global_metrics = defaultdict(list)

        # Statistics cache
        self._stats_cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 1.0  # Cache for 1 second

    def update(self, metrics: Dict[str, Union[float, torch.Tensor]]):
        """
        Update metrics with new values.

        Args:
            metrics: Dictionary of metric names to values
        """
        timestamp = time.time()

        for name, value in metrics.items():
            # Convert tensor to scalar if needed
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = value.item()
                else:
                    value = value.mean().item()

            # Store in moving window
            self.metrics[name].append(value)

            # Store globally (for full history)
            self.global_metrics[name].append({"value": value, "timestamp": timestamp})

        # Invalidate cache
        self._stats_cache.clear()

    def get_current_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get current statistics for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary containing mean, std, min, max, and optionally percentiles
        """
        if metric_name not in self.metrics:
            return {}

        # Check cache
        cache_key = f"{metric_name}_stats"
        current_time = time.time()

        if (
            cache_key in self._stats_cache
            and current_time - self._cache_timestamp < self._cache_ttl
        ):
            return self._stats_cache[cache_key]

        values = list(self.metrics[metric_name])
        if not values:
            return {}

        values_array = np.array(values)

        stats = {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "count": len(values),
            "latest": values[-1],
        }

        # Add percentiles if requested
        if self.track_percentiles and len(values) > 1:
            for p in self.percentiles:
                stats[f"p{p}"] = float(np.percentile(values_array, p))

        # Cache results
        self._stats_cache[cache_key] = stats
        self._cache_timestamp = current_time

        return stats

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked metrics."""
        return {name: self.get_current_stats(name) for name in self.metrics.keys()}

    def get_history(
        self, metric_name: str, last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical values for a metric.

        Args:
            metric_name: Name of the metric
            last_n: Number of recent values to return (None for all)

        Returns:
            List of dictionaries with 'value' and 'timestamp' keys
        """
        if metric_name not in self.global_metrics:
            return []

        history = self.global_metrics[metric_name]
        if last_n is not None:
            history = history[-last_n:]

        return history

    def reset(self, metric_name: Optional[str] = None):
        """
        Reset metrics.

        Args:
            metric_name: Specific metric to reset (None for all)
        """
        if metric_name is not None:
            if metric_name in self.metrics:
                self.metrics[metric_name].clear()
            if metric_name in self.global_metrics:
                self.global_metrics[metric_name].clear()
        else:
            self.metrics.clear()
            self.global_metrics.clear()

        self._stats_cache.clear()


class DistributedLogger:
    """
    Distributed logging system that coordinates logging across multiple processes.

    Handles log aggregation, synchronization, and ensures consistent logging
    across all processes in distributed training.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to files
        log_to_console: Whether to log to console
        aggregate_logs: Whether to aggregate logs from all processes
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        log_level: str = "INFO",
        log_to_file: bool = True,
        log_to_console: bool = True,
        aggregate_logs: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.aggregate_logs = aggregate_logs

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_main = is_main_process()

        # Create log directory (all processes need to write logs)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = self._setup_logger()

        # Log aggregation queue (for distributed logging)
        self.log_queue = queue.Queue() if self.aggregate_logs else None
        self.log_thread = None

        if self.aggregate_logs and self.is_main:
            self._start_log_aggregation()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger with appropriate handlers."""
        logger_name = f"meta_pinn_rank_{self.rank}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.log_level)

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            f"[Rank {self.rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        if self.log_to_console and (self.is_main or not self.aggregate_logs):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler
        if self.log_to_file:
            log_file = self.log_dir / f"training_rank_{self.rank}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _start_log_aggregation(self):
        """Start background thread for log aggregation."""
        if not self.is_main:
            return

        def log_aggregator():
            aggregated_log = self.log_dir / "training_aggregated.log"
            with open(aggregated_log, "a") as f:
                while True:
                    try:
                        log_entry = self.log_queue.get(timeout=1.0)
                        if log_entry is None:  # Shutdown signal
                            break
                        f.write(log_entry + "\n")
                        f.flush()
                    except queue.Empty:
                        continue

        self.log_thread = threading.Thread(target=log_aggregator, daemon=True)
        self.log_thread.start()

    def log(self, level: str, message: str, **kwargs):
        """
        Log a message with specified level.

        Args:
            level: Log level (debug, info, warning, error)
            message: Log message
            **kwargs: Additional context to include
        """
        # Add context information
        if kwargs:
            context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} | {context}"

        # Log using standard logger
        log_func = getattr(self.logger, level.lower())
        log_func(message)

        # Add to aggregation queue if enabled
        if self.aggregate_logs and self.log_queue is not None:
            timestamp = datetime.now().isoformat()
            log_entry = f"[{timestamp}] [Rank {self.rank}] [{level.upper()}] {message}"
            try:
                self.log_queue.put_nowait(log_entry)
            except queue.Full:
                pass  # Skip if queue is full

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log(level="info", message=message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log(level="debug", message=message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log(level="warning", message=message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log(level="error", message=message, **kwargs)

    def shutdown(self):
        """Shutdown logger and cleanup resources."""
        if self.log_queue is not None:
            self.log_queue.put(None)  # Shutdown signal

        if self.log_thread is not None:
            self.log_thread.join(timeout=5.0)
        
        # Close all file handlers to release file locks
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)


class TrainingMonitor:
    """
    Comprehensive training monitoring system.

    Combines metric tracking, logging, and visualization for complete
    monitoring of distributed meta-learning training.

    Args:
        log_dir: Directory for logs and monitoring data
        tensorboard_dir: Directory for TensorBoard logs
        update_frequency: How often to update metrics (in seconds)
        save_frequency: How often to save monitoring data (in steps)
        enable_tensorboard: Whether to enable TensorBoard logging
        enable_profiling: Whether to enable performance profiling
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        tensorboard_dir: Optional[Union[str, Path]] = None,
        update_frequency: float = 1.0,
        save_frequency: int = 100,
        enable_tensorboard: bool = True,
        enable_profiling: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.tensorboard_dir = (
            Path(tensorboard_dir) if tensorboard_dir else self.log_dir / "tensorboard"
        )
        self.update_frequency = update_frequency
        self.save_frequency = save_frequency
        self.enable_tensorboard = enable_tensorboard
        self.enable_profiling = enable_profiling

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_main = is_main_process()

        # Create directories
        if self.is_main:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if self.enable_tensorboard:
                self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.metric_tracker = MetricTracker()
        self.logger = DistributedLogger(self.log_dir)

        # TensorBoard writer (only on main process)
        self.tb_writer = None
        if self.enable_tensorboard and self.is_main:
            self.tb_writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

        # Training state
        self.start_time = time.time()
        self.last_update_time = 0
        self.step_count = 0
        self.epoch_count = 0

        # Performance profiling
        self.profiler = None
        if self.enable_profiling:
            self._setup_profiler()

        # Monitoring data storage
        self.monitoring_data = {
            "metrics_history": [],
            "performance_stats": [],
            "system_info": self._get_system_info(),
        }

        self.logger.info(
            "TrainingMonitor initialized", rank=self.rank, world_size=self.world_size
        )

    def _setup_profiler(self):
        """Setup PyTorch profiler for performance monitoring."""
        if not self.is_main:
            return

        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(self.tensorboard_dir / "profiler")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for monitoring."""
        info = {
            "rank": self.rank,
            "world_size": self.world_size,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
        }

        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name()
            info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory

        return info

    def update_metrics(
        self,
        metrics: Dict[str, Union[float, torch.Tensor]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = "",
    ):
        """
        Update training metrics.

        Args:
            metrics: Dictionary of metrics to update
            step: Current training step
            epoch: Current epoch
            prefix: Prefix for metric names (e.g., "train/", "val/")
        """
        current_time = time.time()

        # Update step/epoch counters
        if step is not None:
            self.step_count = step
        if epoch is not None:
            self.epoch_count = epoch

        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Reduce metrics across processes if distributed
        if self.world_size > 1:
            # Convert to tensors for reduction
            tensor_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    tensor_metrics[k] = v.clone()
                else:
                    tensor_metrics[k] = torch.tensor(float(v))

            # Reduce across processes
            reduced_metrics = reduce_dict(tensor_metrics, average=True)
            metrics = {k: v.item() for k, v in reduced_metrics.items()}

        # Update metric tracker
        self.metric_tracker.update(metrics)

        # Log to TensorBoard (main process only)
        if self.tb_writer is not None and self.step_count > 0:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, self.step_count)

        # Update profiler
        if self.profiler is not None:
            self.profiler.step()

        # Periodic logging and saving
        if current_time - self.last_update_time >= self.update_frequency:
            self._log_current_stats()
            self.last_update_time = current_time

        if self.step_count % self.save_frequency == 0:
            self._save_monitoring_data()

    def log_model_info(self, model: torch.nn.Module):
        """Log model architecture and parameter information."""
        if not self.is_main:
            return

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(
            "Model Information",
            total_params=total_params,
            trainable_params=trainable_params,
            model_type=type(model).__name__,
        )

        # Log to TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_text("Model/Architecture", str(model))
            self.tb_writer.add_scalar("Model/TotalParams", total_params)
            self.tb_writer.add_scalar("Model/TrainableParams", trainable_params)

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        if not self.is_main:
            return

        self.logger.info("Hyperparameters", **hparams)

        # Log to TensorBoard
        if self.tb_writer is not None:
            # Convert to string representation for complex objects
            hparams_str = {k: str(v) for k, v in hparams.items()}
            self.tb_writer.add_hparams(hparams_str, {})

    def log_training_progress(self, progress_info: Dict[str, Any]):
        """Log training progress information."""
        elapsed_time = time.time() - self.start_time

        progress_info.update(
            {
                "elapsed_time": elapsed_time,
                "step": self.step_count,
                "epoch": self.epoch_count,
            }
        )

        self.logger.info("Training Progress", **progress_info)

    def _log_current_stats(self):
        """Log current metric statistics."""
        if not self.is_main:
            return

        stats = self.metric_tracker.get_all_stats()
        if not stats:
            return

        # Log key statistics
        for metric_name, metric_stats in stats.items():
            if "mean" in metric_stats:
                self.logger.debug(
                    f"Metric {metric_name}",
                    mean=metric_stats["mean"],
                    std=metric_stats.get("std", 0),
                    latest=metric_stats.get("latest", 0),
                )

    def _save_monitoring_data(self):
        """Save monitoring data to disk."""
        if not self.is_main:
            return

        # Update monitoring data
        current_stats = self.metric_tracker.get_all_stats()
        self.monitoring_data["metrics_history"].append(
            {
                "step": self.step_count,
                "epoch": self.epoch_count,
                "timestamp": time.time(),
                "stats": current_stats,
            }
        )

        # Save to file
        monitoring_file = self.log_dir / "monitoring_data.json"
        try:
            with open(monitoring_file, "w") as f:
                json.dump(self.monitoring_data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save monitoring data: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        elapsed_time = time.time() - self.start_time
        current_stats = self.metric_tracker.get_all_stats()

        return {
            "elapsed_time": elapsed_time,
            "total_steps": self.step_count,
            "total_epochs": self.epoch_count,
            "current_metrics": current_stats,
            "system_info": self.monitoring_data["system_info"],
        }

    def shutdown(self):
        """Shutdown monitoring system and cleanup resources."""
        self.logger.info("Shutting down TrainingMonitor")

        # Save final monitoring data
        self._save_monitoring_data()

        # Close TensorBoard writer
        if self.tb_writer is not None:
            self.tb_writer.close()

        # Stop profiler
        if self.profiler is not None:
            self.profiler.stop()

        # Shutdown logger
        self.logger.shutdown()


def create_training_monitor(
    log_dir: Union[str, Path], config: Optional[Dict[str, Any]] = None, **kwargs
) -> TrainingMonitor:
    """
    Factory function to create a TrainingMonitor with sensible defaults.

    Args:
        log_dir: Directory for logs
        config: Configuration dictionary for default settings
        **kwargs: Additional arguments for TrainingMonitor

    Returns:
        TrainingMonitor instance
    """
    # Set defaults
    defaults = {
        "update_frequency": 10.0,
        "save_frequency": 100,
        "enable_tensorboard": True,
        "enable_profiling": False,
    }

    # Adjust based on config
    if config:
        if "debug" in config and config["debug"]:
            defaults["update_frequency"] = 1.0
            defaults["enable_profiling"] = True

    # Override with provided kwargs
    defaults.update(kwargs)

    return TrainingMonitor(log_dir, **defaults)
