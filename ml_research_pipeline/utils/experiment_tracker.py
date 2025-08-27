"""
Experiment tracking system for monitoring and logging experiment progress.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading

import torch
import numpy as np
import matplotlib.pyplot as plt

from .logging_utils import LoggerMixin


@dataclass
class MetricEntry:
    """Single metric entry."""

    step: int
    timestamp: float
    value: float
    epoch: Optional[int] = None
    phase: Optional[str] = None  # train, val, test

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentEvent:
    """Experiment event for tracking important occurrences."""

    timestamp: float
    event_type: str  # checkpoint, best_model, error, milestone
    message: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricTracker:
    """Tracker for individual metrics with statistics."""

    def __init__(self, name: str, window_size: int = 100):
        """Initialize metric tracker.

        Args:
            name: Metric name
            window_size: Size of rolling window for statistics
        """
        self.name = name
        self.window_size = window_size
        self.entries: List[MetricEntry] = []
        self.rolling_values = deque(maxlen=window_size)

        # Statistics
        self.best_value = None
        self.best_step = None
        self.total_count = 0
        self.sum_values = 0.0

    def add_entry(self, entry: MetricEntry):
        """Add metric entry."""
        self.entries.append(entry)
        self.rolling_values.append(entry.value)

        # Update statistics
        self.total_count += 1
        self.sum_values += entry.value

        # Update best value (assuming lower is better, can be customized)
        if self.best_value is None or entry.value < self.best_value:
            self.best_value = entry.value
            self.best_step = entry.step

    def get_current_value(self) -> Optional[float]:
        """Get most recent value."""
        return self.entries[-1].value if self.entries else None

    def get_rolling_mean(self) -> float:
        """Get rolling mean of recent values."""
        return np.mean(self.rolling_values) if self.rolling_values else 0.0

    def get_rolling_std(self) -> float:
        """Get rolling standard deviation."""
        return np.std(self.rolling_values) if len(self.rolling_values) > 1 else 0.0

    def get_overall_mean(self) -> float:
        """Get overall mean."""
        return self.sum_values / self.total_count if self.total_count > 0 else 0.0

    def get_trend(self, window: int = 10) -> str:
        """Get trend direction over recent window.

        Args:
            window: Number of recent values to consider

        Returns:
            Trend direction: 'improving', 'worsening', 'stable'
        """
        if len(self.entries) < window:
            return "insufficient_data"

        recent_values = [entry.value for entry in self.entries[-window:]]

        # Simple linear trend
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]

        if abs(slope) < 1e-6:
            return "stable"
        elif slope < 0:
            return "improving"  # Assuming lower is better
        else:
            return "worsening"

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "name": self.name,
            "total_entries": len(self.entries),
            "current_value": self.get_current_value(),
            "best_value": self.best_value,
            "best_step": self.best_step,
            "overall_mean": self.get_overall_mean(),
            "rolling_mean": self.get_rolling_mean(),
            "rolling_std": self.get_rolling_std(),
            "trend": self.get_trend(),
        }


class ExperimentTracker(LoggerMixin):
    """Comprehensive experiment tracking system."""

    def __init__(
        self,
        experiment_name: str,
        output_dir: Union[str, Path],
        auto_save_interval: int = 100,
        enable_plotting: bool = True,
    ):
        """Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save tracking data
            auto_save_interval: Steps between automatic saves
            enable_plotting: Whether to enable automatic plotting
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.auto_save_interval = auto_save_interval
        self.enable_plotting = enable_plotting

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracking data
        self.metrics: Dict[str, MetricTracker] = {}
        self.events: List[ExperimentEvent] = []
        self.hyperparameters: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

        # State tracking
        self.start_time = time.time()
        self.current_step = 0
        self.current_epoch = 0
        self.is_running = False

        # Thread safety
        self.lock = threading.Lock()

        # Files
        self.metrics_file = self.output_dir / "metrics.jsonl"
        self.events_file = self.output_dir / "events.jsonl"
        self.summary_file = self.output_dir / "summary.json"

        self.log_info(f"Initialized experiment tracker: {experiment_name}")

    def start_experiment(self, hyperparameters: Optional[Dict[str, Any]] = None):
        """Start experiment tracking.

        Args:
            hyperparameters: Experiment hyperparameters
        """
        with self.lock:
            self.is_running = True
            self.start_time = time.time()

            if hyperparameters:
                self.hyperparameters = hyperparameters

        # Log start event (outside lock to prevent deadlock)
        self.log_event("experiment_start", "Experiment started")

        self.log_info("Experiment tracking started")

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        phase: Optional[str] = None,
    ):
        """Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Training step
            epoch: Training epoch
            phase: Training phase (train, val, test)
        """
        should_save = False
        should_plot = False

        with self.lock:
            if step is None:
                step = self.current_step

            if epoch is None:
                epoch = self.current_epoch

            # Create metric tracker if not exists
            if name not in self.metrics:
                self.metrics[name] = MetricTracker(name)

            # Add entry
            entry = MetricEntry(
                step=step, timestamp=time.time(), value=value, epoch=epoch, phase=phase
            )

            self.metrics[name].add_entry(entry)

            # Check if we need to save/plot (but don't do it while holding lock)
            should_save = step % self.auto_save_interval == 0
            should_plot = (
                self.enable_plotting and step % (self.auto_save_interval * 5) == 0
            )

        # Perform I/O operations outside the lock to prevent deadlocks
        if should_save:
            self._save_metrics()

        if should_plot:
            self._update_plots()

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        phase: Optional[str] = None,
    ):
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step
            epoch: Training epoch
            phase: Training phase
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step, epoch, phase)

    def log_event(
        self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None
    ):
        """Log an experiment event.

        Args:
            event_type: Type of event
            message: Event message
            data: Additional event data
        """
        event = ExperimentEvent(
            timestamp=time.time(), event_type=event_type, message=message, data=data
        )

        with self.lock:
            self.events.append(event)

        # Save event immediately (outside lock to prevent deadlock)
        self._append_event(event)

        # Log outside the lock to prevent deadlock
        self.log_info(f"Event [{event_type}]: {message}")

    def update_step(self, step: int):
        """Update current step."""
        self.current_step = step

    def update_epoch(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch
        self.log_event("epoch_start", f"Started epoch {epoch}")

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters."""
        with self.lock:
            self.hyperparameters.update(hyperparameters)
            self._save_summary()

    def log_metadata(self, metadata: Dict[str, Any]):
        """Log experiment metadata."""
        with self.lock:
            self.metadata.update(metadata)
            self._save_summary()

    def get_metric_statistics(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Metric statistics or None if metric doesn't exist
        """
        if metric_name in self.metrics:
            return self.metrics[metric_name].get_statistics()
        return None

    def get_all_metrics_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all metrics."""
        return {
            name: tracker.get_statistics() for name, tracker in self.metrics.items()
        }

    def get_best_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get best values for all metrics."""
        best_metrics = {}

        for name, tracker in self.metrics.items():
            if tracker.best_value is not None:
                best_metrics[name] = {
                    "value": tracker.best_value,
                    "step": tracker.best_step,
                }

        return best_metrics

    def is_metric_improving(self, metric_name: str, patience: int = 10) -> bool:
        """Check if a metric is improving.

        Args:
            metric_name: Name of the metric
            patience: Number of steps to look back

        Returns:
            True if metric is improving
        """
        if metric_name not in self.metrics:
            return False

        tracker = self.metrics[metric_name]
        trend = tracker.get_trend(window=patience)

        return trend == "improving"

    def should_stop_early(
        self, metric_name: str, patience: int = 20, min_delta: float = 1e-6
    ) -> bool:
        """Check if early stopping should be triggered.

        Args:
            metric_name: Name of the metric to monitor
            patience: Number of steps without improvement
            min_delta: Minimum change to qualify as improvement

        Returns:
            True if early stopping should be triggered
        """
        if metric_name not in self.metrics:
            return False

        tracker = self.metrics[metric_name]

        if len(tracker.entries) < patience:
            return False

        # Check if there's been improvement in the last 'patience' steps
        recent_entries = tracker.entries[-patience:]
        best_recent = min(entry.value for entry in recent_entries)

        # Compare with best overall value
        if tracker.best_value is None:
            return False

        improvement = tracker.best_value - best_recent

        return improvement < min_delta

    def _save_metrics(self):
        """Save metrics to file."""
        # Capture metrics data while holding lock
        with self.lock:
            metrics_data = []
            for tracker in self.metrics.values():
                for entry in tracker.entries:
                    metrics_data.append(
                        {"metric_name": tracker.name, **entry.to_dict()}
                    )

        # Perform I/O outside the lock
        with open(self.metrics_file, "w") as f:
            for data in metrics_data:
                f.write(json.dumps(data) + "\n")

    def _append_event(self, event: ExperimentEvent):
        """Append event to file."""
        with open(self.events_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def _save_summary(self):
        """Save experiment summary."""
        # Capture data while holding lock
        with self.lock:
            summary = {
                "experiment_name": self.experiment_name,
                "start_time": self.start_time,
                "current_step": self.current_step,
                "current_epoch": self.current_epoch,
                "is_running": self.is_running,
                "hyperparameters": self.hyperparameters.copy(),
                "metadata": self.metadata.copy(),
                "metrics_statistics": self.get_all_metrics_statistics(),
                "best_metrics": self.get_best_metrics(),
                "total_events": len(self.events),
            }

        # Perform I/O outside the lock
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def _update_plots(self):
        """Update metric plots."""
        if not self.enable_plotting:
            return

        # Capture metrics data while holding lock
        with self.lock:
            plot_data = {}
            for name, tracker in self.metrics.items():
                if len(tracker.entries) >= 2:
                    plot_data[name] = {
                        "steps": [entry.step for entry in tracker.entries],
                        "values": [entry.value for entry in tracker.entries],
                    }

        if not plot_data:
            return

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Create plots for each metric (outside the lock)
        for name, data in plot_data.items():
            plt.figure(figsize=(10, 6))

            steps = data["steps"]
            values = data["values"]

            plt.plot(steps, values, label=name, alpha=0.7)

            # Add rolling mean
            if len(values) > 10:
                window = min(len(values) // 10, 50)
                rolling_mean = np.convolve(
                    values, np.ones(window) / window, mode="valid"
                )
                rolling_steps = steps[window - 1 :]
                plt.plot(
                    rolling_steps,
                    rolling_mean,
                    label=f"{name} (rolling mean)",
                    linewidth=2,
                )

            plt.xlabel("Step")
            plt.ylabel(name)
            plt.title(f"{self.experiment_name} - {name}")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save plot
            plot_file = plots_dir / f"{name.replace('/', '_')}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()

    def finish_experiment(self, final_metrics: Optional[Dict[str, float]] = None):
        """Finish experiment tracking.

        Args:
            final_metrics: Final metric values
        """
        duration = None

        with self.lock:
            self.is_running = False
            duration = time.time() - self.start_time

        # Log final metrics outside the lock
        if final_metrics:
            self.log_metrics(final_metrics, phase="final")

        # Log finish event (this method already handles locking properly)
        self.log_event(
            "experiment_finish",
            f"Experiment finished after {duration:.2f} seconds",
            {"duration_seconds": duration},
        )

        # Perform I/O operations outside the lock
        self._save_metrics()
        self._save_summary()

        if self.enable_plotting:
            self._update_plots()

        self.log_info("Experiment tracking finished")

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive experiment report."""
        duration = time.time() - self.start_time if self.is_running else None

        report = {
            "experiment_info": {
                "name": self.experiment_name,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "duration_seconds": duration,
                "is_running": self.is_running,
                "total_steps": self.current_step,
                "total_epochs": self.current_epoch,
            },
            "hyperparameters": self.hyperparameters,
            "metadata": self.metadata,
            "metrics_summary": self.get_all_metrics_statistics(),
            "best_metrics": self.get_best_metrics(),
            "events_summary": {
                "total_events": len(self.events),
                "event_types": list(set(event.event_type for event in self.events)),
            },
            "files": {
                "metrics": str(self.metrics_file),
                "events": str(self.events_file),
                "summary": str(self.summary_file),
                "plots": str(self.output_dir / "plots"),
            },
        }

        return report

    @classmethod
    def load_experiment(cls, output_dir: Union[str, Path]) -> "ExperimentTracker":
        """Load existing experiment tracker.

        Args:
            output_dir: Directory containing experiment data

        Returns:
            Loaded experiment tracker
        """
        output_dir = Path(output_dir)
        summary_file = output_dir / "summary.json"

        if not summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_file}")

        with open(summary_file, "r") as f:
            summary = json.load(f)

        # Create tracker instance
        tracker = cls(
            experiment_name=summary["experiment_name"],
            output_dir=output_dir,
            enable_plotting=False,  # Disable plotting during loading
        )

        # Restore state
        tracker.start_time = summary["start_time"]
        tracker.current_step = summary["current_step"]
        tracker.current_epoch = summary["current_epoch"]
        tracker.is_running = summary["is_running"]
        tracker.hyperparameters = summary["hyperparameters"]
        tracker.metadata = summary["metadata"]

        # Load metrics
        metrics_file = output_dir / "metrics.jsonl"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    metric_name = data.pop("metric_name")

                    if metric_name not in tracker.metrics:
                        tracker.metrics[metric_name] = MetricTracker(metric_name)

                    entry = MetricEntry(**data)
                    tracker.metrics[metric_name].add_entry(entry)

        # Load events
        events_file = output_dir / "events.jsonl"
        if events_file.exists():
            with open(events_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    event = ExperimentEvent(**data)
                    tracker.events.append(event)

        return tracker
