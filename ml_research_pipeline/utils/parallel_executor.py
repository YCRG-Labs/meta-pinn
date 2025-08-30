"""
Parallel execution utilities for physics discovery methods.

This module provides parallel processing capabilities with load balancing,
resource management, and progress monitoring for computationally intensive
physics discovery tasks.
"""

import logging
import multiprocessing as mp
import queue
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result from a parallel task execution."""

    task_id: str
    result: Any
    execution_time: float
    worker_id: int
    success: bool
    error: Optional[str] = None


@dataclass
class ResourceMetrics:
    """System resource usage metrics."""

    cpu_percent: float
    memory_percent: float
    available_memory: float
    cpu_count: int
    timestamp: float


class Task:
    """Represents a task to be executed in parallel."""

    def __init__(
        self, task_id: str, func: Callable, args: tuple = (), kwargs: dict = None
    ):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.priority = 0
        self.estimated_time = None
        self.memory_requirement = None

    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first


class LoadBalancer:
    """Manages load balancing across workers."""

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.worker_loads = [0.0] * num_workers
        self.worker_tasks = [[] for _ in range(num_workers)]
        self.lock = threading.Lock()

    def assign_task(self, task: Task) -> int:
        """Assign task to the least loaded worker."""
        with self.lock:
            # Find worker with minimum load
            worker_id = min(range(self.num_workers), key=lambda i: self.worker_loads[i])

            # Update load estimate
            estimated_time = task.estimated_time or 1.0
            self.worker_loads[worker_id] += estimated_time
            self.worker_tasks[worker_id].append(task.task_id)

            return worker_id

    def task_completed(self, worker_id: int, execution_time: float, task_id: str):
        """Update load when task completes."""
        with self.lock:
            self.worker_loads[worker_id] = max(
                0, self.worker_loads[worker_id] - execution_time
            )
            if task_id in self.worker_tasks[worker_id]:
                self.worker_tasks[worker_id].remove(task_id)

    def get_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution across workers."""
        with self.lock:
            return {
                "loads": self.worker_loads.copy(),
                "avg_load": np.mean(self.worker_loads),
                "max_load": max(self.worker_loads),
                "min_load": min(self.worker_loads),
                "load_variance": np.var(self.worker_loads),
            }


class ResourceMonitor:
    """Monitors system resources and adjusts execution accordingly."""

    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = []
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()

    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = ResourceMetrics(
                    cpu_percent=psutil.cpu_percent(interval=0.1),
                    memory_percent=psutil.virtual_memory().percent,
                    available_memory=psutil.virtual_memory().available
                    / (1024**3),  # GB
                    cpu_count=psutil.cpu_count(),
                    timestamp=time.time(),
                )

                with self.lock:
                    self.metrics_history.append(metrics)
                    # Keep only last 100 measurements
                    if len(self.metrics_history) > 100:
                        self.metrics_history.pop(0)

                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")

    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get the most recent resource metrics."""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None

    def should_throttle(
        self, cpu_threshold: float = 90.0, memory_threshold: float = 85.0
    ) -> bool:
        """Check if execution should be throttled due to resource constraints."""
        metrics = self.get_current_metrics()
        if not metrics:
            return False

        return (
            metrics.cpu_percent > cpu_threshold
            or metrics.memory_percent > memory_threshold
        )

    def get_optimal_worker_count(self) -> int:
        """Suggest optimal number of workers based on current resources."""
        metrics = self.get_current_metrics()
        if not metrics:
            return mp.cpu_count()

        # Adjust based on memory and CPU usage
        cpu_factor = max(0.1, (100 - metrics.cpu_percent) / 100)
        memory_factor = max(0.1, (100 - metrics.memory_percent) / 100)

        optimal_workers = int(metrics.cpu_count * min(cpu_factor, memory_factor))
        return max(1, min(optimal_workers, metrics.cpu_count))


class ProgressTracker:
    """Tracks progress of parallel execution."""

    def __init__(self):
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = None
        self.completion_times = []
        self.lock = threading.Lock()

    def initialize(self, total_tasks: int):
        """Initialize progress tracking."""
        with self.lock:
            self.total_tasks = total_tasks
            self.completed_tasks = 0
            self.failed_tasks = 0
            self.start_time = time.time()
            self.completion_times = []

    def task_completed(self, success: bool, execution_time: float):
        """Record task completion."""
        with self.lock:
            if success:
                self.completed_tasks += 1
            else:
                self.failed_tasks += 1
            self.completion_times.append(execution_time)

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        with self.lock:
            total_processed = self.completed_tasks + self.failed_tasks
            progress_percent = (
                (total_processed / self.total_tasks * 100)
                if self.total_tasks > 0
                else 0
            )

            elapsed_time = time.time() - self.start_time if self.start_time else 0
            avg_task_time = (
                np.mean(self.completion_times) if self.completion_times else 0
            )

            remaining_tasks = self.total_tasks - total_processed
            estimated_remaining_time = (
                remaining_tasks * avg_task_time if avg_task_time > 0 else 0
            )

            return {
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                "progress_percent": progress_percent,
                "elapsed_time": elapsed_time,
                "avg_task_time": avg_task_time,
                "estimated_remaining_time": estimated_remaining_time,
                "success_rate": (
                    (self.completed_tasks / total_processed * 100)
                    if total_processed > 0
                    else 0
                ),
            }


def _execute_task_wrapper(
    task_data: Tuple[str, Callable, tuple, dict, int],
) -> TaskResult:
    """Wrapper function for executing tasks in separate processes."""
    task_id, func, args, kwargs, worker_id = task_data
    start_time = time.time()

    try:
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return TaskResult(
            task_id=task_id,
            result=result,
            execution_time=execution_time,
            worker_id=worker_id,
            success=True,
        )
    except Exception as e:
        execution_time = time.time() - start_time
        return TaskResult(
            task_id=task_id,
            result=None,
            execution_time=execution_time,
            worker_id=worker_id,
            success=False,
            error=str(e),
        )


class ParallelExecutor:
    """
    High-performance parallel executor for physics discovery methods.

    Provides multi-process execution with load balancing, resource management,
    and progress monitoring capabilities.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = True,
        enable_load_balancing: bool = True,
        enable_resource_monitoring: bool = True,
        monitoring_interval: float = 1.0,
        cpu_threshold: float = 90.0,
        memory_threshold: float = 85.0,
    ):
        """
        Initialize the parallel executor.

        Args:
            max_workers: Maximum number of worker processes/threads
            use_processes: Whether to use processes (True) or threads (False)
            enable_load_balancing: Whether to enable load balancing
            enable_resource_monitoring: Whether to monitor system resources
            monitoring_interval: Resource monitoring interval in seconds
            cpu_threshold: CPU usage threshold for throttling (%)
            memory_threshold: Memory usage threshold for throttling (%)
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.enable_load_balancing = enable_load_balancing
        self.enable_resource_monitoring = enable_resource_monitoring
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold

        # Initialize components
        self.load_balancer = (
            LoadBalancer(self.max_workers) if enable_load_balancing else None
        )
        self.resource_monitor = (
            ResourceMonitor(monitoring_interval) if enable_resource_monitoring else None
        )
        self.progress_tracker = ProgressTracker()

        # Execution state
        self.executor = None
        self.is_running = False

    def execute_tasks(
        self,
        tasks: List[Task],
        timeout: Optional[float] = None,
        return_exceptions: bool = False,
    ) -> List[TaskResult]:
        """
        Execute a list of tasks in parallel.

        Args:
            tasks: List of tasks to execute
            timeout: Maximum time to wait for all tasks to complete
            return_exceptions: Whether to return exceptions as results

        Returns:
            List of TaskResult objects
        """
        if not tasks:
            return []

        self.is_running = True
        results = []

        try:
            # Start resource monitoring
            if self.resource_monitor:
                self.resource_monitor.start_monitoring()

            # Initialize progress tracking
            self.progress_tracker.initialize(len(tasks))

            # Adjust worker count based on resources
            if self.resource_monitor:
                optimal_workers = self.resource_monitor.get_optimal_worker_count()
                actual_workers = min(self.max_workers, optimal_workers, len(tasks))
            else:
                actual_workers = min(self.max_workers, len(tasks))

            logger.info(f"Starting parallel execution with {actual_workers} workers")

            # Create executor
            executor_class = (
                ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
            )
            with executor_class(max_workers=actual_workers) as executor:
                self.executor = executor

                # Submit tasks
                future_to_task = {}
                for task in tasks:
                    # Assign worker if load balancing is enabled
                    worker_id = 0
                    if self.load_balancer:
                        worker_id = self.load_balancer.assign_task(task)

                    # Prepare task data for execution
                    task_data = (
                        task.task_id,
                        task.func,
                        task.args,
                        task.kwargs,
                        worker_id,
                    )
                    future = executor.submit(_execute_task_wrapper, task_data)
                    future_to_task[future] = task

                # Collect results
                for future in as_completed(future_to_task, timeout=timeout):
                    task = future_to_task[future]

                    try:
                        result = future.result()
                        results.append(result)

                        # Update progress and load balancing
                        self.progress_tracker.task_completed(
                            result.success, result.execution_time
                        )
                        if self.load_balancer:
                            self.load_balancer.task_completed(
                                result.worker_id, result.execution_time, result.task_id
                            )

                        # Check if we should throttle due to resource constraints
                        if (
                            self.resource_monitor
                            and self.resource_monitor.should_throttle(
                                self.cpu_threshold, self.memory_threshold
                            )
                        ):
                            logger.warning(
                                "High resource usage detected, throttling execution"
                            )
                            time.sleep(0.1)

                    except Exception as e:
                        error_result = TaskResult(
                            task_id=task.task_id,
                            result=None,
                            execution_time=0.0,
                            worker_id=0,
                            success=False,
                            error=str(e),
                        )
                        results.append(error_result)
                        self.progress_tracker.task_completed(False, 0.0)

                        if not return_exceptions:
                            logger.error(f"Task {task.task_id} failed: {e}")

        finally:
            # Cleanup
            self.is_running = False
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()

        return results

    def execute_function_parallel(
        self,
        func: Callable,
        args_list: List[tuple],
        kwargs_list: Optional[List[dict]] = None,
        task_ids: Optional[List[str]] = None,
        **execution_kwargs,
    ) -> List[TaskResult]:
        """
        Execute the same function with different arguments in parallel.

        Args:
            func: Function to execute
            args_list: List of argument tuples for each execution
            kwargs_list: List of keyword argument dicts for each execution
            task_ids: List of task IDs (auto-generated if not provided)
            **execution_kwargs: Additional arguments for execute_tasks

        Returns:
            List of TaskResult objects
        """
        if kwargs_list is None:
            kwargs_list = [{}] * len(args_list)

        if task_ids is None:
            task_ids = [f"task_{i}" for i in range(len(args_list))]

        # Create tasks
        tasks = []
        for i, (args, kwargs, task_id) in enumerate(
            zip(args_list, kwargs_list, task_ids)
        ):
            task = Task(task_id, func, args, kwargs)
            tasks.append(task)

        return self.execute_tasks(tasks, **execution_kwargs)

    def get_progress(self) -> Dict[str, Any]:
        """Get current execution progress."""
        progress = self.progress_tracker.get_progress()

        if self.load_balancer:
            progress["load_distribution"] = self.load_balancer.get_load_distribution()

        if self.resource_monitor:
            progress["resource_metrics"] = self.resource_monitor.get_current_metrics()

        return progress

    def cancel_execution(self):
        """Cancel ongoing execution (best effort)."""
        self.is_running = False
        if self.executor:
            # Note: ProcessPoolExecutor doesn't support cancellation of running tasks
            # This is a limitation of multiprocessing
            logger.warning("Cancellation requested - new tasks will not be submitted")
