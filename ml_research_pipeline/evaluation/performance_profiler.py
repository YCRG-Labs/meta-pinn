"""
Performance profiling and benchmarking system for meta-learning PINNs.

This module provides comprehensive performance analysis including computational
bottleneck identification, memory usage optimization, and scalability testing.
"""

import time
import psutil
import torch
import torch.profiler
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import gc
import functools

from ..utils.distributed_utils import get_rank, get_world_size, is_main_process


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    cpu_time: float = 0.0
    gpu_time: float = 0.0
    memory_peak: float = 0.0
    memory_allocated: float = 0.0
    memory_reserved: float = 0.0
    gpu_utilization: float = 0.0
    throughput: float = 0.0
    latency: float = 0.0
    flops: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def __add__(self, other: 'PerformanceMetrics') -> 'PerformanceMetrics':
        """Add two performance metrics."""
        return PerformanceMetrics(
            cpu_time=self.cpu_time + other.cpu_time,
            gpu_time=self.gpu_time + other.gpu_time,
            memory_peak=max(self.memory_peak, other.memory_peak),
            memory_allocated=self.memory_allocated + other.memory_allocated,
            memory_reserved=self.memory_reserved + other.memory_reserved,
            gpu_utilization=(self.gpu_utilization + other.gpu_utilization) / 2,
            throughput=self.throughput + other.throughput,
            latency=self.latency + other.latency,
            flops=(self.flops or 0) + (other.flops or 0)
        )
    
    def __truediv__(self, scalar: float) -> 'PerformanceMetrics':
        """Divide metrics by scalar."""
        return PerformanceMetrics(
            cpu_time=self.cpu_time / scalar,
            gpu_time=self.gpu_time / scalar,
            memory_peak=self.memory_peak,  # Peak is not averaged
            memory_allocated=self.memory_allocated / scalar,
            memory_reserved=self.memory_reserved / scalar,
            gpu_utilization=self.gpu_utilization / scalar,
            throughput=self.throughput / scalar,
            latency=self.latency / scalar,
            flops=(self.flops / scalar) if self.flops else None
        )


class MemoryMonitor:
    """
    Real-time memory usage monitoring for CPU and GPU.
    
    Tracks memory allocation patterns, peak usage, and provides
    memory optimization recommendations.
    """
    
    def __init__(self, monitor_interval: float = 0.1):
        """
        Initialize memory monitor.
        
        Args:
            monitor_interval: Monitoring interval in seconds
        """
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Memory tracking
        self.cpu_memory_history = deque(maxlen=1000)
        self.gpu_memory_history = deque(maxlen=1000)
        self.peak_cpu_memory = 0.0
        self.peak_gpu_memory = 0.0
        
        # Process handle for CPU monitoring
        self.process = psutil.Process()
        
        # GPU availability
        self.cuda_available = torch.cuda.is_available()
        
        logging.info(f"MemoryMonitor initialized (CUDA: {self.cuda_available})")
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logging.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # CPU memory
                cpu_memory = self.process.memory_info().rss / 1024**3  # GB
                self.cpu_memory_history.append(cpu_memory)
                self.peak_cpu_memory = max(self.peak_cpu_memory, cpu_memory)
                
                # GPU memory
                if self.cuda_available:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                    self.gpu_memory_history.append(gpu_memory)
                    self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_memory)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logging.warning(f"Memory monitoring error: {e}")
                time.sleep(self.monitor_interval)
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        cpu_memory = self.process.memory_info().rss / 1024**3
        
        usage = {
            'cpu_memory_gb': cpu_memory,
            'cpu_memory_percent': self.process.memory_percent(),
            'peak_cpu_memory_gb': self.peak_cpu_memory
        }
        
        if self.cuda_available:
            usage.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'peak_gpu_memory_gb': self.peak_gpu_memory,
                'gpu_memory_percent': (torch.cuda.memory_allocated() / 
                                     torch.cuda.get_device_properties(0).total_memory) * 100
            })
        
        return usage
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            'cpu_memory': {
                'current_gb': self.process.memory_info().rss / 1024**3,
                'peak_gb': self.peak_cpu_memory,
                'percent': self.process.memory_percent(),
                'history_length': len(self.cpu_memory_history)
            }
        }
        
        if self.cuda_available:
            stats['gpu_memory'] = {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'peak_gb': self.peak_gpu_memory,
                'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'percent': (torch.cuda.memory_allocated() / 
                           torch.cuda.get_device_properties(0).total_memory) * 100,
                'history_length': len(self.gpu_memory_history)
            }
        
        return stats
    
    def optimize_memory(self) -> Dict[str, Any]:
        """
        Perform memory optimization and return recommendations.
        
        Returns:
            Dictionary with optimization results and recommendations
        """
        recommendations = []
        actions_taken = []
        
        # Clear Python garbage
        collected = gc.collect()
        if collected > 0:
            actions_taken.append(f"Collected {collected} Python objects")
        
        # Clear GPU cache if available
        if self.cuda_available:
            initial_reserved = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            final_reserved = torch.cuda.memory_reserved()
            freed = (initial_reserved - final_reserved) / 1024**3
            if freed > 0:
                actions_taken.append(f"Freed {freed:.2f} GB GPU cache")
        
        # Generate recommendations based on usage patterns
        current_usage = self.get_current_usage()
        
        if current_usage.get('cpu_memory_percent', 0) > 80:
            recommendations.append("High CPU memory usage - consider reducing batch size")
        
        if current_usage.get('gpu_memory_percent', 0) > 80:
            recommendations.append("High GPU memory usage - consider gradient checkpointing")
        
        if self.peak_gpu_memory > current_usage.get('gpu_memory_allocated_gb', 0) * 2:
            recommendations.append("Large memory spikes detected - consider memory profiling")
        
        return {
            'actions_taken': actions_taken,
            'recommendations': recommendations,
            'memory_usage_after': self.get_current_usage()
        }
    
    def reset_peak_tracking(self):
        """Reset peak memory tracking."""
        self.peak_cpu_memory = 0.0
        self.peak_gpu_memory = 0.0
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()


class ComputationProfiler:
    """
    Detailed computation profiling for identifying bottlenecks.
    
    Uses PyTorch profiler to analyze CPU/GPU utilization, kernel execution,
    and memory access patterns.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        profile_memory: bool = True,
        record_shapes: bool = True,
        with_stack: bool = True
    ):
        """
        Initialize computation profiler.
        
        Args:
            output_dir: Directory to save profiling results
            profile_memory: Whether to profile memory usage
            record_shapes: Whether to record tensor shapes
            with_stack: Whether to record stack traces
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        
        self.profiler = None
        self.profiling_active = False
        
        # Performance tracking
        self.operation_times = defaultdict(list)
        self.bottlenecks = []
        
        logging.info(f"ComputationProfiler initialized, output: {self.output_dir}")
    
    @contextmanager
    def profile_context(
        self,
        name: str = "profile",
        activities: Optional[List] = None,
        schedule: Optional[torch.profiler.schedule] = None
    ):
        """
        Context manager for profiling code blocks.
        
        Args:
            name: Name for the profiling session
            activities: List of profiler activities
            schedule: Profiler schedule
        """
        if activities is None:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ]
        
        if schedule is None:
            schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
        
        # Create profiler
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(self.output_dir / f"{name}_trace")
            ),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack
        )
        
        try:
            self.profiler.start()
            self.profiling_active = True
            yield self.profiler
        finally:
            if self.profiling_active:
                self.profiler.stop()
                self.profiling_active = False
                
                # Save profiler results
                self._save_profiler_results(name)
    
    def _save_profiler_results(self, name: str):
        """Save profiler results to files."""
        if not self.profiler:
            return
        
        try:
            # Export Chrome trace
            trace_file = self.output_dir / f"{name}_trace.json"
            self.profiler.export_chrome_trace(str(trace_file))
            
            # Export stacks (if available)
            if self.with_stack:
                stacks_file = self.output_dir / f"{name}_stacks.txt"
                with open(stacks_file, 'w') as f:
                    f.write(self.profiler.key_averages(group_by_stack_n=5).table(
                        sort_by="self_cpu_time_total", row_limit=20
                    ))
            
            # Export summary table
            summary_file = self.output_dir / f"{name}_summary.txt"
            with open(summary_file, 'w') as f:
                f.write(self.profiler.key_averages().table(
                    sort_by="cpu_time_total", row_limit=20
                ))
            
            logging.info(f"Profiler results saved for {name}")
            
        except Exception as e:
            logging.error(f"Failed to save profiler results: {e}")
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, PerformanceMetrics]:
        """
        Profile a single function call.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (function_result, performance_metrics)
        """
        # Start timing
        start_time = time.perf_counter()
        
        # Memory before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_before = torch.cuda.memory_allocated()
        else:
            gpu_memory_before = 0
        
        cpu_memory_before = psutil.Process().memory_info().rss
        
        # Execute function
        with self.profile_context(name=func.__name__):
            result = func(*args, **kwargs)
        
        # Synchronize and measure
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_after = torch.cuda.memory_allocated()
        else:
            gpu_memory_after = 0
        
        end_time = time.perf_counter()
        cpu_memory_after = psutil.Process().memory_info().rss
        
        # Calculate metrics
        metrics = PerformanceMetrics(
            cpu_time=end_time - start_time,
            gpu_time=end_time - start_time,  # Approximation
            memory_allocated=(gpu_memory_after - gpu_memory_before) / 1024**3,
            memory_peak=gpu_memory_after / 1024**3,
            latency=end_time - start_time
        )
        
        return result, metrics
    
    def identify_bottlenecks(self, threshold_ms: float = 10.0) -> List[Dict[str, Any]]:
        """
        Identify computational bottlenecks from profiling data.
        
        Args:
            threshold_ms: Minimum time in milliseconds to consider as bottleneck
            
        Returns:
            List of bottleneck information dictionaries
        """
        bottlenecks = []
        
        if not self.profiler:
            logging.warning("No profiler data available for bottleneck analysis")
            return bottlenecks
        
        try:
            # Analyze key averages
            key_averages = self.profiler.key_averages()
            
            for event in key_averages:
                cpu_time_ms = event.cpu_time_total / 1000  # Convert to ms
                
                if cpu_time_ms > threshold_ms:
                    bottleneck = {
                        'operation': event.key,
                        'cpu_time_ms': cpu_time_ms,
                        'cpu_time_avg_ms': event.cpu_time / 1000,
                        'count': event.count,
                        'input_shapes': event.input_shapes if hasattr(event, 'input_shapes') else None
                    }
                    
                    if torch.cuda.is_available():
                        bottleneck.update({
                            'cuda_time_ms': event.cuda_time_total / 1000,
                            'cuda_time_avg_ms': event.cuda_time / 1000
                        })
                    
                    bottlenecks.append(bottleneck)
            
            # Sort by total CPU time
            bottlenecks.sort(key=lambda x: x['cpu_time_ms'], reverse=True)
            
        except Exception as e:
            logging.error(f"Error analyzing bottlenecks: {e}")
        
        return bottlenecks
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        bottlenecks = self.identify_bottlenecks()
        
        recommendations = []
        
        # Analyze bottlenecks and generate recommendations
        for bottleneck in bottlenecks[:5]:  # Top 5 bottlenecks
            op_name = bottleneck['operation']
            
            if 'conv' in op_name.lower():
                recommendations.append({
                    'operation': op_name,
                    'suggestion': 'Consider using grouped convolutions or depthwise separable convolutions',
                    'impact': 'High'
                })
            elif 'matmul' in op_name.lower() or 'linear' in op_name.lower():
                recommendations.append({
                    'operation': op_name,
                    'suggestion': 'Consider mixed precision training or tensor parallelism',
                    'impact': 'Medium'
                })
            elif 'backward' in op_name.lower():
                recommendations.append({
                    'operation': op_name,
                    'suggestion': 'Consider gradient checkpointing to trade compute for memory',
                    'impact': 'Medium'
                })
        
        return {
            'bottlenecks': bottlenecks,
            'recommendations': recommendations,
            'total_operations': len(bottlenecks),
            'profiling_overhead': 'Low' if len(bottlenecks) < 100 else 'Medium'
        }


class ScalabilityTester:
    """
    Tests system scalability across different configurations.
    
    Evaluates performance scaling with respect to:
    - Batch size
    - Model size
    - Number of GPUs
    - Dataset size
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize scalability tester.
        
        Args:
            output_dir: Directory to save test results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_monitor = MemoryMonitor()
        self.profiler = ComputationProfiler(self.output_dir / "profiling")
        
        # Test results storage
        self.test_results = defaultdict(list)
        
        logging.info(f"ScalabilityTester initialized, output: {self.output_dir}")
    
    def test_batch_size_scaling(
        self,
        model_fn: Callable,
        batch_sizes: List[int],
        input_shape: Tuple[int, ...],
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Test performance scaling with batch size.
        
        Args:
            model_fn: Function that returns a model instance
            batch_sizes: List of batch sizes to test
            input_shape: Input tensor shape (without batch dimension)
            num_iterations: Number of iterations per test
            
        Returns:
            Dictionary with scaling results
        """
        results = []
        
        for batch_size in batch_sizes:
            logging.info(f"Testing batch size: {batch_size}")
            
            try:
                # Create model and input
                model = model_fn()
                if torch.cuda.is_available():
                    model = model.cuda()
                
                input_tensor = torch.randn(batch_size, *input_shape)
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                
                # Warm up
                for _ in range(3):
                    with torch.no_grad():
                        _ = model(input_tensor)
                
                # Start monitoring
                self.memory_monitor.start_monitoring()
                
                # Time forward passes
                times = []
                for i in range(num_iterations):
                    start_time = time.perf_counter()
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                
                # Stop monitoring and get stats
                self.memory_monitor.stop_monitoring()
                memory_stats = self.memory_monitor.get_memory_stats()
                
                # Calculate metrics
                avg_time = np.mean(times)
                std_time = np.std(times)
                throughput = batch_size / avg_time  # samples per second
                
                result = {
                    'batch_size': batch_size,
                    'avg_time_s': avg_time,
                    'std_time_s': std_time,
                    'throughput_samples_per_s': throughput,
                    'memory_stats': memory_stats,
                    'efficiency': throughput / batch_size  # throughput per sample
                }
                
                results.append(result)
                
                # Cleanup
                del model, input_tensor, output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"Error testing batch size {batch_size}: {e}")
                results.append({
                    'batch_size': batch_size,
                    'error': str(e)
                })
        
        # Save results
        results_file = self.output_dir / "batch_size_scaling.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return {
            'results': results,
            'optimal_batch_size': self._find_optimal_batch_size(results)
        }
    
    def test_model_size_scaling(
        self,
        model_configs: List[Dict[str, Any]],
        input_shape: Tuple[int, ...],
        batch_size: int = 32,
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Test performance scaling with model size.
        
        Args:
            model_configs: List of model configuration dictionaries
            input_shape: Input tensor shape (without batch dimension)
            batch_size: Batch size for testing
            num_iterations: Number of iterations per test
            
        Returns:
            Dictionary with scaling results
        """
        results = []
        
        for i, config in enumerate(model_configs):
            logging.info(f"Testing model config {i+1}/{len(model_configs)}")
            
            try:
                # Create model from config
                from ..core.meta_pinn import MetaPINN
                from ..config.model_config import MetaPINNConfig
                
                model_config = MetaPINNConfig(**config)
                model = MetaPINN(model_config)
                
                if torch.cuda.is_available():
                    model = model.cuda()
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # Create input
                input_tensor = torch.randn(batch_size, *input_shape)
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                
                # Profile forward pass
                _, metrics = self.profiler.profile_function(model, input_tensor)
                
                result = {
                    'config': config,
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'metrics': metrics.to_dict(),
                    'params_per_second': total_params / metrics.cpu_time if metrics.cpu_time > 0 else 0
                }
                
                results.append(result)
                
                # Cleanup
                del model, input_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"Error testing model config {i}: {e}")
                results.append({
                    'config': config,
                    'error': str(e)
                })
        
        # Save results
        results_file = self.output_dir / "model_size_scaling.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return {'results': results}
    
    def test_distributed_scaling(
        self,
        model_fn: Callable,
        world_sizes: List[int],
        batch_size: int = 32,
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Test distributed training scaling efficiency.
        
        Args:
            model_fn: Function that returns a model instance
            world_sizes: List of world sizes to test
            batch_size: Batch size per process
            num_iterations: Number of iterations per test
            
        Returns:
            Dictionary with scaling results
        """
        # Note: This is a simplified version - full distributed testing
        # would require actual multi-process setup
        
        results = []
        baseline_time = None
        
        for world_size in world_sizes:
            logging.info(f"Testing world size: {world_size}")
            
            try:
                # Simulate distributed workload
                effective_batch_size = batch_size * world_size
                
                # Create model
                model = model_fn()
                if torch.cuda.is_available():
                    model = model.cuda()
                
                # Create larger batch to simulate distributed workload
                input_tensor = torch.randn(effective_batch_size, 2)  # Assuming 2D input
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                
                # Time forward passes
                times = []
                for _ in range(num_iterations):
                    start_time = time.perf_counter()
                    
                    with torch.no_grad():
                        _ = model(input_tensor)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                
                # Calculate scaling efficiency
                if baseline_time is None:
                    baseline_time = avg_time
                    scaling_efficiency = 1.0
                else:
                    ideal_time = baseline_time / world_size
                    scaling_efficiency = ideal_time / avg_time
                
                result = {
                    'world_size': world_size,
                    'effective_batch_size': effective_batch_size,
                    'avg_time_s': avg_time,
                    'scaling_efficiency': scaling_efficiency,
                    'throughput_samples_per_s': effective_batch_size / avg_time
                }
                
                results.append(result)
                
                # Cleanup
                del model, input_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"Error testing world size {world_size}: {e}")
                results.append({
                    'world_size': world_size,
                    'error': str(e)
                })
        
        # Save results
        results_file = self.output_dir / "distributed_scaling.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return {'results': results}
    
    def _find_optimal_batch_size(self, results: List[Dict[str, Any]]) -> Optional[int]:
        """Find optimal batch size based on efficiency."""
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return None
        
        # Find batch size with highest efficiency
        best_result = max(valid_results, key=lambda x: x.get('efficiency', 0))
        return best_result['batch_size']
    
    def generate_scalability_report(self) -> Dict[str, Any]:
        """Generate comprehensive scalability report."""
        report = {
            'timestamp': time.time(),
            'system_info': {
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024**3
            },
            'test_results': dict(self.test_results),
            'recommendations': self._generate_scaling_recommendations()
        }
        
        # Save report
        report_file = self.output_dir / "scalability_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_scaling_recommendations(self) -> List[str]:
        """Generate scaling recommendations based on test results."""
        recommendations = []
        
        # Add general recommendations
        recommendations.append("Use mixed precision training to improve memory efficiency")
        recommendations.append("Consider gradient checkpointing for large models")
        recommendations.append("Profile memory usage to identify optimization opportunities")
        
        if torch.cuda.device_count() > 1:
            recommendations.append("Consider distributed training for large datasets")
        
        return recommendations


def benchmark_function(
    func: Callable,
    *args,
    num_runs: int = 10,
    warmup_runs: int = 3,
    **kwargs
) -> PerformanceMetrics:
    """
    Benchmark a function's performance.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        **kwargs: Function keyword arguments
        
    Returns:
        Average performance metrics
    """
    # Warmup runs
    for _ in range(warmup_runs):
        try:
            _ = func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Warmup run failed: {e}")
    
    # Benchmark runs
    metrics_list = []
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        # Memory before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_before = torch.cuda.memory_allocated()
        else:
            gpu_memory_before = 0
        
        try:
            result = func(*args, **kwargs)
            
            # Synchronize and measure
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_memory_after = torch.cuda.memory_allocated()
            else:
                gpu_memory_after = 0
            
            end_time = time.perf_counter()
            
            metrics = PerformanceMetrics(
                cpu_time=end_time - start_time,
                gpu_time=end_time - start_time,
                memory_allocated=(gpu_memory_after - gpu_memory_before) / 1024**3,
                latency=end_time - start_time
            )
            
            metrics_list.append(metrics)
            
        except Exception as e:
            logging.error(f"Benchmark run failed: {e}")
    
    # Average metrics
    if not metrics_list:
        return PerformanceMetrics()
    
    avg_metrics = metrics_list[0]
    for metrics in metrics_list[1:]:
        avg_metrics = avg_metrics + metrics
    
    return avg_metrics / len(metrics_list)


class PerformanceBenchmarkSuite:
    """
    Comprehensive performance benchmark suite for meta-learning PINNs.
    
    Combines all performance testing components into a unified interface.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save all benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.memory_monitor = MemoryMonitor()
        self.profiler = ComputationProfiler(self.output_dir / "profiling")
        self.scalability_tester = ScalabilityTester(self.output_dir / "scalability")
        
        # Results storage
        self.benchmark_results = {}
        
        logging.info(f"PerformanceBenchmarkSuite initialized: {self.output_dir}")
    
    def run_full_benchmark(
        self,
        model_fn: Callable,
        test_configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run complete performance benchmark suite.
        
        Args:
            model_fn: Function that returns model instances
            test_configs: Configuration for different tests
            
        Returns:
            Comprehensive benchmark results
        """
        logging.info("Starting full performance benchmark suite")
        
        results = {
            'timestamp': time.time(),
            'system_info': self._get_system_info(),
            'test_configs': test_configs
        }
        
        try:
            # Memory optimization test
            logging.info("Running memory optimization test")
            results['memory_optimization'] = self._test_memory_optimization()
            
            # Batch size scaling test
            if 'batch_sizes' in test_configs:
                logging.info("Running batch size scaling test")
                results['batch_size_scaling'] = self.scalability_tester.test_batch_size_scaling(
                    model_fn=model_fn,
                    batch_sizes=test_configs['batch_sizes'],
                    input_shape=test_configs.get('input_shape', (2,)),
                    num_iterations=test_configs.get('num_iterations', 10)
                )
            
            # Model size scaling test
            if 'model_configs' in test_configs:
                logging.info("Running model size scaling test")
                results['model_size_scaling'] = self.scalability_tester.test_model_size_scaling(
                    model_configs=test_configs['model_configs'],
                    input_shape=test_configs.get('input_shape', (2,)),
                    batch_size=test_configs.get('batch_size', 32),
                    num_iterations=test_configs.get('num_iterations', 10)
                )
            
            # Distributed scaling test
            if 'world_sizes' in test_configs:
                logging.info("Running distributed scaling test")
                results['distributed_scaling'] = self.scalability_tester.test_distributed_scaling(
                    model_fn=model_fn,
                    world_sizes=test_configs['world_sizes'],
                    batch_size=test_configs.get('batch_size', 32),
                    num_iterations=test_configs.get('num_iterations', 10)
                )
            
            # Generate optimization recommendations
            results['optimization_report'] = self.profiler.generate_optimization_report()
            results['scalability_report'] = self.scalability_tester.generate_scalability_report()
            
        except Exception as e:
            logging.error(f"Error during benchmark execution: {e}")
            results['error'] = str(e)
        
        # Save comprehensive results
        results_file = self.output_dir / "full_benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logging.info(f"Full benchmark completed, results saved to {results_file}")
        return results
    
    def _test_memory_optimization(self) -> Dict[str, Any]:
        """Test memory optimization capabilities."""
        self.memory_monitor.start_monitoring()
        
        # Get initial memory state
        initial_usage = self.memory_monitor.get_current_usage()
        
        # Simulate memory usage
        if torch.cuda.is_available():
            # Create some tensors to use memory
            tensors = []
            for i in range(10):
                tensor = torch.randn(1000, 1000).cuda()
                tensors.append(tensor)
            
            # Get peak usage
            peak_usage = self.memory_monitor.get_current_usage()
            
            # Clean up
            del tensors
            torch.cuda.empty_cache()
        
        # Run optimization
        optimization_result = self.memory_monitor.optimize_memory()
        
        # Get final usage
        final_usage = self.memory_monitor.get_current_usage()
        
        self.memory_monitor.stop_monitoring()
        
        return {
            'initial_usage': initial_usage,
            'peak_usage': peak_usage if torch.cuda.is_available() else initial_usage,
            'final_usage': final_usage,
            'optimization_result': optimization_result
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024**3,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_device_name': torch.cuda.get_device_name(),
                'cuda_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
        
        return info