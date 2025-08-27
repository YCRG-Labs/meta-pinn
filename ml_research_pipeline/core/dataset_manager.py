"""
Large-scale dataset generation and management for meta-learning fluid dynamics tasks.

This module provides efficient generation, caching, and management of large datasets
containing thousands of diverse fluid dynamics tasks for meta-learning applications.
"""

import os
import h5py
import pickle
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import hashlib
import time

from ..config.data_config import DataConfig, TaskConfig
from .task_generator import FluidTaskGenerator, FluidTask
from .analytical_solutions import AnalyticalSolutionGenerator
from .fenicsx_solver import create_fenicsx_solver, SolverConfig

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for generated datasets."""
    
    dataset_name: str
    n_tasks: int
    task_types: List[str]
    generation_time: str
    generator_version: str
    data_config: Dict[str, Any]
    file_format: str
    compression: bool
    total_size_mb: float
    checksum: str


class DatasetGenerator:
    """
    Generates large-scale datasets of fluid dynamics tasks.
    
    Supports parallel generation, multiple solution methods (analytical, FEniCSx),
    and efficient storage formats for meta-learning applications.
    Optimized for generating 1000+ diverse tasks with efficient caching and storage.
    """
    
    def __init__(self, 
                 data_config: DataConfig,
                 use_fenicsx: bool = True,
                 n_workers: Optional[int] = None,
                 cache_dir: Optional[str] = None,
                 memory_limit_gb: float = 8.0):
        """
        Initialize dataset generator.
        
        Args:
            data_config: Configuration for data generation
            use_fenicsx: Whether to use FEniCSx for high-fidelity solutions
            n_workers: Number of parallel workers (default: CPU count)
            cache_dir: Directory for caching intermediate results
            memory_limit_gb: Memory limit in GB for batch processing
        """
        self.data_config = data_config
        self.use_fenicsx = use_fenicsx
        self.n_workers = n_workers or min(mp.cpu_count(), 8)  # Limit to avoid memory issues
        self.cache_dir = Path(cache_dir or data_config.cache_dir)
        self.memory_limit_gb = memory_limit_gb
        
        # Initialize generators
        self.task_generator = FluidTaskGenerator(data_config, seed=42)
        self.analytical_generator = AnalyticalSolutionGenerator()
        
        # Initialize FEniCSx solver if available and requested
        self.fenicsx_solver = None
        if use_fenicsx:
            solver_config = SolverConfig(mesh_resolution=(50, 25))  # Moderate resolution for speed
            self.fenicsx_solver = create_fenicsx_solver(solver_config)
            if self.fenicsx_solver is None:
                logger.warning("FEniCSx solver not available, using analytical solutions only")
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.generation_stats = {
            'total_tasks_generated': 0,
            'total_generation_time': 0.0,
            'average_task_time': 0.0,
            'memory_usage_peak': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"Initialized DatasetGenerator with {self.n_workers} workers, memory limit: {memory_limit_gb}GB")
    
    def generate_dataset(self, 
                        dataset_name: str,
                        n_tasks: int,
                        n_support: int,
                        n_query: int,
                        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                        save_path: Optional[str] = None,
                        enable_caching: bool = True,
                        validate_tasks: bool = True) -> Dict[str, Any]:
        """
        Generate large-scale dataset with train/validation/test splits.
        Optimized for 1000+ tasks with efficient memory management and caching.
        
        Args:
            dataset_name: Name for the dataset
            n_tasks: Total number of tasks to generate
            n_support: Number of support points per task
            n_query: Number of query points per task
            split_ratios: (train, val, test) split ratios
            save_path: Path to save the dataset (if None, uses data_config.data_dir)
            enable_caching: Whether to use task caching for efficiency
            validate_tasks: Whether to validate generated tasks
        
        Returns:
            Dictionary containing dataset information and file paths
        """
        logger.info(f"Generating large-scale dataset '{dataset_name}' with {n_tasks} tasks")
        
        # Validate inputs
        if n_tasks <= 0:
            raise ValueError("Number of tasks must be positive")
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Calculate split sizes
        n_train = int(n_tasks * split_ratios[0])
        n_val = int(n_tasks * split_ratios[1])
        n_test = n_tasks - n_train - n_val
        
        logger.info(f"Dataset splits: train={n_train}, val={n_val}, test={n_test}")
        
        # Estimate memory requirements and adjust batch size
        batch_size = self._calculate_optimal_batch_size(n_support, n_query, n_tasks)
        logger.info(f"Using batch size: {batch_size} for memory efficiency")
        
        # Generate all tasks with progress tracking
        start_time = time.time()
        all_tasks = self._generate_tasks_parallel_optimized(
            n_tasks, n_support, n_query, batch_size, 
            enable_caching, validate_tasks
        )
        generation_time = time.time() - start_time
        
        # Update performance statistics
        self.generation_stats['total_tasks_generated'] += len(all_tasks)
        self.generation_stats['total_generation_time'] += generation_time
        self.generation_stats['average_task_time'] = (
            self.generation_stats['total_generation_time'] / 
            self.generation_stats['total_tasks_generated']
        )
        
        logger.info(f"Generated {len(all_tasks)} tasks in {generation_time:.2f} seconds "
                   f"({generation_time/len(all_tasks)*1000:.2f} ms/task)")
        
        # Split tasks efficiently
        train_tasks, val_tasks, test_tasks = self._split_tasks_efficiently(
            all_tasks, n_train, n_val, n_test
        )
        
        # Save dataset with optimized I/O
        save_dir = Path(save_path or self.data_config.data_dir) / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_info = self._save_dataset_splits_optimized(
            {'train': train_tasks, 'val': val_tasks, 'test': test_tasks},
            save_dir
        )
        
        # Generate comprehensive metadata and statistics
        metadata = self._create_metadata(dataset_name, all_tasks, generation_time, save_dir)
        self._save_metadata(metadata, save_dir / 'metadata.json')
        
        stats = self._compute_dataset_statistics(all_tasks)
        self._save_statistics(stats, save_dir / 'statistics.json')
        
        # Save performance metrics
        self._save_performance_metrics(save_dir / 'performance.json')
        
        logger.info(f"Dataset saved to {save_dir} (total size: {metadata.total_size_mb:.2f} MB)")
        
        return {
            'dataset_name': dataset_name,
            'save_path': str(save_dir),
            'metadata': metadata,
            'statistics': stats,
            'splits': dataset_info,
            'performance_stats': self.generation_stats.copy()
        }
    
    def _calculate_optimal_batch_size(self, n_support: int, n_query: int, n_tasks: int) -> int:
        """Calculate optimal batch size based on memory constraints."""
        # Estimate memory per task (rough approximation)
        points_per_task = n_support + n_query
        # Assume 8 bytes per float, 3 fields (coords, velocity, pressure), 2D coords
        bytes_per_task = points_per_task * (2 + 2 + 1) * 8  # coords(2) + velocity(2) + pressure(1)
        
        # Add overhead for metadata and processing
        bytes_per_task *= 2
        
        # Calculate max tasks per batch based on memory limit
        max_memory_bytes = self.memory_limit_gb * 1024**3
        max_tasks_per_batch = max(1, int(max_memory_bytes * 0.5 / bytes_per_task))  # Use 50% of limit
        
        # Adaptive batch size: smaller for large datasets, larger for small ones
        if n_tasks >= 1000:
            batch_size = min(max_tasks_per_batch, max(10, n_tasks // 50))
        else:
            batch_size = min(max_tasks_per_batch, max(5, n_tasks // 10))
        
        return batch_size
    
    def _generate_tasks_parallel_optimized(self, 
                                         n_tasks: int, 
                                         n_support: int, 
                                         n_query: int,
                                         batch_size: int,
                                         enable_caching: bool,
                                         validate_tasks: bool) -> List[FluidTask]:
        """Generate tasks with optimized parallel processing and caching."""
        all_tasks = []
        n_batches = (n_tasks + batch_size - 1) // batch_size
        
        with tqdm(total=n_tasks, desc="Generating tasks", unit="task") as pbar:
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_tasks)
                current_batch_size = batch_end - batch_start
                
                # Check cache first if enabled
                if enable_caching:
                    cached_tasks, missing_indices = self._check_task_cache(
                        batch_idx, current_batch_size, n_support, n_query
                    )
                    if cached_tasks:
                        all_tasks.extend(cached_tasks)
                        self.generation_stats['cache_hits'] += len(cached_tasks)
                        pbar.update(len(cached_tasks))
                        continue
                
                # Generate batch of tasks
                batch_start_time = time.time()
                
                if self.n_workers > 1 and current_batch_size > 1:
                    batch_tasks = self._generate_batch_parallel(
                        current_batch_size, n_support, n_query, batch_idx
                    )
                else:
                    batch_tasks = self._generate_batch_sequential(
                        current_batch_size, n_support, n_query, batch_idx
                    )
                
                batch_time = time.time() - batch_start_time
                
                # Validate tasks if requested
                if validate_tasks:
                    batch_tasks = self._validate_task_batch(batch_tasks)
                
                # Cache tasks if enabled
                if enable_caching:
                    self._cache_task_batch(batch_idx, batch_tasks, n_support, n_query)
                
                all_tasks.extend(batch_tasks)
                self.generation_stats['cache_misses'] += len(batch_tasks)
                
                pbar.update(len(batch_tasks))
                pbar.set_postfix({
                    'batch_time': f'{batch_time:.2f}s',
                    'tasks/sec': f'{len(batch_tasks)/batch_time:.1f}'
                })
        
        return all_tasks
    
    def _split_tasks_efficiently(self, tasks: List[FluidTask], 
                               n_train: int, n_val: int, n_test: int) -> Tuple[List[FluidTask], List[FluidTask], List[FluidTask]]:
        """Efficiently split tasks while maintaining task type balance."""
        # Simple random shuffle and split for now (can be enhanced for perfect balance later)
        np.random.shuffle(tasks)
        
        train_tasks = tasks[:n_train]
        val_tasks = tasks[n_train:n_train + n_val]
        test_tasks = tasks[n_train + n_val:]
        
        return train_tasks, val_tasks, test_tasks
    
    def _save_dataset_splits_optimized(self, splits: Dict[str, List[FluidTask]], 
                                     save_dir: Path) -> Dict[str, Any]:
        """Save dataset splits with optimized I/O operations."""
        dataset_info = {}
        
        for split_name, tasks in splits.items():
            if not tasks:
                continue
                
            split_start_time = time.time()
            split_info = self._save_task_split(tasks, save_dir / split_name)
            split_time = time.time() - split_start_time
            
            split_info['save_time'] = split_time
            split_info['tasks_per_second'] = len(tasks) / split_time if split_time > 0 else 0
            dataset_info[split_name] = split_info
            
            logger.info(f"Saved {split_name} split: {len(tasks)} tasks in {split_time:.2f}s")
        
        return dataset_info
    
    def _check_task_cache(self, batch_idx: int, batch_size: int, 
                         n_support: int, n_query: int) -> Tuple[List[FluidTask], List[int]]:
        """Check if tasks are available in cache."""
        cache_key = f"batch_{batch_idx}_{batch_size}_{n_support}_{n_query}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_tasks = pickle.load(f)
                return cached_tasks, []
            except Exception as e:
                logger.warning(f"Failed to load cached tasks: {e}")
        
        return [], list(range(batch_size))
    
    def _cache_task_batch(self, batch_idx: int, tasks: List[FluidTask], 
                         n_support: int, n_query: int):
        """Cache a batch of tasks for future use."""
        if not self.data_config.use_cache:
            return
            
        cache_key = f"batch_{batch_idx}_{len(tasks)}_{n_support}_{n_query}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(tasks, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"Failed to cache tasks: {e}")
    
    def _validate_task_batch(self, tasks: List[FluidTask]) -> List[FluidTask]:
        """Validate a batch of tasks and filter out invalid ones."""
        valid_tasks = []
        
        for task in tasks:
            if self.task_generator.validate_task_config(task.config):
                # Additional validation checks
                if self._validate_task_data_integrity(task):
                    valid_tasks.append(task)
                else:
                    logger.warning(f"Task {task.config.task_id} failed data integrity check")
            else:
                logger.warning(f"Task {task.config.task_id} failed configuration validation")
        
        if len(valid_tasks) < len(tasks):
            logger.warning(f"Filtered out {len(tasks) - len(valid_tasks)} invalid tasks")
        
        return valid_tasks
    
    def _validate_task_data_integrity(self, task: FluidTask) -> bool:
        """Validate task data integrity."""
        try:
            # Check tensor shapes and values
            for dataset_name, dataset in [('support_set', task.support_set), 
                                        ('query_set', task.query_set)]:
                for key, tensor in dataset.items():
                    if not torch.all(torch.isfinite(tensor)):
                        return False
                    if tensor.numel() == 0:
                        return False
            
            # Check coordinate bounds
            all_coords = torch.cat([task.support_set['coords'], task.query_set['coords']], dim=0)
            x_bounds = self.data_config.domain_bounds['x']
            y_bounds = self.data_config.domain_bounds['y']
            
            if not (x_bounds[0] <= all_coords[:, 0].min() and all_coords[:, 0].max() <= x_bounds[1]):
                return False
            if not (y_bounds[0] <= all_coords[:, 1].min() and all_coords[:, 1].max() <= y_bounds[1]):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _save_performance_metrics(self, file_path: Path):
        """Save performance metrics for analysis."""
        with open(file_path, 'w') as f:
            json.dump(self.generation_stats, f, indent=2)
    
    def _generate_tasks_parallel(self, 
                               n_tasks: int, 
                               n_support: int, 
                               n_query: int,
                               batch_size: int) -> List[FluidTask]:
        """Generate tasks in parallel batches."""
        all_tasks = []
        
        # Calculate number of batches
        n_batches = (n_tasks + batch_size - 1) // batch_size
        
        with tqdm(total=n_tasks, desc="Generating tasks") as pbar:
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_tasks)
                current_batch_size = batch_end - batch_start
                
                # Generate batch of tasks
                if self.n_workers > 1:
                    batch_tasks = self._generate_batch_parallel(
                        current_batch_size, n_support, n_query, batch_idx
                    )
                else:
                    batch_tasks = self._generate_batch_sequential(
                        current_batch_size, n_support, n_query, batch_idx
                    )
                
                all_tasks.extend(batch_tasks)
                pbar.update(len(batch_tasks))
        
        return all_tasks
    
    def _generate_batch_parallel(self, 
                               batch_size: int, 
                               n_support: int, 
                               n_query: int,
                               batch_idx: int) -> List[FluidTask]:
        """Generate a batch of tasks using parallel processing."""
        # Create worker arguments
        worker_args = []
        tasks_per_worker = max(1, batch_size // self.n_workers)
        
        for worker_idx in range(self.n_workers):
            start_idx = worker_idx * tasks_per_worker
            end_idx = min(start_idx + tasks_per_worker, batch_size)
            
            if start_idx < batch_size:
                n_worker_tasks = end_idx - start_idx
                worker_seed = 42 + batch_idx * 1000 + worker_idx  # Deterministic seed
                
                worker_args.append((
                    n_worker_tasks, n_support, n_query, worker_seed,
                    self.data_config, self.use_fenicsx
                ))
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(_generate_tasks_worker, worker_args))
        
        # Flatten results
        batch_tasks = []
        for worker_tasks in results:
            batch_tasks.extend(worker_tasks)
        
        return batch_tasks
    
    def _generate_batch_sequential(self, 
                                 batch_size: int, 
                                 n_support: int, 
                                 n_query: int,
                                 batch_idx: int) -> List[FluidTask]:
        """Generate a batch of tasks sequentially."""
        seed = 42 + batch_idx * 1000
        return _generate_tasks_worker((
            batch_size, n_support, n_query, seed,
            self.data_config, self.use_fenicsx
        ))
    
    def _save_task_split(self, tasks: List[FluidTask], save_path: Path) -> Dict[str, Any]:
        """Save a split of tasks to disk."""
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.data_config.cache_format == 'hdf5':
            file_path = save_path / 'tasks.h5'
            self._save_tasks_hdf5(tasks, file_path)
        
        elif self.data_config.cache_format == 'pickle':
            file_path = save_path / 'tasks.pkl'
            self._save_tasks_pickle(tasks, file_path)
        
        elif self.data_config.cache_format == 'numpy':
            file_path = save_path / 'tasks.npz'
            self._save_tasks_numpy(tasks, file_path)
        
        else:
            raise ValueError(f"Unsupported cache format: {self.data_config.cache_format}")
        
        return {
            'n_tasks': len(tasks),
            'file_path': str(file_path),
            'file_size_mb': file_path.stat().st_size / (1024 * 1024)
        }
    
    def _save_tasks_hdf5(self, tasks: List[FluidTask], file_path: Path):
        """Save tasks in HDF5 format."""
        with h5py.File(file_path, 'w') as f:
            # Create groups
            configs_group = f.create_group('configs')
            support_group = f.create_group('support_sets')
            query_group = f.create_group('query_sets')
            metadata_group = f.create_group('metadata')
            
            for i, task in enumerate(tasks):
                task_id = f'task_{i:06d}'
                
                # Save configuration as JSON string
                config_str = json.dumps(task.config.to_dict())
                configs_group.create_dataset(task_id, data=config_str)
                
                # Save support set
                support_task_group = support_group.create_group(task_id)
                for key, tensor in task.support_set.items():
                    support_task_group.create_dataset(key, data=tensor.numpy())
                
                # Save query set
                query_task_group = query_group.create_group(task_id)
                for key, tensor in task.query_set.items():
                    query_task_group.create_dataset(key, data=tensor.numpy())
                
                # Save metadata (convert numpy datetime64 to string)
                metadata_serializable = self._make_json_serializable(task.metadata)
                metadata_str = json.dumps(metadata_serializable)
                metadata_group.create_dataset(task_id, data=metadata_str)
    
    def _save_tasks_pickle(self, tasks: List[FluidTask], file_path: Path):
        """Save tasks in pickle format."""
        task_dicts = []
        for task in tasks:
            task_dict = task.to_dict()
            task_dict['metadata'] = self._make_json_serializable(task_dict['metadata'])
            task_dicts.append(task_dict)
        
        with open(file_path, 'wb') as f:
            pickle.dump(task_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _save_tasks_numpy(self, tasks: List[FluidTask], file_path: Path):
        """Save tasks in numpy format."""
        # Convert tasks to numpy arrays
        task_data = {}
        
        for i, task in enumerate(tasks):
            task_id = f'task_{i:06d}'
            
            # Save configuration as JSON string
            task_data[f'{task_id}_config'] = json.dumps(task.config.to_dict())
            
            # Save tensor data
            for key, tensor in task.support_set.items():
                task_data[f'{task_id}_support_{key}'] = tensor.numpy()
            
            for key, tensor in task.query_set.items():
                task_data[f'{task_id}_query_{key}'] = tensor.numpy()
            
            # Save metadata (convert numpy datetime64 to string)
            metadata_serializable = self._make_json_serializable(task.metadata)
            task_data[f'{task_id}_metadata'] = json.dumps(metadata_serializable)
        
        # Save with compression if requested
        if self.data_config.compression:
            np.savez_compressed(file_path, **task_data)
        else:
            np.savez(file_path, **task_data)
    
    def _create_metadata(self, 
                        dataset_name: str, 
                        tasks: List[FluidTask], 
                        generation_time: float,
                        save_dir: Path) -> DatasetMetadata:
        """Create dataset metadata."""
        # Compute dataset size
        total_size = sum(f.stat().st_size for f in save_dir.rglob('*') if f.is_file())
        total_size_mb = total_size / (1024 * 1024)
        
        # Compute checksum
        checksum = self._compute_dataset_checksum(save_dir)
        
        # Get task type distribution
        task_types = list(set(task.config.task_type for task in tasks))
        
        return DatasetMetadata(
            dataset_name=dataset_name,
            n_tasks=len(tasks),
            task_types=task_types,
            generation_time=time.strftime('%Y-%m-%d %H:%M:%S'),
            generator_version='1.0',
            data_config=self.data_config.to_dict(),
            file_format=self.data_config.cache_format,
            compression=self.data_config.compression,
            total_size_mb=total_size_mb,
            checksum=checksum
        )
    
    def _save_metadata(self, metadata: DatasetMetadata, file_path: Path):
        """Save dataset metadata."""
        with open(file_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
    
    def _compute_dataset_statistics(self, tasks: List[FluidTask]) -> Dict[str, Any]:
        """Compute comprehensive dataset statistics."""
        stats = {}
        
        # Task type distribution
        task_types = [task.config.task_type for task in tasks]
        type_counts = {t: task_types.count(t) for t in set(task_types)}
        stats['task_type_distribution'] = type_counts
        
        # Reynolds number statistics
        reynolds_numbers = [task.config.reynolds_number for task in tasks]
        stats['reynolds_statistics'] = {
            'mean': float(np.mean(reynolds_numbers)),
            'std': float(np.std(reynolds_numbers)),
            'min': float(np.min(reynolds_numbers)),
            'max': float(np.max(reynolds_numbers)),
            'median': float(np.median(reynolds_numbers))
        }
        
        # Geometry distribution
        geometries = [task.config.geometry_type for task in tasks]
        geometry_counts = {g: geometries.count(g) for g in set(geometries)}
        stats['geometry_distribution'] = geometry_counts
        
        # Viscosity parameter statistics
        base_viscosities = [task.config.viscosity_params.get('base_viscosity', 0.01) 
                           for task in tasks]
        stats['viscosity_statistics'] = {
            'mean': float(np.mean(base_viscosities)),
            'std': float(np.std(base_viscosities)),
            'min': float(np.min(base_viscosities)),
            'max': float(np.max(base_viscosities))
        }
        
        # Data point statistics
        support_sizes = [task.support_set['coords'].shape[0] for task in tasks]
        query_sizes = [task.query_set['coords'].shape[0] for task in tasks]
        
        stats['data_point_statistics'] = {
            'support_points': {
                'mean': float(np.mean(support_sizes)),
                'min': int(np.min(support_sizes)),
                'max': int(np.max(support_sizes))
            },
            'query_points': {
                'mean': float(np.mean(query_sizes)),
                'min': int(np.min(query_sizes)),
                'max': int(np.max(query_sizes))
            }
        }
        
        return stats
    
    def _save_statistics(self, stats: Dict[str, Any], file_path: Path):
        """Save dataset statistics."""
        with open(file_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _compute_dataset_checksum(self, save_dir: Path) -> str:
        """Compute checksum for dataset integrity verification."""
        hasher = hashlib.md5()
        
        # Sort files for consistent checksum
        files = sorted(save_dir.rglob('*'))
        
        # Exclude files that change after dataset creation
        excluded_files = {'performance.json', 'cache_stats.json'}
        
        for file_path in files:
            if file_path.is_file() and file_path.name not in excluded_files:
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy and other non-JSON-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.datetime64):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


class DatasetLoader:
    """
    Efficient loader for large-scale fluid dynamics datasets.
    
    Supports lazy loading, caching, and batch processing for meta-learning.
    Optimized for datasets with 1000+ tasks with memory-efficient operations.
    """
    
    def __init__(self, dataset_path: str, cache_size: int = 100, 
                 prefetch_size: int = 50, memory_map: bool = True):
        """
        Initialize dataset loader.
        
        Args:
            dataset_path: Path to dataset directory
            cache_size: Number of tasks to keep in memory cache
            prefetch_size: Number of tasks to prefetch for efficiency
            memory_map: Whether to use memory mapping for large files
        """
        self.dataset_path = Path(dataset_path)
        self.cache_size = cache_size
        self.prefetch_size = prefetch_size
        self.memory_map = memory_map
        self.task_cache = {}
        self.prefetch_cache = {}
        
        # Load metadata
        metadata_path = self.dataset_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        # Load statistics
        stats_path = self.dataset_path / 'statistics.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.statistics = json.load(f)
        else:
            self.statistics = {}
        
        # Load performance metrics if available
        perf_path = self.dataset_path / 'performance.json'
        if perf_path.exists():
            with open(perf_path, 'r') as f:
                self.performance_metrics = json.load(f)
        else:
            self.performance_metrics = {}
        
        # Initialize split information
        self.split_info = {}
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_path / split
            if split_path.exists():
                self.split_info[split] = self._get_split_info(split_path)
        
        logger.info(f"Loaded dataset with {self.metadata['n_tasks']} tasks, "
                   f"cache size: {cache_size}, prefetch: {prefetch_size}")
    
    def load_split(self, split: str) -> List[FluidTask]:
        """
        Load a complete dataset split.
        
        Args:
            split: Split name ('train', 'val', 'test')
        
        Returns:
            List of FluidTask objects
        """
        split_path = self.dataset_path / split
        
        if not split_path.exists():
            raise FileNotFoundError(f"Split '{split}' not found at {split_path}")
        
        file_format = self.metadata['file_format']
        
        if file_format == 'hdf5':
            return self._load_tasks_hdf5(split_path / 'tasks.h5')
        elif file_format == 'pickle':
            return self._load_tasks_pickle(split_path / 'tasks.pkl')
        elif file_format == 'numpy':
            return self._load_tasks_numpy(split_path / 'tasks.npz')
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def load_batch(self, split: str, batch_indices: List[int]) -> List[FluidTask]:
        """
        Load a batch of tasks by indices.
        
        Args:
            split: Split name ('train', 'val', 'test')
            batch_indices: List of task indices to load
        
        Returns:
            List of FluidTask objects
        """
        # Check cache first
        cached_tasks = []
        missing_indices = []
        
        for idx in batch_indices:
            cache_key = f"{split}_{idx}"
            if cache_key in self.task_cache:
                cached_tasks.append((idx, self.task_cache[cache_key]))
            else:
                missing_indices.append(idx)
        
        # Load missing tasks
        if missing_indices:
            all_tasks = self.load_split(split)
            
            for idx in missing_indices:
                if idx < len(all_tasks):
                    task = all_tasks[idx]
                    cache_key = f"{split}_{idx}"
                    
                    # Add to cache (with LRU eviction)
                    if len(self.task_cache) >= self.cache_size:
                        # Remove oldest item
                        oldest_key = next(iter(self.task_cache))
                        del self.task_cache[oldest_key]
                    
                    self.task_cache[cache_key] = task
                    cached_tasks.append((idx, task))
        
        # Sort by original indices and return tasks
        cached_tasks.sort(key=lambda x: x[0])
        return [task for _, task in cached_tasks]
    
    def iterate_batches(self, 
                       split: str, 
                       batch_size: int, 
                       shuffle: bool = False,
                       prefetch: bool = True) -> Iterator[List[FluidTask]]:
        """
        Iterate over dataset in batches with optimized loading.
        
        Args:
            split: Split name ('train', 'val', 'test')
            batch_size: Size of each batch
            shuffle: Whether to shuffle the data
            prefetch: Whether to prefetch next batch for efficiency
        
        Yields:
            Batches of FluidTask objects
        """
        if split not in self.split_info:
            raise ValueError(f"Split '{split}' not found in dataset")
        
        n_tasks = self.split_info[split]['n_tasks']
        
        # Create indices
        indices = list(range(n_tasks))
        if shuffle:
            np.random.shuffle(indices)
        
        # Yield batches with optional prefetching
        for i in range(0, n_tasks, batch_size):
            batch_indices = indices[i:i + batch_size]
            
            # Prefetch next batch if enabled
            if prefetch and i + batch_size < n_tasks:
                next_batch_indices = indices[i + batch_size:i + 2 * batch_size]
                self._prefetch_batch(split, next_batch_indices)
            
            batch_tasks = self.load_batch(split, batch_indices)
            yield batch_tasks
    
    def iterate_batches_streaming(self, 
                                split: str, 
                                batch_size: int, 
                                shuffle: bool = False) -> Iterator[List[FluidTask]]:
        """
        Stream batches without loading entire dataset into memory.
        Optimized for very large datasets (10k+ tasks).
        
        Args:
            split: Split name ('train', 'val', 'test')
            batch_size: Size of each batch
            shuffle: Whether to shuffle the data
        
        Yields:
            Batches of FluidTask objects
        """
        if split not in self.split_info:
            raise ValueError(f"Split '{split}' not found in dataset")
        
        file_format = self.metadata['file_format']
        split_path = self.dataset_path / split
        
        if file_format == 'hdf5':
            yield from self._stream_batches_hdf5(split_path / 'tasks.h5', batch_size, shuffle)
        else:
            # Fallback to regular iteration for other formats
            yield from self.iterate_batches(split, batch_size, shuffle, prefetch=False)
    
    def _stream_batches_hdf5(self, file_path: Path, batch_size: int, shuffle: bool) -> Iterator[List[FluidTask]]:
        """Stream batches directly from HDF5 file."""
        with h5py.File(file_path, 'r') as f:
            task_ids = list(f['configs'].keys())
            n_tasks = len(task_ids)
            
            # Create indices
            indices = list(range(n_tasks))
            if shuffle:
                np.random.shuffle(indices)
            
            # Stream batches
            for i in range(0, n_tasks, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_tasks = []
                
                for idx in batch_indices:
                    task_id = task_ids[idx]
                    task = self._load_single_task_hdf5(f, task_id)
                    batch_tasks.append(task)
                
                yield batch_tasks
    
    def _load_single_task_hdf5(self, hdf5_file, task_id: str) -> FluidTask:
        """Load a single task from HDF5 file."""
        # Load configuration
        config_str = hdf5_file['configs'][task_id][()]
        if isinstance(config_str, bytes):
            config_str = config_str.decode('utf-8')
        config_dict = json.loads(config_str)
        config = TaskConfig.from_dict(config_dict)
        
        # Load support set
        support_set = {}
        for key in hdf5_file['support_sets'][task_id].keys():
            support_set[key] = torch.from_numpy(hdf5_file['support_sets'][task_id][key][:])
        
        # Load query set
        query_set = {}
        for key in hdf5_file['query_sets'][task_id].keys():
            query_set[key] = torch.from_numpy(hdf5_file['query_sets'][task_id][key][:])
        
        # Load metadata
        metadata_str = hdf5_file['metadata'][task_id][()]
        if isinstance(metadata_str, bytes):
            metadata_str = metadata_str.decode('utf-8')
        metadata = json.loads(metadata_str)
        
        return FluidTask(
            config=config,
            support_set=support_set,
            query_set=query_set,
            metadata=metadata
        )
    
    def _get_split_info(self, split_path: Path) -> Dict[str, Any]:
        """Get information about a dataset split."""
        info = {'path': str(split_path)}
        
        # Try to get task count efficiently
        file_format = self.metadata['file_format']
        
        if file_format == 'hdf5':
            hdf5_path = split_path / 'tasks.h5'
            if hdf5_path.exists():
                with h5py.File(hdf5_path, 'r') as f:
                    info['n_tasks'] = len(f['configs'].keys())
                    info['file_size_mb'] = hdf5_path.stat().st_size / (1024 * 1024)
        
        elif file_format == 'pickle':
            pkl_path = split_path / 'tasks.pkl'
            if pkl_path.exists():
                info['file_size_mb'] = pkl_path.stat().st_size / (1024 * 1024)
                # For pickle, we need to load to count (expensive)
                with open(pkl_path, 'rb') as f:
                    tasks = pickle.load(f)
                    info['n_tasks'] = len(tasks)
        
        elif file_format == 'numpy':
            npz_path = split_path / 'tasks.npz'
            if npz_path.exists():
                info['file_size_mb'] = npz_path.stat().st_size / (1024 * 1024)
                # Count config keys to get task count
                data = np.load(npz_path, allow_pickle=True)
                config_keys = [k for k in data.keys() if '_config' in k]
                info['n_tasks'] = len(config_keys)
        
        return info
    
    def _prefetch_batch(self, split: str, batch_indices: List[int]):
        """Prefetch a batch of tasks for efficiency."""
        if len(batch_indices) == 0:
            return
        
        prefetch_key = f"{split}_{min(batch_indices)}_{max(batch_indices)}"
        
        # Check if already prefetched
        if prefetch_key in self.prefetch_cache:
            return
        
        # Prefetch in background (simplified - in practice could use threading)
        try:
            prefetched_tasks = self.load_batch(split, batch_indices)
            
            # Store in prefetch cache with size limit
            if len(self.prefetch_cache) >= self.prefetch_size:
                # Remove oldest prefetched batch
                oldest_key = next(iter(self.prefetch_cache))
                del self.prefetch_cache[oldest_key]
            
            self.prefetch_cache[prefetch_key] = prefetched_tasks
            
        except Exception as e:
            logger.warning(f"Prefetch failed: {e}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        return {
            'metadata': self.metadata,
            'statistics': self.statistics,
            'performance_metrics': self.performance_metrics,
            'split_info': self.split_info,
            'cache_stats': {
                'cache_size': len(self.task_cache),
                'prefetch_size': len(self.prefetch_cache),
                'max_cache_size': self.cache_size,
                'max_prefetch_size': self.prefetch_size
            }
        }
    
    def clear_cache(self):
        """Clear all caches to free memory."""
        self.task_cache.clear()
        self.prefetch_cache.clear()
        logger.info("Cleared all caches")
    
    def validate_dataset_integrity(self) -> Dict[str, Any]:
        """Validate dataset integrity and return validation report."""
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'split_validation': {}
        }
        
        try:
            # Check metadata consistency
            expected_checksum = self.metadata.get('checksum')
            if expected_checksum:
                actual_checksum = self._compute_dataset_checksum()
                if actual_checksum != expected_checksum:
                    validation_report['errors'].append(
                        f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
                    )
                    validation_report['valid'] = False
            
            # Validate each split
            for split in ['train', 'val', 'test']:
                if split in self.split_info:
                    split_validation = self._validate_split(split)
                    validation_report['split_validation'][split] = split_validation
                    
                    if not split_validation['valid']:
                        validation_report['valid'] = False
                        validation_report['errors'].extend(split_validation['errors'])
        
        except Exception as e:
            validation_report['valid'] = False
            validation_report['errors'].append(f"Validation failed: {str(e)}")
        
        return validation_report
    
    def _validate_split(self, split: str) -> Dict[str, Any]:
        """Validate a specific dataset split."""
        split_validation = {
            'valid': True,
            'errors': [],
            'n_tasks_checked': 0,
            'n_tasks_valid': 0
        }
        
        try:
            # Sample a few tasks for validation
            n_tasks = self.split_info[split]['n_tasks']
            sample_size = min(10, n_tasks)  # Check up to 10 tasks
            sample_indices = np.random.choice(n_tasks, sample_size, replace=False)
            
            sample_tasks = self.load_batch(split, sample_indices.tolist())
            split_validation['n_tasks_checked'] = len(sample_tasks)
            
            for task in sample_tasks:
                if self._validate_single_task(task):
                    split_validation['n_tasks_valid'] += 1
                else:
                    split_validation['errors'].append(f"Invalid task: {task.config.task_id}")
            
            if split_validation['n_tasks_valid'] < split_validation['n_tasks_checked']:
                split_validation['valid'] = False
        
        except Exception as e:
            split_validation['valid'] = False
            split_validation['errors'].append(f"Split validation failed: {str(e)}")
        
        return split_validation
    
    def _validate_single_task(self, task: FluidTask) -> bool:
        """Validate a single task."""
        try:
            # Check basic structure
            if not hasattr(task, 'config') or not hasattr(task, 'support_set') or not hasattr(task, 'query_set'):
                return False
            
            # Check tensor validity
            for dataset_name, dataset in [('support_set', task.support_set), ('query_set', task.query_set)]:
                for key, tensor in dataset.items():
                    if not torch.all(torch.isfinite(tensor)):
                        return False
                    if tensor.numel() == 0:
                        return False
            
            return True
            
        except Exception:
            return False
    
    def _compute_dataset_checksum(self) -> str:
        """Compute current dataset checksum."""
        hasher = hashlib.md5()
        
        # Sort files for consistent checksum
        files = sorted(self.dataset_path.rglob('*'))
        
        # Exclude files that change after dataset creation
        excluded_files = {'performance.json', 'cache_stats.json'}
        
        for file_path in files:
            if file_path.is_file() and file_path.name not in excluded_files:
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _load_tasks_hdf5(self, file_path: Path) -> List[FluidTask]:
        """Load tasks from HDF5 file."""
        tasks = []
        
        with h5py.File(file_path, 'r') as f:
            configs_group = f['configs']
            support_group = f['support_sets']
            query_group = f['query_sets']
            metadata_group = f['metadata']
            
            for task_id in configs_group.keys():
                # Load configuration
                config_str = configs_group[task_id][()]
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                config_dict = json.loads(config_str)
                config = TaskConfig.from_dict(config_dict)
                
                # Load support set
                support_set = {}
                for key in support_group[task_id].keys():
                    support_set[key] = torch.from_numpy(support_group[task_id][key][:])
                
                # Load query set
                query_set = {}
                for key in query_group[task_id].keys():
                    query_set[key] = torch.from_numpy(query_group[task_id][key][:])
                
                # Load metadata
                metadata_str = metadata_group[task_id][()]
                if isinstance(metadata_str, bytes):
                    metadata_str = metadata_str.decode('utf-8')
                metadata = json.loads(metadata_str)
                
                # Create task
                task = FluidTask(
                    config=config,
                    support_set=support_set,
                    query_set=query_set,
                    metadata=metadata
                )
                tasks.append(task)
        
        return tasks
    
    def _load_tasks_pickle(self, file_path: Path) -> List[FluidTask]:
        """Load tasks from pickle file."""
        with open(file_path, 'rb') as f:
            task_dicts = pickle.load(f)
        
        return [FluidTask.from_dict(task_dict) for task_dict in task_dicts]
    
    def _load_tasks_numpy(self, file_path: Path) -> List[FluidTask]:
        """Load tasks from numpy file."""
        data = np.load(file_path, allow_pickle=True)
        
        # Group data by task
        task_data = {}
        for key in data.keys():
            if '_config' in key:
                task_id = key.replace('_config', '')
                if task_id not in task_data:
                    task_data[task_id] = {}
                task_data[task_id]['config'] = json.loads(str(data[key]))
            
            elif '_support_' in key:
                parts = key.split('_support_')
                task_id, data_key = parts[0], parts[1]
                if task_id not in task_data:
                    task_data[task_id] = {}
                if 'support_set' not in task_data[task_id]:
                    task_data[task_id]['support_set'] = {}
                task_data[task_id]['support_set'][data_key] = torch.from_numpy(data[key])
            
            elif '_query_' in key:
                parts = key.split('_query_')
                task_id, data_key = parts[0], parts[1]
                if task_id not in task_data:
                    task_data[task_id] = {}
                if 'query_set' not in task_data[task_id]:
                    task_data[task_id]['query_set'] = {}
                task_data[task_id]['query_set'][data_key] = torch.from_numpy(data[key])
            
            elif '_metadata' in key:
                task_id = key.replace('_metadata', '')
                if task_id not in task_data:
                    task_data[task_id] = {}
                task_data[task_id]['metadata'] = json.loads(str(data[key]))
        
        # Create tasks
        tasks = []
        for task_id in sorted(task_data.keys()):
            task_dict = task_data[task_id]
            config = TaskConfig.from_dict(task_dict['config'])
            
            task = FluidTask(
                config=config,
                support_set=task_dict['support_set'],
                query_set=task_dict['query_set'],
                metadata=task_dict.get('metadata', {})
            )
            tasks.append(task)
        
        return tasks


def _generate_tasks_worker(args: Tuple) -> List[FluidTask]:
    """
    Worker function for parallel task generation.
    
    This function is defined at module level to be picklable for multiprocessing.
    """
    n_tasks, n_support, n_query, seed, data_config, use_fenicsx = args
    
    # Initialize generators with worker-specific seed
    task_generator = FluidTaskGenerator(data_config, seed=seed)
    analytical_generator = AnalyticalSolutionGenerator()
    
    # Initialize FEniCSx solver if requested
    fenicsx_solver = None
    if use_fenicsx:
        solver_config = SolverConfig(mesh_resolution=(30, 15))  # Coarse for speed
        fenicsx_solver = create_fenicsx_solver(solver_config)
    
    # Generate tasks
    tasks = task_generator.generate_task_batch(n_tasks, n_support, n_query)
    
    # Generate solutions for each task
    for task in tasks:
        try:
            # Try FEniCSx first if available
            if fenicsx_solver is not None:
                try:
                    coords = torch.cat([task.support_set['coords'], task.query_set['coords']], dim=0)
                    solution = fenicsx_solver.solve_task(task.config, coords)
                    
                    # Split solution back to support and query
                    n_support_actual = task.support_set['coords'].shape[0]
                    
                    task.support_set['velocity'] = solution.velocity[:n_support_actual]
                    task.support_set['pressure'] = solution.pressure[:n_support_actual]
                    
                    task.query_set['velocity'] = solution.velocity[n_support_actual:]
                    task.query_set['pressure'] = solution.pressure[n_support_actual:]
                    
                    task.ground_truth = {
                        'velocity': solution.velocity,
                        'pressure': solution.pressure,
                        'viscosity_field': solution.viscosity_field
                    }
                    
                    task.metadata['solution_method'] = 'fenicsx'
                    continue
                    
                except Exception as e:
                    # Fall back to analytical solution
                    pass
            
            # Use analytical solution as fallback
            coords = torch.cat([task.support_set['coords'], task.query_set['coords']], dim=0)
            solution = analytical_generator.generate_solution(task.config, coords)
            
            # Split solution back to support and query
            n_support_actual = task.support_set['coords'].shape[0]
            
            task.support_set['velocity'] = solution.velocity[:n_support_actual]
            task.support_set['pressure'] = solution.pressure[:n_support_actual]
            
            task.query_set['velocity'] = solution.velocity[n_support_actual:]
            task.query_set['pressure'] = solution.pressure[n_support_actual:]
            
            task.ground_truth = {
                'velocity': solution.velocity,
                'pressure': solution.pressure,
                'viscosity_field': solution.viscosity_field
            }
            
            task.metadata['solution_method'] = 'analytical'
            
        except Exception as e:
            # If all else fails, keep placeholder zeros
            task.metadata['solution_method'] = 'placeholder'
            task.metadata['generation_error'] = str(e)
    
    return tasks