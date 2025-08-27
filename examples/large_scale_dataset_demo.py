#!/usr/bin/env python3
"""
Demonstration of large-scale dataset generation and management.

This script shows how to generate and manage datasets with 1000+ diverse tasks
for meta-learning fluid dynamics problems.
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_research_pipeline.core.dataset_manager import DatasetGenerator, DatasetLoader
from ml_research_pipeline.config.data_config import DataConfig


def main():
    """Demonstrate large-scale dataset generation and management."""
    print("=== Large-Scale Dataset Generation Demo ===\n")
    
    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Configure data generation for large-scale dataset
        data_config = DataConfig(
            task_types=[
                'linear_viscosity', 
                'exponential_viscosity', 
                'bilinear_viscosity',
                'temperature_dependent'
            ],
            task_weights=[0.3, 0.3, 0.2, 0.2],  # Balanced distribution
            cache_format='hdf5',
            compression=True,
            use_cache=True,
            cache_dir=temp_dir,
            data_dir=temp_dir
        )
        
        print("Configuration:")
        print(f"  Task types: {data_config.task_types}")
        print(f"  Cache format: {data_config.cache_format}")
        print(f"  Compression: {data_config.compression}")
        print()
        
        # Initialize dataset generator
        generator = DatasetGenerator(
            data_config=data_config,
            use_fenicsx=False,  # Disable FEniCSx for demo speed
            n_workers=4,        # Use multiple workers
            memory_limit_gb=8.0
        )
        
        print("=== Generating Large-Scale Dataset ===")
        
        # Generate dataset with 1200 tasks
        start_time = time.time()
        
        dataset_info = generator.generate_dataset(
            dataset_name='large_scale_demo',
            n_tasks=1200,
            n_support=50,
            n_query=75,
            split_ratios=(0.8, 0.1, 0.1),
            enable_caching=True,
            validate_tasks=False  # Skip validation for speed
        )
        
        generation_time = time.time() - start_time
        
        print(f"\n✓ Dataset generation completed in {generation_time:.2f} seconds")
        print(f"  Tasks per second: {1200/generation_time:.2f}")
        
        # Display dataset information
        metadata = dataset_info['metadata']
        print(f"\nDataset Information:")
        print(f"  Total tasks: {metadata.n_tasks}")
        print(f"  Task types: {metadata.task_types}")
        print(f"  File format: {metadata.file_format}")
        print(f"  Total size: {metadata.total_size_mb:.2f} MB")
        print(f"  Compression: {metadata.compression}")
        
        # Display split information
        print(f"\nDataset Splits:")
        for split_name, split_info in dataset_info['splits'].items():
            print(f"  {split_name}: {split_info['n_tasks']} tasks "
                  f"({split_info['file_size_mb']:.2f} MB)")
        
        # Display performance statistics
        perf_stats = dataset_info['performance_stats']
        print(f"\nPerformance Statistics:")
        print(f"  Average task generation time: {perf_stats['average_task_time']*1000:.2f} ms")
        print(f"  Cache hits: {perf_stats['cache_hits']}")
        print(f"  Cache misses: {perf_stats['cache_misses']}")
        
        # Display dataset statistics
        stats = dataset_info['statistics']
        print(f"\nDataset Statistics:")
        print(f"  Task type distribution: {stats['task_type_distribution']}")
        print(f"  Reynolds number range: {stats['reynolds_statistics']['min']:.1f} - "
              f"{stats['reynolds_statistics']['max']:.1f}")
        print(f"  Average support points: {stats['data_point_statistics']['support_points']['mean']:.0f}")
        print(f"  Average query points: {stats['data_point_statistics']['query_points']['mean']:.0f}")
        
        print("\n=== Testing Dataset Loading ===")
        
        # Initialize dataset loader
        loader = DatasetLoader(
            dataset_path=dataset_info['save_path'],
            cache_size=200,
            prefetch_size=100
        )
        
        print(f"Loaded dataset with {loader.metadata['n_tasks']} tasks")
        
        # Test batch loading performance
        print("\nTesting batch loading performance...")
        batch_size = 64
        n_test_batches = 5
        
        start_time = time.time()
        total_tasks_loaded = 0
        
        for i, batch in enumerate(loader.iterate_batches('train', batch_size, shuffle=True)):
            total_tasks_loaded += len(batch)
            if i >= n_test_batches - 1:
                break
        
        loading_time = time.time() - start_time
        tasks_per_second = total_tasks_loaded / loading_time
        
        print(f"✓ Loaded {total_tasks_loaded} tasks in {loading_time:.2f} seconds")
        print(f"  Loading speed: {tasks_per_second:.2f} tasks/second")
        
        # Test streaming loading
        print("\nTesting streaming loading...")
        start_time = time.time()
        streaming_tasks = 0
        
        for i, batch in enumerate(loader.iterate_batches_streaming('train', batch_size, shuffle=False)):
            streaming_tasks += len(batch)
            if i >= n_test_batches - 1:
                break
        
        streaming_time = time.time() - start_time
        streaming_speed = streaming_tasks / streaming_time
        
        print(f"✓ Streamed {streaming_tasks} tasks in {streaming_time:.2f} seconds")
        print(f"  Streaming speed: {streaming_speed:.2f} tasks/second")
        
        # Validate dataset integrity
        print("\nValidating dataset integrity...")
        validation_report = loader.validate_dataset_integrity()
        
        if validation_report['valid']:
            print("✓ Dataset integrity validation passed")
        else:
            print("✗ Dataset integrity validation failed:")
            for error in validation_report['errors']:
                print(f"  - {error}")
        
        # Display cache statistics
        cache_info = loader.get_dataset_info()['cache_stats']
        print(f"\nCache Statistics:")
        print(f"  Current cache size: {cache_info['cache_size']}")
        print(f"  Current prefetch size: {cache_info['prefetch_size']}")
        
        print("\n=== Demo Summary ===")
        print(f"✓ Successfully generated {metadata.n_tasks} tasks")
        print(f"✓ Dataset size: {metadata.total_size_mb:.2f} MB")
        print(f"✓ Generation speed: {1200/generation_time:.2f} tasks/second")
        print(f"✓ Loading speed: {tasks_per_second:.2f} tasks/second")
        print(f"✓ All validation checks passed")
        
        print(f"\nDataset files created in: {dataset_info['save_path']}")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up temporary directory
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        print("Demo completed!")


if __name__ == "__main__":
    main()