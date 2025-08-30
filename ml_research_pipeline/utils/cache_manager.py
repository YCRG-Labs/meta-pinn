"""
Intelligent caching system for expensive computations in physics discovery.

This module provides memory-efficient caching with intelligent invalidation
strategies, compression, and persistence for computationally intensive
physics discovery operations.
"""

import gzip
import hashlib
import json
import logging
import os
import pickle
import shutil
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""

    key: str
    value: Any
    timestamp: float
    access_count: int
    last_access: float
    size_bytes: int
    ttl: Optional[float] = None  # Time to live in seconds
    dependencies: Optional[List[str]] = None  # Cache keys this entry depends on

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def touch(self):
        """Update access information."""
        self.access_count += 1
        self.last_access = time.time()


class CacheKeyGenerator:
    """Generates consistent cache keys for various input types."""

    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """Generate a consistent cache key from arguments."""
        # Create a deterministic representation
        key_data = {
            "args": CacheKeyGenerator._serialize_args(args),
            "kwargs": CacheKeyGenerator._serialize_kwargs(kwargs),
        }

        # Convert to JSON string for hashing
        key_str = json.dumps(key_data, sort_keys=True, default=str)

        # Generate hash
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    @staticmethod
    def _serialize_args(args: tuple) -> List:
        """Serialize arguments for key generation."""
        serialized = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                # For numpy arrays, use shape, dtype, and hash of data
                serialized.append(
                    {
                        "type": "ndarray",
                        "shape": arg.shape,
                        "dtype": str(arg.dtype),
                        "hash": hashlib.md5(arg.tobytes()).hexdigest()[:8],
                    }
                )
            elif hasattr(arg, "__dict__"):
                # For objects with attributes
                serialized.append(
                    {
                        "type": type(arg).__name__,
                        "attrs": str(sorted(arg.__dict__.items())),
                    }
                )
            else:
                serialized.append(arg)
        return serialized

    @staticmethod
    def _serialize_kwargs(kwargs: dict) -> Dict:
        """Serialize keyword arguments for key generation."""
        serialized = {}
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                serialized[key] = {
                    "type": "ndarray",
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                    "hash": hashlib.md5(value.tobytes()).hexdigest()[:8],
                }
            elif hasattr(value, "__dict__"):
                serialized[key] = {
                    "type": type(value).__name__,
                    "attrs": str(sorted(value.__dict__.items())),
                }
            else:
                serialized[key] = value
        return serialized


class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""

    @abstractmethod
    def select_for_eviction(
        self, entries: Dict[str, CacheEntry], target_size: int
    ) -> List[str]:
        """Select cache entries for eviction."""
        pass


class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""

    def select_for_eviction(
        self, entries: Dict[str, CacheEntry], target_size: int
    ) -> List[str]:
        """Select least recently used entries for eviction."""
        # Sort by last access time
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].last_access)

        evicted = []
        current_size = sum(entry.size_bytes for entry in entries.values())

        for key, entry in sorted_entries:
            if current_size <= target_size:
                break
            evicted.append(key)
            current_size -= entry.size_bytes

        return evicted


class LFUEvictionPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy."""

    def select_for_eviction(
        self, entries: Dict[str, CacheEntry], target_size: int
    ) -> List[str]:
        """Select least frequently used entries for eviction."""
        # Sort by access count, then by last access time
        sorted_entries = sorted(
            entries.items(), key=lambda x: (x[1].access_count, x[1].last_access)
        )

        evicted = []
        current_size = sum(entry.size_bytes for entry in entries.values())

        for key, entry in sorted_entries:
            if current_size <= target_size:
                break
            evicted.append(key)
            current_size -= entry.size_bytes

        return evicted


class SizeBasedEvictionPolicy(EvictionPolicy):
    """Evict largest entries first."""

    def select_for_eviction(
        self, entries: Dict[str, CacheEntry], target_size: int
    ) -> List[str]:
        """Select largest entries for eviction."""
        # Sort by size (largest first)
        sorted_entries = sorted(
            entries.items(), key=lambda x: x[1].size_bytes, reverse=True
        )

        evicted = []
        current_size = sum(entry.size_bytes for entry in entries.values())

        for key, entry in sorted_entries:
            if current_size <= target_size:
                break
            evicted.append(key)
            current_size -= entry.size_bytes

        return evicted


class CacheStorage(ABC):
    """Abstract base class for cache storage backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from storage."""
        pass

    @abstractmethod
    def put(self, key: str, value: Any) -> bool:
        """Store value in storage."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from storage."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all stored values."""
        pass

    @abstractmethod
    def get_size(self, key: str) -> int:
        """Get size of stored value in bytes."""
        pass


class MemoryStorage(CacheStorage):
    """In-memory cache storage."""

    def __init__(self):
        self.storage = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from memory."""
        with self.lock:
            return self.storage.get(key)

    def put(self, key: str, value: Any) -> bool:
        """Store value in memory."""
        with self.lock:
            self.storage[key] = value
            return True

    def delete(self, key: str) -> bool:
        """Delete value from memory."""
        with self.lock:
            if key in self.storage:
                del self.storage[key]
                return True
            return False

    def clear(self):
        """Clear all stored values."""
        with self.lock:
            self.storage.clear()

    def get_size(self, key: str) -> int:
        """Get approximate size of stored value."""
        with self.lock:
            if key in self.storage:
                try:
                    return len(pickle.dumps(self.storage[key]))
                except Exception:
                    return 1024  # Default estimate
            return 0


class DiskStorage(CacheStorage):
    """Disk-based cache storage with compression."""

    def __init__(self, cache_dir: Union[str, Path], compress: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compress = compress
        self.lock = threading.RLock()

    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        extension = ".pkl.gz" if self.compress else ".pkl"
        return self.cache_dir / f"{key}{extension}"

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from disk."""
        file_path = self._get_file_path(key)

        with self.lock:
            if not file_path.exists():
                return None

            try:
                if self.compress:
                    with gzip.open(file_path, "rb") as f:
                        return pickle.load(f)
                else:
                    with open(file_path, "rb") as f:
                        return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache entry {key}: {e}")
                # Clean up corrupted file
                try:
                    file_path.unlink()
                except Exception:
                    pass
                return None

    def put(self, key: str, value: Any) -> bool:
        """Store value on disk."""
        file_path = self._get_file_path(key)

        with self.lock:
            try:
                if self.compress:
                    with gzip.open(file_path, "wb") as f:
                        pickle.dump(value, f)
                else:
                    with open(file_path, "wb") as f:
                        pickle.dump(value, f)
                return True
            except Exception as e:
                logger.warning(f"Failed to store cache entry {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """Delete value from disk."""
        file_path = self._get_file_path(key)

        with self.lock:
            try:
                if file_path.exists():
                    file_path.unlink()
                    return True
                return False
            except Exception as e:
                logger.warning(f"Failed to delete cache entry {key}: {e}")
                return False

    def clear(self):
        """Clear all stored values."""
        with self.lock:
            try:
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to clear cache directory: {e}")

    def get_size(self, key: str) -> int:
        """Get size of stored file."""
        file_path = self._get_file_path(key)

        with self.lock:
            try:
                if file_path.exists():
                    return file_path.stat().st_size
                return 0
            except Exception:
                return 0


class CacheManager:
    """
    Intelligent cache manager for expensive computations.

    Provides memory-efficient caching with configurable eviction policies,
    TTL support, dependency tracking, and both memory and disk storage options.
    """

    def __init__(
        self,
        max_memory_size: int = 1024 * 1024 * 1024,  # 1GB
        eviction_policy: str = "lru",
        enable_disk_cache: bool = False,
        disk_cache_dir: Optional[Union[str, Path]] = None,
        compress_disk_cache: bool = True,
        default_ttl: Optional[float] = None,
        cleanup_interval: float = 300.0,
    ):  # 5 minutes
        """
        Initialize the cache manager.

        Args:
            max_memory_size: Maximum memory usage in bytes
            eviction_policy: Eviction policy ('lru', 'lfu', 'size')
            enable_disk_cache: Whether to enable disk caching
            disk_cache_dir: Directory for disk cache
            compress_disk_cache: Whether to compress disk cache files
            default_ttl: Default time-to-live for cache entries
            cleanup_interval: Interval for cleanup operations in seconds
        """
        self.max_memory_size = max_memory_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        # Initialize storage
        self.memory_storage = MemoryStorage()
        self.disk_storage = None
        if enable_disk_cache:
            cache_dir = disk_cache_dir or Path.cwd() / ".cache" / "physics_discovery"
            self.disk_storage = DiskStorage(cache_dir, compress_disk_cache)

        # Initialize eviction policy
        self.eviction_policy = self._create_eviction_policy(eviction_policy)

        # Cache metadata
        self.entries: Dict[str, CacheEntry] = {}
        self.dependencies: Dict[str, List[str]] = (
            {}
        )  # key -> list of keys that depend on this key
        self.dependents: Dict[str, List[str]] = (
            {}
        )  # key -> list of keys this key depends on

        # Thread safety
        self.lock = threading.RLock()

        # Background cleanup
        self.cleanup_thread = None
        self.cleanup_running = False
        self._start_cleanup_thread()

    def _create_eviction_policy(self, policy_name: str) -> EvictionPolicy:
        """Create eviction policy instance."""
        policies = {
            "lru": LRUEvictionPolicy,
            "lfu": LFUEvictionPolicy,
            "size": SizeBasedEvictionPolicy,
        }

        if policy_name not in policies:
            raise ValueError(f"Unknown eviction policy: {policy_name}")

        return policies[policy_name]()

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.cleanup_running:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired_entries()
                self._enforce_memory_limit()
            except Exception as e:
                logger.warning(f"Cache cleanup error: {e}")

    def _cleanup_expired_entries(self):
        """Remove expired cache entries."""
        with self.lock:
            expired_keys = []
            for key, entry in self.entries.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_entry(key)
                logger.debug(f"Removed expired cache entry: {key}")

    def _enforce_memory_limit(self):
        """Enforce memory usage limits through eviction."""
        with self.lock:
            current_size = sum(entry.size_bytes for entry in self.entries.values())

            if current_size > self.max_memory_size:
                target_size = int(self.max_memory_size * 0.8)  # Evict to 80% of limit
                keys_to_evict = self.eviction_policy.select_for_eviction(
                    self.entries, target_size
                )

                for key in keys_to_evict:
                    self._remove_entry(key)
                    logger.debug(f"Evicted cache entry: {key}")

    def _remove_entry(self, key: str):
        """Remove cache entry and clean up dependencies."""
        with self.lock:
            # Remove from memory
            self.memory_storage.delete(key)

            # Remove from disk if enabled
            if self.disk_storage:
                self.disk_storage.delete(key)

            # Remove metadata
            if key in self.entries:
                del self.entries[key]

            # Clean up dependencies - invalidate entries that depend on this key
            if key in self.dependencies:
                # Get list of dependents before modifying the dictionary
                dependents_to_remove = self.dependencies[key].copy()
                del self.dependencies[key]

                # Invalidate dependent entries
                for dependent_key in dependents_to_remove:
                    if dependent_key in self.entries:
                        self._remove_entry(dependent_key)

            # Remove this key from dependents tracking
            if key in self.dependents:
                # Remove this key from the dependency lists of its dependencies
                for dep_key in self.dependents[key]:
                    if (
                        dep_key in self.dependencies
                        and key in self.dependencies[dep_key]
                    ):
                        self.dependencies[dep_key].remove(key)
                        if not self.dependencies[dep_key]:
                            del self.dependencies[dep_key]
                del self.dependents[key]

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self.lock:
            # Check if entry exists and is not expired
            if key not in self.entries:
                return None

            entry = self.entries[key]
            if entry.is_expired():
                self._remove_entry(key)
                return None

            # Try memory first
            value = self.memory_storage.get(key)
            if value is not None:
                entry.touch()
                return value

            # Try disk if enabled
            if self.disk_storage:
                value = self.disk_storage.get(key)
                if value is not None:
                    # Promote to memory
                    self.memory_storage.put(key, value)
                    entry.touch()
                    return value

            # Entry exists in metadata but not in storage - clean up
            self._remove_entry(key)
            return None

    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        dependencies: Optional[List[str]] = None,
    ) -> bool:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (overrides default)
            dependencies: List of cache keys this entry depends on

        Returns:
            True if successfully cached
        """
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                logger.warning(f"Cannot serialize value for key {key}")
                return False

            # Check if value is too large
            if size_bytes > self.max_memory_size:
                logger.warning(f"Value too large for cache: {key} ({size_bytes} bytes)")
                return False

            # Remove existing entry if present
            if key in self.entries:
                self._remove_entry(key)

            # Create cache entry
            entry_ttl = ttl if ttl is not None else self.default_ttl
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                access_count=1,
                last_access=time.time(),
                size_bytes=size_bytes,
                ttl=entry_ttl,
                dependencies=dependencies or [],
            )

            # Store in memory
            if not self.memory_storage.put(key, value):
                return False

            # Store in disk if enabled
            if self.disk_storage:
                self.disk_storage.put(key, value)

            # Update metadata
            self.entries[key] = entry

            # Update dependencies
            if dependencies:
                self.dependents[key] = dependencies.copy()
                for dep_key in dependencies:
                    if dep_key not in self.dependencies:
                        self.dependencies[dep_key] = []
                    self.dependencies[dep_key].append(key)

            # Enforce memory limits
            self._enforce_memory_limit()

            return True

    def invalidate(self, key: str) -> bool:
        """
        Invalidate cache entry and its dependents.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was found and invalidated
        """
        with self.lock:
            if key in self.entries:
                self._remove_entry(key)
                return True
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache entries matching a pattern.

        Args:
            pattern: Pattern to match (simple string matching)

        Returns:
            Number of entries invalidated
        """
        with self.lock:
            matching_keys = [key for key in self.entries.keys() if pattern in key]

            for key in matching_keys:
                self._remove_entry(key)

            return len(matching_keys)

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.memory_storage.clear()
            if self.disk_storage:
                self.disk_storage.clear()
            self.entries.clear()
            self.dependencies.clear()
            self.dependents.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_size = sum(entry.size_bytes for entry in self.entries.values())
            total_entries = len(self.entries)

            if total_entries > 0:
                avg_size = total_size / total_entries
                avg_access_count = (
                    sum(e.access_count for e in self.entries.values()) / total_entries
                )
            else:
                avg_size = 0
                avg_access_count = 0

            return {
                "total_entries": total_entries,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "memory_usage_percent": (total_size / self.max_memory_size) * 100,
                "avg_entry_size_bytes": avg_size,
                "avg_access_count": avg_access_count,
                "max_memory_size_mb": self.max_memory_size / (1024 * 1024),
                "disk_cache_enabled": self.disk_storage is not None,
            }

    def cached_function(
        self,
        ttl: Optional[float] = None,
        dependencies: Optional[List[str]] = None,
        key_generator: Optional[Callable] = None,
    ):
        """
        Decorator for caching function results.

        Args:
            ttl: Time-to-live for cached results
            dependencies: Static dependencies for all calls
            key_generator: Custom key generation function

        Returns:
            Decorated function
        """

        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_generator:
                    cache_key = key_generator(*args, **kwargs)
                else:
                    func_name = f"{func.__module__}.{func.__name__}"
                    arg_key = CacheKeyGenerator.generate_key(*args, **kwargs)
                    cache_key = f"{func_name}:{arg_key}"

                # Try to get from cache
                result = self.get(cache_key)
                if result is not None:
                    return result

                # Compute result
                result = func(*args, **kwargs)

                # Cache result
                self.put(cache_key, result, ttl=ttl, dependencies=dependencies)

                return result

            return wrapper

        return decorator

    def shutdown(self):
        """Shutdown cache manager and cleanup resources."""
        self.cleanup_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)

        # Final cleanup
        with self.lock:
            self.clear()


# Convenience function for creating a global cache manager
_global_cache_manager = None


def get_global_cache_manager(**kwargs) -> CacheManager:
    """Get or create global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(**kwargs)
    return _global_cache_manager


def cached(
    ttl: Optional[float] = None,
    dependencies: Optional[List[str]] = None,
    key_generator: Optional[Callable] = None,
):
    """
    Convenience decorator using global cache manager.

    Args:
        ttl: Time-to-live for cached results
        dependencies: Static dependencies for all calls
        key_generator: Custom key generation function

    Returns:
        Decorated function
    """
    cache_manager = get_global_cache_manager()
    return cache_manager.cached_function(
        ttl=ttl, dependencies=dependencies, key_generator=key_generator
    )
