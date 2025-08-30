"""
Fallback strategy manager for handling method failures with graceful degradation.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from .error_handler import ErrorCategory, ErrorHandler, ErrorInfo, ErrorSeverity
from .logging_utils import LoggerMixin


class FallbackMode(Enum):
    """Fallback operation modes."""

    AUTOMATIC = "automatic"
    MANUAL = "manual"
    DISABLED = "disabled"


class PerformanceLevel(Enum):
    """Performance levels for graceful degradation."""

    FULL = "full"
    REDUCED = "reduced"
    MINIMAL = "minimal"
    EMERGENCY = "emergency"


@dataclass
class MethodConfig:
    """Configuration for a method with fallback options."""

    name: str
    primary_method: Callable
    fallback_methods: List[Callable] = field(default_factory=list)
    performance_level: PerformanceLevel = PerformanceLevel.FULL
    timeout: Optional[float] = None
    max_retries: int = 3
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "performance_level": self.performance_level.value,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "enabled": self.enabled,
            "fallback_count": len(self.fallback_methods),
        }


@dataclass
class FallbackResult:
    """Result of a fallback operation."""

    success: bool
    result: Any = None
    method_used: str = ""
    performance_level: PerformanceLevel = PerformanceLevel.FULL
    execution_time: float = 0.0
    error_info: Optional[ErrorInfo] = None
    fallback_applied: bool = False
    degradation_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "method_used": self.method_used,
            "performance_level": self.performance_level.value,
            "execution_time": self.execution_time,
            "fallback_applied": self.fallback_applied,
            "degradation_applied": self.degradation_applied,
            "error_occurred": self.error_info is not None,
        }


class FallbackStrategyManager(LoggerMixin):
    """Manager for handling method failures with automatic fallback and graceful degradation."""

    def __init__(
        self,
        error_handler: ErrorHandler,
        fallback_mode: FallbackMode = FallbackMode.AUTOMATIC,
        enable_performance_monitoring: bool = True,
        performance_log_file: Optional[Path] = None,
    ):
        """Initialize fallback strategy manager.

        Args:
            error_handler: ErrorHandler instance for error management
            fallback_mode: Mode for fallback operations
            enable_performance_monitoring: Whether to monitor performance
            performance_log_file: Optional file to log performance metrics
        """
        self.error_handler = error_handler
        self.fallback_mode = fallback_mode
        self.enable_performance_monitoring = enable_performance_monitoring
        self.performance_log_file = performance_log_file

        # Method configurations
        self.method_configs: Dict[str, MethodConfig] = {}

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.method_success_rates: Dict[str, float] = {}
        self.method_avg_times: Dict[str, float] = {}

        # Degradation settings
        self.current_performance_level = PerformanceLevel.FULL
        self.degradation_thresholds = {
            PerformanceLevel.REDUCED: 0.7,  # Switch if success rate < 70%
            PerformanceLevel.MINIMAL: 0.5,  # Switch if success rate < 50%
            PerformanceLevel.EMERGENCY: 0.3,  # Switch if success rate < 30%
        }

        # Fallback statistics
        self.fallback_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "fallback_used": 0,
            "degradation_used": 0,
            "method_failures": {},
        }

    def register_method(
        self,
        name: str,
        primary_method: Callable,
        fallback_methods: Optional[List[Callable]] = None,
        performance_level: PerformanceLevel = PerformanceLevel.FULL,
        timeout: Optional[float] = None,
        max_retries: int = 3,
    ) -> None:
        """Register a method with fallback options.

        Args:
            name: Unique name for the method
            primary_method: Primary method to execute
            fallback_methods: List of fallback methods
            performance_level: Required performance level
            timeout: Optional timeout for method execution
            max_retries: Maximum number of retries
        """
        if fallback_methods is None:
            fallback_methods = []

        config = MethodConfig(
            name=name,
            primary_method=primary_method,
            fallback_methods=fallback_methods,
            performance_level=performance_level,
            timeout=timeout,
            max_retries=max_retries,
        )

        self.method_configs[name] = config
        self.method_success_rates[name] = 1.0  # Start with optimistic rate
        self.method_avg_times[name] = 0.0

        self.log_info(
            f"Registered method '{name}' with {len(fallback_methods)} fallback options"
        )

    def execute_with_fallback(
        self, method_name: str, *args, **kwargs
    ) -> FallbackResult:
        """Execute a method with automatic fallback handling.

        Args:
            method_name: Name of the registered method
            *args: Arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            FallbackResult with execution details
        """
        if method_name not in self.method_configs:
            raise ValueError(f"Method '{method_name}' not registered")

        config = self.method_configs[method_name]

        if not config.enabled:
            return FallbackResult(
                success=False, method_used=method_name, error_info=None
            )

        self.fallback_stats["total_calls"] += 1
        start_time = time.time()

        # Check if current performance level meets requirements
        if self.current_performance_level.value != config.performance_level.value:
            if self._should_degrade_performance(config):
                return self._execute_with_degradation(config, *args, **kwargs)

        # Try primary method first
        result = self._try_method(config.primary_method, method_name, *args, **kwargs)

        if result.success:
            self._update_performance_stats(method_name, result.execution_time, True)
            self.fallback_stats["successful_calls"] += 1
            return result

        # Try fallback methods if primary fails and mode allows it
        if self.fallback_mode == FallbackMode.AUTOMATIC and config.fallback_methods:
            self.log_warning(f"Primary method '{method_name}' failed, trying fallbacks")

            for i, fallback_method in enumerate(config.fallback_methods):
                fallback_name = f"{method_name}_fallback_{i}"
                fallback_result = self._try_method(
                    fallback_method, fallback_name, *args, **kwargs
                )

                if fallback_result.success:
                    fallback_result.fallback_applied = True
                    fallback_result.method_used = fallback_name
                    self._update_performance_stats(
                        method_name, fallback_result.execution_time, True
                    )
                    self.fallback_stats["successful_calls"] += 1
                    self.fallback_stats["fallback_used"] += 1

                    self.log_info(f"Fallback method {i} succeeded for '{method_name}'")
                    return fallback_result

        # All methods failed
        self._update_performance_stats(method_name, time.time() - start_time, False)
        self._record_method_failure(method_name)

        # Try graceful degradation as last resort (only in automatic mode)
        if (
            self.fallback_mode == FallbackMode.AUTOMATIC
            and self._should_apply_emergency_degradation()
        ):
            return self._execute_emergency_fallback(config, *args, **kwargs)

        return result  # Return the original failure

    def _try_method(
        self, method: Callable, method_name: str, *args, **kwargs
    ) -> FallbackResult:
        """Try executing a method with error handling.

        Args:
            method: Method to execute
            method_name: Name for logging
            *args: Method arguments
            **kwargs: Method keyword arguments

        Returns:
            FallbackResult with execution details
        """
        start_time = time.time()

        try:
            # Apply timeout if specified
            config = self.method_configs.get(method_name.split("_fallback_")[0])
            if config and config.timeout:
                # Note: In a real implementation, you might want to use threading or asyncio for timeouts
                pass

            result = method(*args, **kwargs)
            execution_time = time.time() - start_time

            return FallbackResult(
                success=True,
                result=result,
                method_used=method_name,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time

            # Handle error through error handler
            context = {
                "method_name": method_name,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
            }

            error_info = self.error_handler.handle_error(
                e, context, attempt_recovery=False
            )

            return FallbackResult(
                success=False,
                method_used=method_name,
                execution_time=execution_time,
                error_info=error_info,
            )

    def _should_degrade_performance(self, config: MethodConfig) -> bool:
        """Check if performance should be degraded for this method."""
        method_success_rate = self.method_success_rates.get(config.name, 1.0)

        # Check if method's success rate is below threshold for its performance level
        threshold = self.degradation_thresholds.get(config.performance_level, 0.0)
        return method_success_rate < threshold

    def _execute_with_degradation(
        self, config: MethodConfig, *args, **kwargs
    ) -> FallbackResult:
        """Execute method with performance degradation."""
        self.log_warning(f"Applying performance degradation for method '{config.name}'")

        # Create a simplified version of the method call
        # This is a placeholder - in practice, you'd implement method-specific degradation
        degraded_result = FallbackResult(
            success=True,
            result=self._get_degraded_result(config.name, *args, **kwargs),
            method_used=f"{config.name}_degraded",
            performance_level=PerformanceLevel.REDUCED,
            degradation_applied=True,
        )

        self.fallback_stats["degradation_used"] += 1
        return degraded_result

    def _get_degraded_result(self, method_name: str, *args, **kwargs) -> Any:
        """Get a degraded result for the method.

        This is a placeholder implementation. In practice, each method would
        have its own degradation strategy.
        """
        self.log_info(f"Returning degraded result for method '{method_name}'")

        # Return a simple placeholder result
        return {
            "status": "degraded",
            "message": f"Degraded execution of {method_name}",
            "confidence": 0.5,
            "data": None,
        }

    def _should_apply_emergency_degradation(self) -> bool:
        """Check if emergency degradation should be applied."""
        # Only apply emergency degradation if we have enough data points
        if self.fallback_stats["total_calls"] < 10:
            return False

        overall_success_rate = self._calculate_overall_success_rate()
        return (
            overall_success_rate
            < self.degradation_thresholds[PerformanceLevel.EMERGENCY]
        )

    def _execute_emergency_fallback(
        self, config: MethodConfig, *args, **kwargs
    ) -> FallbackResult:
        """Execute emergency fallback with minimal functionality."""
        self.log_critical(f"Applying emergency fallback for method '{config.name}'")

        emergency_result = FallbackResult(
            success=True,
            result=self._get_emergency_result(config.name),
            method_used=f"{config.name}_emergency",
            performance_level=PerformanceLevel.EMERGENCY,
            degradation_applied=True,
            fallback_applied=True,
        )

        self.fallback_stats["degradation_used"] += 1
        self.fallback_stats["fallback_used"] += 1

        return emergency_result

    def _get_emergency_result(self, method_name: str) -> Any:
        """Get an emergency result with minimal functionality."""
        return {
            "status": "emergency",
            "message": f"Emergency fallback for {method_name}",
            "confidence": 0.1,
            "data": None,
            "warning": "System operating in emergency mode with reduced functionality",
        }

    def _update_performance_stats(
        self, method_name: str, execution_time: float, success: bool
    ):
        """Update performance statistics for a method."""
        if not self.enable_performance_monitoring:
            return

        # Update success rate using exponential moving average
        alpha = 0.1  # Learning rate
        current_rate = self.method_success_rates.get(method_name, 1.0)
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
        self.method_success_rates[method_name] = new_rate

        # Update average execution time
        current_avg = self.method_avg_times.get(method_name, 0.0)
        new_avg = alpha * execution_time + (1 - alpha) * current_avg
        self.method_avg_times[method_name] = new_avg

        # Record performance history
        performance_record = {
            "timestamp": time.time(),
            "method_name": method_name,
            "execution_time": execution_time,
            "success": success,
            "success_rate": new_rate,
        }

        self.performance_history.append(performance_record)

        # Maintain history size limit
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

        # Log to file if specified
        if self.performance_log_file:
            self._log_performance_to_file(performance_record)

    def _log_performance_to_file(self, record: Dict[str, Any]):
        """Log performance record to file."""
        try:
            self.performance_log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.performance_log_file, "a") as f:
                json.dump(record, f)
                f.write("\n")
        except Exception as e:
            self.log_warning(f"Failed to log performance to file: {e}")

    def _record_method_failure(self, method_name: str):
        """Record a method failure for statistics."""
        if method_name not in self.fallback_stats["method_failures"]:
            self.fallback_stats["method_failures"][method_name] = 0
        self.fallback_stats["method_failures"][method_name] += 1

    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall system success rate."""
        if self.fallback_stats["total_calls"] == 0:
            return 1.0

        return (
            self.fallback_stats["successful_calls"] / self.fallback_stats["total_calls"]
        )

    def set_performance_level(self, level: PerformanceLevel):
        """Manually set the system performance level."""
        old_level = self.current_performance_level
        self.current_performance_level = level

        self.log_info(
            f"Performance level changed from {old_level.value} to {level.value}"
        )

    def enable_method(self, method_name: str):
        """Enable a method."""
        if method_name in self.method_configs:
            self.method_configs[method_name].enabled = True
            self.log_info(f"Method '{method_name}' enabled")

    def disable_method(self, method_name: str):
        """Disable a method."""
        if method_name in self.method_configs:
            self.method_configs[method_name].enabled = False
            self.log_warning(f"Method '{method_name}' disabled")

    def get_method_status(self, method_name: str) -> Dict[str, Any]:
        """Get status information for a method."""
        if method_name not in self.method_configs:
            return {"error": f"Method '{method_name}' not found"}

        config = self.method_configs[method_name]
        success_rate = self.method_success_rates.get(method_name, 0.0)
        avg_time = self.method_avg_times.get(method_name, 0.0)

        return {
            "name": method_name,
            "enabled": config.enabled,
            "performance_level": config.performance_level.value,
            "success_rate": success_rate,
            "avg_execution_time": avg_time,
            "fallback_methods_count": len(config.fallback_methods),
            "max_retries": config.max_retries,
            "timeout": config.timeout,
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and statistics."""
        overall_success_rate = self._calculate_overall_success_rate()

        return {
            "current_performance_level": self.current_performance_level.value,
            "fallback_mode": self.fallback_mode.value,
            "overall_success_rate": overall_success_rate,
            "total_methods": len(self.method_configs),
            "enabled_methods": sum(
                1 for config in self.method_configs.values() if config.enabled
            ),
            "fallback_statistics": self.fallback_stats.copy(),
            "method_success_rates": self.method_success_rates.copy(),
            "degradation_thresholds": {
                k.value: v for k, v in self.degradation_thresholds.items()
            },
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            "system_status": self.get_system_status(),
            "method_details": {},
            "recent_performance": (
                self.performance_history[-100:] if self.performance_history else []
            ),
            "recommendations": [],
        }

        # Add method details
        for method_name in self.method_configs:
            report["method_details"][method_name] = self.get_method_status(method_name)

        # Add recommendations
        recommendations = self._generate_recommendations()
        report["recommendations"] = recommendations

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on performance data."""
        recommendations = []

        # Check for methods with low success rates
        for method_name, success_rate in self.method_success_rates.items():
            if success_rate < 0.5:
                recommendations.append(
                    f"Method '{method_name}' has low success rate ({success_rate:.2f}). "
                    f"Consider adding more fallback methods or reviewing implementation."
                )

        # Check overall system performance
        overall_rate = self._calculate_overall_success_rate()
        if overall_rate < 0.7:
            recommendations.append(
                f"Overall system success rate is low ({overall_rate:.2f}). "
                f"Consider reviewing system configuration and adding more robust fallback strategies."
            )

        # Check for frequently failing methods
        for method_name, failure_count in self.fallback_stats[
            "method_failures"
        ].items():
            if failure_count > 10:
                recommendations.append(
                    f"Method '{method_name}' has failed {failure_count} times. "
                    f"Consider investigating root cause and improving error handling."
                )

        return recommendations

    def reset_statistics(self):
        """Reset all performance statistics."""
        self.performance_history.clear()
        self.method_success_rates = {name: 1.0 for name in self.method_configs}
        self.method_avg_times = {name: 0.0 for name in self.method_configs}
        self.fallback_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "fallback_used": 0,
            "degradation_used": 0,
            "method_failures": {},
        }

        self.log_info("Performance statistics reset")


# Decorator for automatic fallback handling
def with_fallback(manager: FallbackStrategyManager, method_name: str):
    """Decorator for automatic fallback handling.

    Args:
        manager: FallbackStrategyManager instance
        method_name: Name of the registered method
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Register the method if not already registered
            if method_name not in manager.method_configs:
                manager.register_method(method_name, func)

            result = manager.execute_with_fallback(method_name, *args, **kwargs)

            if result.success:
                return result.result
            else:
                # Re-raise the original exception if available
                if result.error_info:
                    # Try to recreate the original exception type
                    try:
                        exception_class = globals().get(
                            result.error_info.exception_type, Exception
                        )
                        raise exception_class(result.error_info.message)
                    except:
                        raise Exception(result.error_info.message)
                else:
                    raise RuntimeError(
                        f"Method '{method_name}' failed without specific error"
                    )

        return wrapper

    return decorator
