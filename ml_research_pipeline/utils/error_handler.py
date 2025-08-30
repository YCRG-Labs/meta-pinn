"""
Comprehensive error handling and diagnostics for the ML research pipeline.
"""

import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .logging_utils import LoggerMixin


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    DATA_QUALITY = "data_quality"
    ALGORITHM_FAILURE = "algorithm_failure"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_ERROR = "resource_error"
    CONFIGURATION_ERROR = "configuration_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Detailed error information."""

    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception_type: str
    traceback_str: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "exception_type": self.exception_type,
            "traceback": self.traceback_str,
            "context": self.context,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "recovery_strategy": self.recovery_strategy,
        }


class RecoveryStrategy:
    """Base class for error recovery strategies."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Check if this strategy can handle the error."""
        raise NotImplementedError

    def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Attempt to recover from the error."""
        raise NotImplementedError


class RetryStrategy(RecoveryStrategy):
    """Retry strategy with exponential backoff."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        super().__init__("retry", "Retry with exponential backoff")
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_counts: Dict[str, int] = {}

    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Check if retry is appropriate."""
        # Don't retry critical errors or configuration errors
        if error_info.severity == ErrorSeverity.CRITICAL:
            return False
        if error_info.category == ErrorCategory.CONFIGURATION_ERROR:
            return False

        # Check retry count
        retry_count = self.retry_counts.get(error_info.error_id, 0)
        return retry_count < self.max_retries

    def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Attempt recovery by retrying."""
        import time

        retry_count = self.retry_counts.get(error_info.error_id, 0)
        if retry_count >= self.max_retries:
            return False

        # Exponential backoff
        delay = self.base_delay * (2**retry_count)
        time.sleep(delay)

        self.retry_counts[error_info.error_id] = retry_count + 1
        return True


class FallbackStrategy(RecoveryStrategy):
    """Fallback to alternative method strategy."""

    def __init__(self, fallback_methods: Dict[str, Callable]):
        super().__init__("fallback", "Fallback to alternative methods")
        self.fallback_methods = fallback_methods

    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Check if fallback methods are available."""
        return error_info.category in [
            ErrorCategory.ALGORITHM_FAILURE,
            ErrorCategory.VALIDATION_ERROR,
        ]

    def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Attempt recovery using fallback method."""
        method_name = context.get("method_name")
        if method_name and method_name in self.fallback_methods:
            try:
                fallback_method = self.fallback_methods[method_name]
                # This would be called by the fallback manager
                return True
            except Exception:
                return False
        return False


class ErrorHandler(LoggerMixin):
    """Comprehensive error handler with classification and recovery."""

    def __init__(
        self,
        log_file: Optional[Path] = None,
        enable_recovery: bool = True,
        max_error_history: int = 1000,
    ):
        """Initialize error handler.

        Args:
            log_file: Optional file to log errors to
            enable_recovery: Whether to attempt error recovery
            max_error_history: Maximum number of errors to keep in history
        """
        self.log_file = log_file
        self.enable_recovery = enable_recovery
        self.max_error_history = max_error_history

        # Error tracking
        self.error_history: List[ErrorInfo] = []
        self.error_counts: Dict[str, int] = {}

        # Recovery strategies
        self.recovery_strategies: List[RecoveryStrategy] = []
        if enable_recovery:
            self._initialize_default_strategies()

        # Error classification rules
        self.classification_rules = self._initialize_classification_rules()

    def _initialize_default_strategies(self):
        """Initialize default recovery strategies."""
        self.recovery_strategies = [
            RetryStrategy(max_retries=3),
            FallbackStrategy({}),  # Will be populated by fallback manager
        ]

    def _initialize_classification_rules(
        self,
    ) -> Dict[str, Callable[[Exception, Dict], ErrorCategory]]:
        """Initialize error classification rules."""

        def classify_by_exception_and_message(
            exception: Exception, context: Dict
        ) -> ErrorCategory:
            """Classify error by exception type and message."""
            exception_name = type(exception).__name__.lower()
            error_message = str(exception).lower()

            # Check message content first (more specific)
            # Algorithm failures
            algorithm_keywords = [
                "convergence",
                "optimization",
                "numerical",
                "singular",
                "instability",
            ]
            if any(keyword in error_message for keyword in algorithm_keywords):
                return ErrorCategory.ALGORITHM_FAILURE

            # Resource errors
            resource_keywords = ["memory", "cuda", "gpu", "timeout", "resource"]
            if any(keyword in error_message for keyword in resource_keywords):
                return ErrorCategory.RESOURCE_ERROR

            # Configuration errors (check before data quality to catch "invalid parameter")
            config_keywords = ["parameter", "config", "argument", "setting"]
            if any(keyword in error_message for keyword in config_keywords):
                return ErrorCategory.CONFIGURATION_ERROR

            # Data quality errors
            data_keywords = ["data", "shape", "missing", "invalid", "corrupt"]
            if any(keyword in error_message for keyword in data_keywords):
                return ErrorCategory.DATA_QUALITY

            # Validation errors
            validation_keywords = ["validation", "check", "assertion", "verify"]
            if any(keyword in error_message for keyword in validation_keywords):
                return ErrorCategory.VALIDATION_ERROR

            # Network errors
            network_keywords = ["connection", "network", "http", "url", "socket"]
            if any(keyword in error_message for keyword in network_keywords):
                return ErrorCategory.NETWORK_ERROR

            # Fallback to exception type classification
            # Data-related errors by type
            if any(keyword in exception_name for keyword in ["value", "key", "index"]):
                return ErrorCategory.DATA_QUALITY

            # Resource errors by type
            if any(keyword in exception_name for keyword in ["memory", "timeout"]):
                return ErrorCategory.RESOURCE_ERROR

            # Configuration errors by type
            if any(keyword in exception_name for keyword in ["attribute", "type"]):
                return ErrorCategory.CONFIGURATION_ERROR

            return ErrorCategory.UNKNOWN

        return {"exception_and_message": classify_by_exception_and_message}

    def _assess_severity(
        self, exception: Exception, context: Dict[str, Any]
    ) -> ErrorSeverity:
        """Assess error severity based on exception and context."""
        exception_name = type(exception).__name__
        error_message = str(exception).lower()

        # Critical errors that should stop execution
        critical_exceptions = [
            "SystemExit",
            "KeyboardInterrupt",
            "MemoryError",
            "RecursionError",
            "SystemError",
        ]
        if exception_name in critical_exceptions:
            return ErrorSeverity.CRITICAL

        # High severity errors by message content
        high_severity_keywords = [
            "cuda",
            "gpu",
            "memory",
            "corruption",
            "security",
            "out of memory",
        ]
        if any(keyword in error_message for keyword in high_severity_keywords):
            return ErrorSeverity.HIGH

        # Medium severity for algorithm failures by message
        medium_severity_keywords = [
            "convergence",
            "optimization",
            "numerical",
            "instability",
            "singular",
        ]
        if any(keyword in error_message for keyword in medium_severity_keywords):
            return ErrorSeverity.MEDIUM

        # Medium severity for algorithm failures by exception type
        if any(
            keyword in exception_name.lower()
            for keyword in ["convergence", "optimization", "numerical"]
        ):
            return ErrorSeverity.MEDIUM

        # Default to low severity
        return ErrorSeverity.LOW

    def _classify_error(
        self, exception: Exception, context: Dict[str, Any]
    ) -> ErrorCategory:
        """Classify error into appropriate category."""
        for rule_name, rule_func in self.classification_rules.items():
            try:
                category = rule_func(exception, context)
                if category != ErrorCategory.UNKNOWN:
                    return category
            except Exception as e:
                self.log_warning(f"Error in classification rule {rule_name}: {e}")

        return ErrorCategory.UNKNOWN

    def _generate_error_id(self, exception: Exception, context: Dict[str, Any]) -> str:
        """Generate unique error ID."""
        import hashlib

        # Create hash from exception type, message, and relevant context
        error_string = f"{type(exception).__name__}:{str(exception)}"
        if "method_name" in context:
            error_string += f":{context['method_name']}"

        return hashlib.md5(error_string.encode()).hexdigest()[:8]

    def handle_error(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True,
    ) -> ErrorInfo:
        """Handle an error with classification and optional recovery.

        Args:
            exception: The exception that occurred
            context: Additional context information
            attempt_recovery: Whether to attempt recovery

        Returns:
            ErrorInfo object with error details
        """
        if context is None:
            context = {}

        # Create error info
        error_info = ErrorInfo(
            error_id=self._generate_error_id(exception, context),
            timestamp=datetime.now(),
            category=self._classify_error(exception, context),
            severity=self._assess_severity(exception, context),
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback_str=traceback.format_exc(),
            context=context.copy(),
        )

        # Log the error
        self._log_error(error_info)

        # Update error counts
        error_key = f"{error_info.category.value}:{error_info.exception_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Attempt recovery if enabled
        if attempt_recovery and self.enable_recovery:
            error_info.recovery_attempted = True
            error_info.recovery_successful = self._attempt_recovery(error_info, context)

        # Add to history
        self._add_to_history(error_info)

        return error_info

    def _log_error(self, error_info: ErrorInfo):
        """Log error information."""
        log_message = (
            f"Error {error_info.error_id}: {error_info.message} "
            f"[{error_info.category.value}/{error_info.severity.value}]"
        )

        if error_info.severity == ErrorSeverity.CRITICAL:
            self.log_critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.log_error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.log_warning(log_message)
        else:
            self.log_info(log_message)

        # Log to file if specified
        if self.log_file:
            self._log_to_file(error_info)

    def _log_to_file(self, error_info: ErrorInfo):
        """Log error to file."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.log_file, "a") as f:
                json.dump(error_info.to_dict(), f)
                f.write("\n")
        except Exception as e:
            self.log_warning(f"Failed to log error to file: {e}")

    def _attempt_recovery(self, error_info: ErrorInfo, context: Dict[str, Any]) -> bool:
        """Attempt to recover from error using available strategies."""
        for strategy in self.recovery_strategies:
            if strategy.can_recover(error_info):
                try:
                    if strategy.recover(error_info, context):
                        error_info.recovery_strategy = strategy.name
                        self.log_info(
                            f"Successfully recovered from error {error_info.error_id} "
                            f"using strategy: {strategy.name}"
                        )
                        return True
                except Exception as recovery_exception:
                    self.log_warning(
                        f"Recovery strategy {strategy.name} failed: {recovery_exception}"
                    )

        return False

    def _add_to_history(self, error_info: ErrorInfo):
        """Add error to history with size limit."""
        self.error_history.append(error_info)

        # Maintain history size limit
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history :]

    def add_recovery_strategy(self, strategy: RecoveryStrategy):
        """Add a custom recovery strategy."""
        self.recovery_strategies.append(strategy)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and diagnostics."""
        if not self.error_history:
            return {"total_errors": 0}

        # Count by category
        category_counts = {}
        severity_counts = {}
        recovery_stats = {"attempted": 0, "successful": 0}

        for error in self.error_history:
            # Category counts
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1

            # Severity counts
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            # Recovery stats
            if error.recovery_attempted:
                recovery_stats["attempted"] += 1
                if error.recovery_successful:
                    recovery_stats["successful"] += 1

        # Calculate recovery rate
        recovery_rate = 0.0
        if recovery_stats["attempted"] > 0:
            recovery_rate = recovery_stats["successful"] / recovery_stats["attempted"]

        return {
            "total_errors": len(self.error_history),
            "category_counts": category_counts,
            "severity_counts": severity_counts,
            "recovery_stats": recovery_stats,
            "recovery_rate": recovery_rate,
            "most_common_errors": dict(
                sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
        }

    def get_recent_errors(self, count: int = 10) -> List[ErrorInfo]:
        """Get most recent errors."""
        return self.error_history[-count:] if self.error_history else []

    def clear_history(self):
        """Clear error history."""
        self.error_history.clear()
        self.error_counts.clear()


# Decorator for automatic error handling
def handle_errors(
    error_handler: ErrorHandler,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = False,
):
    """Decorator for automatic error handling.

    Args:
        error_handler: ErrorHandler instance
        context: Additional context for error handling
        reraise: Whether to reraise the exception after handling
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                func_context = context.copy() if context else {}
                func_context.update(
                    {
                        "function_name": func.__name__,
                        "args": str(args)[:100],  # Truncate for logging
                        "kwargs": str(kwargs)[:100],
                    }
                )

                error_info = error_handler.handle_error(e, func_context)

                if reraise or error_info.severity == ErrorSeverity.CRITICAL:
                    raise

                return None

        return wrapper

    return decorator
