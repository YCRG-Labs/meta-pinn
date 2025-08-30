"""
Cross-Validation Framework

This module implements comprehensive cross-validation methods including k-fold,
stratified k-fold, time-series aware cross-validation, and nested cross-validation
for hyperparameter optimization.
"""

import copy
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneOut,
    LeavePOut,
    StratifiedKFold,
    TimeSeriesSplit,
)


@dataclass
class CVResult:
    """Container for cross-validation results"""

    scores: List[float]
    mean_score: float
    std_score: float
    fold_results: List[Dict[str, Any]]
    best_params: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    validation_curve: Optional[Dict[str, List[float]]] = None


@dataclass
class NestedCVResult:
    """Container for nested cross-validation results"""

    outer_scores: List[float]
    inner_scores: List[List[float]]
    mean_outer_score: float
    std_outer_score: float
    best_params_per_fold: List[Dict[str, Any]]
    generalization_score: float


class CrossValidationStrategy(ABC):
    """Abstract base class for cross-validation strategies"""

    @abstractmethod
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits"""
        pass

    @abstractmethod
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Get number of splits"""
        pass


class KFoldStrategy(CrossValidationStrategy):
    """K-Fold cross-validation strategy"""

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return self.cv.split(X, y, groups)

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        return self.cv.get_n_splits(X, y, groups)


class StratifiedKFoldStrategy(CrossValidationStrategy):
    """Stratified K-Fold cross-validation strategy"""

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if y is None:
            raise ValueError("Stratified cross-validation requires target labels")
        return self.cv.split(X, y, groups)

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        return self.cv.get_n_splits(X, y, groups)


class TimeSeriesSplitStrategy(CrossValidationStrategy):
    """Time series cross-validation strategy"""

    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
    ):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
        self.cv = TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=max_train_size,
            test_size=test_size,
            gap=gap,
        )

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return self.cv.split(X, y, groups)

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        return self.cv.get_n_splits(X, y, groups)


class GroupKFoldStrategy(CrossValidationStrategy):
    """Group K-Fold cross-validation strategy"""

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.cv = GroupKFold(n_splits=n_splits)

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("Group cross-validation requires group labels")
        return self.cv.split(X, y, groups)

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        return self.cv.get_n_splits(X, y, groups)


class CrossValidationFramework:
    """
    Comprehensive cross-validation framework with support for various CV strategies,
    time-series aware validation, and nested cross-validation for hyperparameter optimization.
    """

    def __init__(
        self,
        strategy: str = "kfold",
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        **strategy_kwargs,
    ):
        """
        Initialize CrossValidationFramework

        Args:
            strategy: CV strategy ('kfold', 'stratified', 'timeseries', 'group', 'loo', 'lpo')
            n_splits: Number of splits for CV
            shuffle: Whether to shuffle data (where applicable)
            random_state: Random seed for reproducibility
            **strategy_kwargs: Additional arguments for specific strategies
        """
        self.strategy_name = strategy
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.strategy_kwargs = strategy_kwargs

        # Initialize CV strategy
        self.cv_strategy = self._create_cv_strategy()

    def _create_cv_strategy(self) -> CrossValidationStrategy:
        """Create appropriate CV strategy based on configuration"""
        if self.strategy_name == "kfold":
            return KFoldStrategy(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        elif self.strategy_name == "stratified":
            return StratifiedKFoldStrategy(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        elif self.strategy_name == "timeseries":
            return TimeSeriesSplitStrategy(
                n_splits=self.n_splits, **self.strategy_kwargs
            )
        elif self.strategy_name == "group":
            return GroupKFoldStrategy(n_splits=self.n_splits)
        else:
            raise ValueError(f"Unknown CV strategy: {self.strategy_name}")

    def cross_validate(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        scoring: Union[str, Callable] = "accuracy",
        groups: Optional[np.ndarray] = None,
        fit_params: Optional[Dict[str, Any]] = None,
        return_train_score: bool = False,
    ) -> CVResult:
        """
        Perform cross-validation

        Args:
            estimator: Estimator to evaluate
            X: Feature matrix
            y: Target vector
            scoring: Scoring function or string
            groups: Group labels for group CV
            fit_params: Parameters to pass to fit method
            return_train_score: Whether to return training scores

        Returns:
            CVResult with cross-validation results
        """
        if fit_params is None:
            fit_params = {}

        scores = []
        train_scores = []
        fold_results = []

        # Get CV splits
        cv_splits = list(self.cv_strategy.split(X, y, groups))

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone estimator for this fold
            fold_estimator = copy.deepcopy(estimator)

            # Fit estimator
            fold_estimator.fit(X_train, y_train, **fit_params)

            # Score on test set
            if callable(scoring):
                test_score = scoring(fold_estimator, X_test, y_test)
            else:
                test_score = fold_estimator.score(X_test, y_test)

            scores.append(test_score)

            # Score on training set if requested
            train_score = None
            if return_train_score:
                if callable(scoring):
                    train_score = scoring(fold_estimator, X_train, y_train)
                else:
                    train_score = fold_estimator.score(X_train, y_train)
                train_scores.append(train_score)

            # Store fold results
            fold_result = {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "test_score": test_score,
                "train_score": train_score,
                "estimator": fold_estimator,
            }
            fold_results.append(fold_result)

        return CVResult(
            scores=scores,
            mean_score=np.mean(scores),
            std_score=np.std(scores, ddof=1),
            fold_results=fold_results,
        )

    def nested_cross_validate(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        inner_cv: Optional["CrossValidationFramework"] = None,
        scoring: Union[str, Callable] = "accuracy",
        groups: Optional[np.ndarray] = None,
    ) -> NestedCVResult:
        """
        Perform nested cross-validation for unbiased performance estimation

        Args:
            estimator: Base estimator to optimize
            X: Feature matrix
            y: Target vector
            param_grid: Parameter grid for hyperparameter optimization
            inner_cv: Inner CV framework (if None, uses same as outer)
            scoring: Scoring function
            groups: Group labels

        Returns:
            NestedCVResult with nested CV results
        """
        if inner_cv is None:
            inner_cv = CrossValidationFramework(
                strategy=self.strategy_name,
                n_splits=min(3, self.n_splits),  # Use fewer splits for inner CV
                shuffle=self.shuffle,
                random_state=self.random_state,
                **self.strategy_kwargs,
            )

        outer_scores = []
        inner_scores = []
        best_params_per_fold = []

        # Outer CV loop
        outer_splits = list(self.cv_strategy.split(X, y, groups))

        for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
            # Split data for outer fold
            X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
            y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]

            # Inner CV for hyperparameter optimization
            best_score = -np.inf
            best_params = None
            fold_inner_scores = []

            # Grid search over parameters
            for params in self._generate_param_combinations(param_grid):
                # Set parameters
                inner_estimator = copy.deepcopy(estimator)
                inner_estimator.set_params(**params)

                # Inner CV evaluation
                inner_cv_result = inner_cv.cross_validate(
                    inner_estimator,
                    X_outer_train,
                    y_outer_train,
                    scoring=scoring,
                    groups=groups[outer_train_idx] if groups is not None else None,
                )

                fold_inner_scores.append(inner_cv_result.mean_score)

                # Track best parameters
                if inner_cv_result.mean_score > best_score:
                    best_score = inner_cv_result.mean_score
                    best_params = params

            inner_scores.append(fold_inner_scores)
            best_params_per_fold.append(best_params)

            # Train final model with best parameters on outer training set
            final_estimator = copy.deepcopy(estimator)
            final_estimator.set_params(**best_params)
            final_estimator.fit(X_outer_train, y_outer_train)

            # Evaluate on outer test set
            if callable(scoring):
                outer_score = scoring(final_estimator, X_outer_test, y_outer_test)
            else:
                outer_score = final_estimator.score(X_outer_test, y_outer_test)

            outer_scores.append(outer_score)

        return NestedCVResult(
            outer_scores=outer_scores,
            inner_scores=inner_scores,
            mean_outer_score=np.mean(outer_scores),
            std_outer_score=np.std(outer_scores, ddof=1),
            best_params_per_fold=best_params_per_fold,
            generalization_score=np.mean(outer_scores),
        )

    def _generate_param_combinations(
        self, param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters from grid"""
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)

        return combinations

    def validation_curve(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_name: str,
        param_range: List[Any],
        scoring: Union[str, Callable] = "accuracy",
        groups: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """
        Generate validation curve for a parameter

        Args:
            estimator: Estimator to evaluate
            X: Feature matrix
            y: Target vector
            param_name: Name of parameter to vary
            param_range: Range of parameter values
            scoring: Scoring function
            groups: Group labels

        Returns:
            Dictionary with train and validation scores for each parameter value
        """
        train_scores = []
        val_scores = []

        for param_value in param_range:
            # Set parameter
            param_estimator = copy.deepcopy(estimator)
            param_estimator.set_params(**{param_name: param_value})

            # Cross-validate
            cv_result = self.cross_validate(
                param_estimator,
                X,
                y,
                scoring=scoring,
                groups=groups,
                return_train_score=True,
            )

            train_scores.append(
                [fold["train_score"] for fold in cv_result.fold_results]
            )
            val_scores.append(cv_result.scores)

        return {
            "param_range": param_range,
            "train_scores": train_scores,
            "val_scores": val_scores,
            "train_scores_mean": [np.mean(scores) for scores in train_scores],
            "train_scores_std": [np.std(scores, ddof=1) for scores in train_scores],
            "val_scores_mean": [np.mean(scores) for scores in val_scores],
            "val_scores_std": [np.std(scores, ddof=1) for scores in val_scores],
        }

    def learning_curve(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        train_sizes: Optional[List[float]] = None,
        scoring: Union[str, Callable] = "accuracy",
        groups: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """
        Generate learning curve showing performance vs training set size

        Args:
            estimator: Estimator to evaluate
            X: Feature matrix
            y: Target vector
            train_sizes: Fractions of training set to use
            scoring: Scoring function
            groups: Group labels

        Returns:
            Dictionary with train and validation scores for each training size
        """
        if train_sizes is None:
            train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

        train_scores = []
        val_scores = []
        actual_train_sizes = []

        # Get CV splits
        cv_splits = list(self.cv_strategy.split(X, y, groups))

        for train_size_frac in train_sizes:
            fold_train_scores = []
            fold_val_scores = []
            fold_train_sizes = []

            for train_idx, test_idx in cv_splits:
                # Determine actual training size
                n_train_samples = int(len(train_idx) * train_size_frac)
                if n_train_samples < 1:
                    n_train_samples = 1

                # Subsample training set
                subsample_idx = np.random.choice(
                    train_idx, size=n_train_samples, replace=False
                )

                # Split data
                X_train, X_test = X[subsample_idx], X[test_idx]
                y_train, y_test = y[subsample_idx], y[test_idx]

                # Train estimator
                fold_estimator = copy.deepcopy(estimator)
                fold_estimator.fit(X_train, y_train)

                # Score on both sets
                if callable(scoring):
                    train_score = scoring(fold_estimator, X_train, y_train)
                    val_score = scoring(fold_estimator, X_test, y_test)
                else:
                    train_score = fold_estimator.score(X_train, y_train)
                    val_score = fold_estimator.score(X_test, y_test)

                fold_train_scores.append(train_score)
                fold_val_scores.append(val_score)
                fold_train_sizes.append(len(X_train))

            train_scores.append(fold_train_scores)
            val_scores.append(fold_val_scores)
            actual_train_sizes.append(np.mean(fold_train_sizes))

        return {
            "train_sizes": actual_train_sizes,
            "train_scores": train_scores,
            "val_scores": val_scores,
            "train_scores_mean": [np.mean(scores) for scores in train_scores],
            "train_scores_std": [np.std(scores, ddof=1) for scores in train_scores],
            "val_scores_mean": [np.mean(scores) for scores in val_scores],
            "val_scores_std": [np.std(scores, ddof=1) for scores in val_scores],
        }

    def time_series_split_validation(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        time_column: Optional[np.ndarray] = None,
        n_splits: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
        scoring: Union[str, Callable] = "accuracy",
    ) -> CVResult:
        """
        Perform time-series aware cross-validation

        Args:
            estimator: Estimator to evaluate
            X: Feature matrix
            y: Target vector
            time_column: Time indices (if None, assumes sequential order)
            n_splits: Number of splits
            test_size: Size of test set
            gap: Gap between train and test sets
            scoring: Scoring function

        Returns:
            CVResult with time-series CV results
        """
        if n_splits is None:
            n_splits = self.n_splits

        # Sort by time if time column provided
        if time_column is not None:
            sort_idx = np.argsort(time_column)
            X = X[sort_idx]
            y = y[sort_idx]

        # Create time series CV strategy
        ts_cv = TimeSeriesSplitStrategy(n_splits=n_splits, test_size=test_size, gap=gap)

        scores = []
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(ts_cv.split(X, y)):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train estimator
            fold_estimator = copy.deepcopy(estimator)
            fold_estimator.fit(X_train, y_train)

            # Score
            if callable(scoring):
                score = scoring(fold_estimator, X_test, y_test)
            else:
                score = fold_estimator.score(X_test, y_test)

            scores.append(score)

            # Store fold results
            fold_result = {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "test_score": score,
                "train_period": (
                    (train_idx[0], train_idx[-1]) if len(train_idx) > 0 else None
                ),
                "test_period": (
                    (test_idx[0], test_idx[-1]) if len(test_idx) > 0 else None
                ),
                "estimator": fold_estimator,
            }
            fold_results.append(fold_result)

        return CVResult(
            scores=scores,
            mean_score=np.mean(scores),
            std_score=np.std(scores, ddof=1),
            fold_results=fold_results,
        )
