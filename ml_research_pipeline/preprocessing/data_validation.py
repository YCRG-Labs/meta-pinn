"""
Data Quality Validation Module

This module implements comprehensive data quality validation for physics data.
Includes outlier detection using isolation forests and statistical methods,
missing value detection and imputation strategies, and data consistency checks
for temporal and spatial data.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler


class OutlierMethod(Enum):
    """Enumeration of available outlier detection methods"""

    ISOLATION_FOREST = "isolation_forest"
    Z_SCORE = "z_score"
    IQR = "iqr"
    MODIFIED_Z_SCORE = "modified_z_score"


class ImputationStrategy(Enum):
    """Enumeration of available imputation strategies"""

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    KNN = "knn"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"


@dataclass
class ValidationConfig:
    """Configuration for data quality validation"""

    outlier_methods: List[OutlierMethod] = None
    outlier_threshold: float = 0.1
    contamination_rate: float = 0.1
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    imputation_strategy: ImputationStrategy = ImputationStrategy.MEDIAN
    knn_neighbors: int = 5
    temporal_consistency_window: int = 10
    spatial_consistency_radius: float = 1.0
    min_data_completeness: float = 0.8
    enable_auto_cleaning: bool = True

    def __post_init__(self):
        if self.outlier_methods is None:
            self.outlier_methods = [
                OutlierMethod.ISOLATION_FOREST,
                OutlierMethod.Z_SCORE,
            ]


@dataclass
class QualityMetrics:
    """Data quality assessment metrics"""

    completeness: float
    outlier_percentage: float
    temporal_consistency: float
    spatial_consistency: float
    overall_quality: float
    missing_value_patterns: Dict[str, float]
    outlier_details: Dict[str, Any]
    consistency_violations: List[str]


@dataclass
class ValidationResult:
    """Result of data quality validation"""

    is_valid: bool
    quality_metrics: QualityMetrics
    cleaned_data: Optional[Dict[str, np.ndarray]]
    validation_report: str
    recommendations: List[str]


class DataQualityValidator:
    """
    Comprehensive data quality validator for physics datasets.

    Implements outlier detection using isolation forests and statistical methods,
    missing value detection and imputation strategies, and data consistency checks
    for temporal and spatial data.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize the DataQualityValidator."""
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize outlier detectors
        self.isolation_forest = IsolationForest(
            contamination=self.config.contamination_rate, random_state=42
        )
        self.scaler = StandardScaler()

        # Initialize imputers
        self.imputers = {
            ImputationStrategy.MEAN: SimpleImputer(strategy="mean"),
            ImputationStrategy.MEDIAN: SimpleImputer(strategy="median"),
            ImputationStrategy.MODE: SimpleImputer(strategy="most_frequent"),
            ImputationStrategy.KNN: KNNImputer(n_neighbors=self.config.knn_neighbors),
        }

        self.validation_history = []

    def validate_data(self, data: Dict[str, np.ndarray]) -> ValidationResult:
        """Perform comprehensive data quality validation."""
        self.logger.info("Starting comprehensive data quality validation")

        # Convert to DataFrame for easier processing
        df = self._dict_to_dataframe(data)

        # Perform validation steps
        quality_metrics = self._calculate_quality_metrics(df)
        outliers = self._detect_outliers(df)
        missing_patterns = self._analyze_missing_patterns(df)
        consistency_issues = self._check_consistency(df)

        # Update quality metrics with outlier information
        if outliers and "combined" in outliers:
            quality_metrics.outlier_percentage = outliers["combined"].mean()
            quality_metrics.outlier_details = {k: v.sum() for k, v in outliers.items()}

        quality_metrics.consistency_violations = consistency_issues

        # Generate recommendations
        recommendations = self._generate_recommendations(
            quality_metrics, outliers, missing_patterns, consistency_issues
        )

        # Clean data if auto-cleaning is enabled
        cleaned_data = None
        if self.config.enable_auto_cleaning:
            cleaned_df = self._clean_data(df, outliers, missing_patterns)
            cleaned_data = self._dataframe_to_dict(cleaned_df)

        # Determine overall validity
        is_valid = (
            quality_metrics.completeness >= self.config.min_data_completeness
            and quality_metrics.overall_quality >= 0.7
        )

        # Generate validation report
        report = self._generate_validation_report(
            quality_metrics, outliers, missing_patterns, consistency_issues
        )

        result = ValidationResult(
            is_valid=is_valid,
            quality_metrics=quality_metrics,
            cleaned_data=cleaned_data,
            validation_report=report,
            recommendations=recommendations,
        )

        self.validation_history.append(result)
        return result

    def _dict_to_dataframe(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Convert dictionary of arrays to DataFrame"""
        if not data:
            raise ValueError("Data dictionary cannot be empty")

        # Ensure all arrays have the same length
        lengths = [len(arr) for arr in data.values()]
        if len(set(lengths)) > 1:
            min_length = min(lengths)
            self.logger.warning(
                f"Arrays have different lengths. Truncating to {min_length}"
            )
            data = {k: v[:min_length] for k, v in data.items()}

        return pd.DataFrame(data)

    def _dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert DataFrame back to dictionary of arrays"""
        return {col: df[col].values for col in df.columns}

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        # Completeness
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0

        # Missing value patterns
        missing_patterns = {}
        for col in df.columns:
            missing_patterns[col] = (
                df[col].isnull().sum() / len(df) if len(df) > 0 else 0.0
            )

        # Temporal consistency (if time-based data)
        temporal_consistency = self._calculate_temporal_consistency(df)

        # Spatial consistency (if spatial data)
        spatial_consistency = self._calculate_spatial_consistency(df)

        # Overall quality score
        overall_quality = (
            completeness + temporal_consistency + spatial_consistency
        ) / 3

        return QualityMetrics(
            completeness=completeness,
            outlier_percentage=0.0,  # Will be updated after outlier detection
            temporal_consistency=temporal_consistency,
            spatial_consistency=spatial_consistency,
            overall_quality=overall_quality,
            missing_value_patterns=missing_patterns,
            outlier_details={},
            consistency_violations=[],
        )

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Detect outliers using multiple methods"""
        outliers = {}

        for method in self.config.outlier_methods:
            if method == OutlierMethod.ISOLATION_FOREST:
                outliers["isolation_forest"] = self._isolation_forest_outliers(df)
            elif method == OutlierMethod.Z_SCORE:
                outliers["z_score"] = self._z_score_outliers(df)
            elif method == OutlierMethod.IQR:
                outliers["iqr"] = self._iqr_outliers(df)
            elif method == OutlierMethod.MODIFIED_Z_SCORE:
                outliers["modified_z_score"] = self._modified_z_score_outliers(df)

        # Combine outlier detections
        if outliers:
            combined_outliers = np.zeros(len(df), dtype=bool)
            for method_outliers in outliers.values():
                combined_outliers |= method_outliers
            outliers["combined"] = combined_outliers

        return outliers

    def _isolation_forest_outliers(self, df: pd.DataFrame) -> np.ndarray:
        """Detect outliers using Isolation Forest"""
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty or len(numeric_df) < 2:
            return np.zeros(len(df), dtype=bool)

        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.median())

        # Scale data
        scaled_data = self.scaler.fit_transform(numeric_df)

        # Detect outliers
        outlier_labels = self.isolation_forest.fit_predict(scaled_data)
        return outlier_labels == -1

    def _z_score_outliers(self, df: pd.DataFrame) -> np.ndarray:
        """Detect outliers using Z-score method"""
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty or len(numeric_df) < 2:
            return np.zeros(len(df), dtype=bool)

        z_scores = np.abs(stats.zscore(numeric_df, nan_policy="omit"))
        return (z_scores > self.config.z_score_threshold).any(axis=1)

    def _iqr_outliers(self, df: pd.DataFrame) -> np.ndarray:
        """Detect outliers using Interquartile Range method"""
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty:
            return np.zeros(len(df), dtype=bool)

        # Initialize result array
        outliers = np.zeros(len(df), dtype=bool)

        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0:  # Avoid issues with constant values
                lower_bound = Q1 - self.config.iqr_multiplier * IQR
                upper_bound = Q3 + self.config.iqr_multiplier * IQR

                col_outliers = (numeric_df[col] < lower_bound) | (
                    numeric_df[col] > upper_bound
                )
                col_outliers_clean = col_outliers.fillna(False).values
                outliers = outliers | col_outliers_clean

        return outliers

    def _modified_z_score_outliers(self, df: pd.DataFrame) -> np.ndarray:
        """Detect outliers using Modified Z-score method"""
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty:
            return np.zeros(len(df), dtype=bool)

        # Initialize result array
        outliers = np.zeros(len(df), dtype=bool)

        for col in numeric_df.columns:
            median = numeric_df[col].median()
            mad = np.median(np.abs(numeric_df[col] - median))

            if mad > 0:  # Avoid division by zero
                modified_z_scores = 0.6745 * (numeric_df[col] - median) / mad
                col_outliers = np.abs(modified_z_scores) > self.config.z_score_threshold
                col_outliers_clean = col_outliers.fillna(False).values
                outliers = outliers | col_outliers_clean

        return outliers

    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing data"""
        missing_patterns = {}

        # Missing value counts per column
        missing_counts = df.isnull().sum()
        missing_patterns["counts"] = missing_counts.to_dict()

        # Missing value percentages
        missing_percentages = (
            (missing_counts / len(df) * 100) if len(df) > 0 else missing_counts * 0
        )
        missing_patterns["percentages"] = missing_percentages.to_dict()

        # Consecutive missing values (for time series)
        consecutive_missing = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                missing_mask = df[col].isnull()
                consecutive_counts = []
                current_count = 0

                for is_missing in missing_mask:
                    if is_missing:
                        current_count += 1
                    else:
                        if current_count > 0:
                            consecutive_counts.append(current_count)
                        current_count = 0

                if current_count > 0:
                    consecutive_counts.append(current_count)

                consecutive_missing[col] = {
                    "max_consecutive": (
                        max(consecutive_counts) if consecutive_counts else 0
                    ),
                    "avg_consecutive": (
                        np.mean(consecutive_counts) if consecutive_counts else 0
                    ),
                }

        missing_patterns["consecutive"] = consecutive_missing
        return missing_patterns

    def _check_consistency(self, df: pd.DataFrame) -> List[str]:
        """Check data consistency for temporal and spatial data"""
        violations = []

        # Temporal consistency checks
        if "time" in df.columns or "t" in df.columns:
            violations.extend(self._check_temporal_consistency_violations(df))

        # Spatial consistency checks
        spatial_cols = [col for col in df.columns if col in ["x", "y", "z", "position"]]
        if spatial_cols:
            violations.extend(
                self._check_spatial_consistency_violations(df, spatial_cols)
            )

        # Physical constraints checks
        violations.extend(self._check_physical_constraints(df))

        return violations

    def _check_temporal_consistency(self, df: pd.DataFrame) -> List[str]:
        """Check temporal consistency in time series data - simplified interface for tests"""
        return self._check_temporal_consistency_violations(df)

    def _check_temporal_consistency_violations(self, df: pd.DataFrame) -> List[str]:
        """Check temporal consistency in time series data"""
        violations = []
        time_col = "time" if "time" in df.columns else "t"

        if time_col not in df.columns:
            return violations

        time_data = df[time_col].dropna()
        if len(time_data) < 2:
            return violations

        # Check for non-monotonic time
        if not time_data.is_monotonic_increasing:
            violations.append("Time series is not monotonically increasing")

        # Check for duplicate time stamps
        if time_data.duplicated().any():
            violations.append("Duplicate time stamps detected")

        # Check for unrealistic time gaps
        time_diffs = time_data.diff().dropna()
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            if median_diff > 0:
                large_gaps = time_diffs > 10 * median_diff
                if large_gaps.any():
                    violations.append(
                        f"Large time gaps detected: {large_gaps.sum()} instances"
                    )

        return violations

    def _check_spatial_consistency(
        self, df: pd.DataFrame, spatial_cols: List[str]
    ) -> List[str]:
        """Check spatial consistency in spatial data - simplified interface for tests"""
        return self._check_spatial_consistency_violations(df, spatial_cols)

    def _check_spatial_consistency_violations(
        self, df: pd.DataFrame, spatial_cols: List[str]
    ) -> List[str]:
        """Check spatial consistency in spatial data"""
        violations = []

        for col in spatial_cols:
            if col in df.columns:
                spatial_data = df[col].dropna()
                if len(spatial_data) > 1:
                    spatial_diffs = spatial_data.diff().dropna()
                    if len(spatial_diffs) > 0:
                        std_diff = spatial_diffs.std()
                        if std_diff > 0:
                            # For linear data (like linspace), the differences should be nearly constant
                            # Only flag as violations if there are truly anomalous jumps
                            # Check coefficient of variation - if it's very low, this is likely linear data
                            cv = (
                                std_diff / np.abs(spatial_diffs).mean()
                                if np.abs(spatial_diffs).mean() > 0
                                else 0
                            )

                            # If coefficient of variation is very low (< 0.001), this is likely uniform data
                            # Use a very small threshold to account for floating point precision
                            if cv > 0.001:
                                median_diff = np.abs(spatial_diffs).median()
                                # Use a more reasonable threshold based on statistical outliers
                                # Use the smaller of: 10x median or 3 standard deviations above mean
                                threshold = min(
                                    10 * median_diff,
                                    np.abs(spatial_diffs).mean() + 3 * std_diff,
                                )
                                large_jumps = np.abs(spatial_diffs) > threshold
                                # Report if there are any significant jumps
                                if large_jumps.any():
                                    violations.append(
                                        f"Large spatial jumps in {col}: {large_jumps.sum()} instances"
                                    )

        return violations

    def _check_physical_constraints(self, df: pd.DataFrame) -> List[str]:
        """Check physical constraints and conservation laws"""
        violations = []

        # Check for negative values where they shouldn't exist
        positive_vars = ["pressure", "density", "temperature", "energy"]
        for var in positive_vars:
            if var in df.columns:
                negative_count = (df[var] < 0).sum()
                if negative_count > 0:
                    violations.append(
                        f"Negative values in {var}: {negative_count} instances"
                    )

        return violations

    def _calculate_temporal_consistency(self, df: pd.DataFrame) -> float:
        """Calculate temporal consistency score"""
        if "time" not in df.columns and "t" not in df.columns:
            return 1.0

        time_col = "time" if "time" in df.columns else "t"
        time_data = df[time_col].dropna()

        if len(time_data) < 2:
            return 1.0

        # Check monotonicity
        monotonic_score = 1.0 if time_data.is_monotonic_increasing else 0.5

        # Check for reasonable time intervals
        time_diffs = time_data.diff().dropna()
        if len(time_diffs) > 0 and time_diffs.mean() > 0:
            cv = time_diffs.std() / time_diffs.mean()
            interval_score = max(0.0, 1.0 - cv)
        else:
            interval_score = 1.0

        return (monotonic_score + interval_score) / 2

    def _calculate_spatial_consistency(self, df: pd.DataFrame) -> float:
        """Calculate spatial consistency score"""
        spatial_cols = [col for col in df.columns if col in ["x", "y", "z"]]
        if not spatial_cols:
            return 1.0

        consistency_scores = []

        for col in spatial_cols:
            spatial_data = df[col].dropna()
            if len(spatial_data) < 2:
                consistency_scores.append(1.0)
                continue

            spatial_diffs = spatial_data.diff().dropna()
            if len(spatial_diffs) > 0 and spatial_diffs.mean() > 0:
                cv = spatial_diffs.std() / spatial_diffs.mean()
                consistency_scores.append(max(0.0, 1.0 - cv))
            else:
                consistency_scores.append(1.0)

        return np.mean(consistency_scores) if consistency_scores else 1.0

    def _clean_data(
        self,
        df: pd.DataFrame,
        outliers: Dict[str, np.ndarray],
        missing_patterns: Dict[str, Any],
    ) -> pd.DataFrame:
        """Clean data by handling outliers and missing values"""
        cleaned_df = df.copy()

        # Handle outliers
        if "combined" in outliers and outliers["combined"].any():
            outlier_mask = outliers["combined"]
            self.logger.info(f"Removing {outlier_mask.sum()} outliers")
            cleaned_df = cleaned_df[~outlier_mask]

        # Handle missing values
        cleaned_df = self._impute_missing_values(cleaned_df)

        return cleaned_df

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using configured strategy"""
        cleaned_df = df.copy()

        for col in df.columns:
            if cleaned_df[col].isnull().any():
                # Check if all values are missing first
                if cleaned_df[col].isnull().all():
                    self.logger.warning(
                        f"Column {col} has all missing values, filling with 0"
                    )
                    cleaned_df[col] = 0.0
                    continue

                # Handle different imputation strategies
                if self.config.imputation_strategy == ImputationStrategy.FORWARD_FILL:
                    cleaned_df[col] = cleaned_df[col].ffill()
                elif (
                    self.config.imputation_strategy == ImputationStrategy.BACKWARD_FILL
                ):
                    cleaned_df[col] = cleaned_df[col].bfill()
                elif self.config.imputation_strategy == ImputationStrategy.INTERPOLATE:
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].interpolate()
                else:
                    # Use sklearn imputers for numeric data only
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        try:
                            imputer = self.imputers[self.config.imputation_strategy]
                            imputed_values = imputer.fit_transform(cleaned_df[[col]])
                            if imputed_values.size > 0 and len(
                                imputed_values.flatten()
                            ) == len(cleaned_df):
                                cleaned_df[col] = imputed_values.flatten()
                            else:
                                self.logger.warning(
                                    f"Imputation returned empty result for column {col}"
                                )
                        except Exception as e:
                            self.logger.warning(
                                f"Imputation failed for column {col}: {e}"
                            )
                            # Fallback to median for failed imputation
                            cleaned_df[col] = cleaned_df[col].fillna(
                                cleaned_df[col].median()
                            )

        return cleaned_df

    def _generate_recommendations(
        self,
        quality_metrics: QualityMetrics,
        outliers: Dict[str, np.ndarray],
        missing_patterns: Dict[str, Any],
        consistency_issues: List[str],
    ) -> List[str]:
        """Generate recommendations for data quality improvement"""
        recommendations = []

        # Completeness recommendations
        if quality_metrics.completeness < 0.9:
            recommendations.append(
                f"Data completeness is {quality_metrics.completeness:.2%}. "
                "Consider collecting more data or using advanced imputation methods."
            )

        # Outlier recommendations
        if outliers and "combined" in outliers:
            outlier_rate = outliers["combined"].mean()
            if outlier_rate > 0.05:
                recommendations.append(
                    f"High outlier rate ({outlier_rate:.2%}). "
                    "Review data collection process and consider outlier treatment."
                )

        # Missing value recommendations
        high_missing_vars = [
            var for var, pct in missing_patterns["percentages"].items() if pct > 20
        ]
        if high_missing_vars:
            recommendations.append(
                f"Variables with high missing rates: {', '.join(high_missing_vars)}. "
                "Consider targeted data collection or alternative measurement methods."
            )

        # Consistency recommendations
        if consistency_issues:
            recommendations.append(
                f"Data consistency issues detected: {len(consistency_issues)} violations. "
                "Review data collection and preprocessing procedures."
            )

        return recommendations

    def _generate_validation_report(
        self,
        quality_metrics: QualityMetrics,
        outliers: Dict[str, np.ndarray],
        missing_patterns: Dict[str, Any],
        consistency_issues: List[str],
    ) -> str:
        """Generate comprehensive validation report"""
        report_lines = [
            "=== Data Quality Validation Report ===",
            "",
            f"Overall Quality Score: {quality_metrics.overall_quality:.3f}",
            f"Data Completeness: {quality_metrics.completeness:.2%}",
            f"Temporal Consistency: {quality_metrics.temporal_consistency:.3f}",
            f"Spatial Consistency: {quality_metrics.spatial_consistency:.3f}",
            "",
            "=== Missing Value Analysis ===",
        ]

        for var, pct in missing_patterns["percentages"].items():
            if pct > 0:
                report_lines.append(f"  {var}: {pct:.1f}% missing")

        if outliers and "combined" in outliers:
            outlier_rate = outliers["combined"].mean()
            report_lines.extend(
                [
                    "",
                    "=== Outlier Analysis ===",
                    f"Overall outlier rate: {outlier_rate:.2%}",
                ]
            )

        if consistency_issues:
            report_lines.extend(
                [
                    "",
                    "=== Consistency Issues ===",
                ]
            )
            for issue in consistency_issues:
                report_lines.append(f"  - {issue}")

        return "\n".join(report_lines)

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation runs"""
        if not self.validation_history:
            return {"message": "No validation runs performed yet"}

        latest_result = self.validation_history[-1]
        return {
            "total_validations": len(self.validation_history),
            "latest_quality_score": latest_result.quality_metrics.overall_quality,
            "latest_completeness": latest_result.quality_metrics.completeness,
            "is_valid": latest_result.is_valid,
            "recommendations_count": len(latest_result.recommendations),
        }
