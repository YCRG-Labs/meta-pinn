"""
Advanced Data Preprocessor Orchestrator

This module implements the main preprocessing pipeline that coordinates all components:
- NoiseReductionEngine for denoising
- PhysicsFeatureEngineer for feature generation
- DataQualityValidator for validation and cleaning
- Quality metrics calculation and reporting
- Preprocessing history tracking for reproducibility
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .data_validation import DataQualityValidator, ValidationConfig, ValidationResult
from .feature_engineering import FeatureEngineeringResult, PhysicsFeatureEngineer
from .noise_reduction import DenoiseMethod, DenoiseResult, NoiseReductionEngine


@dataclass
class PreprocessingConfig:
    """Configuration for the advanced data preprocessor."""

    # Noise reduction settings
    noise_reduction_method: DenoiseMethod = DenoiseMethod.AUTO
    snr_improvement_threshold: float = 1.5
    noise_quality_threshold: float = 0.7

    # Feature engineering settings
    physics_domain: str = "fluid_dynamics"
    enable_dimensional_validation: bool = True
    feature_importance_threshold: float = 0.1

    # Data validation settings
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)

    # Pipeline settings
    enable_noise_reduction: bool = True
    enable_feature_engineering: bool = True
    enable_data_validation: bool = True
    enable_auto_cleaning: bool = True

    # Quality thresholds
    min_overall_quality: float = 0.7
    min_snr_improvement: float = 1.5
    min_feature_count: int = 3

    # Reproducibility settings
    enable_history_tracking: bool = True
    save_intermediate_results: bool = False
    output_directory: Optional[str] = None


@dataclass
class ProcessedData:
    """Enhanced processed data with quality metrics."""

    features: Dict[str, np.ndarray]
    targets: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    quality_metrics: Dict[str, float]
    preprocessing_history: List[str]
    generated_features: Optional[Dict[str, Any]] = None
    validation_report: Optional[str] = None


@dataclass
class PreprocessingResult:
    """Complete result of preprocessing pipeline."""

    processed_data: ProcessedData
    noise_reduction_results: Optional[Dict[str, DenoiseResult]]
    feature_engineering_results: Optional[FeatureEngineeringResult]
    validation_results: Optional[ValidationResult]
    overall_quality_score: float
    processing_time: float
    success: bool
    warnings: List[str]
    recommendations: List[str]


class AdvancedDataPreprocessor:
    """
    Main preprocessing pipeline that coordinates all components.

    Implements Requirements 2.1, 2.2, 2.3, 2.4:
    - Noise reduction techniques that improve SNR by at least 50%
    - Physics-informed feature generation
    - Data quality validation and cleaning
    - Comprehensive preprocessing orchestration
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the advanced data preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize component engines
        self.noise_reducer = None
        self.feature_engineer = None
        self.quality_validator = None

        if self.config.enable_noise_reduction:
            self.noise_reducer = NoiseReductionEngine(
                default_method=self.config.noise_reduction_method,
                snr_improvement_threshold=self.config.snr_improvement_threshold,
                quality_threshold=self.config.noise_quality_threshold,
            )

        if self.config.enable_feature_engineering:
            self.feature_engineer = PhysicsFeatureEngineer(
                domain=self.config.physics_domain,
                enable_dimensional_validation=self.config.enable_dimensional_validation,
                importance_threshold=self.config.feature_importance_threshold,
            )

        if self.config.enable_data_validation:
            self.quality_validator = DataQualityValidator(self.config.validation_config)

        # Processing history
        self.processing_history = []

        # Setup output directory if specified
        if self.config.output_directory:
            self.output_dir = Path(self.config.output_directory)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

    def preprocess(
        self,
        raw_data: Dict[str, np.ndarray],
        spatial_coordinates: Optional[Dict[str, np.ndarray]] = None,
        physical_properties: Optional[Dict[str, float]] = None,
        target_variables: Optional[Dict[str, np.ndarray]] = None,
    ) -> PreprocessingResult:
        """
        Execute the complete preprocessing pipeline.

        Args:
            raw_data: Dictionary of raw input data arrays
            spatial_coordinates: Spatial coordinate arrays (x, y, z)
            physical_properties: Physical properties (viscosity, density, etc.)
            target_variables: Target variables for supervised learning

        Returns:
            Complete preprocessing result with all components
        """
        start_time = datetime.now()
        self.logger.info("Starting advanced data preprocessing pipeline")

        # Initialize result containers
        warnings = []
        recommendations = []
        preprocessing_steps = []

        # Step 1: Data Validation (initial)
        validation_results = None
        if self.config.enable_data_validation:
            try:
                self.logger.info("Step 1: Initial data quality validation")
                validation_results = self.quality_validator.validate_data(raw_data)
                preprocessing_steps.append("initial_validation")

                if not validation_results.is_valid:
                    warnings.append(
                        "Initial data validation failed - proceeding with caution"
                    )
                    recommendations.extend(validation_results.recommendations)

                # Use cleaned data if available and auto-cleaning is enabled
                if (
                    self.config.enable_auto_cleaning
                    and validation_results.cleaned_data is not None
                ):
                    working_data = validation_results.cleaned_data
                    preprocessing_steps.append("auto_cleaning")
                    self.logger.info("Using auto-cleaned data for further processing")
                else:
                    working_data = raw_data.copy()

            except Exception as e:
                self.logger.error(f"Data validation failed: {str(e)}")
                warnings.append(f"Data validation failed: {str(e)}")
                working_data = raw_data.copy()
        else:
            working_data = raw_data.copy()

        # Step 2: Noise Reduction
        noise_reduction_results = None
        if self.config.enable_noise_reduction and self.noise_reducer:
            try:
                self.logger.info("Step 2: Noise reduction")
                noise_reduction_results = self.noise_reducer.denoise_data(working_data)
                preprocessing_steps.append("noise_reduction")

                # Update working data with denoised results
                if isinstance(noise_reduction_results, dict):
                    for key, result in noise_reduction_results.items():
                        if (
                            result.snr_improvement >= self.config.min_snr_improvement
                            and result.quality_score
                            >= self.config.noise_quality_threshold
                        ):
                            working_data[key] = result.denoised_data
                            self.logger.info(
                                f"Applied denoising to {key}: SNR improvement {result.snr_improvement:.2f}"
                            )
                        else:
                            warnings.append(f"Denoising quality insufficient for {key}")
                else:
                    # Single array result
                    if (
                        noise_reduction_results.snr_improvement
                        >= self.config.min_snr_improvement
                        and noise_reduction_results.quality_score
                        >= self.config.noise_quality_threshold
                    ):
                        # Update the first key in working_data
                        first_key = next(iter(working_data.keys()))
                        working_data[first_key] = noise_reduction_results.denoised_data

            except Exception as e:
                self.logger.error(f"Noise reduction failed: {str(e)}")
                warnings.append(f"Noise reduction failed: {str(e)}")

        # Step 3: Feature Engineering
        feature_engineering_results = None
        if self.config.enable_feature_engineering and self.feature_engineer:
            try:
                self.logger.info("Step 3: Physics-informed feature engineering")
                feature_engineering_results = (
                    self.feature_engineer.generate_physics_features(
                        working_data, spatial_coordinates, physical_properties
                    )
                )
                preprocessing_steps.append("feature_engineering")

                # Add generated features to working data
                for (
                    name,
                    feature,
                ) in feature_engineering_results.generated_features.items():
                    working_data[f"feature_{name}"] = feature.data

                if (
                    len(feature_engineering_results.generated_features)
                    < self.config.min_feature_count
                ):
                    warnings.append(
                        f"Only {len(feature_engineering_results.generated_features)} features generated"
                    )

            except Exception as e:
                self.logger.error(f"Feature engineering failed: {str(e)}")
                warnings.append(f"Feature engineering failed: {str(e)}")

        # Step 4: Final Quality Assessment
        final_quality_metrics = self._calculate_final_quality_metrics(
            working_data,
            noise_reduction_results,
            feature_engineering_results,
            validation_results,
        )

        # Determine targets
        if target_variables:
            targets = target_variables.copy()
        else:
            # Try to identify potential targets from the data
            targets = self._identify_potential_targets(working_data)

        # Create processed data object
        processed_data = ProcessedData(
            features=working_data,
            targets=targets,
            metadata={
                "processing_timestamp": start_time.isoformat(),
                "config": self._config_to_dict(),
                "spatial_coordinates": spatial_coordinates,
                "physical_properties": physical_properties,
            },
            quality_metrics=final_quality_metrics,
            preprocessing_history=preprocessing_steps,
            generated_features=(
                feature_engineering_results.generated_features
                if feature_engineering_results
                else None
            ),
            validation_report=(
                validation_results.validation_report if validation_results else None
            ),
        )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Determine overall success
        overall_quality = final_quality_metrics.get("overall_quality", 0.0)
        success = (
            overall_quality >= self.config.min_overall_quality and len(warnings) == 0
        )

        # Generate additional recommendations
        if overall_quality < self.config.min_overall_quality:
            recommendations.append(
                f"Overall quality score {overall_quality:.3f} below threshold {self.config.min_overall_quality}"
            )

        # Create final result
        result = PreprocessingResult(
            processed_data=processed_data,
            noise_reduction_results=noise_reduction_results,
            feature_engineering_results=feature_engineering_results,
            validation_results=validation_results,
            overall_quality_score=overall_quality,
            processing_time=processing_time,
            success=success,
            warnings=warnings,
            recommendations=recommendations,
        )

        # Store in history
        if self.config.enable_history_tracking:
            self.processing_history.append(result)

        # Save intermediate results if requested
        if self.config.save_intermediate_results and self.output_dir:
            self._save_intermediate_results(result)

        self.logger.info(
            f"Preprocessing completed in {processing_time:.2f}s with quality score {overall_quality:.3f}"
        )
        return result

    def _calculate_final_quality_metrics(
        self,
        working_data: Dict[str, np.ndarray],
        noise_results: Optional[Dict[str, DenoiseResult]],
        feature_results: Optional[FeatureEngineeringResult],
        validation_results: Optional[ValidationResult],
    ) -> Dict[str, float]:
        """Calculate comprehensive quality metrics for the final processed data."""
        metrics = {}

        # Data completeness
        total_elements = sum(arr.size for arr in working_data.values())
        if total_elements > 0:
            # Check for NaN/inf values
            valid_elements = sum(
                np.sum(np.isfinite(arr)) for arr in working_data.values()
            )
            metrics["data_completeness"] = valid_elements / total_elements
        else:
            metrics["data_completeness"] = 0.0

        # Noise reduction quality
        if noise_results:
            if isinstance(noise_results, dict):
                snr_improvements = [r.snr_improvement for r in noise_results.values()]
                quality_scores = [r.quality_score for r in noise_results.values()]
            else:
                snr_improvements = [noise_results.snr_improvement]
                quality_scores = [noise_results.quality_score]

            metrics["avg_snr_improvement"] = np.mean(snr_improvements)
            metrics["avg_noise_quality"] = np.mean(quality_scores)
        else:
            metrics["avg_snr_improvement"] = 1.0
            metrics["avg_noise_quality"] = 1.0

        # Feature engineering quality
        if feature_results:
            metrics["features_generated"] = len(feature_results.generated_features)
            if feature_results.feature_importance_ranking:
                avg_importance = np.mean(
                    [score for _, score in feature_results.feature_importance_ranking]
                )
                metrics["avg_feature_importance"] = avg_importance
            else:
                metrics["avg_feature_importance"] = 0.0
        else:
            metrics["features_generated"] = 0
            metrics["avg_feature_importance"] = 0.0

        # Validation quality
        if validation_results:
            metrics["validation_quality"] = (
                validation_results.quality_metrics.overall_quality
            )
            metrics["data_consistency"] = (
                validation_results.quality_metrics.temporal_consistency
                + validation_results.quality_metrics.spatial_consistency
            ) / 2
        else:
            metrics["validation_quality"] = 1.0
            metrics["data_consistency"] = 1.0

        # Overall quality score (weighted combination)
        weights = {
            "data_completeness": 0.25,
            "avg_noise_quality": 0.20,
            "avg_feature_importance": 0.15,
            "validation_quality": 0.25,
            "data_consistency": 0.15,
        }

        overall_quality = sum(
            metrics.get(metric, 0.0) * weight for metric, weight in weights.items()
        )
        metrics["overall_quality"] = min(1.0, max(0.0, overall_quality))

        return metrics

    def _identify_potential_targets(
        self, data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Identify potential target variables from the data."""
        targets = {}

        # Common target variable names in physics
        target_candidates = [
            "pressure",
            "p",
            "temperature",
            "T",
            "density",
            "rho",
            "velocity_magnitude",
            "energy",
            "vorticity",
        ]

        for candidate in target_candidates:
            if candidate in data:
                targets[candidate] = data[candidate]

        # If no obvious targets, use the last few variables as potential targets
        if not targets and len(data) > 2:
            data_items = list(data.items())
            targets = dict(data_items[-2:])  # Last 2 variables as targets

        return targets

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "noise_reduction_method": (
                self.config.noise_reduction_method.value
                if self.config.noise_reduction_method
                else None
            ),
            "physics_domain": self.config.physics_domain,
            "enable_dimensional_validation": self.config.enable_dimensional_validation,
            "feature_importance_threshold": self.config.feature_importance_threshold,
            "min_overall_quality": self.config.min_overall_quality,
            "enable_history_tracking": self.config.enable_history_tracking,
        }

    def _save_intermediate_results(self, result: PreprocessingResult):
        """Save intermediate results to disk for reproducibility."""
        if not self.output_dir:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save processing summary
        summary = {
            "timestamp": timestamp,
            "overall_quality_score": result.overall_quality_score,
            "processing_time": result.processing_time,
            "success": result.success,
            "warnings": result.warnings,
            "recommendations": result.recommendations,
            "preprocessing_steps": result.processed_data.preprocessing_history,
            "quality_metrics": result.processed_data.quality_metrics,
        }

        summary_file = self.output_dir / f"preprocessing_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Save processed data arrays
        data_file = self.output_dir / f"processed_data_{timestamp}.npz"
        np.savez_compressed(
            data_file,
            **result.processed_data.features,
            **{f"target_{k}": v for k, v in result.processed_data.targets.items()},
        )

        self.logger.info(f"Intermediate results saved to {self.output_dir}")

    def get_quality_metrics(self) -> Dict[str, float]:
        """
        Return comprehensive quality metrics for the last preprocessing run.

        Returns:
            Dictionary of quality metrics
        """
        if not self.processing_history:
            return {"message": "No preprocessing operations performed yet"}

        latest_result = self.processing_history[-1]
        return latest_result.processed_data.quality_metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing performance across all operations."""
        if not self.processing_history:
            return {"message": "No preprocessing operations performed yet"}

        quality_scores = [
            result.overall_quality_score for result in self.processing_history
        ]
        processing_times = [
            result.processing_time for result in self.processing_history
        ]
        success_count = sum(1 for result in self.processing_history if result.success)

        # Component-specific summaries
        noise_summary = {}
        if self.noise_reducer:
            noise_summary = self.noise_reducer.get_performance_summary()

        feature_summary = {}
        if self.feature_engineer:
            feature_summary = self.feature_engineer.get_feature_summary()

        validation_summary = {}
        if self.quality_validator:
            validation_summary = self.quality_validator.get_validation_summary()

        return {
            "total_preprocessing_runs": len(self.processing_history),
            "success_rate": success_count / len(self.processing_history),
            "average_quality_score": np.mean(quality_scores),
            "median_quality_score": np.median(quality_scores),
            "average_processing_time": np.mean(processing_times),
            "component_summaries": {
                "noise_reduction": noise_summary,
                "feature_engineering": feature_summary,
                "data_validation": validation_summary,
            },
            "latest_quality_score": quality_scores[-1],
            "quality_trend": (
                "improving"
                if len(quality_scores) > 1 and quality_scores[-1] > quality_scores[-2]
                else "stable"
            ),
        }

    def reset_history(self):
        """Reset processing history for fresh start."""
        self.processing_history.clear()
        if self.noise_reducer:
            self.noise_reducer.denoising_history.clear()
        if self.feature_engineer:
            self.feature_engineer.generation_history.clear()
        if self.quality_validator:
            self.quality_validator.validation_history.clear()

        self.logger.info("Processing history reset")
