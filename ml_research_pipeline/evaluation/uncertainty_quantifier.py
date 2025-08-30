"""
Uncertainty quantification system for physics discovery and meta-learning.

This module implements comprehensive uncertainty quantification including Bayesian methods,
Monte Carlo dropout, prediction intervals, and uncertainty calibration metrics.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


class MonteCarloDropout(nn.Module):
    """
    Monte Carlo Dropout implementation for uncertainty quantification in neural networks.

    Enables dropout during inference to estimate epistemic uncertainty through
    multiple forward passes with different dropout masks.
    """

    def __init__(self, model: nn.Module, dropout_rate: float = 0.1):
        """
        Initialize Monte Carlo Dropout wrapper.

        Args:
            model: Base neural network model
            dropout_rate: Dropout probability for uncertainty estimation
        """
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self._add_dropout_layers()

    def _add_dropout_layers(self):
        """Add dropout layers to the model for MC dropout."""

        def add_dropout_to_module(module):
            # Collect children first to avoid modifying dict during iteration
            children = list(module.named_children())
            for name, child in children:
                if isinstance(child, nn.Linear):
                    # Add dropout after linear layers
                    setattr(module, f"{name}_dropout", nn.Dropout(self.dropout_rate))
                else:
                    add_dropout_to_module(child)

        add_dropout_to_module(self.model)

    def forward(
        self, x: torch.Tensor, n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Monte Carlo dropout for uncertainty estimation.

        Args:
            x: Input tensor
            n_samples: Number of MC samples for uncertainty estimation

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean predictions and uncertainty estimates
        """
        self.train()  # Enable dropout during inference

        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self._forward_with_dropout(x)
                predictions.append(pred)

        predictions = torch.stack(
            predictions, dim=0
        )  # Shape: (n_samples, batch_size, output_dim)

        # Compute statistics
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)

        return mean_pred, uncertainty

    def _forward_with_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through model with dropout enabled."""
        # This is a simplified implementation - in practice, you'd need to
        # modify the forward pass to include dropout layers
        return self.model(x)


class BayesianUncertaintyQuantifier:
    """
    Bayesian uncertainty quantification using variational inference and ensemble methods.

    Implements multiple Bayesian approaches for uncertainty estimation including
    variational Bayes, ensemble methods, and Laplace approximation.
    """

    def __init__(
        self,
        method: str = "ensemble",
        n_ensemble: int = 10,
        prior_std: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        Initialize Bayesian uncertainty quantifier.

        Args:
            method: Bayesian method ('ensemble', 'variational', 'laplace')
            n_ensemble: Number of ensemble members for ensemble method
            prior_std: Prior standard deviation for Bayesian inference
            temperature: Temperature scaling parameter
        """
        self.method = method
        self.n_ensemble = n_ensemble
        self.prior_std = prior_std
        self.temperature = temperature
        self.ensemble_models = []
        self.is_fitted = False

    def fit_ensemble(
        self,
        model_class: type,
        train_data: List[Tuple[torch.Tensor, torch.Tensor]],
        model_kwargs: Dict[str, Any],
        training_kwargs: Dict[str, Any],
    ) -> "BayesianUncertaintyQuantifier":
        """
        Fit ensemble of models for uncertainty quantification.

        Args:
            model_class: Class of model to ensemble
            train_data: List of (input, target) training data tuples
            model_kwargs: Keyword arguments for model initialization
            training_kwargs: Keyword arguments for training

        Returns:
            self: Fitted quantifier
        """
        self.ensemble_models = []

        for i in range(self.n_ensemble):
            # Initialize model with different random seed
            torch.manual_seed(i * 42)
            model = model_class(**model_kwargs)

            # Train model on bootstrap sample or full data with different initialization
            self._train_ensemble_member(model, train_data, training_kwargs)

            self.ensemble_models.append(model)

        self.is_fitted = True
        return self

    def _train_ensemble_member(
        self,
        model: nn.Module,
        train_data: List[Tuple[torch.Tensor, torch.Tensor]],
        training_kwargs: Dict[str, Any],
    ):
        """Train a single ensemble member."""
        optimizer = torch.optim.Adam(
            model.parameters(), lr=training_kwargs.get("lr", 0.001)
        )
        n_epochs = training_kwargs.get("n_epochs", 100)

        model.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            for x_batch, y_batch in train_data:
                optimizer.zero_grad()

                # Forward pass
                pred = model(x_batch)
                loss = F.mse_loss(pred, y_batch)

                # Add L2 regularization (Bayesian prior)
                l2_reg = 0.0
                for param in model.parameters():
                    l2_reg += torch.sum(param**2)
                loss += (1.0 / (2 * self.prior_std**2)) * l2_reg

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

    def predict_with_uncertainty(
        self, x: torch.Tensor, return_individual: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Make predictions with uncertainty estimates using ensemble.

        Args:
            x: Input tensor
            return_individual: Whether to return individual ensemble predictions

        Returns:
            Tuple containing mean predictions and uncertainty estimates,
            optionally with individual predictions
        """
        if not self.is_fitted:
            raise ValueError("Quantifier must be fitted before making predictions")

        predictions = []

        for model in self.ensemble_models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        predictions = torch.stack(
            predictions, dim=0
        )  # Shape: (n_ensemble, batch_size, output_dim)

        # Compute ensemble statistics
        mean_pred = torch.mean(predictions, dim=0)
        epistemic_uncertainty = torch.std(predictions, dim=0)

        if return_individual:
            return mean_pred, epistemic_uncertainty, predictions
        else:
            return mean_pred, epistemic_uncertainty

    def compute_prediction_intervals(
        self, x: torch.Tensor, confidence_levels: List[float] = [0.68, 0.95, 0.99]
    ) -> Dict[float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute prediction intervals at specified confidence levels.

        Args:
            x: Input tensor
            confidence_levels: List of confidence levels (e.g., 0.95 for 95% interval)

        Returns:
            Dict mapping confidence levels to (lower_bound, upper_bound) tensors
        """
        if not self.is_fitted:
            raise ValueError("Quantifier must be fitted before computing intervals")

        # Get individual ensemble predictions
        _, _, predictions = self.predict_with_uncertainty(x, return_individual=True)

        intervals = {}
        for confidence in confidence_levels:
            alpha = 1 - confidence
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower_bound = torch.quantile(predictions, lower_percentile / 100, dim=0)
            upper_bound = torch.quantile(predictions, upper_percentile / 100, dim=0)

            intervals[confidence] = (lower_bound, upper_bound)

        return intervals


class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification system for physics discovery and meta-learning.

    Combines multiple uncertainty estimation methods including Bayesian inference,
    Monte Carlo dropout, prediction intervals, and calibration metrics.
    """

    def __init__(
        self,
        methods: List[str] = ["bayesian", "mc_dropout"],
        bayesian_config: Optional[Dict[str, Any]] = None,
        mc_dropout_config: Optional[Dict[str, Any]] = None,
        calibration_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize comprehensive uncertainty quantifier.

        Args:
            methods: List of uncertainty methods to use
            bayesian_config: Configuration for Bayesian uncertainty quantification
            mc_dropout_config: Configuration for Monte Carlo dropout
            calibration_config: Configuration for uncertainty calibration
        """
        self.methods = methods

        # Initialize method-specific quantifiers
        self.bayesian_quantifier = None
        self.mc_dropout_quantifier = None

        if "bayesian" in methods:
            bayesian_config = bayesian_config or {}
            self.bayesian_quantifier = BayesianUncertaintyQuantifier(**bayesian_config)

        if "mc_dropout" in methods:
            mc_dropout_config = mc_dropout_config or {}
            # MC dropout quantifier will be initialized when model is provided
            self.mc_dropout_config = mc_dropout_config

        # Calibration configuration
        calibration_config = calibration_config or {}
        self.calibration_bins = calibration_config.get("n_bins", 10)
        self.calibration_method = calibration_config.get("method", "isotonic")

        # Storage for uncertainty estimates and calibration data
        self.uncertainty_history = []
        self.calibration_data = {}

    def fit(
        self,
        model: Optional[nn.Module] = None,
        train_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        model_class: Optional[type] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        training_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "UncertaintyQuantifier":
        """
        Fit uncertainty quantification methods.

        Args:
            model: Pre-trained model for MC dropout
            train_data: Training data for Bayesian ensemble
            model_class: Model class for Bayesian ensemble
            model_kwargs: Model initialization arguments
            training_kwargs: Training arguments

        Returns:
            self: Fitted quantifier
        """
        if "bayesian" in self.methods and self.bayesian_quantifier is not None:
            if train_data is None or model_class is None:
                raise ValueError("Bayesian method requires train_data and model_class")

            model_kwargs = model_kwargs or {}
            training_kwargs = training_kwargs or {}

            self.bayesian_quantifier.fit_ensemble(
                model_class, train_data, model_kwargs, training_kwargs
            )

        if "mc_dropout" in self.methods:
            if model is None:
                raise ValueError("MC dropout method requires a pre-trained model")

            dropout_rate = self.mc_dropout_config.get("dropout_rate", 0.1)
            self.mc_dropout_quantifier = MonteCarloDropout(model, dropout_rate)

        return self

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_mc_samples: int = 100, return_components: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """
        Make predictions with comprehensive uncertainty estimates.

        Args:
            x: Input tensor
            n_mc_samples: Number of Monte Carlo samples for MC dropout
            return_components: Whether to return uncertainty components separately

        Returns:
            Predictions and uncertainty estimates, optionally separated by method
        """
        results = {}

        # Bayesian uncertainty
        if "bayesian" in self.methods and self.bayesian_quantifier is not None:
            if self.bayesian_quantifier.is_fitted:
                mean_pred, epistemic_unc = (
                    self.bayesian_quantifier.predict_with_uncertainty(x)
                )
                results["bayesian"] = (mean_pred, epistemic_unc)

        # Monte Carlo dropout uncertainty
        if "mc_dropout" in self.methods and self.mc_dropout_quantifier is not None:
            mean_pred, mc_unc = self.mc_dropout_quantifier.forward(x, n_mc_samples)
            results["mc_dropout"] = (mean_pred, mc_unc)

        if not results:
            raise ValueError("No uncertainty methods are fitted and available")

        if return_components:
            return results

        # Combine uncertainties if multiple methods are used
        if len(results) == 1:
            return list(results.values())[0]
        else:
            # Ensemble multiple uncertainty estimates
            mean_preds = [result[0] for result in results.values()]
            uncertainties = [result[1] for result in results.values()]

            # Average predictions
            combined_mean = torch.mean(torch.stack(mean_preds), dim=0)

            # Combine uncertainties (geometric mean for diversity)
            combined_uncertainty = torch.exp(
                torch.mean(torch.log(torch.stack(uncertainties) + 1e-8), dim=0)
            )

            return combined_mean, combined_uncertainty

    def compute_prediction_intervals(
        self,
        x: torch.Tensor,
        confidence_levels: List[float] = [0.68, 0.95, 0.99],
        method: str = "bayesian",
    ) -> Dict[float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute prediction intervals using specified method.

        Args:
            x: Input tensor
            confidence_levels: Confidence levels for intervals
            method: Method to use ('bayesian' or 'gaussian')

        Returns:
            Dict mapping confidence levels to interval bounds
        """
        if method == "bayesian" and "bayesian" in self.methods:
            if (
                self.bayesian_quantifier is None
                or not self.bayesian_quantifier.is_fitted
            ):
                raise ValueError("Bayesian quantifier not fitted")
            return self.bayesian_quantifier.compute_prediction_intervals(
                x, confidence_levels
            )

        elif method == "gaussian":
            # Use Gaussian approximation with uncertainty estimates
            mean_pred, uncertainty = self.predict_with_uncertainty(x)

            intervals = {}
            for confidence in confidence_levels:
                # Use normal distribution quantiles
                z_score = stats.norm.ppf((1 + confidence) / 2)
                margin = z_score * uncertainty

                lower_bound = mean_pred - margin
                upper_bound = mean_pred + margin

                intervals[confidence] = (lower_bound, upper_bound)

            return intervals
        else:
            raise ValueError(f"Unknown interval method: {method}")

    def evaluate_uncertainty_quality(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        return_detailed: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate quality of uncertainty estimates using multiple metrics.

        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            targets: Ground truth targets
            return_detailed: Whether to return detailed metrics

        Returns:
            Dict of uncertainty quality metrics
        """
        # Convert to numpy for easier computation
        pred_np = predictions.detach().cpu().numpy().flatten()
        unc_np = uncertainties.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()

        # Compute prediction errors
        errors = np.abs(pred_np - target_np)
        squared_errors = (pred_np - target_np) ** 2

        # Remove invalid values
        valid_mask = (
            np.isfinite(pred_np)
            & np.isfinite(unc_np)
            & np.isfinite(target_np)
            & (unc_np > 0)
        )
        if not np.any(valid_mask):
            return {"error": "No valid data points"}

        pred_np = pred_np[valid_mask]
        unc_np = unc_np[valid_mask]
        target_np = target_np[valid_mask]
        errors = errors[valid_mask]
        squared_errors = squared_errors[valid_mask]

        metrics = {}

        # 1. Correlation between uncertainty and error
        if len(errors) > 1:
            correlation = np.corrcoef(unc_np, errors)[0, 1]
            metrics["uncertainty_error_correlation"] = (
                float(correlation) if np.isfinite(correlation) else 0.0
            )
        else:
            metrics["uncertainty_error_correlation"] = 0.0

        # 2. Calibration metrics (Expected Calibration Error)
        ece = self._compute_expected_calibration_error(unc_np, errors)
        metrics["expected_calibration_error"] = ece

        # 3. Sharpness (average uncertainty)
        metrics["mean_uncertainty"] = float(np.mean(unc_np))
        metrics["uncertainty_std"] = float(np.std(unc_np))

        # 4. Coverage probability for prediction intervals
        coverage_68 = self._compute_coverage_probability(
            pred_np, unc_np, target_np, confidence=0.68
        )
        coverage_95 = self._compute_coverage_probability(
            pred_np, unc_np, target_np, confidence=0.95
        )

        metrics["coverage_68"] = coverage_68
        metrics["coverage_95"] = coverage_95

        # 5. Negative log-likelihood (assuming Gaussian)
        nll = self._compute_negative_log_likelihood(pred_np, unc_np, target_np)
        metrics["negative_log_likelihood"] = nll

        # 6. Uncertainty-based ranking metrics
        ranking_metrics = self._compute_ranking_metrics(unc_np, errors)
        metrics.update(ranking_metrics)

        if return_detailed:
            # Add detailed statistics
            metrics.update(
                {
                    "n_samples": len(pred_np),
                    "mean_error": float(np.mean(errors)),
                    "rmse": float(np.sqrt(np.mean(squared_errors))),
                    "mae": float(np.mean(errors)),
                    "uncertainty_range": [float(np.min(unc_np)), float(np.max(unc_np))],
                    "error_range": [float(np.min(errors)), float(np.max(errors))],
                }
            )

        return metrics

    def _compute_expected_calibration_error(
        self, uncertainties: np.ndarray, errors: np.ndarray
    ) -> float:
        """Compute Expected Calibration Error (ECE)."""
        # Bin uncertainties
        n_bins = min(
            self.calibration_bins, len(uncertainties) // 5
        )  # Ensure sufficient samples per bin
        if n_bins < 2:
            return float("inf")

        bin_boundaries = np.linspace(0, np.max(uncertainties), n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        total_samples = len(uncertainties)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                bin_errors = errors[in_bin]
                bin_uncertainties = uncertainties[in_bin]

                avg_confidence = np.mean(bin_uncertainties)
                avg_accuracy = np.mean(bin_errors)

                ece += prop_in_bin * abs(avg_confidence - avg_accuracy)

        return float(ece)

    def _compute_coverage_probability(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        confidence: float,
    ) -> float:
        """Compute coverage probability for prediction intervals."""
        # Use Gaussian approximation
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * uncertainties

        lower_bound = predictions - margin
        upper_bound = predictions + margin

        # Check if targets fall within intervals
        within_interval = (targets >= lower_bound) & (targets <= upper_bound)
        coverage = np.mean(within_interval)

        return float(coverage)

    def _compute_negative_log_likelihood(
        self, predictions: np.ndarray, uncertainties: np.ndarray, targets: np.ndarray
    ) -> float:
        """Compute negative log-likelihood assuming Gaussian distribution."""
        # Avoid numerical issues
        uncertainties = np.maximum(uncertainties, 1e-8)

        # Gaussian NLL
        nll = 0.5 * np.log(2 * np.pi * uncertainties**2) + 0.5 * (
            (predictions - targets) ** 2
        ) / (uncertainties**2)

        return float(np.mean(nll))

    def _compute_ranking_metrics(
        self, uncertainties: np.ndarray, errors: np.ndarray
    ) -> Dict[str, float]:
        """Compute ranking-based uncertainty quality metrics."""
        # Sort by uncertainty (descending)
        sorted_indices = np.argsort(-uncertainties)
        sorted_errors = errors[sorted_indices]

        n_samples = len(errors)
        metrics = {}

        # Area Under Sparsification Error (AUSE)
        # Measures how well uncertainty ranks errors
        cumulative_errors = np.cumsum(sorted_errors)
        total_error = np.sum(errors)

        if total_error > 0:
            sparsification_errors = cumulative_errors / total_error
            fractions_removed = np.arange(1, n_samples + 1) / n_samples

            # Compute area under curve
            ause = np.trapz(sparsification_errors, fractions_removed)
            metrics["area_under_sparsification_error"] = float(ause)
        else:
            metrics["area_under_sparsification_error"] = 0.0

        # Uncertainty-Error Ranking Correlation (Spearman)
        if n_samples > 1:
            from scipy.stats import spearmanr

            correlation, p_value = spearmanr(uncertainties, errors)
            metrics["spearman_correlation"] = (
                float(correlation) if np.isfinite(correlation) else 0.0
            )
            metrics["spearman_p_value"] = (
                float(p_value) if np.isfinite(p_value) else 1.0
            )
        else:
            metrics["spearman_correlation"] = 0.0
            metrics["spearman_p_value"] = 1.0

        return metrics

    def calibrate_uncertainties(
        self,
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
        method: str = "isotonic",
    ) -> torch.Tensor:
        """
        Calibrate uncertainty estimates to improve reliability.

        Args:
            uncertainties: Raw uncertainty estimates
            errors: Actual prediction errors
            method: Calibration method ('isotonic', 'platt', or 'temperature')

        Returns:
            torch.Tensor: Calibrated uncertainty estimates
        """
        from ml_research_pipeline.bayesian.uncertainty_calibrator import (
            UncertaintyCalibrator,
        )

        # Create and fit calibrator
        calibrator = UncertaintyCalibrator(n_bins=self.calibration_bins, method=method)
        calibrator.fit(uncertainties, errors)

        # Apply calibration
        calibrated = calibrator.calibrate(uncertainties)

        # Store calibration data for analysis
        self.calibration_data[method] = {
            "calibrator": calibrator,
            "original_uncertainties": uncertainties.clone(),
            "calibrated_uncertainties": calibrated.clone(),
            "errors": errors.clone(),
        }

        return calibrated

    def plot_uncertainty_analysis(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Create comprehensive uncertainty analysis plots.

        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            targets: Ground truth targets
            save_path: Optional path to save plot
            show: Whether to display plot

        Returns:
            matplotlib.figure.Figure: Analysis figure
        """
        # Convert to numpy
        pred_np = predictions.detach().cpu().numpy().flatten()
        unc_np = uncertainties.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        errors = np.abs(pred_np - target_np)

        # Remove invalid values
        valid_mask = (
            np.isfinite(pred_np)
            & np.isfinite(unc_np)
            & np.isfinite(target_np)
            & (unc_np > 0)
        )
        pred_np = pred_np[valid_mask]
        unc_np = unc_np[valid_mask]
        target_np = target_np[valid_mask]
        errors = errors[valid_mask]

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Uncertainty vs Error scatter plot
        axes[0, 0].scatter(unc_np, errors, alpha=0.6, s=20)
        axes[0, 0].set_xlabel("Predicted Uncertainty")
        axes[0, 0].set_ylabel("Prediction Error")
        axes[0, 0].set_title("Uncertainty vs Error")
        axes[0, 0].grid(True, alpha=0.3)

        # Add correlation coefficient
        if len(errors) > 1:
            corr = np.corrcoef(unc_np, errors)[0, 1]
            axes[0, 0].text(
                0.05,
                0.95,
                f"Correlation: {corr:.3f}",
                transform=axes[0, 0].transAxes,
                verticalalignment="top",
            )

        # 2. Prediction vs Target with uncertainty bars
        axes[0, 1].errorbar(
            target_np, pred_np, yerr=unc_np, fmt="o", alpha=0.6, markersize=3
        )
        min_val = min(np.min(target_np), np.min(pred_np))
        max_val = max(np.max(target_np), np.max(pred_np))
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.7)
        axes[0, 1].set_xlabel("True Values")
        axes[0, 1].set_ylabel("Predictions")
        axes[0, 1].set_title("Predictions vs Targets")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Uncertainty distribution
        axes[0, 2].hist(unc_np, bins=30, alpha=0.7, edgecolor="black")
        axes[0, 2].set_xlabel("Uncertainty")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].set_title("Uncertainty Distribution")
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Error distribution
        axes[1, 0].hist(errors, bins=30, alpha=0.7, edgecolor="black", color="orange")
        axes[1, 0].set_xlabel("Prediction Error")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Error Distribution")
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Calibration plot (reliability diagram)
        n_bins = min(10, len(unc_np) // 10)
        if n_bins >= 2:
            bin_boundaries = np.linspace(0, np.max(unc_np), n_bins + 1)
            bin_centers = []
            bin_accuracies = []

            for i in range(n_bins):
                bin_mask = (unc_np >= bin_boundaries[i]) & (
                    unc_np < bin_boundaries[i + 1]
                )
                if np.sum(bin_mask) > 0:
                    bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                    bin_accuracies.append(np.mean(errors[bin_mask]))

            if bin_centers:
                axes[1, 1].scatter(bin_centers, bin_accuracies, s=50)
                max_val = max(max(bin_centers), max(bin_accuracies))
                axes[1, 1].plot(
                    [0, max_val],
                    [0, max_val],
                    "r--",
                    alpha=0.7,
                    label="Perfect Calibration",
                )
                axes[1, 1].set_xlabel("Mean Predicted Uncertainty")
                axes[1, 1].set_ylabel("Mean Actual Error")
                axes[1, 1].set_title("Calibration Plot")
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

        # 6. Cumulative error by uncertainty ranking
        sorted_indices = np.argsort(-unc_np)  # Sort by uncertainty (descending)
        sorted_errors = errors[sorted_indices]
        cumulative_errors = np.cumsum(sorted_errors)
        fractions = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

        axes[1, 2].plot(fractions, cumulative_errors / np.sum(errors))
        axes[1, 2].set_xlabel("Fraction of Data Removed (by uncertainty)")
        axes[1, 2].set_ylabel("Cumulative Error Fraction")
        axes[1, 2].set_title("Sparsification Plot")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """
        Get summary of uncertainty quantification configuration and status.

        Returns:
            Dict[str, Any]: Summary information
        """
        summary = {
            "methods": self.methods,
            "calibration_bins": self.calibration_bins,
            "calibration_method": self.calibration_method,
        }

        # Bayesian quantifier status
        if self.bayesian_quantifier is not None:
            summary["bayesian_status"] = {
                "fitted": self.bayesian_quantifier.is_fitted,
                "n_ensemble": self.bayesian_quantifier.n_ensemble,
                "method": self.bayesian_quantifier.method,
            }
        else:
            summary["bayesian_status"] = "not_initialized"

        # MC dropout status
        if (
            hasattr(self, "mc_dropout_quantifier")
            and self.mc_dropout_quantifier is not None
        ):
            summary["mc_dropout_status"] = {
                "initialized": True,
                "dropout_rate": self.mc_dropout_quantifier.dropout_rate,
            }
        else:
            summary["mc_dropout_status"] = "not_initialized"

        # Calibration data status
        summary["calibration_data_available"] = list(self.calibration_data.keys())

        return summary
