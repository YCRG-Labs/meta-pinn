"""
Uncertainty calibration system for Bayesian meta-learning PINNs.

This module implements uncertainty calibration using isotonic regression and
provides tools for evaluating calibration quality through reliability diagrams.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


class UncertaintyCalibrator:
    """
    Uncertainty calibration system using isotonic regression.

    Calibrates uncertainty estimates to improve reliability and provides
    tools for evaluating calibration quality.
    """

    def __init__(self, n_bins: int = 10, method: str = "isotonic"):
        """
        Initialize uncertainty calibrator.

        Args:
            n_bins: Number of bins for reliability diagram
            method: Calibration method ('isotonic' or 'platt')
        """
        self.n_bins = n_bins
        self.method = method
        self.calibrator = None
        self.is_fitted = False

        # Calibration statistics
        self.calibration_error = None
        self.reliability_data = None

    def fit(
        self,
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> "UncertaintyCalibrator":
        """
        Fit calibration model using uncertainty estimates and actual errors.

        Args:
            uncertainties: Predicted uncertainty values, shape (n_samples,)
            errors: Actual prediction errors, shape (n_samples,)
            weights: Optional sample weights, shape (n_samples,)

        Returns:
            self: Fitted calibrator
        """
        # Convert to numpy for sklearn compatibility
        uncertainties_np = uncertainties.detach().cpu().numpy().flatten()
        errors_np = errors.detach().cpu().numpy().flatten()

        if weights is not None:
            weights_np = weights.detach().cpu().numpy().flatten()
        else:
            weights_np = None

        # Remove invalid values
        valid_mask = np.isfinite(uncertainties_np) & np.isfinite(errors_np)
        uncertainties_np = uncertainties_np[valid_mask]
        errors_np = errors_np[valid_mask]

        if weights_np is not None:
            weights_np = weights_np[valid_mask]

        if len(uncertainties_np) == 0:
            raise ValueError("No valid uncertainty-error pairs found")

        # Fit calibration model
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(uncertainties_np, errors_np, sample_weight=weights_np)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True

        # Compute calibration statistics
        self._compute_calibration_stats(uncertainties_np, errors_np)

        return self

    def calibrate(self, uncertainties: torch.Tensor) -> torch.Tensor:
        """
        Apply calibration to uncertainty estimates.

        Args:
            uncertainties: Raw uncertainty estimates

        Returns:
            torch.Tensor: Calibrated uncertainty estimates
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")

        # Convert to numpy
        uncertainties_np = uncertainties.detach().cpu().numpy()
        original_shape = uncertainties_np.shape
        uncertainties_flat = uncertainties_np.flatten()

        # Apply calibration
        calibrated_flat = self.calibrator.predict(uncertainties_flat)
        calibrated_np = calibrated_flat.reshape(original_shape)

        # Convert back to tensor
        calibrated = torch.from_numpy(calibrated_np).to(uncertainties.device).float()

        return calibrated

    def _compute_calibration_stats(self, uncertainties: np.ndarray, errors: np.ndarray):
        """Compute calibration statistics."""
        # Bin uncertainties and compute reliability
        bin_boundaries = np.linspace(0, np.max(uncertainties), self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Compute statistics for this bin
                bin_errors = errors[in_bin]
                bin_uncertainties = uncertainties[in_bin]

                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(np.mean(bin_errors))
                bin_confidences.append(np.mean(bin_uncertainties))
                bin_counts.append(np.sum(in_bin))

        self.reliability_data = {
            "bin_centers": np.array(bin_centers),
            "bin_accuracies": np.array(bin_accuracies),
            "bin_confidences": np.array(bin_confidences),
            "bin_counts": np.array(bin_counts),
        }

        # Compute Expected Calibration Error (ECE)
        if len(bin_centers) > 0:
            total_samples = len(uncertainties)
            ece = 0.0
            for i in range(len(bin_centers)):
                bin_weight = bin_counts[i] / total_samples
                ece += bin_weight * abs(bin_confidences[i] - bin_accuracies[i])
            self.calibration_error = ece
        else:
            self.calibration_error = float("inf")

    def evaluate_calibration(
        self,
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
        return_detailed: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate calibration quality of uncertainty estimates.

        Args:
            uncertainties: Uncertainty estimates
            errors: Actual prediction errors
            return_detailed: Whether to return detailed statistics

        Returns:
            Dict[str, float]: Calibration metrics
        """
        uncertainties_np = uncertainties.detach().cpu().numpy().flatten()
        errors_np = errors.detach().cpu().numpy().flatten()

        # Remove invalid values
        valid_mask = np.isfinite(uncertainties_np) & np.isfinite(errors_np)
        uncertainties_np = uncertainties_np[valid_mask]
        errors_np = errors_np[valid_mask]

        if len(uncertainties_np) == 0:
            return {"ece": float("inf"), "mce": float("inf")}

        # Compute reliability diagram data
        bin_boundaries = np.linspace(0, np.max(uncertainties_np), self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (uncertainties_np > bin_lower) & (uncertainties_np <= bin_upper)

            if np.sum(in_bin) > 0:
                bin_errors = errors_np[in_bin]
                bin_uncertainties = uncertainties_np[in_bin]

                bin_accuracies.append(np.mean(bin_errors))
                bin_confidences.append(np.mean(bin_uncertainties))
                bin_counts.append(np.sum(in_bin))

        if len(bin_accuracies) == 0:
            return {"ece": float("inf"), "mce": float("inf")}

        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)

        # Expected Calibration Error (ECE)
        total_samples = len(uncertainties_np)
        bin_weights = bin_counts / total_samples
        ece = np.sum(bin_weights * np.abs(bin_confidences - bin_accuracies))

        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(bin_confidences - bin_accuracies))

        metrics = {
            "ece": float(ece),
            "mce": float(mce),
            "n_samples": int(total_samples),
            "n_bins": len(bin_accuracies),
        }

        if return_detailed:
            metrics.update(
                {
                    "bin_accuracies": bin_accuracies.tolist(),
                    "bin_confidences": bin_confidences.tolist(),
                    "bin_counts": bin_counts.tolist(),
                    "bin_weights": bin_weights.tolist(),
                }
            )

        return metrics

    def plot_reliability_diagram(
        self,
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot reliability diagram for uncertainty calibration.

        Args:
            uncertainties: Uncertainty estimates
            errors: Actual prediction errors
            save_path: Optional path to save the plot
            show: Whether to display the plot

        Returns:
            matplotlib.figure.Figure: The reliability diagram figure
        """
        # Evaluate calibration to get bin data
        metrics = self.evaluate_calibration(uncertainties, errors, return_detailed=True)

        if metrics["n_bins"] == 0:
            warnings.warn("No valid bins for reliability diagram")
            return None

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Reliability diagram
        bin_confidences = np.array(metrics["bin_confidences"])
        bin_accuracies = np.array(metrics["bin_accuracies"])
        bin_counts = np.array(metrics["bin_counts"])

        # Plot perfect calibration line
        max_val = max(np.max(bin_confidences), np.max(bin_accuracies))
        ax1.plot(
            [0, max_val], [0, max_val], "k--", alpha=0.7, label="Perfect calibration"
        )

        # Plot actual calibration
        ax1.scatter(
            bin_confidences,
            bin_accuracies,
            s=bin_counts / np.max(bin_counts) * 200,
            alpha=0.7,
            c="blue",
            label="Observed",
        )

        ax1.set_xlabel("Mean Predicted Uncertainty")
        ax1.set_ylabel("Mean Actual Error")
        ax1.set_title(f'Reliability Diagram\nECE: {metrics["ece"]:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Histogram of uncertainties
        uncertainties_np = uncertainties.detach().cpu().numpy().flatten()
        valid_mask = np.isfinite(uncertainties_np)
        uncertainties_np = uncertainties_np[valid_mask]

        ax2.hist(
            uncertainties_np,
            bins=self.n_bins,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        ax2.set_xlabel("Predicted Uncertainty")
        ax2.set_ylabel("Count")
        ax2.set_title("Uncertainty Distribution")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def physics_informed_calibration(
        self,
        model,
        coords: torch.Tensor,
        task_info: Dict[str, Any],
        params: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Compute physics-informed uncertainty estimates from residuals.

        Args:
            model: BayesianMetaPINN model
            coords: Input coordinates
            task_info: Task information
            params: Optional model parameters

        Returns:
            torch.Tensor: Physics-informed uncertainty estimates
        """
        # Compute physics residuals
        physics_losses = model.physics_loss(
            coords, task_info, params, create_graph=False
        )

        # Extract individual residual components
        momentum_x_residual = physics_losses.get("momentum_x", torch.tensor(0.0))
        momentum_y_residual = physics_losses.get("momentum_y", torch.tensor(0.0))
        continuity_residual = physics_losses.get("continuity", torch.tensor(0.0))

        # Combine residuals into uncertainty estimate
        total_residual = momentum_x_residual + momentum_y_residual + continuity_residual

        # Convert to per-point uncertainty
        batch_size = coords.shape[0]
        output_dim = model.config.get_layer_sizes()[-1]

        # Scale residual to uncertainty range
        uncertainty_scale = torch.sqrt(torch.clamp(total_residual, min=1e-8))
        physics_uncertainty = uncertainty_scale.expand(batch_size, output_dim)

        return physics_uncertainty

    def adaptive_calibration(
        self,
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
        physics_residuals: torch.Tensor,
        physics_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        Perform adaptive calibration combining data-driven and physics-informed uncertainty.

        Args:
            uncertainties: Model uncertainty estimates
            errors: Actual prediction errors
            physics_residuals: Physics residual magnitudes
            physics_weight: Weight for physics-informed component

        Returns:
            torch.Tensor: Adaptively calibrated uncertainties
        """
        # Calibrate data-driven uncertainties
        if self.is_fitted:
            calibrated_uncertainties = self.calibrate(uncertainties)
        else:
            calibrated_uncertainties = uncertainties

        # Ensure all inputs are finite and positive
        physics_residuals = torch.clamp(physics_residuals, min=1e-8)
        physics_residuals = torch.where(
            torch.isfinite(physics_residuals),
            physics_residuals,
            torch.tensor(1e-3, device=physics_residuals.device),
        )

        # Normalize physics residuals to uncertainty scale
        mean_residual = torch.mean(physics_residuals)
        if not torch.isfinite(mean_residual) or mean_residual <= 0:
            mean_residual = torch.tensor(1e-3, device=physics_residuals.device)

        physics_residuals_norm = physics_residuals / (mean_residual + 1e-8)
        physics_uncertainty = torch.sqrt(torch.clamp(physics_residuals_norm, min=1e-8))

        # Ensure physics uncertainty is finite
        physics_uncertainty = torch.where(
            torch.isfinite(physics_uncertainty),
            physics_uncertainty,
            torch.tensor(1e-3, device=physics_uncertainty.device),
        )

        # Combine uncertainties
        combined_uncertainty = (
            1 - physics_weight
        ) * calibrated_uncertainties + physics_weight * physics_uncertainty

        # Final safety check
        combined_uncertainty = torch.clamp(combined_uncertainty, min=1e-8)
        combined_uncertainty = torch.where(
            torch.isfinite(combined_uncertainty),
            combined_uncertainty,
            torch.tensor(1e-3, device=combined_uncertainty.device),
        )

        return combined_uncertainty

    def get_calibration_summary(self) -> Dict[str, Any]:
        """
        Get summary of calibration statistics.

        Returns:
            Dict[str, Any]: Calibration summary
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}

        summary = {
            "status": "fitted",
            "method": self.method,
            "n_bins": self.n_bins,
            "calibration_error": self.calibration_error,
        }

        if self.reliability_data is not None:
            summary.update(
                {
                    "n_reliability_bins": len(self.reliability_data["bin_centers"]),
                    "mean_bin_accuracy": float(
                        np.mean(self.reliability_data["bin_accuracies"])
                    ),
                    "mean_bin_confidence": float(
                        np.mean(self.reliability_data["bin_confidences"])
                    ),
                    "total_samples": int(np.sum(self.reliability_data["bin_counts"])),
                }
            )

        return summary

    def save_calibrator(self, path: str):
        """Save calibrator to file."""
        import pickle

        calibrator_data = {
            "n_bins": self.n_bins,
            "method": self.method,
            "calibrator": self.calibrator,
            "is_fitted": self.is_fitted,
            "calibration_error": self.calibration_error,
            "reliability_data": self.reliability_data,
        }

        with open(path, "wb") as f:
            pickle.dump(calibrator_data, f)

    @classmethod
    def load_calibrator(cls, path: str) -> "UncertaintyCalibrator":
        """Load calibrator from file."""
        import pickle

        with open(path, "rb") as f:
            calibrator_data = pickle.load(f)

        calibrator = cls(
            n_bins=calibrator_data["n_bins"], method=calibrator_data["method"]
        )

        calibrator.calibrator = calibrator_data["calibrator"]
        calibrator.is_fitted = calibrator_data["is_fitted"]
        calibrator.calibration_error = calibrator_data["calibration_error"]
        calibrator.reliability_data = calibrator_data["reliability_data"]

        return calibrator


class CalibrationEvaluator:
    """
    Utility class for evaluating uncertainty calibration across multiple models and tasks.
    """

    def __init__(self):
        self.results = []

    def evaluate_model_calibration(
        self,
        model,
        test_tasks: List[Dict],
        calibrator: Optional[UncertaintyCalibrator] = None,
        n_mc_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Evaluate calibration quality across multiple test tasks.

        Args:
            model: BayesianMetaPINN model
            test_tasks: List of test tasks
            calibrator: Optional pre-fitted calibrator
            n_mc_samples: Number of Monte Carlo samples

        Returns:
            Dict[str, Any]: Calibration evaluation results
        """
        all_uncertainties = []
        all_errors = []
        all_physics_residuals = []

        model.eval()

        for task in test_tasks:
            query_coords = task["query_coords"]
            query_data = task["query_data"]
            task_info = task["task_info"]

            # Get model predictions with uncertainty
            with torch.no_grad():
                mean_pred, uncertainty = model.forward_with_uncertainty(
                    query_coords, n_samples=n_mc_samples
                )

                # Compute prediction errors
                errors = torch.abs(mean_pred - query_data)

                # Compute physics residuals
                physics_losses = model.physics_loss(
                    query_coords, task_info, create_graph=False
                )
                physics_residual = physics_losses["total_pde"]

                all_uncertainties.append(uncertainty.flatten())
                all_errors.append(errors.flatten())
                all_physics_residuals.append(
                    physics_residual.expand_as(errors).flatten()
                )

        # Concatenate all results
        all_uncertainties = torch.cat(all_uncertainties)
        all_errors = torch.cat(all_errors)
        all_physics_residuals = torch.cat(all_physics_residuals)

        # Evaluate calibration
        if calibrator is None:
            calibrator = UncertaintyCalibrator()

        metrics = calibrator.evaluate_calibration(all_uncertainties, all_errors)

        # Add physics-informed metrics
        physics_uncertainty = torch.sqrt(torch.clamp(all_physics_residuals, min=1e-8))
        physics_metrics = calibrator.evaluate_calibration(
            physics_uncertainty, all_errors
        )

        results = {
            "epistemic_calibration": metrics,
            "physics_calibration": physics_metrics,
            "n_tasks": len(test_tasks),
            "n_samples": len(all_uncertainties),
            "mean_uncertainty": float(torch.mean(all_uncertainties)),
            "mean_error": float(torch.mean(all_errors)),
            "mean_physics_residual": float(torch.mean(all_physics_residuals)),
        }

        self.results.append(results)
        return results

    def compare_calibration_methods(
        self,
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
        methods: List[str] = ["isotonic"],
    ) -> Dict[str, Dict]:
        """
        Compare different calibration methods.

        Args:
            uncertainties: Uncertainty estimates
            errors: Actual errors
            methods: List of calibration methods to compare

        Returns:
            Dict[str, Dict]: Comparison results
        """
        results = {}

        for method in methods:
            calibrator = UncertaintyCalibrator(method=method)

            # Split data for training and testing calibration
            n_samples = len(uncertainties)
            n_train = n_samples // 2

            train_unc = uncertainties[:n_train]
            train_err = errors[:n_train]
            test_unc = uncertainties[n_train:]
            test_err = errors[n_train:]

            # Fit calibrator
            calibrator.fit(train_unc, train_err)

            # Evaluate on test set
            calibrated_unc = calibrator.calibrate(test_unc)
            metrics = calibrator.evaluate_calibration(calibrated_unc, test_err)

            results[method] = {"metrics": metrics, "calibrator": calibrator}

        return results
