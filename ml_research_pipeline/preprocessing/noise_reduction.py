"""
Noise Reduction Engine for Physics Data Preprocessing

This module implements multiple denoising techniques for physics data including:
- Savitzky-Golay filtering for smooth data
- Gaussian filtering for general noise reduction
- Wavelet denoising for complex noise patterns
- Automatic method selection based on signal characteristics
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pywt
import scipy.signal as signal


class DenoiseMethod(Enum):
    """Available denoising methods."""

    SAVITZKY_GOLAY = "savitzky_golay"
    GAUSSIAN = "gaussian"
    WAVELET = "wavelet"
    AUTO = "auto"


@dataclass
class DenoiseResult:
    """Result of denoising operation."""

    denoised_data: np.ndarray
    method_used: str
    snr_original: float
    snr_denoised: float
    snr_improvement: float
    parameters_used: Dict[str, Any]
    quality_score: float


class NoiseReductionEngine:
    """
    Advanced noise reduction engine with multiple denoising techniques.

    Implements Requirement 2.1: "WHEN raw physics data is processed THEN the system
    SHALL implement noise reduction techniques that improve signal-to-noise ratio by at least 50%"
    """

    def __init__(
        self,
        default_method: DenoiseMethod = DenoiseMethod.AUTO,
        snr_improvement_threshold: float = 1.5,
        quality_threshold: float = 0.7,
    ):
        """
        Initialize noise reduction engine.

        Args:
            default_method: Default denoising method to use
            snr_improvement_threshold: Minimum SNR improvement factor (1.5 = 50% improvement)
            quality_threshold: Minimum quality score for accepting denoised result
        """
        self.default_method = default_method
        self.snr_improvement_threshold = snr_improvement_threshold
        self.quality_threshold = quality_threshold

        # Method-specific parameters
        self.savgol_params = {
            "window_length": None,  # Will be auto-determined
            "polyorder": 3,
            "mode": "nearest",
        }

        self.gaussian_params = {
            "sigma": None,  # Will be auto-determined
            "mode": "nearest",
            "truncate": 4.0,
        }

        self.wavelet_params = {"wavelet": "db4", "mode": "symmetric", "method": "soft"}

        # Performance tracking
        self.denoising_history = []

    def denoise_data(
        self,
        data: Union[np.ndarray, Dict[str, np.ndarray]],
        method: Optional[DenoiseMethod] = None,
        **kwargs,
    ) -> Union[DenoiseResult, Dict[str, DenoiseResult]]:
        """
        Denoise input data using specified or automatic method selection.

        Args:
            data: Input data array or dictionary of arrays
            method: Denoising method to use (None for default)
            **kwargs: Method-specific parameters

        Returns:
            DenoiseResult or dictionary of results for each data array
        """
        if method is None:
            method = self.default_method

        if isinstance(data, dict):
            # Process multiple data arrays
            results = {}
            for key, array in data.items():
                results[key] = self._denoise_single_array(array, method, **kwargs)
            return results
        else:
            # Process single array
            return self._denoise_single_array(data, method, **kwargs)

    def _denoise_single_array(
        self, data: np.ndarray, method: DenoiseMethod, **kwargs
    ) -> DenoiseResult:
        """Denoise a single data array."""
        if data.ndim > 2:
            raise ValueError("Only 1D and 2D arrays are supported")

        # Calculate original SNR
        snr_original = self.calculate_snr(data)

        # Apply denoising based on method
        if method == DenoiseMethod.AUTO:
            denoised_data, method_used, params = self._auto_select_method(
                data, **kwargs
            )
        elif method == DenoiseMethod.SAVITZKY_GOLAY:
            denoised_data, params = self._apply_savitzky_golay(data, **kwargs)
            method_used = "savitzky_golay"
        elif method == DenoiseMethod.GAUSSIAN:
            denoised_data, params = self._apply_gaussian_filter(data, **kwargs)
            method_used = "gaussian"
        elif method == DenoiseMethod.WAVELET:
            denoised_data, params = self._apply_wavelet_denoising(data, **kwargs)
            method_used = "wavelet"
        else:
            raise ValueError(f"Unknown denoising method: {method}")

        # Calculate denoised SNR
        snr_denoised = self.calculate_snr(denoised_data)
        snr_improvement = snr_denoised / max(snr_original, 1e-10)

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            data, denoised_data, snr_improvement
        )

        # Create result
        result = DenoiseResult(
            denoised_data=denoised_data,
            method_used=method_used,
            snr_original=snr_original,
            snr_denoised=snr_denoised,
            snr_improvement=snr_improvement,
            parameters_used=params,
            quality_score=quality_score,
        )

        # Store in history
        self.denoising_history.append(result)

        return result

    def _auto_select_method(
        self, data: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, str, Dict]:
        """Automatically select the best denoising method for the data."""
        # Test all methods and select the best one
        methods_to_test = [
            (DenoiseMethod.SAVITZKY_GOLAY, self._apply_savitzky_golay),
            (DenoiseMethod.GAUSSIAN, self._apply_gaussian_filter),
            (DenoiseMethod.WAVELET, self._apply_wavelet_denoising),
        ]

        best_result = None
        best_score = -1
        best_method = "gaussian"  # fallback
        best_params = {}

        original_snr = self.calculate_snr(data)

        for method_enum, method_func in methods_to_test:
            try:
                denoised_data, params = method_func(data, **kwargs)
                denoised_snr = self.calculate_snr(denoised_data)
                snr_improvement = denoised_snr / max(original_snr, 1e-10)

                quality_score = self._calculate_quality_score(
                    data, denoised_data, snr_improvement
                )

                if quality_score > best_score:
                    best_score = quality_score
                    best_result = denoised_data
                    best_method = method_enum.value
                    best_params = params

            except Exception as e:
                warnings.warn(f"Method {method_enum.value} failed: {str(e)}")
                continue

        if best_result is None:
            # Fallback to original data if all methods fail
            best_result = data.copy()
            best_method = "none"
            best_params = {}

        return best_result, best_method, best_params

    def _apply_savitzky_golay(
        self, data: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """Apply Savitzky-Golay filtering."""
        params = self.savgol_params.copy()
        params.update(kwargs)

        if data.ndim == 1:
            # 1D Savitzky-Golay
            if params["window_length"] is None:
                # Auto-determine window length (should be odd and < data length)
                window_length = min(max(5, len(data) // 10), len(data) - 1)
                if window_length % 2 == 0:
                    window_length -= 1
                params["window_length"] = max(3, window_length)

            # Ensure window length is valid
            if params["window_length"] >= len(data):
                params["window_length"] = len(data) - 1 if len(data) > 1 else 1
            if params["window_length"] % 2 == 0:
                params["window_length"] -= 1
            if params["window_length"] < 3:
                params["window_length"] = 3

            # Ensure polyorder is valid
            if params["polyorder"] >= params["window_length"]:
                params["polyorder"] = params["window_length"] - 1

            denoised = signal.savgol_filter(
                data, params["window_length"], params["polyorder"], mode=params["mode"]
            )
        else:
            # 2D Savitzky-Golay (apply to each row/column)
            denoised = np.zeros_like(data)
            for i in range(data.shape[0]):
                if params["window_length"] is None:
                    window_length = min(max(5, data.shape[1] // 10), data.shape[1] - 1)
                    if window_length % 2 == 0:
                        window_length -= 1
                    window_length = max(3, window_length)
                else:
                    window_length = params["window_length"]

                if window_length >= data.shape[1]:
                    window_length = data.shape[1] - 1 if data.shape[1] > 1 else 1
                if window_length % 2 == 0:
                    window_length -= 1
                if window_length < 3:
                    window_length = 3

                polyorder = min(params["polyorder"], window_length - 1)

                denoised[i, :] = signal.savgol_filter(
                    data[i, :], window_length, polyorder, mode=params["mode"]
                )

            params["window_length"] = window_length

        return denoised, params

    def _apply_gaussian_filter(
        self, data: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """Apply Gaussian filtering."""
        from scipy import ndimage

        params = self.gaussian_params.copy()
        params.update(kwargs)

        if params["sigma"] is None:
            # Auto-determine sigma based on data characteristics
            if data.ndim == 1:
                params["sigma"] = max(1.0, len(data) / 100)
            else:
                params["sigma"] = max(1.0, min(data.shape) / 50)

        denoised = ndimage.gaussian_filter(
            data,
            sigma=params["sigma"],
            mode=params["mode"],
            truncate=params["truncate"],
        )

        return denoised, params

    def _apply_wavelet_denoising(
        self, data: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """Apply wavelet denoising."""
        params = self.wavelet_params.copy()
        params.update(kwargs)

        if data.ndim == 1:
            # 1D wavelet denoising
            coeffs = pywt.wavedec(data, params["wavelet"], mode=params["mode"])

            # Estimate noise level using median absolute deviation
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745

            # Calculate threshold
            threshold = sigma * np.sqrt(2 * np.log(len(data)))

            # Apply soft thresholding
            coeffs_thresh = list(coeffs)
            coeffs_thresh[1:] = [
                pywt.threshold(detail, threshold, mode=params["method"])
                for detail in coeffs_thresh[1:]
            ]

            denoised = pywt.waverec(
                coeffs_thresh, params["wavelet"], mode=params["mode"]
            )

            # Ensure same length as original
            if len(denoised) != len(data):
                denoised = denoised[: len(data)]

        else:
            # 2D wavelet denoising
            coeffs = pywt.wavedec2(data, params["wavelet"], mode=params["mode"])

            # Estimate noise level
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(data.size))

            # Apply thresholding
            coeffs_thresh = list(coeffs)
            coeffs_thresh[1:] = [
                tuple(
                    [
                        pywt.threshold(detail, threshold, mode=params["method"])
                        for detail in coeff_tuple
                    ]
                )
                for coeff_tuple in coeffs_thresh[1:]
            ]

            denoised = pywt.waverec2(
                coeffs_thresh, params["wavelet"], mode=params["mode"]
            )

            # Ensure same shape as original
            if denoised.shape != data.shape:
                denoised = denoised[: data.shape[0], : data.shape[1]]

        params["threshold"] = threshold
        params["sigma_estimated"] = sigma

        return denoised, params

    def calculate_snr(self, data: np.ndarray) -> float:
        """
        Calculate signal-to-noise ratio of data.

        Args:
            data: Input data array

        Returns:
            SNR value in dB
        """
        if data.size == 0:
            return 0.0

        # Calculate signal power (variance of the signal)
        signal_power = np.var(data)

        if signal_power == 0:
            return float("inf")

        # Estimate noise power using high-frequency components
        if data.ndim == 1:
            if len(data) > 2:
                # Use second derivative as noise estimate
                noise_estimate = np.diff(data, n=2)
                noise_power = (
                    np.var(noise_estimate)
                    if len(noise_estimate) > 0
                    else signal_power * 0.1
                )
            else:
                noise_power = signal_power * 0.1
        else:
            # For 2D data, use Laplacian as noise estimate
            from scipy import ndimage

            laplacian = ndimage.laplace(data)
            noise_power = np.var(laplacian)

        if noise_power == 0:
            return float("inf")

        # SNR in dB
        snr_db = 10 * np.log10(signal_power / noise_power)
        return max(snr_db, 0.0)  # Ensure non-negative SNR

    def _calculate_quality_score(
        self, original: np.ndarray, denoised: np.ndarray, snr_improvement: float
    ) -> float:
        """Calculate overall quality score for denoising result."""
        # SNR improvement component (0-0.4)
        snr_score = min(0.4, max(0.0, (snr_improvement - 1.0) * 0.4))

        # Correlation with original (0-0.3)
        correlation = np.corrcoef(original.flatten(), denoised.flatten())[0, 1]
        correlation_score = max(0.0, correlation) * 0.3

        # Smoothness improvement (0-0.2)
        original_smoothness = self._calculate_smoothness(original)
        denoised_smoothness = self._calculate_smoothness(denoised)
        smoothness_improvement = (denoised_smoothness - original_smoothness) / max(
            original_smoothness, 1e-10
        )
        smoothness_score = min(0.2, max(0.0, smoothness_improvement * 0.2))

        # Edge preservation (0-0.1)
        edge_score = self._calculate_edge_preservation(original, denoised) * 0.1

        total_score = snr_score + correlation_score + smoothness_score + edge_score
        return min(1.0, max(0.0, total_score))

    def _calculate_smoothness(self, data: np.ndarray) -> float:
        """Calculate smoothness metric for data."""
        if data.ndim == 1:
            if len(data) > 1:
                gradients = np.diff(data)
                return 1.0 / (1.0 + np.std(gradients))
            else:
                return 1.0
        else:
            # For 2D data, use total variation
            grad_x = np.diff(data, axis=1)
            grad_y = np.diff(data, axis=0)
            total_variation = np.sum(np.abs(grad_x)) + np.sum(np.abs(grad_y))
            return 1.0 / (1.0 + total_variation / data.size)

    def _calculate_edge_preservation(
        self, original: np.ndarray, denoised: np.ndarray
    ) -> float:
        """Calculate how well edges are preserved."""
        if original.ndim == 1:
            # For 1D, use gradient comparison
            if len(original) > 1:
                orig_grad = np.abs(np.diff(original))
                denoised_grad = np.abs(np.diff(denoised))

                # Find significant edges in original
                edge_threshold = np.percentile(orig_grad, 75)
                edge_mask = orig_grad > edge_threshold

                if np.any(edge_mask):
                    edge_preservation = np.corrcoef(
                        orig_grad[edge_mask], denoised_grad[edge_mask]
                    )[0, 1]
                    return max(0.0, edge_preservation)
                else:
                    return 1.0
            else:
                return 1.0
        else:
            # For 2D, use Sobel edge detection
            from scipy import ndimage

            orig_edges = ndimage.sobel(original)
            denoised_edges = ndimage.sobel(denoised)

            # Compare edge magnitudes
            correlation = np.corrcoef(orig_edges.flatten(), denoised_edges.flatten())[
                0, 1
            ]
            return max(0.0, correlation)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of denoising performance across all operations."""
        if not self.denoising_history:
            return {"message": "No denoising operations performed yet"}

        snr_improvements = [result.snr_improvement for result in self.denoising_history]
        quality_scores = [result.quality_score for result in self.denoising_history]
        methods_used = [result.method_used for result in self.denoising_history]

        return {
            "total_operations": len(self.denoising_history),
            "average_snr_improvement": np.mean(snr_improvements),
            "median_snr_improvement": np.median(snr_improvements),
            "average_quality_score": np.mean(quality_scores),
            "methods_used": {
                method: methods_used.count(method) for method in set(methods_used)
            },
            "operations_meeting_threshold": sum(
                1
                for result in self.denoising_history
                if result.snr_improvement >= self.snr_improvement_threshold
            ),
            "success_rate": sum(
                1
                for result in self.denoising_history
                if result.snr_improvement >= self.snr_improvement_threshold
            )
            / len(self.denoising_history),
        }
