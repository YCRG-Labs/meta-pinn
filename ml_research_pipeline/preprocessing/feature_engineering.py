"""
Physics-Informed Feature Engineering Module

This module implements physics-informed feature generation for fluid dynamics and other physics domains:
- Reynolds number derivatives and related dimensionless numbers
- Vorticity and strain rate calculations
- Dimensional analysis validation
- Feature importance scoring based on physics principles
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.ndimage as ndimage


class PhysicsFeatureType(Enum):
    """Types of physics-informed features."""

    REYNOLDS_NUMBER = "reynolds_number"
    VORTICITY = "vorticity"
    STRAIN_RATE = "strain_rate"
    VELOCITY_GRADIENT = "velocity_gradient"
    PRESSURE_GRADIENT = "pressure_gradient"
    KINETIC_ENERGY = "kinetic_energy"
    ENSTROPHY = "enstrophy"
    Q_CRITERION = "q_criterion"
    LAMBDA2_CRITERION = "lambda2_criterion"
    DIMENSIONLESS_GROUPS = "dimensionless_groups"


@dataclass
class PhysicsFeature:
    """Represents a physics-informed feature."""

    name: str
    data: np.ndarray
    feature_type: PhysicsFeatureType
    dimensions: Dict[str, float]  # Physical dimensions (L, T, M, etc.)
    importance_score: float
    physics_principle: str
    validation_passed: bool
    metadata: Dict[str, Any]


@dataclass
class FeatureEngineeringResult:
    """Result of physics feature engineering."""

    original_features: Dict[str, np.ndarray]
    generated_features: Dict[str, PhysicsFeature]
    feature_importance_ranking: List[Tuple[str, float]]
    dimensional_analysis: Dict[str, Dict[str, float]]
    validation_summary: Dict[str, bool]
    total_features_generated: int


class PhysicsFeatureEngineer:
    """
    Physics-informed feature engineering for fluid dynamics and physics data.

    Implements Requirement 2.2: "WHEN feature engineering is performed THEN the system
    SHALL generate physics-informed features that capture relevant domain knowledge"
    """

    def __init__(
        self,
        domain: str = "fluid_dynamics",
        enable_dimensional_validation: bool = True,
        importance_threshold: float = 0.1,
    ):
        """
        Initialize physics feature engineer.

        Args:
            domain: Physics domain (fluid_dynamics, heat_transfer, etc.)
            enable_dimensional_validation: Whether to validate dimensional consistency
            importance_threshold: Minimum importance score for feature inclusion
        """
        self.domain = domain
        self.enable_dimensional_validation = enable_dimensional_validation
        self.importance_threshold = importance_threshold

        # Physical constants and properties
        self.physical_constants = {
            "fluid_dynamics": {
                "kinematic_viscosity": 1e-6,  # m²/s for water at 20°C
                "density": 1000.0,  # kg/m³ for water
                "dynamic_viscosity": 1e-3,  # Pa·s for water
            }
        }

        # Dimensional analysis base units [L, T, M, Θ] (Length, Time, Mass, Temperature)
        self.base_dimensions = {
            "velocity": {"L": 1, "T": -1, "M": 0, "Θ": 0},
            "pressure": {"L": -1, "T": -2, "M": 1, "Θ": 0},
            "density": {"L": -3, "T": 0, "M": 1, "Θ": 0},
            "viscosity": {"L": -1, "T": -1, "M": 1, "Θ": 0},
            "temperature": {"L": 0, "T": 0, "M": 0, "Θ": 1},
            "length": {"L": 1, "T": 0, "M": 0, "Θ": 0},
            "time": {"L": 0, "T": 1, "M": 0, "Θ": 0},
            "acceleration": {"L": 1, "T": -2, "M": 0, "Θ": 0},
        }

        # Feature generation history
        self.generation_history = []

    def generate_physics_features(
        self,
        data: Dict[str, np.ndarray],
        spatial_coordinates: Optional[Dict[str, np.ndarray]] = None,
        physical_properties: Optional[Dict[str, float]] = None,
    ) -> FeatureEngineeringResult:
        """
        Generate physics-informed features from input data.

        Args:
            data: Dictionary of input data arrays (velocity, pressure, etc.)
            spatial_coordinates: Spatial coordinate arrays (x, y, z)
            physical_properties: Physical properties (viscosity, density, etc.)

        Returns:
            Complete feature engineering result
        """
        print("Generating physics-informed features...")

        # Initialize result containers
        generated_features = {}
        dimensional_analysis = {}
        validation_summary = {}

        # Update physical properties if provided
        if physical_properties:
            if self.domain in self.physical_constants:
                self.physical_constants[self.domain].update(physical_properties)

        # Generate different types of physics features
        feature_generators = [
            self._generate_reynolds_features,
            self._generate_vorticity_features,
            self._generate_strain_rate_features,
            self._generate_kinetic_energy_features,
            self._generate_pressure_gradient_features,
            self._generate_dimensionless_groups,
        ]

        for generator in feature_generators:
            try:
                new_features = generator(data, spatial_coordinates)
                generated_features.update(new_features)
            except Exception as e:
                warnings.warn(
                    f"Feature generator {generator.__name__} failed: {str(e)}"
                )
                continue

        # Perform dimensional analysis validation
        if self.enable_dimensional_validation:
            dimensional_analysis = self._perform_dimensional_analysis(
                generated_features
            )
            validation_summary = self._validate_dimensional_consistency(
                dimensional_analysis
            )

        # Calculate feature importance scores
        importance_ranking = self._calculate_feature_importance(
            generated_features, data
        )

        # Filter features by importance threshold
        filtered_features = {
            name: feature
            for name, feature in generated_features.items()
            if feature.importance_score >= self.importance_threshold
        }

        # Create result
        result = FeatureEngineeringResult(
            original_features=data.copy(),
            generated_features=filtered_features,
            feature_importance_ranking=importance_ranking,
            dimensional_analysis=dimensional_analysis,
            validation_summary=validation_summary,
            total_features_generated=len(generated_features),
        )

        # Store in history
        self.generation_history.append(result)

        print(
            f"Generated {len(filtered_features)} physics features (filtered from {len(generated_features)})"
        )
        return result

    def _generate_reynolds_features(
        self,
        data: Dict[str, np.ndarray],
        spatial_coords: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, PhysicsFeature]:
        """Generate Reynolds number and related dimensionless features."""
        features = {}

        # Check if we have velocity data
        velocity_components = []
        for comp in ["u", "v", "w", "velocity_x", "velocity_y", "velocity_z"]:
            if comp in data:
                velocity_components.append(data[comp])

        if not velocity_components:
            return features

        # Calculate velocity magnitude
        if len(velocity_components) == 1:
            velocity_magnitude = np.abs(velocity_components[0])
        else:
            velocity_magnitude = np.sqrt(sum(v**2 for v in velocity_components))

        # Get physical properties
        props = self.physical_constants.get(self.domain, {})
        nu = props.get("kinematic_viscosity", 1e-6)

        # Estimate characteristic length scale
        if spatial_coords and "x" in spatial_coords:
            L_char = np.ptp(spatial_coords["x"])  # Range of x-coordinates
        else:
            # Use data array size as proxy for length scale
            L_char = np.sqrt(velocity_magnitude.size) * 0.01  # Assume 1cm grid spacing

        # Reynolds number
        reynolds_number = velocity_magnitude * L_char / nu

        features["reynolds_number"] = PhysicsFeature(
            name="reynolds_number",
            data=reynolds_number,
            feature_type=PhysicsFeatureType.REYNOLDS_NUMBER,
            dimensions={"L": 0, "T": 0, "M": 0, "Θ": 0},  # Dimensionless
            importance_score=0.9,  # High importance for fluid dynamics
            physics_principle="Reynolds number characterizes flow regime (laminar vs turbulent)",
            validation_passed=True,
            metadata={"characteristic_length": L_char, "kinematic_viscosity": nu},
        )

        # Local Reynolds number (using local velocity gradients)
        if len(velocity_components) >= 2 and spatial_coords:
            try:
                # Calculate local velocity gradients
                dudx = np.gradient(velocity_components[0], axis=-1)
                local_strain_rate = np.abs(dudx)
                local_reynolds = local_strain_rate * L_char**2 / nu

                features["local_reynolds"] = PhysicsFeature(
                    name="local_reynolds",
                    data=local_reynolds,
                    feature_type=PhysicsFeatureType.REYNOLDS_NUMBER,
                    dimensions={"L": 0, "T": 0, "M": 0, "Θ": 0},
                    importance_score=0.7,
                    physics_principle="Local Reynolds number based on strain rate",
                    validation_passed=True,
                    metadata={"based_on": "velocity_gradient"},
                )
            except Exception:
                pass

        return features

    def _generate_vorticity_features(
        self,
        data: Dict[str, np.ndarray],
        spatial_coords: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, PhysicsFeature]:
        """Generate vorticity-related features."""
        features = {}

        # Need at least 2D velocity field for vorticity
        u = data.get("u")
        if u is None:
            u = data.get("velocity_x")
        v = data.get("v")
        if v is None:
            v = data.get("velocity_y")

        if u is None or v is None:
            return features

        try:
            # Calculate vorticity (ω = ∂v/∂x - ∂u/∂y)
            if u.ndim >= 2 and v.ndim >= 2:
                dvdx = np.gradient(v, axis=1)  # ∂v/∂x
                dudy = np.gradient(u, axis=0)  # ∂u/∂y
                vorticity = dvdx - dudy

                features["vorticity"] = PhysicsFeature(
                    name="vorticity",
                    data=vorticity,
                    feature_type=PhysicsFeatureType.VORTICITY,
                    dimensions={"L": 0, "T": -1, "M": 0, "Θ": 0},  # 1/time
                    importance_score=0.8,
                    physics_principle="Vorticity measures local rotation of fluid elements",
                    validation_passed=True,
                    metadata={"calculation": "dvdx - dudy"},
                )

                # Enstrophy (vorticity squared)
                enstrophy = 0.5 * vorticity**2

                features["enstrophy"] = PhysicsFeature(
                    name="enstrophy",
                    data=enstrophy,
                    feature_type=PhysicsFeatureType.ENSTROPHY,
                    dimensions={"L": 0, "T": -2, "M": 0, "Θ": 0},  # 1/time²
                    importance_score=0.6,
                    physics_principle="Enstrophy measures vorticity intensity",
                    validation_passed=True,
                    metadata={"calculation": "0.5 * vorticity^2"},
                )

        except Exception as e:
            warnings.warn(f"Vorticity calculation failed: {str(e)}")

        return features

    def _generate_strain_rate_features(
        self,
        data: Dict[str, np.ndarray],
        spatial_coords: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, PhysicsFeature]:
        """Generate strain rate tensor components."""
        features = {}

        u = data.get("u")
        if u is None:
            u = data.get("velocity_x")
        v = data.get("v")
        if v is None:
            v = data.get("velocity_y")

        if u is None or v is None:
            return features

        try:
            if u.ndim >= 2 and v.ndim >= 2:
                # Strain rate tensor components
                # S11 = ∂u/∂x
                dudx = np.gradient(u, axis=1)
                # S22 = ∂v/∂y
                dvdy = np.gradient(v, axis=0)
                # S12 = S21 = 0.5 * (∂u/∂y + ∂v/∂x)
                dudy = np.gradient(u, axis=0)
                dvdx = np.gradient(v, axis=1)
                s12 = 0.5 * (dudy + dvdx)

                # Strain rate magnitude
                strain_rate_magnitude = np.sqrt(2 * (dudx**2 + dvdy**2 + 2 * s12**2))

                features["strain_rate_magnitude"] = PhysicsFeature(
                    name="strain_rate_magnitude",
                    data=strain_rate_magnitude,
                    feature_type=PhysicsFeatureType.STRAIN_RATE,
                    dimensions={"L": 0, "T": -1, "M": 0, "Θ": 0},  # 1/time
                    importance_score=0.7,
                    physics_principle="Strain rate characterizes fluid deformation",
                    validation_passed=True,
                    metadata={"components": ["S11", "S22", "S12"]},
                )

                # Shear strain rate (S12 component)
                features["shear_strain_rate"] = PhysicsFeature(
                    name="shear_strain_rate",
                    data=s12,
                    feature_type=PhysicsFeatureType.STRAIN_RATE,
                    dimensions={"L": 0, "T": -1, "M": 0, "Θ": 0},
                    importance_score=0.6,
                    physics_principle="Shear strain rate measures fluid shearing",
                    validation_passed=True,
                    metadata={"calculation": "0.5 * (dudy + dvdx)"},
                )

        except Exception as e:
            warnings.warn(f"Strain rate calculation failed: {str(e)}")

        return features

    def _generate_kinetic_energy_features(
        self,
        data: Dict[str, np.ndarray],
        spatial_coords: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, PhysicsFeature]:
        """Generate kinetic energy related features."""
        features = {}

        # Collect velocity components
        velocity_components = []
        for comp in ["u", "v", "w", "velocity_x", "velocity_y", "velocity_z"]:
            if comp in data:
                velocity_components.append(data[comp])

        if not velocity_components:
            return features

        try:
            # Kinetic energy per unit mass
            kinetic_energy = 0.5 * sum(v**2 for v in velocity_components)

            features["kinetic_energy"] = PhysicsFeature(
                name="kinetic_energy",
                data=kinetic_energy,
                feature_type=PhysicsFeatureType.KINETIC_ENERGY,
                dimensions={"L": 2, "T": -2, "M": 0, "Θ": 0},  # velocity²
                importance_score=0.7,
                physics_principle="Kinetic energy represents fluid motion intensity",
                validation_passed=True,
                metadata={"components": len(velocity_components)},
            )

            # Turbulent kinetic energy (fluctuation from mean)
            if kinetic_energy.size > 10:  # Need sufficient data points
                mean_ke = np.mean(kinetic_energy)
                turbulent_ke = kinetic_energy - mean_ke

                features["turbulent_kinetic_energy"] = PhysicsFeature(
                    name="turbulent_kinetic_energy",
                    data=turbulent_ke,
                    feature_type=PhysicsFeatureType.KINETIC_ENERGY,
                    dimensions={"L": 2, "T": -2, "M": 0, "Θ": 0},
                    importance_score=0.6,
                    physics_principle="Turbulent kinetic energy measures flow unsteadiness",
                    validation_passed=True,
                    metadata={"mean_subtracted": True},
                )

        except Exception as e:
            warnings.warn(f"Kinetic energy calculation failed: {str(e)}")

        return features

    def _generate_pressure_gradient_features(
        self,
        data: Dict[str, np.ndarray],
        spatial_coords: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, PhysicsFeature]:
        """Generate pressure gradient features."""
        features = {}

        pressure = data.get("p")
        if pressure is None:
            pressure = data.get("pressure")
        if pressure is None:
            return features

        try:
            if pressure.ndim >= 2:
                # Pressure gradients
                dpdx = np.gradient(pressure, axis=1)
                dpdy = np.gradient(pressure, axis=0)

                # Pressure gradient magnitude
                pressure_grad_magnitude = np.sqrt(dpdx**2 + dpdy**2)

                features["pressure_gradient_magnitude"] = PhysicsFeature(
                    name="pressure_gradient_magnitude",
                    data=pressure_grad_magnitude,
                    feature_type=PhysicsFeatureType.PRESSURE_GRADIENT,
                    dimensions={"L": -2, "T": -2, "M": 1, "Θ": 0},  # pressure/length
                    importance_score=0.6,
                    physics_principle="Pressure gradient drives fluid motion",
                    validation_passed=True,
                    metadata={"components": ["dpdx", "dpdy"]},
                )

        except Exception as e:
            warnings.warn(f"Pressure gradient calculation failed: {str(e)}")

        return features

    def _generate_dimensionless_groups(
        self,
        data: Dict[str, np.ndarray],
        spatial_coords: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, PhysicsFeature]:
        """Generate important dimensionless groups for the physics domain."""
        features = {}

        if self.domain == "fluid_dynamics":
            # Froude number (if gravity effects are relevant)
            velocity_components = []
            for comp in ["u", "v", "velocity_x", "velocity_y"]:
                if comp in data:
                    velocity_components.append(data[comp])

            if velocity_components:
                try:
                    velocity_magnitude = np.sqrt(sum(v**2 for v in velocity_components))

                    # Estimate characteristic length
                    if spatial_coords and "x" in spatial_coords:
                        L_char = np.ptp(spatial_coords["x"])
                    else:
                        L_char = np.sqrt(velocity_magnitude.size) * 0.01

                    g = 9.81  # gravitational acceleration
                    froude_number = velocity_magnitude / np.sqrt(g * L_char)

                    features["froude_number"] = PhysicsFeature(
                        name="froude_number",
                        data=froude_number,
                        feature_type=PhysicsFeatureType.DIMENSIONLESS_GROUPS,
                        dimensions={"L": 0, "T": 0, "M": 0, "Θ": 0},
                        importance_score=0.5,
                        physics_principle="Froude number characterizes gravity effects",
                        validation_passed=True,
                        metadata={"gravity": g, "characteristic_length": L_char},
                    )

                except Exception as e:
                    warnings.warn(f"Froude number calculation failed: {str(e)}")

        return features

    def _perform_dimensional_analysis(
        self, features: Dict[str, PhysicsFeature]
    ) -> Dict[str, Dict[str, float]]:
        """Perform dimensional analysis on generated features."""
        dimensional_analysis = {}

        for name, feature in features.items():
            dimensional_analysis[name] = feature.dimensions.copy()

        return dimensional_analysis

    def _validate_dimensional_consistency(
        self, dimensional_analysis: Dict[str, Dict[str, float]]
    ) -> Dict[str, bool]:
        """Validate dimensional consistency of features."""
        validation_summary = {}

        for name, dimensions in dimensional_analysis.items():
            # Check if dimensions are physically reasonable
            is_valid = True

            # Check for extreme dimension values
            for dim, power in dimensions.items():
                if abs(power) > 10:  # Unreasonably high dimension power
                    is_valid = False
                    break

            # Check for known invalid combinations
            if dimensions.get("M", 0) < 0 and dimensions.get("L", 0) > 0:
                # Negative mass with positive length is unusual
                pass  # Allow for now, might be valid in some contexts

            validation_summary[name] = is_valid

        return validation_summary

    def _calculate_feature_importance(
        self, features: Dict[str, PhysicsFeature], original_data: Dict[str, np.ndarray]
    ) -> List[Tuple[str, float]]:
        """Calculate feature importance scores based on physics principles."""
        importance_scores = []

        for name, feature in features.items():
            # Start with the feature's inherent importance score
            score = feature.importance_score

            # Adjust based on data characteristics
            if feature.data.size > 0:
                # Penalize features with too many zeros or constants
                non_zero_ratio = np.count_nonzero(feature.data) / feature.data.size
                if non_zero_ratio < 0.1:
                    score *= 0.5  # Reduce importance for mostly zero features

                # Penalize constant features
                if np.std(feature.data) < 1e-10:
                    score *= 0.1

                # Boost importance for features with reasonable dynamic range
                dynamic_range = np.ptp(feature.data) / (
                    np.mean(np.abs(feature.data)) + 1e-10
                )
                if 0.1 < dynamic_range < 10:
                    score *= 1.2

            # Boost importance for dimensionless features (often more fundamental)
            if all(abs(dim) < 1e-10 for dim in feature.dimensions.values()):
                score *= 1.1

            importance_scores.append((name, score))

        # Sort by importance score (descending)
        importance_scores.sort(key=lambda x: x[1], reverse=True)

        return importance_scores

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering operations."""
        if not self.generation_history:
            return {"message": "No feature engineering operations performed yet"}

        total_features = sum(
            result.total_features_generated for result in self.generation_history
        )
        total_filtered = sum(
            len(result.generated_features) for result in self.generation_history
        )

        # Collect feature types
        feature_types = {}
        for result in self.generation_history:
            for feature in result.generated_features.values():
                feature_type = feature.feature_type.value
                feature_types[feature_type] = feature_types.get(feature_type, 0) + 1

        return {
            "total_operations": len(self.generation_history),
            "total_features_generated": total_features,
            "total_features_after_filtering": total_filtered,
            "filtering_efficiency": total_filtered / max(total_features, 1),
            "feature_types_generated": feature_types,
            "domain": self.domain,
            "dimensional_validation_enabled": self.enable_dimensional_validation,
        }
