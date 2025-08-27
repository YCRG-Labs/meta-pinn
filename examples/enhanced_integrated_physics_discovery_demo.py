#!/usr/bin/env python3
"""
Enhanced Integrated Physics Discovery Demo

This demo showcases the enhanced integrated physics discovery system that combines
causal discovery, symbolic regression, and meta-learning validation with comprehensive
natural language interpretation capabilities.

This demonstrates the implementation of:
- Task 7.3: Integrate physics discovery with meta-learning validation
- Requirement 5.3: Validation scores > 0.8 using meta-learning performance
- Requirement 5.5: Natural language interpretations of discovered physics
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from ml_research_pipeline.physics_discovery.integrated_discovery import IntegratedPhysicsDiscovery


def generate_complex_physics_data(n_samples=200, noise_level=0.05, random_seed=42):
    """Generate complex physics data with multiple interacting variables."""
    np.random.seed(random_seed)
    
    # Generate base variables
    reynolds = np.random.uniform(50, 2000, n_samples)
    temperature = np.random.uniform(270, 350, n_samples)
    pressure = np.random.uniform(1e5, 5e5, n_samples)
    shear_rate = np.random.uniform(1, 100, n_samples)
    
    # Complex viscosity relationship with multiple physics effects
    # Temperature dependence (Arrhenius-like)
    temp_effect = np.exp(-0.02 * (temperature - 300))
    
    # Reynolds number dependence (turbulent effects)
    reynolds_effect = 1000 / (reynolds + 100)
    
    # Pressure dependence (compressibility)
    pressure_effect = (pressure / 1e5) ** 0.1
    
    # Shear rate dependence (non-Newtonian behavior)
    shear_effect = 1 / (1 + 0.01 * shear_rate)
    
    # Combined viscosity with interactions
    viscosity = (reynolds_effect * temp_effect * pressure_effect * shear_effect + 
                np.random.normal(0, noise_level, n_samples))
    
    # Derived variables
    velocity_x = np.sqrt(reynolds * viscosity * 0.001) + np.random.normal(0, 0.1, n_samples)
    velocity_y = 0.3 * velocity_x + np.random.normal(0, 0.05, n_samples)
    
    return {
        'reynolds_number': reynolds,
        'temperature': temperature,
        'pressure': pressure,
        'shear_rate': shear_rate,
        'velocity_x': velocity_x,
        'velocity_y': velocity_y,
        'viscosity': viscosity
    }


def demonstrate_enhanced_features(discovery_system, physics_data):
    """Demonstrate the enhanced features of the integrated discovery system."""
    print("\n🚀 ENHANCED FEATURES DEMONSTRATION")
    print("=" * 50)
    
    # Perform discovery with enhanced validation
    print("1. Enhanced Physics Discovery with Meta-Learning Validation...")
    discovery_result = discovery_system.discover_physics_relationships(
        physics_data,
        target_variable='viscosity',
        meta_learning_baseline=0.65
    )
    
    # Demonstrate enhanced natural language generation (Requirement 5.5)
    print("\n2. Enhanced Natural Language Interpretation (Requirement 5.5)...")
    hypothesis = discovery_result.hypothesis
    
    print(f"\n📊 Validation Score: {hypothesis.validation_score:.3f}")
    print(f"🎯 Confidence Score: {hypothesis.confidence_score:.3f}")
    print(f"📈 Meta-Learning Improvement: {hypothesis.meta_learning_improvement:.1%}")
    
    # Show enhanced natural language description
    print("\n📝 ENHANCED NATURAL LANGUAGE DESCRIPTION:")
    print("-" * 50)
    print(hypothesis.natural_language_description)
    
    # Demonstrate comprehensive physics report generation
    print("\n3. Comprehensive Physics Report Generation...")
    comprehensive_report = discovery_system.generate_comprehensive_physics_report(
        discovery_result,
        include_validation=True
    )
    
    # Save the comprehensive report
    results_dir = Path("results/enhanced_integrated_physics_discovery_demo")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "comprehensive_physics_report.txt", "w", encoding='utf-8') as f:
        f.write(comprehensive_report)
    
    print(f"✅ Comprehensive report saved to: {results_dir / 'comprehensive_physics_report.txt'}")
    
    # Demonstrate enhanced meta-learning validation (Requirement 5.3)
    print("\n4. Enhanced Meta-Learning Validation (Requirement 5.3)...")
    
    validation_tasks = [
        {'task_id': i, 'complexity': np.random.uniform(0.2, 0.9), 'domain': f'domain_{i%3}'}
        for i in range(25)
    ]
    
    baseline_config = {
        'adaptation_steps': 15,
        'learning_rate': 0.01,
        'meta_lr': 0.001
    }
    
    physics_config = {
        'adaptation_steps': 8,  # Faster with physics
        'learning_rate': 0.01,
        'meta_lr': 0.001,
        'physics_weight': 1.0,
        'use_discovered_constraints': True
    }
    
    class MockMetaPINN:
        def __init__(self, config):
            self.config = config
    
    validation_metrics = discovery_system.validate_discovered_physics_with_meta_learning(
        hypothesis,
        MockMetaPINN,
        validation_tasks,
        baseline_config,
        physics_config
    )
    
    print("\n📊 ENHANCED VALIDATION METRICS:")
    print("-" * 40)
    for metric, value in validation_metrics.items():
        if isinstance(value, (int, float)):
            if 'improvement' in metric or 'speedup' in metric:
                print(f"  📈 {metric}: {value:.3f}")
            elif 'accuracy' in metric or 'efficiency' in metric:
                print(f"  🎯 {metric}: {value:.3f}")
            elif 'score' in metric:
                print(f"  📊 {metric}: {value:.3f}")
            else:
                print(f"  📋 {metric}: {value:.3f}")
        elif isinstance(value, bool):
            emoji = "✅" if value else "❌"
            print(f"  {emoji} {metric}: {value}")
        else:
            print(f"  📝 {metric}: {value}")
    
    # Check if validation threshold is met (Requirement 5.3)
    meets_threshold = validation_metrics.get('meets_validation_threshold', False)
    overall_score = validation_metrics.get('overall_validation_score', 0.0)
    
    print(f"\n🎯 REQUIREMENT 5.3 VALIDATION:")
    print(f"   Target: Validation scores > 0.8")
    print(f"   Achieved: {overall_score:.3f}")
    print(f"   Status: {'✅ PASSED' if meets_threshold else '❌ NEEDS IMPROVEMENT'}")
    
    return discovery_result, validation_metrics


def main():
    """Run the enhanced integrated physics discovery demo."""
    print("🔬 ENHANCED INTEGRATED PHYSICS DISCOVERY DEMO")
    print("=" * 60)
    print("Demonstrating Requirements 5.3 and 5.5 implementation")
    print("=" * 60)
    
    # Generate complex physics data
    print("📊 Generating Complex Physics Dataset...")
    physics_data = generate_complex_physics_data(n_samples=300, noise_level=0.03)
    
    print(f"Dataset contains {len(physics_data)} variables with {len(physics_data['viscosity'])} samples each")
    for var, data in physics_data.items():
        print(f"  {var}: range [{data.min():.3f}, {data.max():.3f}]")
    
    # Initialize enhanced integrated discovery system
    print("\n🧠 Initializing Enhanced Integrated Physics Discovery System...")
    discovery_system = IntegratedPhysicsDiscovery(
        variables=list(physics_data.keys()),
        causal_config={
            'significance_threshold': 0.01,
            'min_mutual_info': 0.05
        },
        symbolic_config={
            'population_size': 50,
            'max_generations': 20,
            'complexity_penalty': 0.02
        },
        validation_config={
            'min_validation_score': 0.8,  # Requirement 5.3
            'min_improvement_threshold': 0.1,
            'confidence_threshold': 0.6
        },
        random_state=42
    )
    
    # Demonstrate enhanced features
    discovery_result, validation_metrics = demonstrate_enhanced_features(
        discovery_system, physics_data
    )
    
    # Export enhanced results
    print("\n💾 Exporting Enhanced Results...")
    results_dir = Path("results/enhanced_integrated_physics_discovery_demo")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = discovery_system.export_discovery_results(
        results_dir,
        include_plots=True
    )
    
    print("📁 Saved files:")
    for file_type, file_path in saved_files.items():
        print(f"  📄 {file_type}: {file_path}")
    
    # Display enhanced discovery summary
    print("\n📈 Enhanced Discovery Summary:")
    summary = discovery_system.get_discovery_summary()
    for key, value in summary.items():
        if key != 'discovery_timeline':
            print(f"  📊 {key}: {value}")
    
    # Final assessment
    print("\n" + "=" * 60)
    print("🎯 FINAL ASSESSMENT")
    print("=" * 60)
    
    hypothesis = discovery_result.hypothesis
    
    print(f"Overall Performance:")
    print(f"  🔬 Physics Discovery Quality: {hypothesis.validation_score:.3f}/1.0")
    print(f"  🎯 Confidence Level: {hypothesis.confidence_score:.3f}/1.0")
    print(f"  📈 Expected Meta-Learning Improvement: {hypothesis.meta_learning_improvement:.1%}")
    
    # Requirements compliance check
    print(f"\nRequirements Compliance:")
    
    # Requirement 5.3: validation scores > 0.8
    req_5_3_met = validation_metrics.get('meets_validation_threshold', False)
    print(f"  📋 Requirement 5.3 (Validation > 0.8): {'✅ MET' if req_5_3_met else '⚠️  PARTIAL'}")
    
    # Requirement 5.5: natural language interpretations
    has_nl_description = len(hypothesis.natural_language_description) > 200
    print(f"  📝 Requirement 5.5 (Natural Language): {'✅ MET' if has_nl_description else '❌ NOT MET'}")
    
    # Overall system readiness
    system_ready = req_5_3_met and has_nl_description
    print(f"\n🚀 System Readiness: {'✅ READY FOR PRODUCTION' if system_ready else '⚠️  NEEDS REFINEMENT'}")
    
    print(f"\n✅ Enhanced demo completed successfully!")
    print(f"📁 Check the results directory: {results_dir}")
    print(f"📊 Comprehensive report available at: {results_dir / 'comprehensive_physics_report.txt'}")


if __name__ == "__main__":
    main()