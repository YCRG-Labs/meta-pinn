#!/usr/bin/env python3
"""
Integrated Physics Discovery Demo

This script demonstrates the integrated physics discovery system that combines
causal discovery, symbolic regression, and meta-learning validation to discover
and validate physics relationships in fluid dynamics data.

This implements task 7.3: "Integrate physics discovery with meta-learning validation"
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from ml_research_pipeline.physics_discovery.integrated_discovery import IntegratedPhysicsDiscovery


def generate_synthetic_physics_data(n_samples: int = 200, noise_level: float = 0.05) -> dict:
    """
    Generate synthetic fluid dynamics data with known physics relationships.
    
    Args:
        n_samples: Number of data samples to generate
        noise_level: Amount of noise to add to the data
    
    Returns:
        Dictionary containing synthetic physics data
    """
    np.random.seed(42)
    
    # Generate input variables
    reynolds_number = np.random.uniform(100, 2000, n_samples)
    temperature = np.random.uniform(280, 350, n_samples)
    pressure = np.random.uniform(1e5, 3e5, n_samples)
    
    # Create realistic physics relationships
    # Viscosity depends on Reynolds number (inverse), temperature (Arrhenius), and pressure
    viscosity = (
        (800 / reynolds_number) *  # Inverse Reynolds relationship
        np.exp(-0.015 * (temperature - 300)) *  # Arrhenius temperature dependence
        (pressure / 1e5) ** 0.05 +  # Weak pressure dependence
        np.random.normal(0, noise_level, n_samples)  # Noise
    )
    
    # Velocity related to Reynolds number and viscosity
    velocity_x = np.sqrt(reynolds_number * viscosity * 0.001) + np.random.normal(0, 0.1, n_samples)
    velocity_y = 0.3 * velocity_x + np.random.normal(0, 0.05, n_samples)
    
    # Shear rate related to velocity gradients
    shear_rate = np.abs(velocity_x - velocity_y) * 10 + np.random.normal(0, 0.5, n_samples)
    
    return {
        'reynolds_number': reynolds_number,
        'temperature': temperature,
        'pressure': pressure,
        'velocity_x': velocity_x,
        'velocity_y': velocity_y,
        'shear_rate': shear_rate,
        'viscosity': viscosity
    }


def demonstrate_integrated_discovery():
    """Demonstrate the integrated physics discovery pipeline."""
    
    print("🔬 Integrated Physics Discovery Demonstration")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic fluid dynamics data...")
    physics_data = generate_synthetic_physics_data(n_samples=150, noise_level=0.03)
    
    print(f"   Generated {len(physics_data)} variables with {len(physics_data['viscosity'])} samples each")
    print(f"   Variables: {list(physics_data.keys())}")
    
    # Step 2: Initialize integrated discovery system
    print("\n2. Initializing integrated physics discovery system...")
    
    variables = list(physics_data.keys())
    
    # Configure the discovery system
    causal_config = {
        'significance_threshold': 0.05,
        'min_mutual_info': 0.1
    }
    
    symbolic_config = {
        'population_size': 50,
        'max_generations': 20,
        'max_expression_depth': 4,
        'complexity_penalty': 0.02
    }
    
    validation_config = {
        'min_validation_score': 0.6,
        'min_improvement_threshold': 0.05,
        'confidence_threshold': 0.7
    }
    
    discovery_system = IntegratedPhysicsDiscovery(
        variables=variables,
        causal_config=causal_config,
        symbolic_config=symbolic_config,
        validation_config=validation_config,
        random_state=42
    )
    
    print("   ✓ Discovery system initialized")
    
    # Step 3: Perform integrated physics discovery
    print("\n3. Performing integrated physics discovery...")
    print("   This combines causal discovery, symbolic regression, and meta-learning validation")
    
    discovery_result = discovery_system.discover_physics_relationships(
        flow_data=physics_data,
        target_variable='viscosity',
        meta_learning_baseline=0.65  # Simulated baseline performance
    )
    
    print("   ✓ Physics discovery completed")
    
    # Step 4: Display results
    print("\n4. Discovery Results Summary")
    print("-" * 40)
    
    hypothesis = discovery_result.hypothesis
    
    print(f"   Validation Score: {hypothesis.validation_score:.3f}")
    print(f"   Confidence Score: {hypothesis.confidence_score:.3f}")
    print(f"   Meta-Learning Improvement: {hypothesis.meta_learning_improvement:.3f}")
    print(f"   Causal Relationships Found: {len(hypothesis.causal_relationships)}")
    print(f"   Symbolic Expressions Found: {len(hypothesis.symbolic_expressions)}")
    
    # Step 5: Display causal relationships
    if hypothesis.causal_relationships:
        print("\n5. Discovered Causal Relationships")
        print("-" * 40)
        
        for i, rel in enumerate(hypothesis.causal_relationships[:5], 1):
            print(f"   {i}. {rel.source} → {rel.target}")
            print(f"      Strength: {rel.strength:.3f}")
            print(f"      P-value: {rel.p_value:.2e}")
            print()
    
    # Step 6: Display symbolic expressions
    if hypothesis.symbolic_expressions:
        print("\n6. Discovered Mathematical Relationships")
        print("-" * 40)
        
        for i, expr in enumerate(hypothesis.symbolic_expressions[:3], 1):
            print(f"   Expression {i}:")
            print(f"      Mathematical form: {expr.expression}")
            print(f"      R² score: {expr.r2_score:.3f}")
            print(f"      Complexity: {expr.complexity}")
            print()
    
    # Step 7: Display natural language hypothesis
    print("\n7. Natural Language Hypothesis")
    print("-" * 40)
    print(hypothesis.natural_language_description)
    
    # Step 8: Validate with meta-learning
    print("\n8. Meta-Learning Validation")
    print("-" * 40)
    
    # Create mock validation tasks
    validation_tasks = [
        {'task_id': i, 'complexity': np.random.uniform(0.3, 0.9)}
        for i in range(20)
    ]
    
    baseline_config = {'adaptation_steps': 10, 'learning_rate': 0.01}
    physics_config = {'adaptation_steps': 8, 'learning_rate': 0.01, 'physics_weight': 1.0}
    
    # Mock MetaPINN class for validation
    class MockMetaPINN:
        pass
    
    validation_metrics = discovery_system.validate_discovered_physics_with_meta_learning(
        hypothesis,
        MockMetaPINN,
        validation_tasks,
        baseline_config,
        physics_config
    )
    
    print(f"   Baseline Accuracy: {validation_metrics['baseline_accuracy']:.3f}")
    print(f"   Physics-Informed Accuracy: {validation_metrics['physics_informed_accuracy']:.3f}")
    print(f"   Accuracy Improvement: {validation_metrics['accuracy_improvement']:.3f}")
    print(f"   Adaptation Speedup: {validation_metrics['adaptation_speedup']:.2f}x")
    print(f"   Overall Validation Score: {validation_metrics['overall_validation_score']:.3f}")
    print(f"   Meets Validation Threshold: {validation_metrics['meets_validation_threshold']}")
    
    # Step 9: Export results
    print("\n9. Exporting Results")
    print("-" * 40)
    
    results_dir = Path("results/physics_discovery_demo")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = discovery_system.export_discovery_results(
        save_dir=results_dir,
        include_plots=True
    )
    
    print("   Exported files:")
    for file_type, file_path in saved_files.items():
        print(f"     {file_type}: {file_path}")
    
    # Step 10: Summary and recommendations
    print("\n10. Summary and Recommendations")
    print("-" * 40)
    
    if hypothesis.validation_score >= 0.8:
        print("   🎉 EXCELLENT: High-confidence physics relationships discovered!")
        print("   Recommendation: Integrate discovered physics into meta-learning models")
    elif hypothesis.validation_score >= 0.6:
        print("   ✅ GOOD: Moderate-confidence physics relationships found")
        print("   Recommendation: Consider additional validation before integration")
    elif hypothesis.validation_score >= 0.4:
        print("   ⚠️  FAIR: Weak physics relationships detected")
        print("   Recommendation: Collect more data or refine discovery methods")
    else:
        print("   ❌ POOR: Limited physics relationships found")
        print("   Recommendation: Check data quality and discovery parameters")
    
    print(f"\n   Final validation score: {hypothesis.validation_score:.3f}")
    print(f"   Confidence level: {discovery_system._assess_confidence_level(hypothesis.validation_score)}")
    
    return discovery_result


def plot_discovery_results(discovery_result, save_path: str = None):
    """
    Create visualization plots for the discovery results.
    
    Args:
        discovery_result: Result from integrated physics discovery
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Integrated Physics Discovery Results', fontsize=16, fontweight='bold')
    
    hypothesis = discovery_result.hypothesis
    
    # Plot 1: Causal relationship strengths
    ax1 = axes[0, 0]
    if hypothesis.causal_relationships:
        relationships = hypothesis.causal_relationships[:8]  # Top 8
        sources = [r.source for r in relationships]
        strengths = [r.strength for r in relationships]
        
        bars = ax1.barh(sources, strengths, alpha=0.7)
        
        # Color bars by strength
        for bar, strength in zip(bars, strengths):
            if strength > 0.5:
                bar.set_color('green')
            elif strength > 0.3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax1.set_xlabel('Causal Strength')
        ax1.set_title('Discovered Causal Relationships')
        ax1.set_xlim(0, 1)
    else:
        ax1.text(0.5, 0.5, 'No causal relationships found', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Causal Relationships')
    
    # Plot 2: Validation metrics
    ax2 = axes[0, 1]
    metrics = {
        'Validation Score': hypothesis.validation_score,
        'Confidence Score': hypothesis.confidence_score,
        'Meta-Learning\nImprovement': hypothesis.meta_learning_improvement * 5  # Scale for visibility
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax2.bar(metric_names, metric_values, alpha=0.7)
    
    # Color bars based on values
    for bar, value in zip(bars, metric_values):
        if value > 0.7:
            bar.set_color('green')
        elif value > 0.4:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.set_ylim(0, 1)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 3: Symbolic expression quality
    ax3 = axes[1, 0]
    if hypothesis.symbolic_expressions:
        expr_data = []
        for i, expr in enumerate(hypothesis.symbolic_expressions[:5]):
            expr_data.append({
                'Expression': f'Expr {i+1}',
                'R² Score': expr.r2_score,
                'Complexity': expr.complexity / 50  # Normalize for plotting
            })
        
        if expr_data:
            expressions = [d['Expression'] for d in expr_data]
            r2_scores = [d['R² Score'] for d in expr_data]
            complexities = [d['Complexity'] for d in expr_data]
            
            x = np.arange(len(expressions))
            width = 0.35
            
            ax3.bar(x - width/2, r2_scores, width, label='R² Score', alpha=0.7)
            ax3.bar(x + width/2, complexities, width, label='Complexity (normalized)', alpha=0.7)
            
            ax3.set_xlabel('Symbolic Expressions')
            ax3.set_ylabel('Score')
            ax3.set_title('Symbolic Expression Quality')
            ax3.set_xticks(x)
            ax3.set_xticklabels(expressions)
            ax3.legend()
            ax3.set_ylim(0, 1)
    else:
        ax3.text(0.5, 0.5, 'No symbolic expressions found', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Symbolic Expressions')
    
    # Plot 4: Overall assessment
    ax4 = axes[1, 1]
    
    # Create a simple assessment visualization
    assessment_data = {
        'Causal Discovery': len(hypothesis.causal_relationships) / 10,  # Normalize
        'Symbolic Regression': len(hypothesis.symbolic_expressions) / 5,  # Normalize
        'Physics Consistency': hypothesis.validation_score,
        'Meta-Learning Potential': hypothesis.meta_learning_improvement * 10  # Scale
    }
    
    categories = list(assessment_data.keys())
    values = [min(1.0, max(0.0, v)) for v in assessment_data.values()]  # Clamp to [0,1]
    
    # Create radar-like plot using bar chart
    bars = ax4.bar(categories, values, alpha=0.7)
    
    for bar, value in zip(bars, values):
        if value > 0.7:
            bar.set_color('green')
        elif value > 0.4:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax4.set_ylabel('Assessment Score')
    ax4.set_title('Overall Discovery Assessment')
    ax4.set_ylim(0, 1)
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Run the demonstration
    print("Starting Integrated Physics Discovery Demonstration...")
    
    try:
        result = demonstrate_integrated_discovery()
        
        # Create visualization
        print("\n11. Creating Visualization")
        print("-" * 40)
        
        plot_save_path = "results/physics_discovery_demo/discovery_visualization.png"
        plot_discovery_results(result, save_path=plot_save_path)
        
        print("\n🎉 Demonstration completed successfully!")
        print("\nThis demonstration showcased:")
        print("  ✓ Causal discovery for identifying variable relationships")
        print("  ✓ Symbolic regression for mathematical expression discovery")
        print("  ✓ Meta-learning validation for physics assessment")
        print("  ✓ Natural language hypothesis generation")
        print("  ✓ Comprehensive validation pipeline")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()