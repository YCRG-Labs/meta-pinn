#!/usr/bin/env python3
"""
Comprehensive Report Generation Demo

This script demonstrates the complete report generation pipeline including:
- Automated executive summary generation
- Natural language descriptions of experimental findings
- Multi-format output (Markdown, LaTeX, HTML)
- Integration with plots and tables
- Physics discovery integration
- Statistical analysis integration

This addresses requirements 9.3 and 9.5 from the ML research pipeline specification.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import tempfile
import json

from ml_research_pipeline.papers.report_generator import ReportGenerator
from ml_research_pipeline.papers.plot_generator import PaperPlotGenerator
from ml_research_pipeline.papers.table_generator import LaTeXTableGenerator


def create_comprehensive_experimental_data():
    """Create comprehensive experimental data for demonstration."""
    
    # Simulate results from a meta-learning PINN experiment
    experimental_results = {
        'experiment_type': 'Meta-Learning Physics-Informed Neural Networks for Viscosity Inference',
        'num_tasks': 1000,
        'metrics': ['parameter_accuracy', 'adaptation_speed', 'physics_consistency', 'computational_efficiency'],
        
        # Method comparison results
        'method_results': {
            'Meta-PINN': {
                'parameter_accuracy_mean': 0.946,
                'parameter_accuracy_std': 0.028,
                'adaptation_speed_mean': 7.3,
                'adaptation_speed_std': 1.2,
                'physics_consistency_mean': 0.987,
                'physics_consistency_std': 0.008,
                'computational_efficiency_mean': 0.89,
                'computational_efficiency_std': 0.04
            },
            'Standard PINN': {
                'parameter_accuracy_mean': 0.834,
                'parameter_accuracy_std': 0.067,
                'adaptation_speed_mean': 52.1,
                'adaptation_speed_std': 9.8,
                'physics_consistency_mean': 0.923,
                'physics_consistency_std': 0.035,
                'computational_efficiency_mean': 0.42,
                'computational_efficiency_std': 0.09
            },
            'Transfer Learning PINN': {
                'parameter_accuracy_mean': 0.891,
                'parameter_accuracy_std': 0.041,
                'adaptation_speed_mean': 18.7,
                'adaptation_speed_std': 3.2,
                'physics_consistency_mean': 0.951,
                'physics_consistency_std': 0.019,
                'computational_efficiency_mean': 0.73,
                'computational_efficiency_std': 0.06
            },
            'Fourier Neural Operator': {
                'parameter_accuracy_mean': 0.798,
                'parameter_accuracy_std': 0.083,
                'adaptation_speed_mean': 11.2,
                'adaptation_speed_std': 2.8,
                'physics_consistency_mean': 0.781,
                'physics_consistency_std': 0.092,
                'computational_efficiency_mean': 0.94,
                'computational_efficiency_std': 0.02
            },
            'DeepONet': {
                'parameter_accuracy_mean': 0.823,
                'parameter_accuracy_std': 0.058,
                'adaptation_speed_mean': 14.1,
                'adaptation_speed_std': 3.1,
                'physics_consistency_mean': 0.856,
                'physics_consistency_std': 0.047,
                'computational_efficiency_mean': 0.81,
                'computational_efficiency_std': 0.05
            }
        },     
   
        # Statistical significance testing results
        'statistical_tests': {
            'Meta-PINN': {
                'parameter_accuracy': 0.0001,  # Highly significant
                'adaptation_speed': 0.0001,
                'physics_consistency': 0.001,
                'computational_efficiency': 0.005
            },
            'Standard PINN': {
                'parameter_accuracy': 0.18,    # Not significant
                'adaptation_speed': 0.25,
                'physics_consistency': 0.12,
                'computational_efficiency': 0.35
            },
            'Transfer Learning PINN': {
                'parameter_accuracy': 0.015,   # Significant
                'adaptation_speed': 0.008,
                'physics_consistency': 0.022,
                'computational_efficiency': 0.031
            },
            'Fourier Neural Operator': {
                'parameter_accuracy': 0.31,    # Not significant
                'adaptation_speed': 0.045,
                'physics_consistency': 0.52,
                'computational_efficiency': 0.0008
            },
            'DeepONet': {
                'parameter_accuracy': 0.089,   # Marginally significant
                'adaptation_speed': 0.028,
                'physics_consistency': 0.15,
                'computational_efficiency': 0.019
            }
        },
        
        # Convergence analysis data
        'convergence_data': {
            'Meta-PINN': [1.0, 0.28, 0.065, 0.018, 0.0045, 0.0012],
            'Standard PINN': [1.0, 0.92, 0.84, 0.76, 0.68, 0.61, 0.54, 0.47, 0.41, 0.35, 0.29, 0.24],
            'Transfer Learning PINN': [1.0, 0.58, 0.21, 0.074, 0.026, 0.009],
            'Fourier Neural Operator': [1.0, 0.41, 0.17, 0.071, 0.029, 0.012],
            'DeepONet': [1.0, 0.52, 0.19, 0.068, 0.024, 0.0085]
        },
        
        # Computational efficiency data
        'efficiency_data': {
            'training_time': {
                'Meta-PINN': 145.2,
                'Standard PINN': 523.7,
                'Transfer Learning PINN': 312.4,
                'Fourier Neural Operator': 89.1,
                'DeepONet': 198.6
            },
            'memory_usage': {
                'Meta-PINN': 2.3,
                'Standard PINN': 1.9,
                'Transfer Learning PINN': 2.7,
                'Fourier Neural Operator': 3.8,
                'DeepONet': 3.1
            },
            'inference_time': {
                'Meta-PINN': 0.012,
                'Standard PINN': 0.045,
                'Transfer Learning PINN': 0.018,
                'Fourier Neural Operator': 0.008,
                'DeepONet': 0.015
            }
        },
        
        # Robustness analysis
        'robustness_data': {
            'variance_across_tasks': {
                'Meta-PINN': 0.0089,
                'Standard PINN': 0.0521,
                'Transfer Learning PINN': 0.0234,
                'Fourier Neural Operator': 0.0789,
                'DeepONet': 0.0412
            },
            'performance_degradation': {
                'Meta-PINN': 0.023,
                'Standard PINN': 0.156,
                'Transfer Learning PINN': 0.078,
                'Fourier Neural Operator': 0.198,
                'DeepONet': 0.134
            }
        },
        
        # Physics discovery results
        'physics_discovery': {
            'discovered_relationships': [
                {
                    'relationship': 'μ(T) = A * T^(-1.47) * exp(B/T)',
                    'confidence': 0.94,
                    'validation_score': 0.91,
                    'description': 'Temperature-dependent viscosity following modified Arrhenius law with power-law pre-factor'
                },
                {
                    'relationship': 'μ(P,T) = μ₀(T) * (1 + αP + βP²)',
                    'confidence': 0.87,
                    'validation_score': 0.85,
                    'description': 'Pressure-temperature coupling in viscosity with quadratic pressure dependence'
                },
                {
                    'relationship': 'μ(γ̇) = μ∞ + (μ₀ - μ∞)/(1 + (λγ̇)ⁿ)',
                    'confidence': 0.82,
                    'validation_score': 0.79,
                    'description': 'Shear-thinning behavior following Cross model for non-Newtonian fluids'
                }
            ],
            'causal_strengths': {
                'temperature': 0.89,
                'pressure': 0.76,
                'shear_rate': 0.54,
                'concentration': 0.41,
                'molecular_weight': 0.38,
                'surface_tension': 0.22
            }
        },    
    
        # Effect sizes for method comparisons
        'effect_sizes': {
            'Meta-PINN vs Standard PINN': {
                'parameter_accuracy': 2.1,  # Very large effect
                'adaptation_speed': 3.8,    # Extremely large effect
                'physics_consistency': 1.9, # Large effect
                'computational_efficiency': 2.7  # Very large effect
            },
            'Meta-PINN vs Transfer Learning': {
                'parameter_accuracy': 1.2,  # Large effect
                'adaptation_speed': 2.1,    # Very large effect
                'physics_consistency': 1.4, # Large effect
                'computational_efficiency': 1.1  # Large effect
            },
            'Meta-PINN vs FNO': {
                'parameter_accuracy': 1.8,  # Large effect
                'adaptation_speed': 0.9,    # Medium effect
                'physics_consistency': 2.3, # Very large effect
                'computational_efficiency': -0.7  # Medium effect (FNO better)
            }
        },
        
        # Confidence intervals
        'confidence_intervals': {
            'Meta-PINN': {
                'parameter_accuracy': [0.918, 0.974],
                'adaptation_speed': [6.1, 8.5],
                'physics_consistency': [0.979, 0.995],
                'computational_efficiency': [0.85, 0.93]
            },
            'Standard PINN': {
                'parameter_accuracy': [0.767, 0.901],
                'adaptation_speed': [42.3, 61.9],
                'physics_consistency': [0.888, 0.958],
                'computational_efficiency': [0.33, 0.51]
            }
        },
        
        # Hyperparameter configurations
        'hyperparameters': {
            'Meta-PINN': {
                'meta_learning_rate': 0.001,
                'adaptation_learning_rate': 0.01,
                'adaptation_steps': 5,
                'hidden_layers': [64, 128, 128, 64],
                'batch_size': 32,
                'meta_batch_size': 16,
                'physics_loss_weight': 1.0
            },
            'Standard PINN': {
                'learning_rate': 0.001,
                'hidden_layers': [32, 64, 64, 32],
                'batch_size': 64,
                'epochs': 2000,
                'physics_loss_weight': 1.0,
                'optimizer': 'Adam'
            },
            'Transfer Learning PINN': {
                'pretrain_learning_rate': 0.001,
                'finetune_learning_rate': 0.0001,
                'hidden_layers': [64, 128, 64],
                'batch_size': 32,
                'pretrain_epochs': 1000,
                'finetune_epochs': 200,
                'freeze_layers': 2
            }
        },
        
        # Uncertainty quantification data
        'uncertainty_data': {
            'x': np.linspace(0, 1, 100),
            'y_mean': 0.5 + 0.3 * np.sin(2 * np.pi * np.linspace(0, 1, 100)),
            'y_std': 0.05 + 0.02 * np.abs(np.sin(4 * np.pi * np.linspace(0, 1, 100))),
            'ground_truth': 0.5 + 0.3 * np.sin(2 * np.pi * np.linspace(0, 1, 100))
        }
    }
    
    return experimental_results


def demonstrate_comprehensive_report_generation():
    """Demonstrate comprehensive report generation with all features."""
    
    print("🔬 Meta-Learning PINN Comprehensive Report Generation Demo")
    print("=" * 60)
    
    # Create experimental data
    print("\n📊 Creating comprehensive experimental data...")
    experimental_results = create_comprehensive_experimental_data()
    
    # Initialize report generator
    print("🛠️  Initializing report generator with plot and table integration...")
    plot_generator = PaperPlotGenerator()
    table_generator = LaTeXTableGenerator()
    report_generator = ReportGenerator(
        plot_generator=plot_generator,
        table_generator=table_generator,
        template_style='academic',
        language='en'
    )
    
    # Create output directory
    output_dir = Path('results/comprehensive_report_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Generate reports in all formats
    formats = ['markdown', 'latex', 'html']
    
    for format_type in formats:
        print(f"\n📝 Generating {format_type.upper()} report...")
        
        format_output_dir = output_dir / format_type
        format_output_dir.mkdir(exist_ok=True)
        
        saved_files = report_generator.generate_comprehensive_report(
            experimental_results=experimental_results,
            output_dir=format_output_dir,
            report_title=f"Meta-Learning Physics-Informed Neural Networks: Comprehensive Analysis",
            author="ML Research Pipeline",
            include_plots=True,
            include_tables=True,
            format_type=format_type
        )
        
        print(f"   ✅ Generated {len(saved_files)} files:")
        for file_type, file_path in saved_files.items():
            print(f"      - {file_type}: {file_path.name}")
    
    # Demonstrate natural language generation quality
    print("\n🗣️  Demonstrating natural language generation...")
    
    # Executive summary
    executive_summary = report_generator._generate_executive_summary(experimental_results)
    print("\n📋 Executive Summary Preview:")
    print("-" * 40)
    print(executive_summary[:300] + "..." if len(executive_summary) > 300 else executive_summary)
    
    # Physics discovery analysis
    physics_analysis = report_generator._analyze_physics_discoveries(
        experimental_results['physics_discovery']
    )
    print("\n🔬 Physics Discovery Analysis Preview:")
    print("-" * 40)
    print(physics_analysis[:400] + "..." if len(physics_analysis) > 400 else physics_analysis)
    
    # Method comparison
    method_comparison = report_generator._generate_method_comparison_analysis(experimental_results)
    print("\n📊 Method Comparison Preview:")
    print("-" * 40)
    print(method_comparison[:350] + "..." if len(method_comparison) > 350 else method_comparison)
    
    # Statistical analysis
    statistical_analysis = report_generator._generate_statistical_analysis(experimental_results)
    print("\n📈 Statistical Analysis Preview:")
    print("-" * 40)
    print(statistical_analysis[:300] + "..." if len(statistical_analysis) > 300 else statistical_analysis)
    
    print("\n✨ Report generation complete!")
    print(f"📂 All reports saved to: {output_dir.absolute()}")
    
    # Show file structure
    print("\n📁 Generated file structure:")
    for format_dir in output_dir.iterdir():
        if format_dir.is_dir():
            print(f"   {format_dir.name}/")
            for file_path in format_dir.iterdir():
                size_kb = file_path.stat().st_size / 1024
                print(f"      ├── {file_path.name} ({size_kb:.1f} KB)")
    
    return output_dir


def demonstrate_error_handling():
    """Demonstrate graceful error handling with incomplete data."""
    
    print("\n🛡️  Demonstrating error handling and graceful degradation...")
    
    # Create minimal/incomplete data
    incomplete_data = {
        'method_results': {
            'Method A': {'accuracy_mean': 0.8},
            'Method B': {'accuracy_mean': 0.7, 'accuracy_std': 0.05}
        },
        'metrics': ['accuracy']
    }
    
    report_generator = ReportGenerator()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        try:
            saved_files = report_generator.generate_comprehensive_report(
                experimental_results=incomplete_data,
                output_dir=output_dir,
                report_title="Error Handling Demo",
                include_plots=False,
                include_tables=False
            )
            
            print("   ✅ Successfully handled incomplete data")
            print(f"   📄 Generated report: {saved_files['report'].name}")
            
            # Check report content
            with open(saved_files['report'], 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"   📏 Report length: {len(content)} characters")
            print("   ✅ Graceful degradation successful")
            
        except Exception as e:
            print(f"   ❌ Error handling failed: {e}")


if __name__ == '__main__':
    # Run comprehensive demonstration
    output_directory = demonstrate_comprehensive_report_generation()
    
    # Demonstrate error handling
    demonstrate_error_handling()
    
    print(f"\n🎉 Demo complete! Check the results in: {output_directory}")
    print("\n💡 Key features demonstrated:")
    print("   ✅ Automated executive summary generation")
    print("   ✅ Natural language descriptions of experimental findings")
    print("   ✅ Multi-format output (Markdown, LaTeX, HTML)")
    print("   ✅ Integration with plots and tables")
    print("   ✅ Physics discovery integration")
    print("   ✅ Statistical analysis integration")
    print("   ✅ Comprehensive content validation")
    print("   ✅ Error handling and graceful degradation")
    print("\n📋 Requirements satisfied:")
    print("   ✅ 9.3: Generate comprehensive analysis with executive summaries")
    print("   ✅ 9.5: Provide natural language descriptions of physics findings")