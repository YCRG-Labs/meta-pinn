# Presentation Figures for Physics-Informed Meta-Learning

This directory contains all the data-based visualizations generated from the experimental results reported in the MDPI paper. These figures are designed for use in presentation slides.

## Generated Data-Based Figures

### Slide 3: Motivating Example - Fluid Dynamics
- **File**: `computational_cost_comparison.png`
- **Description**: Bar chart showing training times across different methods
- **Data Source**: Table A3 from paper (training times: Standard PINN 12.4h, Transfer PINN 8.7h, MAML 6.2h, PI-MAML 4.1h)

### Slide 10: Theoretical Convergence Guarantees
- **File**: `theoretical_convergence.png`
- **Description**: Log-log plot showing theoretical upper bound vs empirical convergence
- **Data Source**: Theoretical analysis from paper (convergence rate O(1/T + √(log T/T)))

### Slide 11: Sample Complexity Analysis
- **File**: `sample_complexity.png`
- **Description**: Log-log plot showing sample complexity vs problem dimension for different ε values
- **Data Source**: Theoretical bounds from paper with physics regularization benefit

### Slide 13: Statistical Analysis Methodology
- **File**: `statistical_analysis.png`
- **Description**: Bootstrap distribution and confidence intervals comparison
- **Data Source**: Statistical analysis from paper (bootstrap with 1000 iterations, 95% CI)

### Slide 14: Main Experimental Results
- **File**: `main_experimental_results.png`
- **Description**: Dual panel showing validation accuracy and adaptation efficiency
- **Data Source**: Table 1 from paper
- **Key Results**:
  - PI-MAML: 92.4% accuracy (95% CI: [88.2%, 96.6%])
  - Transfer PINN baseline: 83.0% accuracy
  - PI-MAML requires only 50 adaptation steps vs 150 for Transfer PINN

### Slide 15: Detailed Performance Breakdown
- **File**: `performance_breakdown.png`
- **Description**: Line plot showing performance vs number of shots (5, 10, 20)
- **Data Source**: Table 1 from paper with error bars (95% confidence intervals)

### Slide 16: Physics Discovery Results
- **File**: `physics_discovery_results.png`
- **Description**: Bar chart showing discovery accuracies with confidence intervals
- **Data Source**: Physics discovery results from paper
- **Key Results**:
  - Reynolds number dependence: 94% ± 3%
  - Pressure-velocity coupling: 91% ± 4%
  - Boundary layer effects: 89% ± 5%
  - Heat transfer correlations: 92% ± 3%

### Slide 17: Convergence Analysis
- **File**: `convergence_analysis.png`
- **Description**: Convergence curves during meta-training (log scale)
- **Data Source**: Based on convergence behavior described in paper
- **Note**: Shows faster convergence with physics constraints

### Slide 18: Ablation Study Results
- **File**: `ablation_study.png`
- **Description**: Scatter plot showing accuracy vs efficiency trade-offs
- **Data Source**: Ablation study results from paper
- **Key Results**:
  - Full PI-MAML: 0.924 ± 0.042
  - Without adaptive weighting: 0.887 ± 0.055
  - Without physics discovery: 0.901 ± 0.049
  - Without physics constraints: 0.801 ± 0.072
  - Without meta-learning: 0.830 ± 0.057

### Slide 19: Computational Efficiency Analysis
- **File**: `computational_efficiency.png`
- **Description**: Four-panel comparison of computational metrics
- **Data Source**: Table A3 from paper
- **Metrics**: Training time, memory usage, GPU utilization, energy consumption

## Paper Figures to Use Directly

**IMPORTANT**: For these slides, use the original figures from the paper rather than generating new ones:

### Slide 14: Main Experimental Results
- **Use**: Figure 1 from paper (`MDPI/figures/Figure 1.png`)
- **Description**: Performance comparison (validation accuracy and adaptation efficiency)

### Slide 17: Convergence Analysis
- **Use**: Figure 2 from paper (`MDPI/figures/Figure 2.png`)
- **Description**: Convergence comparison during meta-training

### Slide 18: Ablation Study Results
- **Use**: Figure 3 from paper (`MDPI/figures/Figure 3.png`)
- **Description**: Accuracy vs efficiency trade-offs

## Tables to Reference from Paper

For any slides requiring tabular data, reference these tables from the MDPI paper:

- **Table 1**: Few-shot adaptation results (main experimental results)
- **Table A1**: Meta-learning hyperparameters
- **Table A2**: Navier-Stokes results by Reynolds number
- **Table A3**: Computational efficiency analysis
- **Table A4**: Physics discovery accuracy metrics

## Statistical Significance Notes

All results include proper statistical analysis:
- **Sample sizes**: n=50 for each condition
- **Confidence intervals**: 95% CI using bootstrap resampling (1000 iterations)
- **Statistical tests**: Two-tailed t-tests with Bonferroni correction
- **Effect sizes**: Cohen's d reported where applicable
- **Significance level**: α = 0.05

## Key Statistical Results

- **PI-MAML vs Transfer PINN**: p < 0.001, Cohen d = 2.1 (large effect)
- **Physics constraints contribution**: Cohen d = 1.8 (large effect)
- **All physics discovery accuracies**: p < 0.001 vs chance level (25%)

## Usage Instructions

1. **For data-based slides**: Use the generated PNG files from this directory
2. **For paper figure slides**: Use the original figures from `paper/MDPI/figures/`
3. **For tables**: Reference the full MDPI paper (`paper/mdpi.tex`)
4. **For statistical details**: All confidence intervals and significance tests are included in the generated figures

## Figure Quality

- **Resolution**: 300 DPI for high-quality presentation
- **Format**: PNG with transparent backgrounds where appropriate
- **Style**: Professional presentation style with clear labels and legends
- **Colors**: Colorblind-friendly palette
- **Fonts**: Consistent sizing for readability in presentations

## Conceptual Visuals (Not Generated)

The following slides require conceptual/illustrative visuals that should be created using design tools:

- Slide 1: Title slide with logos
- Slide 2: PINN limitations diagram
- Slide 4: Meta-learning conceptual framework
- Slide 5: Research contributions overview
- Slide 6: Problem formulation diagrams
- Slide 7: Algorithmic flowchart
- Slide 8: Physics loss implementation
- Slide 9: Adaptive constraint weighting
- Slide 12: Experimental setup overview
- Slides 20-28: Limitations, future work, and conclusions

These should focus on clear, pedagogical illustrations that complement the data-based figures provided here.