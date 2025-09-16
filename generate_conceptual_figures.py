#!/usr/bin/env python3
"""
Generate Conceptual Visualizations for Physics-Informed Meta-Learning Presentation

This script generates all the conceptual/illustrative figures needed for the presentation slides.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow
import numpy as np
import seaborn as sns
from pathlib import Path
# QR code generation would require: pip install qrcode[pil]
# from PIL import Image, ImageDraw, ImageFont

# Set style for professional presentation figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Create output directory
output_dir = Path("conceptual_figures")
output_dir.mkdir(exist_ok=True)

def create_title_slide():
    """Slide 1: Title slide with clean layout"""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'Meta-Learning Physics-Informed Neural Networks', 
            ha='center', va='center', fontsize=28, fontweight='bold', color='#1f77b4')
    ax.text(5, 6.8, 'for Few-Shot Parameter Inference', 
            ha='center', va='center', fontsize=24, fontweight='bold', color='#1f77b4')
    
    # Authors
    ax.text(5, 5.5, 'Brandon Yee¹, Wilson Collins¹, Benjamin Pellegrini¹, Caden Wang²', 
            ha='center', va='center', fontsize=16, color='#333333')
    
    # Affiliations
    ax.text(5, 4.8, '¹ Yee Collins Research Group', 
            ha='center', va='center', fontsize=14, color='#666666')
    ax.text(5, 4.4, '² Department of Computer Science, New York University', 
            ha='center', va='center', fontsize=14, color='#666666')
    
    # GitHub link
    ax.text(5, 3.2, 'GitHub: https://github.com/YCRG-Labs/meta-pinn', 
            ha='center', va='center', fontsize=14, color='#0066cc',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f8ff", alpha=0.8))
    
    # Date
    ax.text(5, 2.0, 'AAAI 2026', 
            ha='center', va='center', fontsize=16, fontweight='bold', color='#333333')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'title_slide.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def create_pinn_limitations():
    """Slide 2: Traditional PINN workflow and limitations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Traditional PINN workflow
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Traditional PINN Workflow', fontsize=16, fontweight='bold', pad=20)
    
    # Workflow boxes
    boxes = [
        (5, 8.5, 'New Physics Problem'),
        (5, 7, 'Initialize Network\nfrom Scratch'),
        (5, 5.5, 'Train on Limited Data\n(Expensive)'),
        (5, 4, 'Solve Single Problem'),
        (5, 2.5, 'Repeat for Each\nNew Problem')
    ]
    
    for x, y, text in boxes:
        box = FancyBboxPatch((x-1.5, y-0.4), 3, 0.8, boxstyle="round,pad=0.1",
                            facecolor='#ffcccc', edgecolor='#cc0000', linewidth=2)
        ax1.add_patch(box)
        ax1.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrows
    for i in range(len(boxes)-1):
        ax1.arrow(5, boxes[i][1]-0.5, 0, -0.6, head_width=0.2, head_length=0.1, 
                 fc='#cc0000', ec='#cc0000', linewidth=2)
    
    # Timeline comparison
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Computational Cost Timeline', fontsize=16, fontweight='bold', pad=20)
    
    # Timeline bars
    methods = ['Traditional\nPINN', 'Our\nPI-MAML']
    times = [12.4, 4.1]
    colors = ['#ff7f7f', '#7fbf7f']
    
    for i, (method, time, color) in enumerate(zip(methods, times, colors)):
        y_pos = 7 - i * 2
        # Bar
        bar = Rectangle((1, y_pos-0.3), time*0.6, 0.6, facecolor=color, alpha=0.8)
        ax2.add_patch(bar)
        # Label
        ax2.text(0.5, y_pos, method, ha='right', va='center', fontsize=12, fontweight='bold')
        # Time
        ax2.text(time*0.6 + 1.2, y_pos, f'{time:.1f}h', ha='left', va='center', 
                fontsize=12, fontweight='bold')
    
    ax2.text(5, 3, '67% Reduction in Training Time', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#2ca02c',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e8", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pinn_limitations.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_fluid_dynamics_examples():
    """Slide 3: Four flow visualizations for different Reynolds numbers"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    reynolds_numbers = [100, 200, 500, 1000]
    
    for i, (ax, re) in enumerate(zip(axes.flat, reynolds_numbers)):
        # Create flow field visualization
        x = np.linspace(0, 4, 20)
        y = np.linspace(0, 2, 10)
        X, Y = np.meshgrid(x, y)
        
        # Simulate flow patterns based on Reynolds number
        # Higher Re = more turbulent/complex patterns
        turbulence_factor = re / 1000
        U = 1 - Y**2 + turbulence_factor * 0.3 * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
        V = turbulence_factor * 0.2 * np.cos(2*np.pi*X) * np.sin(2*np.pi*Y)
        
        # Streamplot
        ax.streamplot(X, Y, U, V, density=1.5, color=U, cmap='viridis')
        
        # Cylinder (obstacle)
        circle = Circle((1, 1), 0.2, facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 2)
        ax.set_title(f'Reynolds Number = {re}', fontsize=14, fontweight='bold')
        ax.set_xlabel('x/D')
        ax.set_ylabel('y/D')
        
        # Add flow characteristics
        if re <= 200:
            ax.text(3, 0.2, 'Laminar Flow', fontsize=12, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        else:
            ax.text(3, 0.2, 'Turbulent Flow', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    
    plt.suptitle('Fluid Dynamics: Flow Patterns at Different Reynolds Numbers', 
                fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(output_dir / 'fluid_dynamics_examples.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_meta_learning_concept():
    """Slide 4: Meta-learning conceptual framework"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Why Meta-Learning for Physics?', ha='center', va='center', 
            fontsize=20, fontweight='bold')
    
    # Traditional Learning
    ax.text(3, 8.5, 'Traditional Learning', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='#d62728')
    
    # Traditional boxes
    trad_tasks = ['Task 1\n(Re=100)', 'Task 2\n(Re=200)', 'Task 3\n(Re=500)']
    for i, task in enumerate(trad_tasks):
        y_pos = 7.5 - i * 1.5
        # Task box
        box = FancyBboxPatch((1.5, y_pos-0.3), 3, 0.6, boxstyle="round,pad=0.1",
                            facecolor='#ffcccc', edgecolor='#d62728', linewidth=2)
        ax.add_patch(box)
        ax.text(3, y_pos, task, ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Arrow to training
        ax.arrow(4.5, y_pos, 1, 0, head_width=0.1, head_length=0.1, 
                fc='#d62728', ec='#d62728')
        ax.text(5.8, y_pos, 'Train from\nScratch', ha='center', va='center', 
               fontsize=10, color='#d62728')
    
    # Meta-Learning
    ax.text(9, 8.5, 'Meta-Learning Approach', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='#2ca02c')
    
    # Meta-learning diagram
    # Support tasks
    support_box = FancyBboxPatch((7.5, 7), 3, 1, boxstyle="round,pad=0.1",
                                facecolor='#ccffcc', edgecolor='#2ca02c', linewidth=2)
    ax.add_patch(support_box)
    ax.text(9, 7.5, 'Support Tasks\n(Re=100,200,300,...)', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # Arrow down
    ax.arrow(9, 6.8, 0, -0.8, head_width=0.2, head_length=0.1, 
            fc='#2ca02c', ec='#2ca02c', linewidth=2)
    
    # Meta-model
    meta_box = FancyBboxPatch((7.5, 5.5), 3, 0.8, boxstyle="round,pad=0.1",
                             facecolor='#90EE90', edgecolor='#2ca02c', linewidth=2)
    ax.add_patch(meta_box)
    ax.text(9, 5.9, 'Meta-Model\n(Learns to Learn)', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # Arrow down
    ax.arrow(9, 5.3, 0, -0.8, head_width=0.2, head_length=0.1, 
            fc='#2ca02c', ec='#2ca02c', linewidth=2)
    
    # New task
    new_task_box = FancyBboxPatch((7.5, 4), 3, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='#98FB98', edgecolor='#2ca02c', linewidth=2)
    ax.add_patch(new_task_box)
    ax.text(9, 4.4, 'New Task (Re=750)\nFast Adaptation', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # Benefits
    ax.text(6, 2.5, 'Key Benefits:', ha='center', va='center', 
           fontsize=16, fontweight='bold')
    benefits = [
        '• 3× Fewer Adaptation Steps (50 vs 150)',
        '• 15% Better Generalization Performance', 
        '• Leverages Prior Physics Knowledge',
        '• Maintains Physical Consistency'
    ]
    
    for i, benefit in enumerate(benefits):
        ax.text(6, 2 - i*0.4, benefit, ha='center', va='center', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f8ff", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'meta_learning_concept.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_research_contributions():
    """Slide 5: Research contributions overview"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Research Contributions', ha='center', va='center', 
            fontsize=20, fontweight='bold')
    
    # Central framework
    center_box = FancyBboxPatch((4.5, 4.5), 3, 1.5, boxstyle="round,pad=0.2",
                               facecolor='#1f77b4', edgecolor='#1f77b4', linewidth=3, alpha=0.8)
    ax.add_patch(center_box)
    ax.text(6, 5.25, 'Physics-Informed\nMeta-Learning\nFramework', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='white')
    
    # Contributions around the center
    contributions = [
        (2, 8, 'Novel Meta-Learning\nAlgorithm', '#ff7f0e'),
        (10, 8, 'Theoretical\nGuarantees', '#2ca02c'),
        (1.5, 5.25, 'Adaptive Constraint\nWeighting', '#d62728'),
        (10.5, 5.25, 'Automated Physics\nDiscovery', '#9467bd'),
        (2, 2.5, 'Comprehensive\nExperimental\nValidation', '#8c564b'),
        (10, 2.5, 'Statistical\nRigor', '#e377c2')
    ]
    
    for x, y, text, color in contributions:
        # Contribution box
        box = FancyBboxPatch((x-1, y-0.5), 2, 1, boxstyle="round,pad=0.1",
                            facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Arrow to center
        dx = 6 - x
        dy = 5.25 - y
        length = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / length * 1.5
        dy_norm = dy / length * 1.5
        
        ax.arrow(x + dx_norm*0.3, y + dy_norm*0.3, dx_norm*0.4, dy_norm*0.4,
                head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.7)
    
    # Key equations preview
    ax.text(6, 1.5, r'$\theta^* = \arg\min_\theta \mathbb{E}_{\mathcal{T}} [\mathcal{L}_{total}(\phi_{\mathcal{T}}, \mathcal{T})]$', 
           ha='center', va='center', fontsize=14, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffacd", alpha=0.8))
    
    ax.text(6, 0.8, r'$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda(\mathcal{T}) \mathcal{L}_{physics}$', 
           ha='center', va='center', fontsize=14,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffacd", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'research_contributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_problem_formulation():
    """Slide 6: Problem formulation with domain visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Domain visualization
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 3)
    ax1.set_title('Physics Domain Ω with Boundary ∂Ω', fontsize=14, fontweight='bold')
    
    # Domain boundary
    domain = Rectangle((0.5, 0.5), 3, 2, facecolor='lightblue', alpha=0.3, 
                      edgecolor='blue', linewidth=3)
    ax1.add_patch(domain)
    
    # Boundary conditions
    ax1.text(2, 2.7, 'Boundary Conditions\n𝓑[u](x) = 0', ha='center', va='center', 
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Interior points
    np.random.seed(42)
    x_interior = np.random.uniform(0.7, 3.3, 15)
    y_interior = np.random.uniform(0.7, 2.3, 15)
    ax1.scatter(x_interior, y_interior, c='red', s=30, alpha=0.7, label='Interior Points')
    
    # Boundary points
    x_boundary = np.concatenate([
        np.linspace(0.5, 3.5, 8),  # bottom
        np.linspace(0.5, 3.5, 8),  # top
        np.full(6, 0.5),           # left
        np.full(6, 3.5)            # right
    ])
    y_boundary = np.concatenate([
        np.full(8, 0.5),           # bottom
        np.full(8, 2.5),           # top
        np.linspace(0.5, 2.5, 6),  # left
        np.linspace(0.5, 2.5, 6)   # right
    ])
    ax1.scatter(x_boundary, y_boundary, c='blue', s=30, alpha=0.7, label='Boundary Points')
    
    ax1.text(2, 1.5, 'PDE: 𝓕[u](x) = 0\nx ∈ Ω', ha='center', va='center', 
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    
    # Mathematical formulation
    ax2.axis('off')
    ax2.set_title('Mathematical Formulation', fontsize=14, fontweight='bold')
    
    equations = [
        r'Task Distribution: $\mathcal{T} \sim p(\mathcal{T})$',
        r'Domain: $\Omega_i \subset \mathbb{R}^d$ with boundary $\partial\Omega_i$',
        r'Governing PDE: $\mathcal{F}_i[u_i](x) = 0, \quad x \in \Omega_i$',
        r'Boundary Conditions: $\mathcal{B}_i[u_i](x) = 0, \quad x \in \partial\Omega_i$',
        r'Limited Data: $\mathcal{D}_i = \{(x_j, u_j)\}_{j=1}^{N_i}$ where $N_i$ is small',
        '',
        r'Goal: Learn meta-model $\theta^*$ for rapid adaptation',
        r'$\theta^* = \arg\min_\theta \mathbb{E}_{\mathcal{T}} [\mathcal{L}_{total}(\phi_{\mathcal{T}}, \mathcal{T})]$'
    ]
    
    for i, eq in enumerate(equations):
        if eq:  # Skip empty strings
            ax2.text(0.05, 0.9 - i*0.12, eq, transform=ax2.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f8ff", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'problem_formulation.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_framework_flowchart():
    """Slide 7: Physics-informed meta-learning framework flowchart"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(6, 11.5, 'Physics-Informed Meta-Learning Framework', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    
    # Outer loop (Meta-learning)
    outer_box = FancyBboxPatch((0.5, 8), 11, 3, boxstyle="round,pad=0.2",
                              facecolor='#e6f3ff', edgecolor='#1f77b4', linewidth=3)
    ax.add_patch(outer_box)
    ax.text(1, 10.5, 'Outer Loop (Meta-Learning)', fontsize=14, fontweight='bold', color='#1f77b4')
    
    # Meta-update components
    meta_components = [
        (2.5, 9.5, 'Sample Tasks\n𝒯 ~ p(𝒯)'),
        (5, 9.5, 'Inner Loop\nAdaptation'),
        (7.5, 9.5, 'Meta-Gradient\n∇θ ℒmeta'),
        (10, 9.5, 'Update θ')
    ]
    
    for i, (x, y, text) in enumerate(meta_components):
        box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8, boxstyle="round,pad=0.1",
                            facecolor='#cce7ff', edgecolor='#1f77b4', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        if i < len(meta_components) - 1:
            ax.arrow(x+0.8, y, 1.4, 0, head_width=0.1, head_length=0.1, 
                    fc='#1f77b4', ec='#1f77b4')
    
    # Inner loop (Task adaptation)
    inner_box = FancyBboxPatch((0.5, 4.5), 11, 3, boxstyle="round,pad=0.2",
                              facecolor='#ffe6e6', edgecolor='#d62728', linewidth=3)
    ax.add_patch(inner_box)
    ax.text(1, 7, 'Inner Loop (Task Adaptation)', fontsize=14, fontweight='bold', color='#d62728')
    
    # Inner loop components
    inner_components = [
        (2.5, 6, 'Initialize\nφ = θ'),
        (5, 6, 'Compute Loss\nℒtotal'),
        (7.5, 6, 'Gradient Step\nφ ← φ - α∇φℒ'),
        (10, 6, 'Adapted\nParameters φ')
    ]
    
    for i, (x, y, text) in enumerate(inner_components):
        box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8, boxstyle="round,pad=0.1",
                            facecolor='#ffcccc', edgecolor='#d62728', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        if i < len(inner_components) - 1:
            ax.arrow(x+0.8, y, 1.4, 0, head_width=0.1, head_length=0.1, 
                    fc='#d62728', ec='#d62728')
    
    # Physics loss components
    physics_box = FancyBboxPatch((0.5, 1), 11, 3, boxstyle="round,pad=0.2",
                                facecolor='#e6ffe6', edgecolor='#2ca02c', linewidth=3)
    ax.add_patch(physics_box)
    ax.text(1, 3.5, 'Physics Loss Components', fontsize=14, fontweight='bold', color='#2ca02c')
    
    # Physics components
    ax.text(3, 2.5, 'Data Loss\nℒdata(φ,𝒯)', ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#ccffcc", alpha=0.8))
    
    ax.text(6, 2.5, '+', ha='center', va='center', fontsize=16, fontweight='bold')
    
    ax.text(9, 2.5, 'Physics Loss\nλ(𝒯)ℒphysics(φ,𝒯)', ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#ccffcc", alpha=0.8))
    
    # Adaptive weighting
    ax.text(6, 1.5, 'λ(𝒯) = σ(Wλh𝒯 + bλ)', ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#90EE90", alpha=0.8))
    
    # Arrows connecting loops
    ax.arrow(5, 8.5, 0, -0.8, head_width=0.2, head_length=0.1, 
            fc='purple', ec='purple', linewidth=3)
    ax.arrow(7.5, 4.5, 0, -0.8, head_width=0.2, head_length=0.1, 
            fc='purple', ec='purple', linewidth=3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'framework_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_physics_loss_diagram():
    """Slide 8: Physics loss implementation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Domain sampling visualization
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 3)
    ax1.set_title('Physics Loss: Domain Sampling', fontsize=14, fontweight='bold')
    
    # Domain
    domain = Rectangle((0.5, 0.5), 3, 2, facecolor='lightblue', alpha=0.3, 
                      edgecolor='blue', linewidth=2)
    ax1.add_patch(domain)
    
    # Interior sampling points for PDE residual
    np.random.seed(42)
    x_pde = np.random.uniform(0.7, 3.3, 20)
    y_pde = np.random.uniform(0.7, 2.3, 20)
    ax1.scatter(x_pde, y_pde, c='red', s=40, alpha=0.8, label='PDE Residual Points', marker='o')
    
    # Boundary sampling points
    x_bc = np.concatenate([
        np.linspace(0.5, 3.5, 10),
        np.linspace(0.5, 3.5, 10),
        np.full(8, 0.5),
        np.full(8, 3.5)
    ])
    y_bc = np.concatenate([
        np.full(10, 0.5),
        np.full(10, 2.5),
        np.linspace(0.5, 2.5, 8),
        np.linspace(0.5, 2.5, 8)
    ])
    ax1.scatter(x_bc, y_bc, c='blue', s=40, alpha=0.8, label='Boundary Points', marker='s')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    
    # Add annotations
    ax1.annotate('PDE: 𝓕[uφ](x) = 0', xy=(2, 1.5), xytext=(1, 0.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax1.annotate('BC: 𝓑[uφ](x) = 0', xy=(3.5, 1.5), xytext=(3.2, 0.2),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=12, fontweight='bold', color='blue',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Physics loss formulation
    ax2.axis('off')
    ax2.set_title('Physics Loss Formulation', fontsize=14, fontweight='bold')
    
    # Main physics loss equation
    ax2.text(0.5, 0.8, r'$\mathcal{L}_{physics}(\phi, \mathcal{T}) =$', 
            transform=ax2.transAxes, fontsize=16, fontweight='bold', ha='center')
    
    # PDE residual term
    ax2.text(0.5, 0.65, r'$\mathbb{E}_{x \sim \Omega} \left[ |\mathcal{F}[u_\phi](x)|^2 \right]$', 
            transform=ax2.transAxes, fontsize=14, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffcccc", alpha=0.8))
    
    ax2.text(0.5, 0.55, '+', transform=ax2.transAxes, fontsize=16, fontweight='bold', ha='center')
    
    # Boundary condition term
    ax2.text(0.5, 0.4, r'$\mathbb{E}_{x \sim \partial\Omega} \left[ |\mathcal{B}[u_\phi](x)|^2 \right]$', 
            transform=ax2.transAxes, fontsize=14, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ccccff", alpha=0.8))
    
    # Explanation
    explanations = [
        'PDE Residual: Measures how well the network',
        'satisfies the governing differential equation',
        '',
        'Boundary Residual: Ensures boundary conditions',
        'are satisfied at domain boundaries',
        '',
        'Total Loss: ℒtotal = ℒdata + λ(𝒯)ℒphysics'
    ]
    
    for i, text in enumerate(explanations):
        if text:  # Skip empty strings
            ax2.text(0.05, 0.25 - i*0.03, text, transform=ax2.transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f8ff", alpha=0.6))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'physics_loss_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_adaptive_weighting():
    """Slide 9: Adaptive constraint weighting mechanism"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Neural network diagram for task embedding
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    ax1.set_title('Task Embedding Network', fontsize=14, fontweight='bold')
    
    # Input layer (task characteristics)
    input_features = ['Reynolds\nNumber', 'Boundary\nType', 'Geometry', 'PDE\nType']
    for i, feature in enumerate(input_features):
        y_pos = 6.5 - i * 1.5
        circle = Circle((1.5, y_pos), 0.3, facecolor='lightblue', edgecolor='blue')
        ax1.add_patch(circle)
        ax1.text(1.5, y_pos, feature, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Hidden layer
    for i in range(3):
        y_pos = 5.5 - i * 1.5
        circle = Circle((4, y_pos), 0.3, facecolor='lightgreen', edgecolor='green')
        ax1.add_patch(circle)
        ax1.text(4, y_pos, f'h{i+1}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Connections from input
        for j in range(4):
            input_y = 6.5 - j * 1.5
            ax1.plot([1.8, 3.7], [input_y, y_pos], 'k-', alpha=0.3, linewidth=1)
    
    # Task embedding
    embedding_circle = Circle((6.5, 4), 0.4, facecolor='yellow', edgecolor='orange', linewidth=2)
    ax1.add_patch(embedding_circle)
    ax1.text(6.5, 4, 'h𝒯', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Connections to embedding
    for i in range(3):
        y_pos = 5.5 - i * 1.5
        ax1.plot([4.3, 6.1], [y_pos, 4], 'k-', alpha=0.5, linewidth=2)
    
    # Weight computation
    weight_box = FancyBboxPatch((8, 3.5), 1.5, 1, boxstyle="round,pad=0.1",
                               facecolor='pink', edgecolor='red', linewidth=2)
    ax1.add_patch(weight_box)
    ax1.text(8.75, 4, 'λ(𝒯)\n= σ(Wλh𝒯 + bλ)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # Arrow to weight
    ax1.arrow(6.9, 4, 0.9, 0, head_width=0.1, head_length=0.1, 
             fc='orange', ec='orange', linewidth=2)
    
    # Examples of different λ values
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    ax2.set_title('Adaptive Weighting Examples', fontsize=14, fontweight='bold')
    
    # Example scenarios
    scenarios = [
        ('High Re (Turbulent)', 0.8, 'High physics weight\nfor complex dynamics'),
        ('Low Re (Laminar)', 0.3, 'Lower physics weight\nfor simpler flow'),
        ('Complex Geometry', 0.9, 'Very high weight for\ncomplex boundaries'),
        ('Simple Domain', 0.2, 'Minimal physics weight\nfor basic cases')
    ]
    
    for i, (scenario, weight, description) in enumerate(scenarios):
        y_pos = 7 - i * 1.8
        
        # Scenario box
        scenario_box = FancyBboxPatch((0.5, y_pos-0.3), 2.5, 0.6, boxstyle="round,pad=0.1",
                                     facecolor='lightblue', edgecolor='blue', linewidth=1)
        ax2.add_patch(scenario_box)
        ax2.text(1.75, y_pos, scenario, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrow
        ax2.arrow(3.2, y_pos, 1, 0, head_width=0.1, head_length=0.1, 
                 fc='black', ec='black')
        
        # Weight value
        weight_color = plt.cm.Reds(weight)
        weight_box = FancyBboxPatch((4.5, y_pos-0.2), 1, 0.4, boxstyle="round,pad=0.05",
                                   facecolor=weight_color, edgecolor='red', linewidth=2)
        ax2.add_patch(weight_box)
        ax2.text(5, y_pos, f'λ = {weight}', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white' if weight > 0.5 else 'black')
        
        # Description
        ax2.text(7.5, y_pos, description, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))
    
    # Benefits text
    ax2.text(5, 0.5, 'Benefits: Task-specific physics regularization\nImproves adaptation efficiency and accuracy', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'adaptive_weighting.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_experimental_setup():
    """Slide 12: Experimental setup overview"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Experimental Setup Overview', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    
    # Problem types grid
    problems = [
        ('Navier-Stokes\nEquations', 'Re = 100-1000', '#ff7f0e'),
        ('Heat Transfer\nProblems', 'Varying BCs', '#2ca02c'),
        ('Burgers\nEquation', 'Different ν', '#d62728'),
        ('Lid-Driven\nCavity', 'Various Geometries', '#9467bd')
    ]
    
    # Create 2x2 grid
    positions = [(3, 7.5), (9, 7.5), (3, 5.5), (9, 5.5)]
    
    for (problem, param, color), (x, y) in zip(problems, positions):
        # Problem box
        box = FancyBboxPatch((x-1.2, y-0.8), 2.4, 1.6, boxstyle="round,pad=0.2",
                            facecolor=color, alpha=0.3, edgecolor=color, linewidth=3)
        ax.add_patch(box)
        
        # Problem name
        ax.text(x, y+0.3, problem, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Parameters
        ax.text(x, y-0.3, param, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Dataset statistics
    stats_box = FancyBboxPatch((1, 3), 10, 1.5, boxstyle="round,pad=0.2",
                              facecolor='#f0f8ff', edgecolor='#1f77b4', linewidth=2)
    ax.add_patch(stats_box)
    
    ax.text(6, 4, 'Dataset Statistics', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#1f77b4')
    
    stats = [
        '• 200 Training Tasks per Problem Type',
        '• 50 Test Tasks per Problem Type', 
        '• 20-100 Data Points per Task (Few-Shot Setting)',
        '• Rigorous Statistical Analysis (n=50, α=0.05, Bootstrap CI)'
    ]
    
    for i, stat in enumerate(stats):
        ax.text(1.5, 3.6 - i*0.25, stat, ha='left', va='center', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    
    # Evaluation metrics
    metrics_box = FancyBboxPatch((1, 1), 10, 1.5, boxstyle="round,pad=0.2",
                                facecolor='#fff0f0', edgecolor='#d62728', linewidth=2)
    ax.add_patch(metrics_box)
    
    ax.text(6, 2, 'Evaluation Metrics', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#d62728')
    
    metrics = [
        '• Validation Accuracy with 95% Confidence Intervals',
        '• Adaptation Efficiency (Number of Steps Required)',
        '• Physics Discovery Accuracy (Precision, Recall, F1-Score)',
        '• Computational Efficiency (Time, Memory, Energy)'
    ]
    
    for i, metric in enumerate(metrics):
        ax.text(1.5, 1.6 - i*0.25, metric, ha='left', va='center', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'experimental_setup.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_limitations_domain():
    """Slide 20: Limitations - Domain specificity"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Domain scope visualization
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Current Domain Scope', fontsize=14, fontweight='bold')
    
    # Current scope (fluid dynamics)
    current_circle = Circle((5, 5), 2.5, facecolor='lightblue', alpha=0.6, 
                           edgecolor='blue', linewidth=3)
    ax1.add_patch(current_circle)
    ax1.text(5, 5, 'Fluid\nDynamics\n\n✓ Validated', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='blue')
    
    # Potential extensions
    extensions = [
        (2, 8, 'Heat\nTransfer', 'orange'),
        (8, 8, 'Structural\nMechanics', 'green'),
        (2, 2, 'Electromagnetics', 'red'),
        (8, 2, 'Quantum\nMechanics', 'purple')
    ]
    
    for x, y, domain, color in extensions:
        circle = Circle((x, y), 1, facecolor=color, alpha=0.3, 
                       edgecolor=color, linewidth=2, linestyle='--')
        ax1.add_patch(circle)
        ax1.text(x, y, f'{domain}\n\n? Untested', ha='center', va='center', 
                fontsize=10, fontweight='bold', color=color)
        
        # Arrow from current to extension
        dx = x - 5
        dy = y - 5
        length = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / length * 1.5
        dy_norm = dy / length * 1.5
        
        ax1.arrow(5 + dx_norm, 5 + dy_norm, dx_norm*0.5, dy_norm*0.5,
                 head_width=0.2, head_length=0.2, fc=color, ec=color, 
                 alpha=0.5, linestyle='--')
    
    # Assumptions and constraints
    ax2.axis('off')
    ax2.set_title('Key Assumptions & Constraints', fontsize=14, fontweight='bold')
    
    assumptions = [
        'Domain Specificity:',
        '• Current validation limited to fluid dynamics',
        '• Physics discovery tuned for CFD problems',
        '• Boundary condition types specific to flow',
        '',
        'Scalability Constraints:',
        '• Performance on very high-dimensional problems unclear',
        '• Memory requirements scale with problem complexity',
        '• GPU utilization may vary across domains',
        '',
        'Theoretical Limitations:',
        '• Convergence analysis assumes standard regularity',
        '• Sample complexity bounds may not hold universally',
        '• Physics regularization benefit domain-dependent'
    ]
    
    for i, assumption in enumerate(assumptions):
        if assumption.endswith(':'):
            # Category headers
            ax2.text(0.05, 0.95 - i*0.06, assumption, transform=ax2.transAxes, 
                    fontsize=12, fontweight='bold', color='#d62728')
        elif assumption.startswith('•'):
            # Bullet points
            ax2.text(0.05, 0.95 - i*0.06, assumption, transform=ax2.transAxes, 
                    fontsize=11, color='#333333')
        elif assumption:
            # Regular text
            ax2.text(0.05, 0.95 - i*0.06, assumption, transform=ax2.transAxes, 
                    fontsize=11, color='#666666')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'limitations_domain.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_limitations_theoretical():
    """Slide 21: Limitations - Theoretical assumptions"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Theoretical Assumptions & Limitations', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    
    # Mathematical assumptions
    math_box = FancyBboxPatch((0.5, 6.5), 5, 2.5, boxstyle="round,pad=0.2",
                             facecolor='#fff0f0', edgecolor='#d62728', linewidth=2)
    ax.add_patch(math_box)
    ax.text(3, 8.5, 'Mathematical Assumptions', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#d62728')
    
    math_assumptions = [
        '• L-Lipschitz continuity (L ≤ C₁)',
        '• Bounded gradient variance (σ² ≤ C₂)', 
        '• μ-strongly convex physics constraints',
        '• Standard regularity conditions',
        '• Finite sample approximation validity'
    ]
    
    for i, assumption in enumerate(math_assumptions):
        ax.text(1, 8.2 - i*0.3, assumption, ha='left', va='center', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    
    # Performance degradation scenarios
    perf_box = FancyBboxPatch((6.5, 6.5), 5, 2.5, boxstyle="round,pad=0.2",
                             facecolor='#f0f0ff', edgecolor='#1f77b4', linewidth=2)
    ax.add_patch(perf_box)
    ax.text(9, 8.5, 'Performance Degradation Scenarios', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#1f77b4')
    
    degradation_scenarios = [
        '• Highly nonlinear physics (turbulence)',
        '• Discontinuous boundary conditions',
        '• Multi-scale phenomena',
        '• Sparse or noisy training data',
        '• Domain shift between tasks'
    ]
    
    for i, scenario in enumerate(degradation_scenarios):
        ax.text(7, 8.2 - i*0.3, scenario, ha='left', va='center', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    
    # Convergence rate visualization
    convergence_box = FancyBboxPatch((0.5, 3.5), 11, 2.5, boxstyle="round,pad=0.2",
                                    facecolor='#f0fff0', edgecolor='#2ca02c', linewidth=2)
    ax.add_patch(convergence_box)
    ax.text(6, 5.5, 'Convergence Rate Dependencies', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#2ca02c')
    
    # Show how convergence depends on parameters
    ax.text(6, 4.8, r'$\mathbb{E}[\|\nabla \mathcal{L}(\theta_T)\|^2] \leq \frac{C_1}{T} + C_2 \sqrt{\frac{\log T}{T}}$', 
           ha='center', va='center', fontsize=14,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    dependencies = [
        'C₁ = L²data + λ²L²physics/μ  (depends on physics regularization)',
        'C₂ = λ²σ²physics  (depends on physics constraint variance)',
        'Rate improves with strong physics regularization (large λ, μ)'
    ]
    
    for i, dep in enumerate(dependencies):
        ax.text(6, 4.3 - i*0.3, dep, ha='center', va='center', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.1", facecolor="lightyellow", alpha=0.8))
    
    # Practical implications
    impl_box = FancyBboxPatch((0.5, 0.5), 11, 2.5, boxstyle="round,pad=0.2",
                             facecolor='#fff8f0', edgecolor='#ff7f0e', linewidth=2)
    ax.add_patch(impl_box)
    ax.text(6, 2.5, 'Practical Implications', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#ff7f0e')
    
    implications = [
        '• Method works best for well-conditioned physics problems',
        '• Performance may degrade for highly chaotic or discontinuous systems',
        '• Theoretical guarantees require careful hyperparameter tuning',
        '• Sample complexity benefits depend on physics regularization strength'
    ]
    
    for i, impl in enumerate(implications):
        ax.text(1, 2.1 - i*0.3, impl, ha='left', va='center', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'limitations_theoretical.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_future_work_domains():
    """Slide 22: Future work - Broader physics domains"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Future Work: Broader Physics Domains', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    
    # Timeline/roadmap
    timeline_y = 7.5
    
    # Timeline line
    ax.plot([1, 11], [timeline_y, timeline_y], 'k-', linewidth=3, alpha=0.7)
    
    # Timeline milestones
    milestones = [
        (2, 'Year 1\nHeat Transfer\nExtension', '#ff7f0e'),
        (4.5, 'Year 2\nStructural\nMechanics', '#2ca02c'),
        (7, 'Year 3\nElectromagnetics', '#d62728'),
        (9.5, 'Year 4\nMulti-Physics\nCoupling', '#9467bd')
    ]
    
    for x, text, color in milestones:
        # Milestone marker
        circle = Circle((x, timeline_y), 0.2, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Milestone box
        box = FancyBboxPatch((x-0.8, timeline_y+0.5), 1.6, 1.2, boxstyle="round,pad=0.1",
                            facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, timeline_y+1.1, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrow
        ax.arrow(x, timeline_y+0.3, 0, -0.2, head_width=0.1, head_length=0.05, 
                fc=color, ec=color)
    
    # Methodological improvements
    method_box = FancyBboxPatch((0.5, 4.5), 5, 2, boxstyle="round,pad=0.2",
                               facecolor='#e6f3ff', edgecolor='#1f77b4', linewidth=2)
    ax.add_patch(method_box)
    ax.text(3, 6, 'Methodological Improvements', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#1f77b4')
    
    improvements = [
        '• Hierarchical meta-learning for multi-scale',
        '• Domain adaptation techniques',
        '• Transfer learning across physics domains',
        '• Automated architecture search',
        '• Uncertainty quantification enhancement'
    ]
    
    for i, improvement in enumerate(improvements):
        ax.text(1, 5.7 - i*0.25, improvement, ha='left', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    
    # Technical challenges
    challenge_box = FancyBboxPatch((6.5, 4.5), 5, 2, boxstyle="round,pad=0.2",
                                  facecolor='#ffe6e6', edgecolor='#d62728', linewidth=2)
    ax.add_patch(challenge_box)
    ax.text(9, 6, 'Technical Challenges', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#d62728')
    
    challenges = [
        '• Different PDE types and formulations',
        '• Varying boundary condition complexity',
        '• Multi-physics coupling mechanisms',
        '• Computational scalability issues',
        '• Validation across diverse domains'
    ]
    
    for i, challenge in enumerate(challenges):
        ax.text(7, 5.7 - i*0.25, challenge, ha='left', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    
    # Expected outcomes
    outcome_box = FancyBboxPatch((2, 1.5), 8, 2, boxstyle="round,pad=0.2",
                                facecolor='#f0fff0', edgecolor='#2ca02c', linewidth=2)
    ax.add_patch(outcome_box)
    ax.text(6, 3, 'Expected Outcomes', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#2ca02c')
    
    outcomes = [
        '• Universal physics-informed meta-learning framework',
        '• Automated physics discovery across multiple domains',
        '• Significant computational savings for scientific computing',
        '• Enhanced capability for multi-physics simulations',
        '• Broader impact on scientific machine learning community'
    ]
    
    for i, outcome in enumerate(outcomes):
        ax.text(2.5, 2.7 - i*0.25, outcome, ha='left', va='center', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'future_work_domains.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_future_work_theoretical():
    """Slide 23: Future work - Theoretical extensions"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Future Work: Theoretical Extensions', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    
    # Central theoretical framework
    center_box = FancyBboxPatch((4.5, 4), 3, 2, boxstyle="round,pad=0.2",
                               facecolor='#1f77b4', alpha=0.8, edgecolor='#1f77b4', linewidth=3)
    ax.add_patch(center_box)
    ax.text(6, 5, 'Enhanced\nTheoretical\nFramework', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='white')
    
    # Theoretical extensions
    extensions = [
        (2, 8, 'Non-Convex\nPhysics\nConstraints', '#ff7f0e'),
        (10, 8, 'Stochastic\nPDE\nSystems', '#2ca02c'),
        (1, 5, 'Multi-Scale\nConvergence\nAnalysis', '#d62728'),
        (11, 5, 'Adaptive\nSample\nComplexity', '#9467bd'),
        (2, 2, 'Robustness\nGuarantees', '#8c564b'),
        (10, 2, 'Generalization\nBounds', '#e377c2')
    ]
    
    for x, y, text, color in extensions:
        # Extension box
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2, boxstyle="round,pad=0.1",
                            facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrow to center
        dx = 6 - x
        dy = 5 - y
        length = np.sqrt(dx**2 + dy**2)
        if length > 2:  # Only draw arrow if not too close
            dx_norm = dx / length * 1.2
            dy_norm = dy / length * 1.2
            
            ax.arrow(x + dx_norm*0.4, y + dy_norm*0.4, dx_norm*0.4, dy_norm*0.4,
                    head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.7)
    
    # Key theoretical questions
    questions_box = FancyBboxPatch((0.5, 0.2), 11, 1.2, boxstyle="round,pad=0.1",
                                  facecolor='#fffacd', edgecolor='#daa520', linewidth=2)
    ax.add_patch(questions_box)
    ax.text(6, 1.1, 'Key Theoretical Questions to Address', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='#b8860b')
    
    questions = [
        '• How do convergence rates change with non-convex physics constraints?',
        '• What are optimal sample complexity bounds for different PDE classes?',
        '• How can we guarantee robustness to domain shift and noise?'
    ]
    
    for i, question in enumerate(questions):
        ax.text(1, 0.8 - i*0.2, question, ha='left', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'future_work_theoretical.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_broader_impact():
    """Slide 24: Broader impact and applications"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Broader Impact and Applications', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    
    # Application domains map
    domains = [
        (2, 7.5, 'Climate\nModeling', '#1f77b4', 'Weather prediction\nClimate change studies'),
        (6, 8, 'Aerospace\nEngineering', '#ff7f0e', 'Aircraft design\nPropulsion systems'),
        (10, 7.5, 'Biomedical\nEngineering', '#2ca02c', 'Blood flow modeling\nDrug delivery'),
        (1.5, 5, 'Energy\nSystems', '#d62728', 'Renewable energy\nPower grid optimization'),
        (6, 5, 'Manufacturing', '#9467bd', 'Process optimization\nQuality control'),
        (10.5, 5, 'Materials\nScience', '#8c564b', 'Material discovery\nProperty prediction'),
        (2, 2.5, 'Environmental\nScience', '#e377c2', 'Pollution modeling\nEcosystem dynamics'),
        (6, 2, 'Automotive\nIndustry', '#7f7f7f', 'Vehicle aerodynamics\nEngine optimization'),
        (10, 2.5, 'Oil & Gas', '#bcbd22', 'Reservoir simulation\nPipeline design')
    ]
    
    for x, y, domain, color, applications in domains:
        # Domain circle
        circle = Circle((x, y), 0.8, facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y+0.2, domain, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x, y-0.3, applications, ha='center', va='center', fontsize=8, style='italic')
    
    # Central impact
    impact_circle = Circle((6, 5), 1.2, facecolor='gold', alpha=0.8, edgecolor='orange', linewidth=3)
    ax.add_patch(impact_circle)
    ax.text(6, 5, 'Physics-Informed\nMeta-Learning\nImpact', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='darkred')
    
    # Timeline of impact
    timeline_box = FancyBboxPatch((0.5, 0.2), 11, 1, boxstyle="round,pad=0.1",
                                 facecolor='#f0f8ff', edgecolor='#4682b4', linewidth=2)
    ax.add_patch(timeline_box)
    
    timeline_items = [
        'Short-term (1-2 years): Fluid dynamics applications',
        'Medium-term (3-5 years): Multi-physics simulations', 
        'Long-term (5+ years): Universal scientific computing framework'
    ]
    
    for i, item in enumerate(timeline_items):
        ax.text(1, 0.9 - i*0.2, item, ha='left', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'broader_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_conclusion_comparison():
    """Slide 25: Conclusion - Before/after comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Before (Traditional approach)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Before: Traditional PINNs', fontsize=16, fontweight='bold', color='#d62728')
    
    # Problems with traditional approach
    problems = [
        (5, 8.5, 'New Problem\nArrives', '#ffcccc'),
        (5, 7, 'Train from\nScratch', '#ffcccc'),
        (5, 5.5, '12.4 hours\nTraining', '#ffcccc'),
        (5, 4, '150 adaptation\nsteps', '#ffcccc'),
        (5, 2.5, '78.3% accuracy', '#ffcccc'),
        (5, 1, 'Repeat for each\nnew problem', '#ffcccc')
    ]
    
    for x, y, text, color in problems:
        box = FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='#d62728', linewidth=2)
        ax1.add_patch(box)
        ax1.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
        
        if y > 1.5:  # Don't draw arrow from last box
            ax1.arrow(x, y-0.5, 0, -0.6, head_width=0.2, head_length=0.1, 
                     fc='#d62728', ec='#d62728', linewidth=2)
    
    # After (Our approach)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('After: Physics-Informed Meta-Learning', fontsize=16, fontweight='bold', color='#2ca02c')
    
    # Benefits of our approach
    benefits = [
        (5, 8.5, 'New Problem\nArrives', '#ccffcc'),
        (5, 7, 'Fast Adaptation\nfrom Meta-Model', '#ccffcc'),
        (5, 5.5, '4.1 hours\nTraining', '#ccffcc'),
        (5, 4, '50 adaptation\nsteps', '#ccffcc'),
        (5, 2.5, '92.4% accuracy', '#ccffcc'),
        (5, 1, 'Leverage prior\nknowledge', '#ccffcc')
    ]
    
    for x, y, text, color in benefits:
        box = FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='#2ca02c', linewidth=2)
        ax2.add_patch(box)
        ax2.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
        
        if y > 1.5:  # Don't draw arrow from last box
            ax2.arrow(x, y-0.5, 0, -0.6, head_width=0.2, head_length=0.1, 
                     fc='#2ca02c', ec='#2ca02c', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'conclusion_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_technical_contributions():
    """Slide 26: Technical contributions summary"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Technical Contributions Summary', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    
    # Main contributions with connections
    contributions = [
        (3, 7.5, 'Novel Algorithm\nPhysics + Meta-Learning', '#1f77b4'),
        (9, 7.5, 'Theoretical Analysis\nConvergence Guarantees', '#ff7f0e'),
        (3, 5.5, 'Adaptive Weighting\nTask-Specific λ(𝒯)', '#2ca02c'),
        (9, 5.5, 'Physics Discovery\nAutomated + Interpretable', '#d62728'),
        (3, 3.5, 'Experimental Validation\n92.4% Accuracy', '#9467bd'),
        (9, 3.5, 'Statistical Rigor\n95% CI, p < 0.001', '#8c564b')
    ]
    
    for x, y, text, color in contributions:
        # Contribution box
        box = FancyBboxPatch((x-1.2, y-0.6), 2.4, 1.2, boxstyle="round,pad=0.1",
                            facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Central framework connection
    center_box = FancyBboxPatch((5, 1.5), 2, 1, boxstyle="round,pad=0.1",
                               facecolor='gold', alpha=0.8, edgecolor='orange', linewidth=3)
    ax.add_patch(center_box)
    ax.text(6, 2, 'Unified\nFramework', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='darkred')
    
    # Draw connections
    for x, y, _, color in contributions:
        # Arrow to center
        dx = 6 - x
        dy = 2 - y
        length = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / length * 1.5
        dy_norm = dy / length * 1.5
        
        ax.arrow(x + dx_norm*0.3, y + dy_norm*0.3, dx_norm*0.4, dy_norm*0.4,
                head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.7)
    
    # Key quantitative achievements
    achievements_box = FancyBboxPatch((0.5, 0.2), 11, 0.8, boxstyle="round,pad=0.1",
                                     facecolor='#f0f8ff', edgecolor='#4682b4', linewidth=2)
    ax.add_patch(achievements_box)
    
    achievements = [
        '15% accuracy improvement • 3× faster adaptation • 67% training time reduction • 94% physics discovery accuracy'
    ]
    
    ax.text(6, 0.6, achievements[0], ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'technical_contributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_research_impact():
    """Slide 27: Impact on physics-informed machine learning"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Impact on Physics-Informed Machine Learning', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    
    # Central impact node
    center = Circle((6, 5), 1, facecolor='#1f77b4', alpha=0.8, edgecolor='#1f77b4', linewidth=3)
    ax.add_patch(center)
    ax.text(6, 5, 'PI-MAML\nFramework', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='white')
    
    # Connected research areas
    research_areas = [
        (2, 8, 'Meta-Learning\nCommunity', '#ff7f0e', 'Extends MAML to\nphysics domains'),
        (10, 8, 'Scientific ML\nCommunity', '#2ca02c', 'Enables few-shot\nphysics learning'),
        (1, 5, 'PINN\nResearch', '#d62728', 'Adds adaptation\ncapabilities'),
        (11, 5, 'Automated\nDiscovery', '#9467bd', 'Physics-aware\nsymbolic regression'),
        (2, 2, 'Computational\nFluid Dynamics', '#8c564b', 'Faster simulation\nsetup and solving'),
        (10, 2, 'Transfer Learning\nin Science', '#e377c2', 'Cross-domain\nknowledge transfer')
    ]
    
    for x, y, area, color, impact in research_areas:
        # Research area node
        circle = Circle((x, y), 0.8, facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y+0.1, area, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x, y-0.4, impact, ha='center', va='center', fontsize=8, style='italic')
        
        # Connection to center
        dx = 6 - x
        dy = 5 - y
        length = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Draw bidirectional connection
        ax.plot([x + dx_norm*0.8, 6 - dx_norm*1], [y + dy_norm*0.8, 5 - dy_norm*1], 
               color=color, linewidth=3, alpha=0.7)
        
        # Add arrowheads
        ax.arrow(x + dx_norm*1.2, y + dy_norm*1.2, dx_norm*0.3, dy_norm*0.3,
                head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.7)
        ax.arrow(6 - dx_norm*1.3, 5 - dy_norm*1.3, -dx_norm*0.3, -dy_norm*0.3,
                head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.7)
    
    # Future research directions
    future_box = FancyBboxPatch((3, 0.2), 6, 0.8, boxstyle="round,pad=0.1",
                               facecolor='#fffacd', edgecolor='#daa520', linewidth=2)
    ax.add_patch(future_box)
    ax.text(6, 0.6, 'Opens New Research Directions: Multi-physics coupling, Hierarchical meta-learning,\nRobust physics discovery, Universal scientific computing frameworks', 
           ha='center', va='center', fontsize=11, fontweight='bold', color='#b8860b')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'research_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_questions_discussion():
    """Slide 28: Questions and discussion"""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 8.5, 'Questions & Discussion', ha='center', va='center', 
            fontsize=24, fontweight='bold', color='#1f77b4')
    
    # Key contributions summary
    ax.text(5, 7, 'Key Contributions:', ha='center', va='center', 
           fontsize=16, fontweight='bold', color='#333333')
    
    contributions = [
        '✓ 92.4% validation accuracy with 95% confidence intervals',
        '✓ 3× faster adaptation (50 vs 150 steps)',
        '✓ 67% reduction in training time (12.4h → 4.1h)', 
        '✓ 94% automated physics discovery accuracy',
        '✓ Theoretical convergence guarantees'
    ]
    
    for i, contrib in enumerate(contributions):
        ax.text(5, 6.3 - i*0.4, contrib, ha='center', va='center', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8f5e8", alpha=0.8))
    
    # Contact information
    contact_box = FancyBboxPatch((1.5, 2.5), 7, 1.5, boxstyle="round,pad=0.2",
                                facecolor='#f0f8ff', edgecolor='#4682b4', linewidth=2)
    ax.add_patch(contact_box)
    
    ax.text(5, 3.6, 'Contact Information', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#4682b4')
    
    ax.text(5, 3.1, 'Brandon Yee: b.yee@ycrg-labs.org', ha='center', va='center', fontsize=12)
    ax.text(5, 2.8, 'GitHub: https://github.com/YCRG-Labs/meta-pinn', ha='center', va='center', 
           fontsize=12, color='#0066cc')
    
    # Thank you
    ax.text(5, 1.5, 'Thank you for your attention!', ha='center', va='center', 
           fontsize=18, fontweight='bold', color='#2ca02c',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fff0", alpha=0.8))
    
    # QR code placeholder (would need qrcode library for actual QR code)
    qr_box = Rectangle((7.5, 0.5), 1.5, 1.5, facecolor='lightgray', edgecolor='black')
    ax.add_patch(qr_box)
    ax.text(8.25, 1.25, 'QR Code\nGitHub', ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'questions_discussion.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def main():
    """Generate all conceptual figures"""
    print("Generating conceptual figures...")
    
    # Create all conceptual visualizations
    create_title_slide()
    print("✓ Title slide")
    
    create_pinn_limitations()
    print("✓ PINN limitations")
    
    create_fluid_dynamics_examples()
    print("✓ Fluid dynamics examples")
    
    create_meta_learning_concept()
    print("✓ Meta-learning concept")
    
    create_research_contributions()
    print("✓ Research contributions")
    
    create_problem_formulation()
    print("✓ Problem formulation")
    
    create_framework_flowchart()
    print("✓ Framework flowchart")
    
    create_physics_loss_diagram()
    print("✓ Physics loss diagram")
    
    create_adaptive_weighting()
    print("✓ Adaptive weighting")
    
    create_experimental_setup()
    print("✓ Experimental setup")
    
    create_limitations_domain()
    print("✓ Limitations - domain")
    
    create_limitations_theoretical()
    print("✓ Limitations - theoretical")
    
    create_future_work_domains()
    print("✓ Future work - domains")
    
    create_future_work_theoretical()
    print("✓ Future work - theoretical")
    
    create_broader_impact()
    print("✓ Broader impact")
    
    create_conclusion_comparison()
    print("✓ Conclusion comparison")
    
    create_technical_contributions()
    print("✓ Technical contributions")
    
    create_research_impact()
    print("✓ Research impact")
    
    create_questions_discussion()
    print("✓ Questions and discussion")
    
    print(f"\nAll conceptual figures saved to: {output_dir.absolute()}")
    print("\nGenerated figures:")
    for fig_file in sorted(output_dir.glob("*.png")):
        print(f"  - {fig_file.name}")

if __name__ == "__main__":
    main()