'use client'

import React, { useEffect, useState } from 'react'

// Add these types at the top of the file, after the imports
interface FlowData {
  u_velocity: string;
  v_velocity: string;
  pressure: string;
  vorticity: string;
}

interface ApiMetadata {
  grid_shape: [number, number];
  [key: string]: any;
}

interface ApiResponse {
  success: boolean;
  data: FlowData[];
  metadata: ApiMetadata;
}

const FluidViscosityExplainer = () => {
  const [isScrolled, setIsScrolled] = useState(false)
  const [selectedLevel, setSelectedLevel] = useState('')
  const [showContent, setShowContent] = useState(false)
  const [isClient, setIsClient] = useState(false)
  const [apiData, setApiData] = useState<ApiResponse | null>(null)
  const [loadingData, setLoadingData] = useState(false)
  const [apiError, setApiError] = useState<string | null>(null)
  const [fadeOutOverlay, setFadeOutOverlay] = useState(false)
  const [selectedScenario, setSelectedScenario] = useState<string>('')
  const [scenarios, setScenarios] = useState<any[]>([])

  // Constants (not user-editable)
  const BACKEND_URL = "http://localhost:8000";
  const MODEL_PATH = "backend/results/trained_model.pth";

  // State for parameter inputs (sliders)
  const [reynoldsNumber, setReynoldsNumber] = useState<number>(100);
  const [nuBaseTrue, setNuBaseTrue] = useState<number>(0.1);
  const [aTrue, setATrue] = useState<number>(0.05);
  const [uMaxInlet, setUMaxInlet] = useState<number>(1.0);
  const [xMax] = useState<number>(2.0);
  const [yMax] = useState<number>(1.0);
  const [xMin] = useState<number>(0.0);
  const [yMin] = useState<number>(0.0);
  const [nGridX] = useState<number>(25);
  const [nGridY] = useState<number>(25);
  const [nTimeSlices] = useState<number>(5);
  const [name] = useState<string>("Frontend Visualization");
  
  // Add effect to log state changes
  useEffect(() => {
    console.log('State updated:', {
      reynoldsNumber,
      nuBaseTrue,
      aTrue,
      uMaxInlet
    });
  }, [reynoldsNumber, nuBaseTrue, aTrue, uMaxInlet]);

  // Fix hydration by ensuring client-side rendering
  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    if (!isClient) return

    const handleScroll = () => {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop
      setIsScrolled(scrollTop > 50)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [isClient])

  // Fetch available scenarios on component mount
  useEffect(() => {
    const fetchScenarios = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/scenarios`);
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.scenarios) {
            setScenarios(data.scenarios);
            console.log('Fetched scenarios:', data.scenarios); // Debug log
          } else {
            console.error('Invalid response format:', data);
          }
        } else {
          console.error('Failed to fetch scenarios:', response.status);
        }
      } catch (error) {
        console.error('Error fetching scenarios:', error);
      }
    };
    fetchScenarios();
  }, []);

  // Enhanced fetch function with better error handling
  const fetchPINNData = async () => {
    if (!selectedScenario) {
      setApiError('Please select a scenario first');
      return;
    }

    setLoadingData(true);
    setApiError(null);

    try {
      // First get the scenario metadata
      const metadataResponse = await fetch(`${BACKEND_URL}/scenarios/${selectedScenario}`);
      if (!metadataResponse.ok) {
        throw new Error(`Failed to fetch scenario metadata: ${metadataResponse.status}`);
      }
      const metadata = await metadataResponse.json();

      // Then get the flow field data
      const dataResponse = await fetch(`${BACKEND_URL}/scenarios/${selectedScenario}/data/inferred_flow_3d_complete.csv`);
      if (!dataResponse.ok) {
        throw new Error(`Failed to fetch flow data: ${dataResponse.status}`);
      }
      const data = await dataResponse.json();

      if (data.success) {
        setApiData({
          ...data,
          metadata: metadata
        });
      } else {
        throw new Error(data.error_message || 'API returned error');
      }
    } catch (error: any) {
      console.error('Error fetching PINN data:', error);
      setApiError(error.message || error.toString());
    } finally {
      setLoadingData(false);
    }
  };

  // Test backend connection
  const testBackendConnection = async () => {
    if (!BACKEND_URL) {
        console.warn('Backend URL is not set.');
        return false;
    }
    try {
      const response = await fetch(`${BACKEND_URL}/health`);
      if (response.ok) {
        console.log('Backend connection successful');
        alert('Backend connection successful!');
        return true;
      } else {
        console.warn('Backend health check failed');
        alert(`Backend health check failed: ${response.status}`);
        return false;
      }
    } catch (error) {
      console.error('Backend connection failed:', error);
      alert(`Backend connection failed: ${error}`);
      return false;
    }
  };

  // Load Plotly and create plots
  useEffect(() => {
    if (!isClient || typeof window === 'undefined' || !showContent) return;

    const loadScript = (src: string) => {
      return new Promise<void>((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) {
          resolve(undefined);
          return;
        }
        const script = document.createElement('script');
        script.src = src;
        script.onload = () => resolve(undefined);
        script.onerror = reject;
        document.head.appendChild(script);
      });
    };

    const initializePlots = async () => {
      try {
        if (!(window as any).d3) {
          await loadScript('https://d3js.org/d3.v5.min.js');
        }

        const PlotlyModule = await import('plotly.js-dist');
        const Plotly = PlotlyModule.default;

        if (apiData && apiData.data) {
          const { data, metadata } = apiData;
          const gridShape = metadata.grid_shape || [120, 60]; // Default grid shape if not provided
          const [nx, ny] = gridShape;

          // Extract data from the CSV
          const uVelocity = data.map((row: any) => parseFloat(row.u_velocity));
          const vVelocity = data.map((row: any) => parseFloat(row.v_velocity));
          const pressure = data.map((row: any) => parseFloat(row.pressure));
          const vorticity = data.map((row: any) => parseFloat(row.vorticity));

          // Calculate velocity magnitude
          const velocityMagnitude = uVelocity.map((u: number, i: number) => 
            Math.sqrt(u * u + vVelocity[i] * vVelocity[i])
          );

          // Reshape data into 2D grids
          const uVelocityGrid = reshapeToGrid(uVelocity, nx, ny);
          const vVelocityGrid = reshapeToGrid(vVelocity, nx, ny);
          const pressureGrid = reshapeToGrid(pressure, nx, ny);
          const vorticityGrid = reshapeToGrid(vorticity, nx, ny);
          const velocityMagnitudeGrid = reshapeToGrid(velocityMagnitude, nx, ny);

          const commonLayoutProps = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            scene: {
              bgcolor: 'rgba(0,0,0,0)',
              xaxis: { title: 'X Position', gridcolor: 'white', color: 'white' },
              yaxis: { title: 'Y Position', gridcolor: 'white', color: 'white' },
              zaxis: { gridcolor: 'white', color: 'white' }
            },
            autosize: true,
            margin: { l: 0, r: 0, b: 0, t: 50 }
          };

          // Plot U Velocity
          if (document.getElementById('velocityPlot')) {
            Plotly.newPlot('velocityPlot', [{
              z: uVelocityGrid,
              type: 'surface',
              colorscale: 'Viridis',
              name: 'U Velocity'
            }], {
              ...commonLayoutProps,
              title: { text: 'U Velocity Field (3D Surface)', font: { color: 'white' } },
              scene: { ...commonLayoutProps.scene, zaxis: { ...commonLayoutProps.scene.zaxis, title: 'Velocity' } }
            }, { responsive: true });
          }

          // Plot Pressure
          if (document.getElementById('pressurePlot')) {
            Plotly.newPlot('pressurePlot', [{
              z: pressureGrid,
              type: 'surface',
              colorscale: 'RdBu',
              name: 'Pressure'
            }], {
              ...commonLayoutProps,
              title: { text: 'Pressure Field (3D Surface)', font: { color: 'white' } },
              scene: { ...commonLayoutProps.scene, zaxis: { ...commonLayoutProps.scene.zaxis, title: 'Pressure' } }
            }, { responsive: true });
          }

          // Plot Velocity Magnitude
          if (document.getElementById('velocityMagPlot')) {
            Plotly.newPlot('velocityMagPlot', [{
              z: velocityMagnitudeGrid,
              type: 'surface',
              colorscale: 'Viridis',
              name: 'Velocity Magnitude'
            }], {
              ...commonLayoutProps,
              title: { text: 'Velocity Magnitude (3D Surface)', font: { color: 'white' } },
              scene: { ...commonLayoutProps.scene, zaxis: { ...commonLayoutProps.scene.zaxis, title: 'Velocity Magnitude' } }
            }, { responsive: true });
          }

          // Plot Vorticity
          if (document.getElementById('vorticityPlot')) {
            const maxAbsVorticity = Math.max(...vorticityGrid.flat().map(Math.abs));
            Plotly.newPlot('vorticityPlot', [{
              z: vorticityGrid,
              type: 'surface',
              colorscale: 'RdBu',
              name: 'Vorticity',
              zmin: -maxAbsVorticity,
              zmax: maxAbsVorticity
            }], {
              ...commonLayoutProps,
              title: { text: 'Vorticity Field (3D Surface)', font: { color: 'white' } },
              scene: { ...commonLayoutProps.scene, zaxis: { ...commonLayoutProps.scene.zaxis, title: 'Vorticity' } }
            }, { responsive: true });
          }
        }
      } catch (error) {
        console.error('Error loading/initializing Plotly:', error);
      }
    };

    initializePlots();
  }, [isClient, apiData, showContent]);

  const reshapeToGrid = (dataArray: number[], nx: number, ny: number) => {
    const grid: number[][] = [];
    if (!dataArray || dataArray.length !== nx * ny) {
      console.warn("Data array is null, undefined, or has incorrect length for reshaping. Using empty values.");
      for (let i = 0; i < ny; i++) {
        const row: number[] = [];
        for (let j = 0; j < nx; j++) {
          row.push(0);
        }
        grid.push(row);
      }
      return grid;
    }
    for (let i = 0; i < ny; i++) {
      const row: number[] = [];
      for (let j = 0; j < nx; j++) {
        const idx = i * nx + j;
        row.push(dataArray[idx]);
      }
      grid.push(row);
    }
    return grid;
  };

  const handleLevelSelect = (level: string) => {
    setSelectedLevel(level);
    setFadeOutOverlay(true);
    setTimeout(() => {
      setShowContent(true);
      setFadeOutOverlay(false);
      setTimeout(() => {
        const el = document.getElementById('explanation-content');
        if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 50); // slight delay to ensure content is rendered
    }, 500); // Duration matches the transition
  }

  const beginnerContent = {
    title: "Understanding Fluid Viscosity - Beginner Level",
    sections: [
      {
        title: "🍯 What is Viscosity?",
        content: `Think of viscosity as how "thick" or "sticky" a liquid is:
• Water has low viscosity - it flows easily
• Honey has high viscosity - it flows slowly and sticks
• Motor oil is in between

In real life, liquids don't always have the same thickness everywhere - like how honey might be thicker when it's cold at the bottom of the jar.`
      },
      {
        title: "🤖 What This Research Does",
        content: `Scientists wanted to create a smart computer program that can:
1. Watch how a liquid flows (like water through a pipe)
2. Figure out if the liquid is thicker in some places than others
3. Do this without having to stick sensors everywhere in the liquid

It's like being a detective - you see how the liquid moves and guess its "thickness recipe."`
      },
      {
        title: "🧠 The Smart Computer (Neural Network)",
        content: `They used artificial intelligence called a "neural network" - think of it as a very smart computer brain that:
• Learns patterns by looking at examples
• Follows the rules of physics (like how liquids must behave)
• Makes educated guesses about what it can't directly see

It's like teaching a computer to be a liquid flow expert!`
      },
      {
        title: "🎯 The Results",
        content: `The good news: The computer got really good at predicting how the liquid flows!
The challenge: It wasn't great at figuring out the exact "thickness recipe."

Why? Imagine trying to guess a cake recipe just by looking at the finished cake - it's really hard! The liquid flow might look similar even with different thickness patterns.`
      },
      {
        title: "🔬 Why This Matters",
        content: `This research helps us understand liquids in:
• Blood flow in our bodies (thicker in some arteries)
• Oil flowing through pipelines
• Paint or chocolate flowing in factories
• Weather patterns in the atmosphere

Better understanding means better designs for everything from medical devices to manufacturing!`
      }
    ]
  }

  const intermediateContent = {
    title: "Physics-Informed Neural Networks for Viscosity Inference - Intermediate Level",
    sections: [
      {
        title: "📊 The Mathematical Foundation",
        content: `This research tackles fluid dynamics using the Navier-Stokes equations - the fundamental equations that describe how fluids move:

• **Momentum equations**: How forces cause fluid motion
• **Continuity equation**: Mass conservation (fluid doesn't disappear)
• **Viscosity model**: ν(y) = νbase + a·y (linear variation)

The key innovation is inferring the parameter 'a' (viscosity gradient) from sparse flow measurements.`
      },
      {
        title: "🧮 Physics-Informed Neural Networks (PINNs)",
        content: `PINNs are special because they combine:

**Data-driven learning**: Learn from actual measurements
**Physics constraints**: Must obey fundamental laws
**Inverse problem solving**: Work backwards from effects to causes

The loss function includes three terms:
• PDE residuals (physics compliance)
• Boundary conditions (realistic constraints)
• Data fitting (match observations)

This ensures the AI solution is both accurate and physically meaningful.`
      },
      {
        title: "🔧 Advanced Training Techniques",
        content: `The researchers used sophisticated methods to improve training:

**Fourier Feature Embeddings**: Help the network learn high-frequency patterns
**Adaptive Loss Weighting**: Automatically balance different objectives
**Curriculum Learning**: Start simple, gradually increase complexity
**Parameter Re-initialization**: Escape local minima in optimization

These techniques address common challenges in training neural networks for physics problems.`
      },
      {
        title: "📈 Experimental Setup",
        content: `**Test Case**: 2D channel flow (like flow between parallel plates)
**Reynolds Number**: 100 (moderate flow speed)
**Data**: 100 sparse measurement points
**Domain**: 2×1 rectangular channel
**True Parameter**: a = 0.05 (small viscosity gradient)

The sparse data simulates realistic scenarios where you can't measure everywhere.`
      },
      {
        title: "⚠️ Key Findings & Challenges",
        content: `**Success**: Excellent flow field reconstruction with low PDE residuals (~10⁻⁵)
**Challenge**: Poor parameter estimation (inferred a ≈ 1.195 vs. true a = 0.05)

**Why this happens**:
• Ill-posed inverse problem (multiple solutions possible)
• Weak parameter sensitivity in sparse data
• Complex optimization landscape
• Parameter-field coupling effects

This highlights the difference between fitting data and correctly identifying underlying physics.`
      }
    ]
  }

  const expertContent = {
    title: "Advanced Analysis: Spatial Viscosity Inference via PINNs - Expert Level",
    sections: [
      {
        title: "🔬 Mathematical Formulation & Non-dimensionalization",
        content: `**Governing Equations** (steady, incompressible, 2D):

Momentum: ρ(u·∇)u = -∇P + ∇·τ
Continuity: ∇·u = 0
Constitutive: τ = 2μS, where μ = ρν(y)

**Non-dimensional form** with characteristic scales (Lc, Uc, νbase,true):
Rx = uux + vuy + Px - (1/Rebase)Vx(u,v,ν̃) = 0
Ry = ubx + vvy + Py - (1/Rebase)Vy(u,v,ν̃) = 0
Rc = ux + vy = 0

Where ν̃(y) = 1 + ã·y and ã = a·Lc/νbase,true is the target parameter.`
      },
      {
        title: "⚡ Network Architecture & Feature Engineering",
        content: `**MLP Structure**: [2, 64, 128, 128, 64, 3] with tanh activation
**Input**: (x,y) → (û,v̂,P̂)
**Parameter**: ã as additional trainable scalar

**Fourier Feature Embeddings**:
γ(x) = [cos(2πBx), sin(2πBx)]ᵀ
Maps R² → R²ᵐ to mitigate spectral bias

**Automatic Differentiation**: Essential for computing higher-order derivatives:
Vx = ν̃(uxx + uyy) + ν̃yuy
Vy = ν̃(vxx + vyy) + ν̃yvy
where ν̃y = ∂ν̃/∂y = ã`
      },
      {
        title: "🎯 Loss Function Architecture & Optimization",
        content: `**Multi-objective Loss**:
Ltotal = λPDE·LPDE + λBC·LBC + λdata·Ldata

**Adaptive Weighting Schemes**:
• Gradient-based (Wang et al.): λᵢ ∝ |∇θLᵢ|
• Uncertainty-based (Kendall et al.): Balance homoscedastic/heteroscedastic

**Advanced Sampling**:
• Latin Hypercube/Sobol for collocation points
• Adaptive refinement in high-residual regions

**Optimization Strategy**:
Adam with exponential LR decay + curriculum learning + periodic re-initialization`
      },
      {
        title: "📊 Identifiability Analysis & Sensitivity",
        content: `**Parameter Sensitivity Matrix**:
S = ∂u/∂ã evaluated at measurement locations

**Fisher Information Matrix**:
F = SᵀΣ⁻¹S (where Σ is measurement covariance)

**Practical Identifiability**:
cond(F) and eigenvalue spectrum indicate parameter estimability

**Key Issue**: ã appears only through ν̃y = ã in viscous terms
Linear coupling → weak sensitivity, especially with sparse data
Multiple (û,v̂,P̂,ã) combinations can satisfy PDEs with similar accuracy`
      },
      {
        title: "🔍 Results Analysis & Inverse Problem Pathology",
        content: `**Quantitative Performance**:
- PDE residuals: O(10⁻⁴) - O(10⁻⁵) ✓
- Flow field MSE: Low, physically consistent ✓
- Parameter error: |ãinf - ãtrue|/|ãtrue| = 2290% ✗

**Root Causes**:
1. **Non-uniqueness**: Multiple ã values yield similar flow patterns
2. **Regularization deficiency**: No prior constraints on ã
3. **Information content**: Nd=100 insufficient for unique identification
4. **Compensation mechanisms**: Network adjusts flow fields to maintain PDE compliance

**Theoretical Implications**:
This exemplifies classical inverse problem pathology where data fitting ≠ parameter recovery`
      },
      {
        title: "🚀 Future Directions & Methodological Improvements",
        content: `**Enhanced Identifiability**:
• Optimal sensor placement (D-optimal design)
• Multi-physics constraints (temperature coupling)
• Temporal data incorporation

**Bayesian Framework**:
• Uncertainty quantification via variational inference
• Prior regularization on parameter space
• Ensemble methods for robustness

**Advanced Architectures**:
• Multi-fidelity networks
• Domain decomposition PINNs
• Operator learning approaches

**Regularization Strategies**:
• Sobolev space constraints
• Maximum entropy regularization
• Physics-informed priors`
      }
    ]
  }

  const getContent = () => {
    switch (selectedLevel) {
      case 'beginner': return beginnerContent
      case 'intermediate': return intermediateContent
      case 'expert': return expertContent
      default: return null
    }
  }

  const currentContent = getContent()

  const renderLearningLevelButtons = () => (
    <div className="flex flex-col md:flex-row justify-center gap-4 md:gap-8">
      <button
        onClick={() => handleLevelSelect('beginner')}
        className="px-8 py-4 backdrop-blur-xl border border-white/30 rounded-full font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-xl bg-blue-600/20 text-white border-blue-400 shadow-blue-500/50"
      >
        🌟 Beginner
      </button>
      <button
        onClick={() => handleLevelSelect('intermediate')}
        className="px-8 py-4 backdrop-blur-xl border border-white/30 rounded-full font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-xl bg-green-600/20 text-white border-green-400 shadow-green-500/50"
      >
        🎓 Intermediate
      </button>
      <button
        onClick={() => handleLevelSelect('expert')}
        className="px-8 py-4 backdrop-blur-xl border border-white/30 rounded-full font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-xl bg-purple-600/20 text-white border-purple-400 shadow-purple-500/50"
      >
        🔬 Expert
      </button>
    </div>
  );

  // Replace the sliders section with this dropdown menu
  const renderScenarioSelector = () => (
    <div className="mb-4">
      <label htmlFor="scenario-select" className="block text-sm font-medium text-blue-200 mb-1">
        Select Scenario:
      </label>
      <select
        id="scenario-select"
        value={selectedScenario}
        onChange={(e) => setSelectedScenario(e.target.value)}
        className="w-full p-2 rounded-lg bg-gray-100 border border-white/20 text-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="">Select a scenario...</option>
        {scenarios.map((scenario) => (
          <option key={scenario.id} value={scenario.id}>
            {scenario.name} - {scenario.description}
          </option>
        ))}
      </select>
      {scenarios.length === 0 && (
        <p className="mt-2 text-yellow-400 text-sm">
          Loading scenarios...
        </p>
      )}
    </div>
  );

  if (!isClient) {
    return <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 flex items-center justify-center">
      <div className="text-white text-xl">Loading...</div>
    </div>
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900" style={{
      scrollSnapType: 'y mandatory'
    }}>
      {!showContent && (
        <div
          className={`fixed inset-0 z-50 flex flex-col items-center justify-center bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 bg-opacity-95 transition-opacity duration-400 ${fadeOutOverlay ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}
          style={{ minHeight: '100vh' }}
        >
          <div className="max-w-2xl w-full p-8 rounded-2xl shadow-2xl bg-white/10 border border-white/20 text-center">
            <h2 className="text-4xl font-bold text-white mb-8">Choose Your Learning Level</h2>
            {renderLearningLevelButtons()}
          </div>
        </div>
      )}

      {/* Header/Navbar */}
      <header className="sticky top-0 z-40 w-full py-6 px-4 bg-gradient-to-r from-blue-800/80 via-indigo-900/80 to-purple-900/80 shadow-lg flex items-center justify-center">
        <h1 className="font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-300 via-white to-indigo-300 text-3xl md:text-4xl tracking-tight">
          Inferring Spatial Fluid Viscosity
        </h1>
      </header>

      {/* Content Section */}
      <div className={`transition-all duration-1000`}>
        <section className="relative py-16 px-4 min-h-screen flex items-center" style={{
          scrollSnapAlign: 'start'
        }}>
          <div className="w-full">
            <h2 className="text-4xl font-bold text-white mb-8 text-center">Choose Your Learning Level</h2>
            <div className="max-w-4xl mx-auto">
              <div className="border border-white/10 rounded-3xl p-8 md:p-12 shadow-2xl backdrop-blur-md bg-white/5 text-center">
                {renderLearningLevelButtons()}
              </div>
            </div>
          </div>
        </section>

        {/* Explanation Content & Visualization - only shown after level selection */}
        {showContent && currentContent && (
          <>
            <section id="explanation-content" className="py-30 px-4 min-h-screen" style={{
              scrollSnapAlign: 'start'
            }}>
              <div className="max-w-6xl mx-auto">
                <h2 className="text-5xl font-bold text-white mb-10 text-center">
                  {currentContent.title}
                </h2>

                <div className="space-y-12">
                  {currentContent.sections.map((section, index) => (
                    <div
                      key={index}
                      className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-8 shadow-xl transform transition-all duration-500"
                    >
                      <h3 className="text-2xl font-bold text-blue-300 mb-6 flex items-center gap-3">
                        {section.title}
                      </h3>
                      <div className="text-gray-100 leading-relaxed whitespace-pre-line text-lg">
                        {section.content}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </section>

            {/* Paper Summary */}
            <section className="py-16 px-4 min-h-screen flex items-center" style={{
              scrollSnapAlign: 'start'
            }}>
              <div className="max-w-6xl mx-auto w-full">
                <div className="backdrop-blur-xl bg-gradient-to-r from-blue-900/30 to-purple-900/30 border border-white/20 rounded-2xl p-8 shadow-xl">
                  <h3 className="text-3xl font-bold text-center text-white mb-6">Key Takeaways from the Research</h3>
                  <div className="grid md:grid-cols-2 gap-8 text-gray-100">
                    <div>
                      <h4 className="text-xl font-semibold text-green-300 mb-4">✅ What Worked Well</h4>
                      <ul className="space-y-2 text-lg list-disc list-inside">
                        <li>Excellent flow field reconstruction (velocity, pressure).</li>
                        <li>Low PDE residuals, meaning solutions align with physics.</li>
                        <li>Robust neural network training using advanced techniques.</li>
                        <li>Successful application of Fourier features & adaptive weights.</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="text-xl font-semibold text-orange-300 mb-4">⚠️ Key Challenges & Limitations</h4>
                      <ul className="space-y-2 text-lg list-disc list-inside">
                        <li>Poor parameter estimation for the viscosity gradient ('a').</li>
                        <li>Inverse problem was ill-posed with sparse data.</li>
                        <li>Difficulty in distinguishing true parameters from compensatory effects.</li>
                        <li>Model struggled with identifiability of the spatial viscosity.</li>
                      </ul>
                    </div>
                  </div>
                   <p className="text-center mt-8 text-blue-200 text-lg">
                    This highlights a common challenge: matching observed data doesn't always guarantee recovery of the true underlying physical parameters, especially in complex systems with limited measurements.
                  </p>
                </div>
              </div>
            </section>

            {/* Data Visualization Interface Section */}
            <section className="py-16 px-4 min-h-screen flex flex-col items-center justify-center" style={{ scrollSnapAlign: 'start' }}>
              <div className="max-w-7xl mx-auto w-full">
                <h2 className="text-4xl font-bold text-white mb-10 text-center">Live PINN Inference & Visualization</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
                  {/* Parameter Inputs Column */}
                  <div className="md:col-span-1 backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-6 shadow-xl">
                    <h3 className="text-2xl font-semibold text-blue-300 mb-6">Select Scenario</h3>
                    {renderScenarioSelector()}
                    <button
                      onClick={fetchPINNData}
                      disabled={loadingData || !selectedScenario}
                      className="w-full mt-6 bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:shadow-lg transition duration-150 ease-in-out disabled:opacity-50 flex items-center justify-center"
                    >
                      {loadingData ? (
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                      ) : "🚀 Load Scenario"}
                    </button>
                    <button
                      onClick={testBackendConnection}
                      className="w-full mt-3 bg-gradient-to-r from-gray-500 to-gray-600 hover:from-gray-600 hover:to-gray-700 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:shadow-lg transition duration-150 ease-in-out"
                    >
                      🔌 Test Backend
                    </button>
                    {apiError && <p className="mt-4 text-red-400 text-sm">Error: {apiError}</p>}
                  </div>

                  {/* Plots Column */}
                  <div className="md:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div id="velocityPlot" className="w-full h-[400px] bg-white/5 border border-white/10 rounded-xl shadow-xl p-2"></div>
                    <div id="pressurePlot" className="w-full h-[400px] bg-white/5 border border-white/10 rounded-xl shadow-xl p-2"></div>
                    <div id="velocityMagPlot" className="w-full h-[400px] bg-white/5 border border-white/10 rounded-xl shadow-xl p-2"></div>
                    <div id="vorticityPlot" className="w-full h-[400px] bg-white/5 border border-white/10 rounded-xl shadow-xl p-2"></div>
                  </div>
                </div>
                { !apiData && !loadingData && (
                    <p className="text-center text-blue-200 text-lg mt-8">
                        Select a learning level and adjust parameters, then click "Run Inference" to generate and visualize fluid dynamics fields. Sample plots are shown by default.
                    </p>
                )}
                 { apiData && (
                    <div className="mt-8 p-6 backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-xl text-blue-200">
                        <h4 className="text-xl font-semibold mb-4">Graph Explanations</h4>
                        <div className="mb-4">
                          <h5 className="text-lg font-bold text-blue-300 mb-1">U Velocity Field (3D Surface)</h5>
                          <p>This plot shows the distribution of the horizontal (U) velocity component across the domain. The height and color represent the speed of the fluid in the X direction, with higher regions indicating faster flow.</p>
                        </div>
                        <div className="mb-4">
                          <h5 className="text-lg font-bold text-blue-300 mb-1">Pressure Field (3D Surface)</h5>
                          <p>This plot visualizes the pressure at each point in the fluid domain. The color and height indicate the pressure magnitude, helping to identify regions of high and low pressure that drive the flow.</p>
                        </div>
                        <div className="mb-4">
                          <h5 className="text-lg font-bold text-blue-300 mb-1">Velocity Magnitude (3D Surface)</h5>
                          <p>This graph displays the overall speed of the fluid at each location, regardless of direction. It combines both horizontal and vertical velocity components to show where the fluid is moving fastest.</p>
                        </div>
                        <div>
                          <h5 className="text-lg font-bold text-blue-300 mb-1">Vorticity Field (3D Surface)</h5>
                          <p>This plot represents the vorticity, or the local spinning motion of the fluid. High vorticity regions indicate strong rotational flow, which is important for understanding turbulence and mixing.</p>
                        </div>
                    </div>
                )}
              </div>
            </section>
          </>
        )}
      </div>
    </main>
  )
}

export default FluidViscosityExplainer