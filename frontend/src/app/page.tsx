'use client'

import React, { useEffect, useState } from 'react'

const FluidViscosityExplainer = () => {
  const [isScrolled, setIsScrolled] = useState(false)
  const [selectedLevel, setSelectedLevel] = useState('')
  const [showContent, setShowContent] = useState(false)
  const [isClient, setIsClient] = useState(false)
  const [apiData, setApiData] = useState(null)
  const [loadingData, setLoadingData] = useState(false)
  const [apiError, setApiError] = useState(null)

  // State for parameter inputs
  const [reynoldsNumber, setReynoldsNumber] = useState(100)
  const [nuBaseTrue, setNuBaseTrue] = useState(0.01)
  const [aTrue, setATrue] = useState(0.05)
  const [uMaxInlet, setUMaxInlet] = useState(1.0);
    const [xMax, setXMax] = useState(2.0);
    const [yMax, setYMax] = useState(1.0);
    const [xMin, setXMin] = useState(0.0);
    const [yMin, setYMin] = useState(0.0);
    const [nGridX, setNGridX] = useState(50);
    const [nGridY, setNGridY] = useState(25);
    const [nTimeSlices, setNTimeSlices] = useState(5);
    const [name, setName] = useState("Frontend Visualization");
  
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

  // Fetch data from PINN API
    const fetchPINNData = async () => {
        setLoadingData(true);
        setApiError(null);

        try {
            const response = await fetch('http://localhost:8000/inference/single', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    parameters: {
                        reynolds_number: reynoldsNumber,
                        nu_base_true: nuBaseTrue,
                        a_true: aTrue,
                        u_max_inlet: uMaxInlet,
                        x_max: xMax,
                        y_max: yMax,
                        x_min: xMin,
                        y_min: yMin,
                        n_grid_x: nGridX,
                        n_grid_y: nGridY,
                        n_time_slices: nTimeSlices,
                        name: name,
                    },
                    model_path: "/home/brand/pinn_viscosity/backend/results/trained_model.pth",
                    include_boundary: true,
                    include_centerline: true,
                    include_viscosity: true
                })
            });

            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }

            const data = await response.json();
            if (data.success) {
                setApiData(data);
            } else {
                throw new Error(data.error_message || 'API returned error');
            }
        } catch (error: any) {
            console.error('Error fetching PINN data:', error);
            setApiError(error);
        } finally {
            setLoadingData(false);
        }
    };
    

  // Load Plotly and create plots
  useEffect(() => {
    if (!isClient || typeof window === 'undefined') return

    const loadScript = (src: any) => {
      return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) {
          resolve(void 0)
          return
        }
        const script = document.createElement('script')
        script.src = src
        script.onload = resolve
        script.onerror = reject
        document.head.appendChild(script)
      })
    }

    const initializePlots = async () => {
      try {
        // Load d3 if not already loaded
        if (!window.d3) {
          await loadScript('https://d3js.org/d3.v5.min.js')
        }

        // Dynamically import Plotly
        const PlotlyModule = await import('plotly.js-dist')
        const Plotly = PlotlyModule.default

        // Create sample data if no API data available
        const createSampleData = () => {
          const size = 20
          const data = []
          for (let i = 0; i < size; i++) {
            const row = []
            for (let j = 0; j < size; j++) {
              row.push(Math.sin(i * 0.3) * Math.cos(j * 0.3) * 10 + Math.random() * 2)
            }
            data.push(row)
          }
          return data
        }

        // Function to reshape 1D data to 2D grid
        const reshapeToGrid = (data: any, nx: any, ny: any) => {
          const grid = []
          for (let i = 0; i < ny; i++) {
            const row = []
            for (let j = 0; j < nx; j++) {
              const idx = i * nx + j
              row.push(data[idx] || 0)
            }
            grid.push(row)
          }
          return grid
        }

        // Use API data if available, otherwise use sample data
        let velocityData, pressureData, viscosityData
        let gridX = 50, gridY = 25

        if (apiData && (apiData as any).flow_field) {
          const { flow_field } = apiData
          gridX = (flow_field as any).grid_shape[0]
          gridY = (flow_field as any).grid_shape[1]

          velocityData = reshapeToGrid((flow_field as any).velocity_magnitude, gridX, gridY)
          pressureData = reshapeToGrid((flow_field as any).pressure, gridX, gridY)
          viscosityData = reshapeToGrid((flow_field as any).viscosity, gridX, gridY)
        } else {
          velocityData = createSampleData()
          pressureData = createSampleData()
          viscosityData = createSampleData()
        }

        // Velocity Magnitude Plot
        const data1 = [{
          z: velocityData,
          type: 'surface',
          colorscale: 'Viridis',
          name: 'Velocity Magnitude'
        }]

        const layout1 = {
          title: {
            text: apiData ? `Velocity Magnitude Field (Re=${(apiData as any).model_info?.reynolds_number})` : 'Sample Velocity Field',
            font: { color: 'white' }
          },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          scene: {
            bgcolor: 'rgba(0,0,0,0)',
            xaxis: {
              title: 'X Position',
              gridcolor: 'white',
              color: 'white'
            },
            yaxis: {
              title: 'Y Position',
              gridcolor: 'white',
              color: 'white'
            },
            zaxis: {
              title: 'Velocity Magnitude',
              gridcolor: 'white',
              color: 'white'
            }
          },
          autosize: true,
          margin: { l: 0, r: 0, b: 0, t: 50 }
        }

        if (document.getElementById('velocityPlot')) {
          Plotly.newPlot('velocityPlot', data1, layout1, { responsive: true })
        }

        // Pressure Field Plot
        const data2 = [{
          z: pressureData,
          type: 'surface',
          colorscale: 'RdBu',
          name: 'Pressure'
        }]

        const layout2 = {
          title: {
            text: apiData ? `Pressure Field (Learned ν param: ${(apiData as any).learned_viscosity_param?.toFixed(4)})` : 'Sample Pressure Field',
            font: { color: 'white' }
          },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          scene: {
            bgcolor: 'rgba(0,0,0,0)',
            camera: { eye: { x: 1.87, y: 0.88, z: -0.64 } },
            xaxis: {
              title: 'X Position',
              gridcolor: 'white',
              color: 'white'
            },
            yaxis: {
              title: 'Y Position',
              gridcolor: 'white',
              color: 'white'
            },
            zaxis: {
              title: 'Pressure',
              gridcolor: 'white',
              color: 'white'
            }
          },
          autosize: true,
          margin: { l: 0, r: 0, b: 0, t: 50 }
        }

        if (document.getElementById('pressurePlot')) {
          Plotly.newPlot('pressurePlot', data2, layout2, { responsive: true })
        }

        // Viscosity Field Plot
        const data3 = [{
          z: viscosityData,
          type: 'surface',
          colorscale: 'Plasma',
          name: 'Viscosity'
        }]

        const layout3 = {
          title: {
            text: apiData ? `Viscosity Field Distribution (${(apiData as any).total_points} points)` : 'Sample Viscosity Field',
            font: { color: 'white' }
          },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          scene: {
            bgcolor: 'rgba(0,0,0,0)',
            xaxis: {
              title: 'X Position',
              gridcolor: 'white',
              color: 'white'
            },
            yaxis: {
              title: 'Y Position',
              gridcolor: 'white',
              color: 'white'
            },
            zaxis: {
              title: 'Viscosity',
              gridcolor: 'white',
              color: 'white'
            }
          },
          autosize: true,
          margin: { l: 0, r: 0, b: 0, t: 50 }
        }

        if (document.getElementById('viscosityPlot')) {
          Plotly.newPlot('viscosityPlot', data3, layout3, { responsive: true })
        }

        // Additional comparison plot
        const data4 = [
          {
            z: velocityData,
            type: 'surface',
            colorscale: 'Viridis',
            opacity: 0.8,
            name: 'Velocity'
          },
          {
            z: pressureData.map(row => row.map(val => val * 0.5)),
            type: 'surface',
            colorscale: 'RdBu',
            opacity: 0.6,
            showscale: false,
            name: 'Pressure (scaled)'
          }
        ]

        const layout4 = {
          title: {
            text: 'Combined Velocity & Pressure Fields',
            font: { color: 'white' }
          },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          scene: {
            bgcolor: 'rgba(0,0,0,0)',
            xaxis: {
              title: 'X Position',
              gridcolor: 'white',
              color: 'white'
            },
            yaxis: {
              title: 'Y Position',
              gridcolor: 'white',
              color: 'white'
            },
            zaxis: {
              title: 'Field Values',
              gridcolor: 'white',
              color: 'white'
            }
          },
          autosize: true,
          margin: { l: 0, r: 0, b: 0, t: 50 }
        }

        if (document.getElementById('combinedPlot')) {
          Plotly.newPlot('combinedPlot', data4, layout4, { responsive: true })
        }

      } catch (error) {
        console.error('Error loading Plotly:', error)
      }
    }

    initializePlots()
  }, [isClient, apiData])

  const handleLevelSelect = (level: any) => {
    setSelectedLevel(level)
    setShowContent(true)
    setTimeout(() => {
      document.getElementById('explanation-content')?.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      })
    }, 100)
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

  if (!isClient) {
    return <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 flex items-center justify-center">
      <div className="text-white text-xl">Loading...</div>
    </div>
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900" style={{
      scrollSnapType: 'y mandatory'
    }}>
      {/* Hero Section */}
      <section className={`relative transition-all duration-1000 ${isScrolled ? 'h-16' : 'h-screen'}`} style={{
        scrollSnapAlign: 'start'
      }}>
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20 backdrop-blur-sm"></div>
        <div className="relative h-full flex items-center justify-center">
          <div className="text-center px-4">
            <h1 className={`transition-all duration-500 py-3 font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-300 via-white to-indigo-300 ${
              isScrolled ? 'text-3xl md:text-5xl' : 'text-4xl md:text-8xl'
            }`}>
              Inferring Spatial Fluid Viscosity
            </h1>
            {!isScrolled && (
              <div className="mt-8 text-blue-200 animate-bounce">↓ Scroll Down ↓</div>
            )}
          </div>
        </div>
      </section>

      {/* Content Section */}
      <div className={`transition-all duration-1000 ${isScrolled ? 'opacity-100' : 'opacity-0'}`}>
        <section className="relative py-16 px-4 min-h-screen flex items-center" style={{
          scrollSnapAlign: 'start'
        }}>
          <div className="w-full">
            <h2 className="text-4xl font-bold text-white mb-8 text-center">Choose Your Learning Level</h2>
            <div className="max-w-4xl mx-auto">
              <div className="border border-white/10 rounded-3xl p-8 md:p-12 shadow-2xl">
                <div className="flex flex-col md:flex-row justify-center gap-4 md:gap-8">
                  <button
                    onClick={() => handleLevelSelect('beginner')}
                    className={`px-8 py-4 backdrop-blur-xl border border-white/30 rounded-full font-semibold
                      transition-all duration-300 transform hover:scale-105 hover:shadow-xl
                      ${selectedLevel === 'beginner'
                        ? 'bg-blue-600/20 text-white border-blue-400'
                        : 'text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-white to-indigo-400 hover:from-blue-300 hover:to-indigo-300'
                      }`}
                  >
                    🌟 Beginner
                  </button>
                  <button
                    onClick={() => handleLevelSelect('intermediate')}
                    className={`px-8 py-4 backdrop-blur-xl border border-white/30 rounded-full font-semibold
                      transition-all duration-300 transform hover:scale-105 hover:shadow-xl
                      ${selectedLevel === 'intermediate'
                        ? 'bg-blue-600/20 text-white border-blue-400'
                        : 'text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-white to-indigo-400 hover:from-blue-300 hover:to-indigo-300'
                      }`}
                  >
                    🎓 Intermediate
                  </button>
                  <button
                    onClick={() => handleLevelSelect('expert')}
                    className={`px-8 py-4 backdrop-blur-xl border border-white/30 rounded-full font-semibold
                      transition-all duration-300 transform hover:scale-105 hover:shadow-xl
                      ${selectedLevel === 'expert'
                        ? 'bg-blue-600/20 text-white border-blue-400'
                        : 'text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-white to-indigo-400 hover:from-blue-300 hover:to-indigo-300'
                      }`}
                  >
                    🔬 Expert
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Explanation Content */}
        {showContent && currentContent && (
          <>
            <section id="explanation-content" className="py-16 px-4 min-h-screen" style={{
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
                      className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-8 shadow-xl transform transition-all duration-500 hover:scale-[1.02] hover:shadow-2xl"
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
                  <h3 className="text-3xl font-bold text-center text-white mb-6">Key Takeaways</h3>
                  <div className="grid md:grid-cols-2 gap-8 text-gray-100">
                    <div>
                      <h4 className="text-xl font-semibold text-blue-300 mb-4">✅ What Worked</h4>
                      <ul className="space-y-2 text-lg">
                        <li>• Excellent flow field reconstruction</li>
                        <li>• Low physics equation violations</li>
                        <li>• Robust neural network training</li>
                        <li>• Advanced optimization techniques</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="text-xl font-semibold text-red-300 mb-4">⚠️ Challenges</h4>
                      <ul className="space-y-2 text-lg">
                        <li>• Poor parameter identification</li>
                        <li>• Ill-posed inverse problem</li>
                        <li>• Need for more/better data</li>
                        <li>• Multiple valid solutions</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </>
        )}
      </div>

      {/* PINN Data Visualization Section */}
      <section className="py-16 px-4 min-h-screen" style={{
        scrollSnapAlign: 'start'
      }}>
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-white mb-10 text-center">PINN Model Predictions</h2>

          {/* API Controls */}
          <div className="mb-12 text-center">

            <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-6 mb-8 shadow-xl">
                            <h3 className="text-xl font-bold text-blue-300 mb-4">PINN Model Parameters</h3>

                            {/* Parameter Inputs */}
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                                <div>
                                    <label htmlFor="reynoldsNumber" className="block text-gray-300 text-sm font-bold mb-2">Reynolds Number</label>
                                    <input
                                        type="number"
                                        id="reynoldsNumber"
                                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                                        value={reynoldsNumber}
                                        onChange={(e) => setReynoldsNumber(parseFloat(e.target.value))}
                                    />
                                </div>

                                <div>
                                    <label htmlFor="nuBaseTrue" className="block text-gray-300 text-sm font-bold mb-2">Base Viscosity</label>
                                    <input
                                        type="number"
                                        id="nuBaseTrue"
                                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                                        value={nuBaseTrue}
                                        onChange={(e) => setNuBaseTrue(parseFloat(e.target.value))}
                                    />
                                </div>

                                <div>
                                    <label htmlFor="aTrue" className="block text-gray-300 text-sm font-bold mb-2">Viscosity Gradient</label>
                                    <input
                                        type="number"
                                        id="aTrue"
                                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                                        value={aTrue}
                                        onChange={(e) => setATrue(parseFloat(e.target.value))}
                                    />
                                </div>
                                 <div>
                                    <label htmlFor="uMaxInlet" className="block text-gray-300 text-sm font-bold mb-2">Max Inlet Velocity</label>
                                    <input
                                        type="number"
                                        id="uMaxInlet"
                                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                                        value={uMaxInlet}
                                        onChange={(e) => setUMaxInlet(parseFloat(e.target.value))}
                                    />
                                </div>
                                <div>
                                    <label htmlFor="xMax" className="block text-gray-300 text-sm font-bold mb-2">Max X</label>
                                    <input
                                        type="number"
                                        id="xMax"
                                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                                        value={xMax}
                                        onChange={(e) => setXMax(parseFloat(e.target.value))}
                                    />
                                </div>
                                 <div>
                                    <label htmlFor="yMax" className="block text-gray-300 text-sm font-bold mb-2">Max Y</label>
                                    <input
                                        type="number"
                                        id="yMax"
                                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight  focus:outline-none focus:shadow-outline"
                                        value={yMax}
                                        onChange={(e) => setYMax(parseFloat(e.target.value))}
                                    />
                                </div>
                                <div>
                                    <label htmlFor="xMin" className="block text-gray-300 text-sm font-bold mb-2">Min X</label>
                                    <input
                                        type="number"
                                        id="xMin"
                                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                                        value={xMin}
                                        onChange={(e) => setXMin(parseFloat(e.target.value))}
                                    />
                                </div>
                                 <div>
                                    <label htmlFor="yMin" className="block text-gray-300 text-sm font-bold mb-2">Min Y</label>
                                    <input
                                        type="number"
                                        id="yMin"
                                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                                        value={yMin}
                                        onChange={(e) => setYMin(parseFloat(e.target.value))}
                                    />
                                </div>
                                 <div>
                                    <label htmlFor="nGridX" className="block text-gray-300 text-sm font-bold mb-2">Grid X</label>
                                    <input
                                        type="number"
                                        id="nGridX"
                                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                                        value={nGridX}
                                        onChange={(e) => setNGridX(parseFloat(e.target.value))}
                                    />
                                </div>
                                 <div>
                                    <label htmlFor="nGridY" className="block text-gray-300 text-sm font-bold mb-2">Grid Y</label>
                                    <input
                                        type="number"
                                        id="nGridY"
                                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                                        value={nGridY}
                                        onChange={(e) => setNGridY(parseFloat(e.target.value))}
                                    />
                                </div>
                                 <div>
                                    <label htmlFor="nTimeSlices" className="block text-gray-300 text-sm font-bold mb-2">Time Slices</label>
                                    <input
                                        type="number"
                                        id="nTimeSlices"
                                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                                        value={nTimeSlices}
                                        onChange={(e) => setNTimeSlices(parseFloat(e.target.value))}
                                    />
                                </div>
                                <div>
                                    <label htmlFor="name" className="block text-gray-300 text-sm font-bold mb-2">Name</label>
                                    <input
                                        type="text"
                                        id="name"
                                        className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                                        value={name}
                                        onChange={(e) => setName(e.target.value)}
                                    />
                                </div>
                            </div>

                            <button
                                onClick={fetchPINNData}
                                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                                type="button"
                                disabled={loadingData}
                            >
                                {loadingData ? 'Loading...' : 'Fetch PINN Data'}
                            </button>
                            {apiError && <div className="text-red-500 mt-4">Error: {String(apiError)}</div>}
                        </div>

            {/* Velocity Field Plot */}
            <h3 className="text-2xl font-bold text-blue-300 mb-6">Velocity Magnitude Field</h3>
            <div id="velocityPlot" className="w-full h-[500px] mx-auto bg-black/20 rounded-lg"></div>
            <p className="text-gray-300 mt-4 text-center">
              Interactive 3D visualization of fluid velocity distribution across the domain
            </p>

            {/* Pressure Field Plot */}
            <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-8 shadow-xl">
              <h3 className="text-2xl font-bold text-blue-300 mb-6">Pressure Field Distribution</h3>
              <div id="pressurePlot" className="w-full h-[500px] mx-auto bg-black/20 rounded-lg"></div>
              <p className="text-gray-300 mt-4 text-center">
                Pressure field showing how pressure varies throughout the fluid domain
              </p>
            </div>

            {/* Viscosity Field Plot */}
            <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-8 shadow-xl">
              <h3 className="text-2xl font-bold text-blue-300 mb-6">Viscosity Field Distribution</h3>
              <div id="viscosityPlot" className="w-full h-[500px] mx-auto bg-black/20 rounded-lg"></div>
              <p className="text-gray-300 mt-4 text-center">
                Spatial variation of fluid viscosity as predicted by the PINN model
              </p>
            </div>
          </div>

          {/* Model Performance Metrics */}
          {apiData && (
            <div className="mt-16 backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-8 shadow-xl">
              <h3 className="text-2xl font-bold text-blue-300 mb-6 text-center">Model Performance Metrics</h3>
              <div className="grid md:grid-cols-3 gap-8 text-center">
                <div className="bg-gradient-to-br from-blue-600/20 to-cyan-600/20 p-6 rounded-xl">
                  <h4 className="text-lg font-semibold text-cyan-300 mb-2">Reynolds Number</h4>
                  <p className="text-3xl font-bold text-white">{(apiData as any)?.model_info?.reynolds_number}</p>
                </div>
                <div className="bg-gradient-to-br from-purple-600/20 to-pink-600/20 p-6 rounded-xl">
                  <h4 className="text-lg font-semibold text-pink-300 mb-2">Learned Parameter</h4>
                  <p className="text-3xl font-bold text-white">{(apiData as any)?.learned_viscosity_param?.toFixed(4) || 'N/A'}</p>
                </div>
                <div className="bg-gradient-to-br from-green-600/20 to-emerald-600/20 p-6 rounded-xl">
                  <h4 className="text-lg font-semibold text-emerald-300 mb-2">Processing Time</h4>
                  <p className="text-3xl font-bold text-white">{(apiData as any)?.processing_time?.toFixed(2)}s</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Footer Section */}
      <section className="py-16 px-4" style={{
        scrollSnapAlign: 'start'
      }}>
        <div className="max-w-4xl mx-auto text-center">
          <div className="backdrop-blur-xl bg-gradient-to-r from-slate-800/30 to-gray-800/30 border border-white/10 rounded-2xl p-8 shadow-xl">
            <h3 className="text-2xl font-bold text-white mb-4">Research Impact</h3>
            <p className="text-gray-300 text-lg leading-relaxed">
              This work demonstrates both the potential and limitations of Physics-Informed Neural Networks
              for inverse problems in fluid dynamics. While excellent at flow field reconstruction, the challenge
              of parameter identification highlights the need for improved methodologies in computational fluid dynamics.
            </p>
            <div className="mt-8 flex justify-center space-x-4">
              <div className="text-blue-300 font-semibold">🔬 Computational Physics</div>
              <div className="text-purple-300 font-semibold">🤖 Machine Learning</div>
              <div className="text-cyan-300 font-semibold">🌊 Fluid Dynamics</div>
            </div>
          </div>
        </div>
      </section>
    </main>
  )
}

export default FluidViscosityExplainer