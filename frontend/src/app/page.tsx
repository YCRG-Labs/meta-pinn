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
  
  // Add model path configuration
  const [modelPath, setModelPath] = useState("./results/trained_model.pth");
  const [backendUrl, setBackendUrl] = useState("http://localhost:8000");
  
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

  // Enhanced fetch function with better error handling
  const fetchPINNData = async () => {
    setLoadingData(true);
    setApiError(null);

    try {
      const response = await fetch(`${backendUrl}/inference/single`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          parameters: {
            reynolds_number: parseFloat(String(reynoldsNumber)),
            nu_base_true: parseFloat(String(nuBaseTrue)),
            a_true: parseFloat(String(aTrue)),
            u_max_inlet: parseFloat(String(uMaxInlet)),
            x_max: parseFloat(String(xMax)),
            y_max: parseFloat(String(yMax)),
            x_min: parseFloat(String(xMin)),
            y_min: parseFloat(String(yMin)),
            n_grid_x: parseInt(String(nGridX), 10),
            n_grid_y: parseInt(String(nGridY), 10),
            n_time_slices: parseInt(String(nTimeSlices), 10),
            name: name,
          },
          model_path: modelPath,
          include_boundary: true,
          include_centerline: true,
          include_viscosity: true
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API request failed: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      if (data.success) {
        setApiData(data);
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
    if (!backendUrl) {
        console.warn('Backend URL is not set.');
        return false;
    }
    try {
      const response = await fetch(`${backendUrl}/health`);
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

    const loadScript = (src: string) => { // Explicitly type src as string
      return new Promise<void>((resolve, reject) => { // Explicitly type Promise
        if (document.querySelector(`script[src="${src}"]`)) {
          resolve(undefined) // Use undefined for void Promise
          return
        }
        const script = document.createElement('script')
        script.src = src
        script.onload = () => resolve(undefined); // Use undefined
        script.onerror = reject
        document.head.appendChild(script)
      })
    }

    const initializePlots = async () => {
      try {
        if (!(window as any).d3) {
          await loadScript('https://d3js.org/d3.v5.min.js')
        }

        const PlotlyModule = await import('plotly.js-dist')
        const Plotly = PlotlyModule.default

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

        const reshapeToGrid = (dataArray: number[], nx: number, ny: number) => { // Typed parameters
          const grid: number[][] = [] // Typed grid
          if (!dataArray || dataArray.length !== nx * ny) {
            console.warn("Data array is null, undefined, or has incorrect length for reshaping. Using empty values.");
            for (let i = 0; i < ny; i++) {
                const row: number[] = [];
                for (let j = 0; j < nx; j++) {
                    row.push(0); // Push default value
                }
                grid.push(row);
            }
            return grid;
          }
          for (let i = 0; i < ny; i++) {
            const row: number[] = [] // Typed row
            for (let j = 0; j < nx; j++) {
              const idx = i * nx + j
              row.push(dataArray[idx])
            }
            grid.push(row)
          }
          return grid
        }
        
        let velocityData, pressureData, viscosityData
        // Use state values for default grid size from parameter inputs
        let currentGridX = typeof nGridX === 'string' ? parseInt(nGridX, 10) : nGridX;
        let currentGridY = typeof nGridY === 'string' ? parseInt(nGridY, 10) : nGridY;


        if (apiData && (apiData as any).flow_field) {
          const flow_field = (apiData as any).flow_field;
          currentGridX = (flow_field as any).grid_shape[0];
          currentGridY = (flow_field as any).grid_shape[1];

          velocityData = reshapeToGrid((flow_field as any).velocity_magnitude, currentGridX, currentGridY);
          pressureData = reshapeToGrid((flow_field as any).pressure, currentGridX, currentGridY);
          viscosityData = reshapeToGrid((flow_field as any).viscosity, currentGridX, currentGridY);
        } else {
          // When no API data, use sample data with current grid settings
          velocityData = createSampleData() // Sample data is 20x20, not tied to nGridX/Y
          pressureData = createSampleData()
          viscosityData = createSampleData()
        }
        
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

        if (document.getElementById('velocityPlot')) {
          Plotly.newPlot('velocityPlot', [{ z: velocityData, type: 'surface', colorscale: 'Viridis', name: 'Velocity' }] as any, {
            ...commonLayoutProps,
            title: { text: apiData ? `Velocity Magnitude (Re=${(apiData as any).model_info?.reynolds_number})` : 'Sample Velocity', font: { color: 'white' } },
            scene: { ...commonLayoutProps.scene, zaxis: { ...commonLayoutProps.scene.zaxis, title: 'Velocity Mag.' } }
          } as any, { responsive: true });
        }

        if (document.getElementById('pressurePlot')) {
          Plotly.newPlot('pressurePlot', [{ z: pressureData, type: 'surface', colorscale: 'RdBu', name: 'Pressure' }] as any, {
            ...commonLayoutProps,
            title: { text: apiData ? `Pressure Field (Learned ν: ${(apiData as any).learned_viscosity_param?.toFixed(4)})` : 'Sample Pressure', font: { color: 'white' } },
            scene: { ...commonLayoutProps.scene, camera: { eye: { x: 1.87, y: 0.88, z: -0.64 } }, zaxis: { ...commonLayoutProps.scene.zaxis, title: 'Pressure' } }
          } as any, { responsive: true });
        }

        if (document.getElementById('viscosityPlot')) {
          Plotly.newPlot('viscosityPlot', [{ z: viscosityData, type: 'surface', colorscale: 'Plasma', name: 'Viscosity' }] as any, {
            ...commonLayoutProps,
            title: { text: apiData ? `Viscosity Field (${(apiData as any).total_points} pts)` : 'Sample Viscosity', font: { color: 'white' } },
            scene: { ...commonLayoutProps.scene, zaxis: { ...commonLayoutProps.scene.zaxis, title: 'Viscosity' } }
          } as any, { responsive: true });
        }
        
        if (document.getElementById('combinedPlot') && velocityData && pressureData) {
            const combinedData: any[] = [
                { z: velocityData, type: 'surface', colorscale: 'Viridis', opacity: 0.8, name: 'Velocity' },
                { z: pressureData.map(row => row.map(val => val * 0.5)), type: 'surface', colorscale: 'RdBu', opacity: 0.6, showscale: false, name: 'Pressure (scaled)' }
            ];
            Plotly.newPlot('combinedPlot', combinedData, {
                ...commonLayoutProps,
                title: { text: 'Combined Velocity & Scaled Pressure', font: { color: 'white' } },
                scene: { ...commonLayoutProps.scene, zaxis: { ...commonLayoutProps.scene.zaxis, title: 'Field Values' } }
            } as any, { responsive: true });
        }


      } catch (error) {
        console.error('Error loading/initializing Plotly:', error)
      }
    }

    initializePlots()
    // Cleanup function (optional, use if plots cause issues on re-renders without full page navigation)
    // return () => {
    //   const Plotly = (window as any).Plotly; // Get Plotly instance
    //   if (Plotly) {
    //       ['velocityPlot', 'pressurePlot', 'viscosityPlot', 'combinedPlot'].forEach(id => {
    //           const plotDiv = document.getElementById(id);
    //           if (plotDiv && plotDiv.data) { // Check if plot exists
    //               try { Plotly.purge(id); } catch (e) { console.warn(`Could not purge plot ${id}:`, e); }
    //           }
    //       });
    //   }
    // };
  }, [isClient, apiData, showContent, nGridX, nGridY]) // Re-run if apiData, showContent, or grid dimensions change

  const handleLevelSelect = (level: string) => {
    setSelectedLevel(level)
    setShowContent(true)
    setApiData(null); // Clear previous API data/plots when changing level to show fresh sample plots
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

  const renderInputField = (label: string, id: string, value: string | number, setter: (value: any) => void, type = "number", step = "any") => (
    <div className="mb-4">
      <label htmlFor={id} className="block text-sm font-medium text-blue-200 mb-1">{label}:</label>
      <input
        type={type}
        id={id}
        value={value}
        onChange={(e) => setter(type === "number" ? parseFloat(e.target.value) || 0 : e.target.value)}
        step={step}
        className="w-full p-2.5 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:ring-blue-500 focus:border-blue-500 shadow-sm"
      />
    </div>
  );


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
              <div className="border border-white/10 rounded-3xl p-8 md:p-12 shadow-2xl backdrop-blur-md bg-white/5">
                <div className="flex flex-col md:flex-row justify-center gap-4 md:gap-8">
                  <button
                    onClick={() => handleLevelSelect('beginner')}
                    className={`px-8 py-4 backdrop-blur-xl border border-white/30 rounded-full font-semibold
                      transition-all duration-300 transform hover:scale-105 hover:shadow-xl
                      ${selectedLevel === 'beginner'
                        ? 'bg-blue-600/20 text-white border-blue-400 shadow-blue-500/50'
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
                        ? 'bg-green-600/20 text-white border-green-400 shadow-green-500/50'
                        : 'text-transparent bg-clip-text bg-gradient-to-r from-green-400 via-white to-teal-400 hover:from-green-300 hover:to-teal-300'
                      }`}
                  >
                    🎓 Intermediate
                  </button>
                  <button
                    onClick={() => handleLevelSelect('expert')}
                    className={`px-8 py-4 backdrop-blur-xl border border-white/30 rounded-full font-semibold
                      transition-all duration-300 transform hover:scale-105 hover:shadow-xl
                      ${selectedLevel === 'expert'
                        ? 'bg-purple-600/20 text-white border-purple-400 shadow-purple-500/50'
                        : 'text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-white to-pink-400 hover:from-purple-300 hover:to-pink-300'
                      }`}
                  >
                    🔬 Expert
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Explanation Content & Visualization - only shown after level selection */}
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
                    <h3 className="text-2xl font-semibold text-blue-300 mb-6">Simulation Parameters</h3>
                    {renderInputField("Reynolds Number", "reynoldsNumber", reynoldsNumber, setReynoldsNumber)}
                    {renderInputField("Base Viscosity (ν_base)", "nuBaseTrue", nuBaseTrue, setNuBaseTrue, "number", "0.001")}
                    {renderInputField("Viscosity Gradient (a_true)", "aTrue", aTrue, setATrue, "number", "0.01")}
                    {renderInputField("Max Inlet Velocity (U_max)", "uMaxInlet", uMaxInlet, setUMaxInlet)}
                    
                    <h4 className="text-xl font-semibold text-blue-200 mt-6 mb-3">Domain & Grid</h4>
                    {renderInputField("X Max", "xMax", xMax, setXMax)}
                    {renderInputField("Y Max", "yMax", yMax, setYMax)}
                    {renderInputField("X Min", "xMin", xMin, setXMin)}
                    {renderInputField("Y Min", "yMin", yMin, setYMin)}
                    {renderInputField("Grid Points X (n_grid_x)", "nGridX", nGridX, setNGridX, "number", "1")}
                    {renderInputField("Grid Points Y (n_grid_y)", "nGridY", nGridY, setNGridY, "number", "1")}
                    {/* {renderInputField("Time Slices (n_time_slices)", "nTimeSlices", nTimeSlices, setNTimeSlices, "number", "1")} */}
                    {/* {renderInputField("Case Name", "name", name, setName, "text")} */}

                    <h4 className="text-xl font-semibold text-blue-200 mt-6 mb-3">Backend & Model</h4>
                    {renderInputField("Backend URL", "backendUrl", backendUrl, setBackendUrl, "text")}
                    {renderInputField("Model Path", "modelPath", modelPath, setModelPath, "text")}


                    <button
                      onClick={fetchPINNData}
                      disabled={loadingData}
                      className="w-full mt-6 bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:shadow-lg transition duration-150 ease-in-out disabled:opacity-50 flex items-center justify-center"
                    >
                      {loadingData ? (
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                      ) : "🚀 Run Inference"}
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
                  <div className="md:col-span-2 grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div id="velocityPlot" className="w-full h-[400px] md:h-[500px] bg-white/5 border border-white/10 rounded-xl shadow-xl p-2"></div>
                    <div id="pressurePlot" className="w-full h-[400px] md:h-[500px] bg-white/5 border border-white/10 rounded-xl shadow-xl p-2"></div>
                    <div id="viscosityPlot" className="w-full h-[400px] md:h-[500px] bg-white/5 border border-white/10 rounded-xl shadow-xl p-2"></div>
                    <div id="combinedPlot" className="w-full h-[400px] md:h-[500px] bg-white/5 border border-white/10 rounded-xl shadow-xl p-2"></div>
                  </div>
                </div>
                { !apiData && !loadingData && (
                    <p className="text-center text-blue-200 text-lg mt-8">
                        Select a learning level and adjust parameters, then click "Run Inference" to generate and visualize fluid dynamics fields. Sample plots are shown by default.
                    </p>
                )}
                 { apiData && (
                    <div className="mt-8 p-6 backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl shadow-xl text-blue-200">
                        <h4 className="text-xl font-semibold mb-2">Inference Details:</h4>
                        <p>• Reynolds Number Used: {(apiData as any).model_info?.reynolds_number}</p>
                        <p>• Learned Viscosity Parameter (ã): {(apiData as any).learned_viscosity_param?.toFixed(5)}</p>
                        <p>• Total Points in Flow Field: {(apiData as any).total_points}</p>
                        <p>• Grid Shape: [{(apiData as any).flow_field?.grid_shape.join(', ')}]</p>
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