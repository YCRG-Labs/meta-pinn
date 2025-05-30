'use client'

import React, { useEffect, useState } from 'react'

const FluidViscosityExplainer = () => {
  const [isScrolled, setIsScrolled] = useState(false)
  const [selectedLevel, setSelectedLevel] = useState('')
  const [showContent, setShowContent] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop
      setIsScrolled(scrollTop > 50)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const handleLevelSelect = (level: 'beginner' | 'intermediate' | 'expert') => {
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
    switch(selectedLevel) {
      case 'beginner': return beginnerContent
      case 'intermediate': return intermediateContent
      case 'expert': return expertContent
      default: return null
    }
  }

  const currentContent = getContent()

  return (
    <main className="min-h-screen" style={{
      scrollSnapType: 'y mandatory'
    }}>
      {/* Hero Section */}
      <section className={`relative transition-all duration-1000 ${isScrolled ? 'h-32' : 'h-screen'}`} style={{
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
            <section id="explanation-content" className=" px-4 min-h-screen" style={{
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
    </main>
  )
}

export default FluidViscosityExplainer