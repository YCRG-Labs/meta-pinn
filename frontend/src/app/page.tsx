import React from 'react'

const Page = () => {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-20 left-10 w-32 h-32 bg-blue-400 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-40 right-20 w-24 h-24 bg-indigo-400 rounded-full blur-2xl animate-pulse delay-1000"></div>
        <div className="absolute bottom-40 left-1/4 w-40 h-40 bg-purple-400 rounded-full blur-3xl animate-pulse delay-2000"></div>
        <div className="absolute bottom-20 right-1/3 w-28 h-28 bg-cyan-400 rounded-full blur-2xl animate-pulse delay-3000"></div>
      </div>

      {/* Hero Section */}
      <section className="relative py-32 px-4">
        <div className="max-w-7xl mx-auto text-center">
          
          <h1 className="text-5xl md:text-7xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-300 via-white to-indigo-300 mb-8 leading-tight">
            Inferring Spatial Fluid Viscosity
          </h1>
          <p className="text-xl md:text-2xl text-blue-200 mb-12 font-light max-w-4xl mx-auto">
            Using Physics-Informed Neural Networks in Navier-Stokes Flow
          </p>
          
          <div className="flex flex-wrap justify-center gap-3 mb-12">
            <span className="px-6 py-3 bg-gradient-to-r from-blue-500/30 to-cyan-500/30 border border-blue-400/40 text-blue-100 rounded-full text-sm font-medium backdrop-blur-sm hover:from-blue-500/40 hover:to-cyan-500/40 transition-all duration-300 cursor-default">
              Physics-Informed Neural Networks
            </span>
            <span className="px-6 py-3 bg-gradient-to-r from-indigo-500/30 to-purple-500/30 border border-indigo-400/40 text-indigo-100 rounded-full text-sm font-medium backdrop-blur-sm hover:from-indigo-500/40 hover:to-purple-500/40 transition-all duration-300 cursor-default">
              Fluid Dynamics
            </span>
            <span className="px-6 py-3 bg-gradient-to-r from-purple-500/30 to-pink-500/30 border border-purple-400/40 text-purple-100 rounded-full text-sm font-medium backdrop-blur-sm hover:from-purple-500/40 hover:to-pink-500/40 transition-all duration-300 cursor-default">
              Machine Learning
            </span>
          </div>
          
          <div className="text-blue-200/80 mb-12 space-y-2">
            <p className="text-lg">Authors: <span className="text-white font-medium">Brandon Yee*, Wilson Collins, Benjamin Pellegrini, Mihir Tekal</span></p>
            <p className="text-blue-300/70 italic">*Corresponding author: b.yee@ycrg-labs.org</p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a 
              href="https://github.com/YCRC-Labs/pinn-viscosity" 
              className="group px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl font-semibold hover:from-blue-500 hover:to-indigo-500 transition-all duration-300 transform hover:scale-105 hover:shadow-2xl hover:shadow-blue-500/25"
              target="_blank"
              rel="noopener noreferrer"
            >
              <span className="flex items-center justify-center gap-2">
                View on GitHub
                <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </span>
            </a>
            <a 
              href="#" 
              className="group px-8 py-4 border-2 border-blue-400/50 text-blue-100 rounded-xl font-semibold hover:bg-blue-400/10 hover:border-blue-400 transition-all duration-300 backdrop-blur-sm transform hover:scale-105"
            >
              Read Full Paper
            </a>
          </div>
        </div>
      </section>

      {/* Abstract Section */}
      <section className="relative py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-12 shadow-2xl">
            <h2 className="text-4xl font-bold text-white mb-8 text-center">Abstract</h2>
            <p className="text-blue-100 leading-relaxed text-lg text-center max-w-4xl mx-auto">
              This investigation presents a computational framework based on Physics-Informed Neural Networks (PINNs) 
              for the inference of spatially heterogeneous viscosity distributions. We examine linear spatial variations 
              within two-dimensional, steady, incompressible Navier-Stokes flow regimes.
            </p>
          </div>
        </div>
      </section>

      {/* Key Features Section */}
      <section className="relative py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-bold text-white mb-16 text-center">Key Contributions</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="group p-8 bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-xl border border-white/20 rounded-2xl hover:from-white/15 hover:to-white/10 transition-all duration-500 transform hover:scale-105 hover:shadow-2xl hover:shadow-blue-500/20">
              <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl mb-6 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-blue-300 transition-colors">PINN Framework</h3>
              <p className="text-blue-200 leading-relaxed">
                Development of a specialized PINN framework for inferring scalar parameter 'a' 
                and flow fields simultaneously.
              </p>
            </div>
            
            <div className="group p-8 bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-xl border border-white/20 rounded-2xl hover:from-white/15 hover:to-white/10 transition-all duration-500 transform hover:scale-105 hover:shadow-2xl hover:shadow-indigo-500/20">
              <div className="w-12 h-12 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-xl mb-6 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-indigo-300 transition-colors">Advanced Techniques</h3>
              <p className="text-blue-200 leading-relaxed">
                Integration of Fourier feature embeddings, adaptive loss weighting, 
                and curriculum learning strategies.
              </p>
            </div>
            
            <div className="group p-8 bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-xl border border-white/20 rounded-2xl hover:from-white/15 hover:to-white/10 transition-all duration-500 transform hover:scale-105 hover:shadow-2xl hover:shadow-purple-500/20">
              <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl mb-6 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-purple-300 transition-colors">Performance Evaluation</h3>
              <p className="text-blue-200 leading-relaxed">
                Quantitative evaluation on benchmark channel flow problems with 
                Reynolds number of 100.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative py-12 px-4 border-t border-white/10">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-blue-300/60">© 2025 YCRC Labs. All rights reserved.</p>
        </div>
      </footer>
    </main>
  )
}

export default Page