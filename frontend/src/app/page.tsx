'use client'

import React, { useEffect, useState } from 'react'

const Page = () => {
  const [isScrolled, setIsScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop
      setIsScrolled(scrollTop > 50)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <main className="min-h-screen bg-fixed">
      {/* Hero Section */}
      <section className={`hero-section ${isScrolled ? 'shrunk' : ''}`}>
        <div className="h-full flex items-center justify-center">
          <div className="text-center">
            <h1 className={`transition-all duration-500 py-3 font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-300 via-white to-indigo-300 ${
              isScrolled ? 'text-5xl' : 'text-8xl'
            }`}>
              Inferring Spatial Fluid Viscosity
            </h1>
            {!isScrolled && (
              <div className="scroll-indicator text-blue-200">↓ Scroll Down ↓</div>
            )}
          </div>
        </div>
      </section>

      {/* Content Section */}
      <div className={`content-section ${isScrolled ? '' : 'hide'}`}>
        {/* Your existing content sections */}
        <section className="relative px-4">
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
        {/* ... other sections ... */}
      </div>
    </main>
  )
}

export default Page