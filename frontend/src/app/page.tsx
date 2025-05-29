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
              <h2 className="text-4xl font-bold text-white mb-8 text-center">Complexity Level</h2>

            <div className="flex justify-center space-x-4 mb-8">
              <button id='beginner' className='px-10 py-4 backdrop-blur-xl border border-white rounded-full bg-blue-600 font-semibold hover:bg-blue-700 transition duration-300 text-transparent bg-clip-text bg-gradient-to-r from-blue-600 via-zinc-100 to-indigo-600'>beginner</button>
              <button id='intermediate' className='px-10 py-4 backdrop-blur-xl border border-white rounded-full bg-blue-600 font-semibold hover:bg-blue-700 transition duration-300 text-transparent bg-clip-text bg-gradient-to-r from-blue-600 via-zinc-100 to-indigo-600'>intermediate</button>
              <button id='expert' className='px-10 py-4 backdrop-blur-xl border border-white rounded-full bg-blue-600 font-semibold hover:bg-blue-700 transition duration-300 text-transparent bg-clip-text bg-gradient-to-r from-blue-600 via-zinc-100 to-indigo-600'>expert</button>
            </div>
            </div>  
          </div>
        </section>
        {/* ... other sections ... */}
      </div>
    </main>
  )
}

export default Page