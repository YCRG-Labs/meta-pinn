#!/bin/bash

# FEniCSx Installation Script for WSL/Ubuntu
# This script installs FEniCSx (DOLFINx) and all required dependencies
# Compatible with Ubuntu 20.04+ and WSL2

set -e  # Exit on any error

echo "🚀 FEniCSx Installation Script for WSL/Ubuntu"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on WSL/Linux
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    print_error "This script is for WSL/Linux. Please run in WSL environment."
    exit 1
fi

print_status "Checking system requirements..."

# Check Ubuntu version
if command -v lsb_release &> /dev/null; then
    UBUNTU_VERSION=$(lsb_release -rs)
    print_status "Detected Ubuntu version: $UBUNTU_VERSION"
    
    # Check if version is supported
    if [[ $(echo "$UBUNTU_VERSION >= 20.04" | bc -l) -eq 0 ]]; then
        print_warning "Ubuntu version $UBUNTU_VERSION may not be fully supported. Recommended: 20.04+"
    fi
else
    print_warning "Could not detect Ubuntu version. Proceeding anyway..."
fi

# Update system packages
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential build tools and dependencies
print_status "Installing essential build tools..."
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    curl \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Python development packages
print_status "Installing Python development packages..."
sudo apt install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-wheel

# Install mathematical libraries
print_status "Installing mathematical and scientific libraries..."
sudo apt install -y \
    libblas-dev \
    liblapack-dev \
    libopenmpi-dev \
    openmpi-bin \
    libhdf5-openmpi-dev \
    libboost-all-dev \
    libeigen3-dev \
    libparmetis-dev \
    libscotch-dev \
    libsuitesparse-dev \
    libsuperlu-dev \
    libhypre-dev \
    petsc-dev \
    slepc-dev

# Install additional dependencies for FEniCSx
print_status "Installing FEniCSx-specific dependencies..."
sudo apt install -y \
    libpugixml-dev \
    libadios2-dev \
    pybind11-dev \
    python3-pybind11 \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-h5py \
    python3-mpi4py

# Install pip packages
print_status "Installing Python packages via pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Install scientific Python packages
python3 -m pip install --user \
    numpy \
    scipy \
    matplotlib \
    h5py \
    mpi4py \
    petsc4py \
    slepc4py \
    pybind11[global]

# Install FEniCSx components
print_status "Installing FEniCSx components..."

# Method 1: Try conda-forge first (recommended)
if command -v conda &> /dev/null; then
    print_status "Conda detected. Installing FEniCSx via conda-forge..."
    conda install -c conda-forge fenics-dolfinx mpich pyvista
    print_success "FEniCSx installed via conda!"
else
    # Method 2: Install via pip (may require compilation)
    print_status "Installing FEniCSx via pip..."
    
    # Install fenics-basix first
    python3 -m pip install --user fenics-basix
    
    # Install fenics-ufl
    python3 -m pip install --user fenics-ufl
    
    # Install fenics-ffcx
    python3 -m pip install --user fenics-ffcx
    
    # Install fenics-dolfinx (main package)
    python3 -m pip install --user fenics-dolfinx
    
    print_success "FEniCSx installed via pip!"
fi

# Install additional useful packages
print_status "Installing additional scientific packages..."
python3 -m pip install --user \
    meshio \
    pyvista \
    gmsh \
    pygmsh \
    matplotlib \
    seaborn \
    pandas \
    jupyter \
    ipython

# Verify installation
print_status "Verifying FEniCSx installation..."

# Create a test script
cat > test_fenicsx.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify FEniCSx installation."""

import sys

def test_import(module_name, description):
    try:
        __import__(module_name)
        print(f"✅ {description}: OK")
        return True
    except ImportError as e:
        print(f"❌ {description}: FAILED - {e}")
        return False

def main():
    print("🧪 Testing FEniCSx Installation")
    print("=" * 40)
    
    success_count = 0
    total_tests = 0
    
    # Test core components
    tests = [
        ("dolfinx", "DOLFINx (main FEniCSx library)"),
        ("basix", "Basix (finite element basis functions)"),
        ("ufl", "UFL (Unified Form Language)"),
        ("ffcx", "FFCx (FEniCS Form Compiler)"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("h5py", "HDF5 support"),
        ("mpi4py", "MPI support"),
    ]
    
    for module, description in tests:
        if test_import(module, description):
            success_count += 1
        total_tests += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {success_count}/{total_tests} passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! FEniCSx is ready to use.")
        return 0
    else:
        print("⚠️  Some components failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# Run the test
print_status "Running installation verification test..."
python3 test_fenicsx.py

# Check test result
if [ $? -eq 0 ]; then
    print_success "FEniCSx installation verification passed!"
else
    print_warning "Some components may not be working correctly."
fi

# Create a simple FEniCSx example
print_status "Creating a simple FEniCSx example..."

cat > fenicsx_example.py << 'EOF'
#!/usr/bin/env python3
"""
Simple FEniCSx example: Poisson equation on a unit square.
This demonstrates that FEniCSx is working correctly.
"""

import numpy as np
try:
    import dolfinx
    from dolfinx import mesh, fem, io, nls, log
    from dolfinx.fem.petsc import LinearProblem
    import ufl
    from mpi4py import MPI
    from petsc4py.PETSc import ScalarType
    
    def main():
        print("🧮 Running simple FEniCSx Poisson equation example...")
        
        # Create mesh
        domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.triangle)
        
        # Define function space
        V = fem.FunctionSpace(domain, ("Lagrange", 1))
        
        # Define boundary condition
        uD = fem.Function(V)
        uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
        
        # Create boundary condition
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(domain.topology)
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(uD, boundary_dofs)
        
        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        f = fem.Constant(domain, ScalarType(-6))
        a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f * v * ufl.dx
        
        # Solve linear problem
        problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()
        
        # Compute L2 error
        V2 = fem.FunctionSpace(domain, ("Lagrange", 2))
        uex = fem.Function(V2)
        uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
        
        # Interpolate finite element solution on higher-order space
        u_ex = fem.Function(V2)
        u_ex.interpolate(uh)
        
        # Compute L2 error
        error_L2 = fem.form(ufl.inner(u_ex - uex, u_ex - uex) * ufl.dx)
        error_local = fem.assemble_scalar(error_L2)
        error_global = domain.comm.allreduce(error_local, op=MPI.SUM)
        
        print(f"L2 error: {np.sqrt(error_global):.2e}")
        print("✅ FEniCSx example completed successfully!")
        
        return 0

    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ Cannot run FEniCSx example: {e}")
    print("Please check your FEniCSx installation.")
EOF

# Run the example
print_status "Running FEniCSx example..."
python3 fenicsx_example.py

# Set up environment variables
print_status "Setting up environment variables..."

# Add to bashrc if not already present
if ! grep -q "# FEniCSx environment" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# FEniCSx environment" >> ~/.bashrc
    echo "export PYTHONPATH=\$HOME/.local/lib/python3.*/site-packages:\$PYTHONPATH" >> ~/.bashrc
    echo "export PATH=\$HOME/.local/bin:\$PATH" >> ~/.bashrc
    print_success "Environment variables added to ~/.bashrc"
else
    print_status "Environment variables already configured"
fi

# Clean up test files
rm -f test_fenicsx.py fenicsx_example.py

# Final summary
echo ""
echo "🎉 FEniCSx Installation Complete!"
echo "=================================="
print_success "FEniCSx has been installed successfully!"
echo ""
echo "📋 What was installed:"
echo "  • DOLFINx (main FEniCSx library)"
echo "  • Basix (finite element basis functions)"
echo "  • UFL (Unified Form Language)"
echo "  • FFCx (FEniCS Form Compiler)"
echo "  • All required dependencies"
echo "  • Additional scientific packages"
echo ""
echo "🔧 Next steps:"
echo "  1. Restart your terminal or run: source ~/.bashrc"
echo "  2. Test the installation with: python3 -c 'import dolfinx; print(\"FEniCSx works!\")'"
echo "  3. Check the FEniCSx documentation: https://docs.fenicsx.org/"
echo ""
echo "💡 Troubleshooting:"
echo "  • If you encounter import errors, try: pip3 install --user --upgrade fenics-dolfinx"
echo "  • For MPI issues, ensure OpenMPI is properly configured"
echo "  • Check system requirements: https://github.com/FEniCS/dolfinx"
echo ""
print_success "Installation script completed!"