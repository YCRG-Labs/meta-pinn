import torch
import torch.nn as nn
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import cfg

class FourierFeatureTransform(nn.Module):
    """
    Fourier feature mapping for better convergence on high-frequency functions
    """
    def __init__(self, input_dim, mapping_size=64, scale=10, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        self.device = device if device is not None else torch.device('cpu')
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size, device=self.device) * scale, requires_grad=False)
        
    def forward(self, x):
        # Ensure input is on the same device as parameters
        x = x.to(self.B.device)
        x_proj = x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class MLP(nn.Module):
    """
    Multi-layer perceptron with specified architecture and optional Fourier features
    """
    def __init__(self, layers, activation=nn.Tanh(), use_fourier_features=False, fourier_scale=10, device=None):
        super(MLP, self).__init__()
        self.layers = layers
        self.activation = activation
        self.use_fourier_features = use_fourier_features
        self.device = device if device is not None else torch.device('cpu')
        
        if use_fourier_features:
            self.fourier_transform = FourierFeatureTransform(
                input_dim=layers[0], 
                mapping_size=64, 
                scale=fourier_scale,
                device=self.device
            )
            # Adjust first layer to account for expanded Fourier features
            self.net = self._build_network_with_fourier()
        else:
            self.net = self._build_network()
        
        # Move the entire network to the specified device
        self.to(self.device)
        
    def _build_network(self):
        layers_list = []
        for i in range(len(self.layers) - 2):
            layers_list.append(nn.Linear(self.layers[i], self.layers[i+1], device=self.device))
            layers_list.append(self.activation)
        layers_list.append(nn.Linear(self.layers[-2], self.layers[-1], device=self.device))
        return nn.Sequential(*layers_list)
    
    def _build_network_with_fourier(self):
        # First layer takes expanded Fourier features (2*mapping_size)
        fourier_output_dim = 2 * 64  # 2 * mapping_size
        
        layers_list = []
        # First layer after Fourier transform
        layers_list.append(nn.Linear(fourier_output_dim, self.layers[1], device=self.device))
        layers_list.append(self.activation)
        
        # Remaining layers
        for i in range(1, len(self.layers) - 2):
            layers_list.append(nn.Linear(self.layers[i], self.layers[i+1], device=self.device))
            layers_list.append(self.activation)
        
        layers_list.append(nn.Linear(self.layers[-2], self.layers[-1], device=self.device))
        return nn.Sequential(*layers_list)
    
    def forward(self, x):
        # Ensure input is on the same device as parameters
        x = x.to(self.device)
        if self.use_fourier_features:
            x = self.fourier_transform(x)
        return self.net(x)

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for inferring spatially varying viscosity
    with support for Navier-Stokes equations and advanced features
    """
    def __init__(self, config=None):
        super(PINN, self).__init__()
        if config is None:
            config = cfg
            
        self.config = config
        self.device = config.DEVICE
        
        # Neural network for predicting (u, v, p)
        self.net = MLP(
            config.PINN_LAYERS, 
            activation=nn.Tanh(),
            use_fourier_features=config.USE_FOURIER_FEATURES,
            fourier_scale=config.FOURIER_SCALE,
            device=self.device
        )
        
        # Trainable parameter for viscosity variation: nu(y) = nu_base + a * y
        # Initialize with a better starting point
        initial_a = torch.tensor([0.05], device=self.device)  # Start closer to true value
        self.a_param = nn.Parameter(initial_a)
        
        # Known base viscosity (fixed)
        self.nu_base = config.NU_BASE_TRUE
        
        # Domain bounds
        self.x_min, self.x_max = config.X_MIN, config.X_MAX
        self.y_min, self.y_max = config.Y_MIN, config.Y_MAX
        
        # For unsteady flow (if enabled)
        self.unsteady = config.UNSTEADY_FLOW
        self.t_min, self.t_max = config.T_MIN, config.T_MAX
        
        # For adaptive loss weighting
        self.use_adaptive_weights = config.USE_ADAPTIVE_WEIGHTS
        if self.use_adaptive_weights:
            # Initialize log weights for each loss term with better values
            self.log_weight_pde_momentum_x = nn.Parameter(torch.tensor([0.0], device=self.device))
            self.log_weight_pde_momentum_y = nn.Parameter(torch.tensor([0.0], device=self.device))
            self.log_weight_pde_continuity = nn.Parameter(torch.tensor([0.0], device=self.device))
            self.log_weight_bc = nn.Parameter(torch.tensor([2.0], device=self.device))  # Higher initial BC weight
            self.log_weight_data = nn.Parameter(torch.tensor([2.0], device=self.device))  # Higher initial data weight
        
        # Move the entire model to the specified device
        self.to(self.device)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, 2] containing (x,y) coordinates
               or [batch_size, 3] containing (x,y,t) if unsteady
            
        Returns:
            Tensor of shape [batch_size, 3] containing (u,v,p) predictions
        """
        # Ensure input is on the same device as model
        x = x.to(self.device)
        return self.net(x)
    
    def uvp(self, x, y, t=None):
        """
        Predict velocity components and pressure at given coordinates
        
        Args:
            x: x-coordinates tensor
            y: y-coordinates tensor
            t: time coordinates tensor (optional, for unsteady flow)
            
        Returns:
            Tuple of (u, v, p) tensors
        """
        # Ensure inputs are tensors on the correct device
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)
            
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y = y.to(self.device)
            
        # Reshape if needed
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
            
        # Stack coordinates
        if self.unsteady and t is not None:
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.float32, device=self.device)
            else:
                t = t.to(self.device)
                
            if t.dim() == 1:
                t = t.unsqueeze(1)
            xy = torch.cat([x, y, t], dim=1)
        else:
            xy = torch.cat([x, y], dim=1)
        
        # Get predictions
        output = self.forward(xy)
        
        # Split into u, v, p
        u = output[:, 0:1]
        v = output[:, 1:2]
        p = output[:, 2:3]
        
        return u, v, p
    
    def get_viscosity(self, y):
        """
        Compute spatially varying viscosity: nu(y) = nu_base + a * y
        """
        # Add small epsilon to prevent zero viscosity
        return self.nu_base + self.a_param * y + 1e-6
    
    def pde_residual(self, x, y, t=None):
        """
        Calculate PDE residuals for the Navier-Stokes equations with varying viscosity
        
        Args:
            x: x-coordinates tensor
            y: y-coordinates tensor
            t: time coordinates tensor (optional, for unsteady flow)
            
        Returns:
            Dictionary of residuals for momentum and continuity equations
        """
        # Ensure inputs are tensors with gradients on the correct device
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=self.device, requires_grad=True)
        else:
            x = x.to(self.device)
            x.requires_grad_(True)
            
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32, device=self.device, requires_grad=True)
        else:
            y = y.to(self.device)
            y.requires_grad_(True)
            
        # Reshape if needed
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
            
        # Handle time for unsteady flow
        if self.unsteady and t is not None:
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.float32, device=self.device, requires_grad=True)
            else:
                t = t.to(self.device)
                t.requires_grad_(True)
                
            if t.dim() == 1:
                t = t.unsqueeze(1)
            xy = torch.cat([x, y, t], dim=1)
        else:
            xy = torch.cat([x, y], dim=1)
        
        # Get predictions
        output = self.forward(xy)
        u = output[:, 0:1]
        v = output[:, 1:2]
        p = output[:, 2:3]
        
        # Create ones_like tensors on the correct device
        ones_u = torch.ones_like(u, device=self.device)
        ones_v = torch.ones_like(v, device=self.device)
        ones_p = torch.ones_like(p, device=self.device)
        
        # Calculate spatial derivatives
        # First-order derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=ones_u, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=ones_u, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=ones_v, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=ones_v, create_graph=True)[0]
        p_x = torch.autograd.grad(p, x, grad_outputs=ones_p, create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=ones_p, create_graph=True)[0]
        
        # Create ones_like tensors for second derivatives
        ones_u_x = torch.ones_like(u_x, device=self.device)
        ones_u_y = torch.ones_like(u_y, device=self.device)
        ones_v_x = torch.ones_like(v_x, device=self.device)
        ones_v_y = torch.ones_like(v_y, device=self.device)
        
        # Second-order derivatives
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones_u_x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones_u_y, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones_v_x, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=ones_v_y, create_graph=True)[0]
        
        # Calculate viscosity and its derivative
        nu = self.get_viscosity(y)
        ones_nu = torch.ones_like(nu, device=self.device)
        nu_y = torch.autograd.grad(nu, y, grad_outputs=ones_nu, create_graph=True)[0]
        
        # Time derivatives for unsteady flow
        if self.unsteady and t is not None:
            u_t = torch.autograd.grad(u, t, grad_outputs=ones_u, create_graph=True)[0]
            v_t = torch.autograd.grad(v, t, grad_outputs=ones_v, create_graph=True)[0]
        
        # Reynolds number for Navier-Stokes
        Re = self.config.REYNOLDS_NUMBER
        
        # Navier-Stokes equations with varying viscosity
        if self.unsteady and t is not None:
            # Unsteady Navier-Stokes
            # Momentum-x: u_t + u*u_x + v*u_y = -p_x + (1/Re) * [div(nu * grad(u))]
            momentum_x = u_t + u*u_x + v*u_y + p_x - (1/Re) * (nu*u_xx + nu_y*u_y + nu*u_yy)
            
            # Momentum-y: v_t + u*v_x + v*v_y = -p_y + (1/Re) * [div(nu * grad(v))]
            momentum_y = v_t + u*v_x + v*v_y + p_y - (1/Re) * (nu*v_xx + nu_y*v_y + nu*v_yy)
        else:
            # Steady Navier-Stokes
            # Momentum-x: u*u_x + v*u_y = -p_x + (1/Re) * [div(nu * grad(u))]
            momentum_x = u*u_x + v*u_y + p_x - (1/Re) * (nu*u_xx + nu_y*u_y + nu*u_yy)
            
            # Momentum-y: u*v_x + v*v_y = -p_y + (1/Re) * [div(nu * grad(v))]
            momentum_y = u*v_x + v*v_y + p_y - (1/Re) * (nu*v_xx + nu_y*v_y + nu*v_yy)
        
        # Continuity: div(u) = 0
        continuity = u_x + v_y
        
        return {
            'momentum_x': momentum_x,
            'momentum_y': momentum_y,
            'continuity': continuity
        }
    
    def boundary_conditions(self, boundary_points):
        """
        Calculate boundary condition residuals
        
        Args:
            boundary_points: Dictionary with keys 'inlet', 'outlet', 'walls' containing boundary points
            
        Returns:
            Dictionary of boundary condition residuals
        """
        bc_residuals = {}
        
        # Inlet boundary: u = parabolic profile, v = 0
        inlet_points = boundary_points['inlet'].to(self.device)
        x_inlet = inlet_points[:, 0:1]
        y_inlet = inlet_points[:, 1:2]
        
        # Add time dimension for unsteady flow if needed
        if self.unsteady and inlet_points.shape[1] > 2:
            t_inlet = inlet_points[:, 2:3]
            u_inlet, v_inlet, _ = self.uvp(x_inlet, y_inlet, t_inlet)
        else:
            u_inlet, v_inlet, _ = self.uvp(x_inlet, y_inlet)
        
        # Calculate parabolic profile
        h = self.y_max - self.y_min
        u_inlet_target = 4 * self.config.U_MAX_INLET * (y_inlet - self.y_min) * (self.y_max - y_inlet) / (h**2)
        
        bc_residuals['inlet_u'] = u_inlet - u_inlet_target
        bc_residuals['inlet_v'] = v_inlet
        
        # Outlet boundary: Natural outflow (no explicit BC in PINN, handled through PDE)
        # Could add p=0 or other conditions if needed
        
        # Wall boundaries: u = 0, v = 0 (no-slip)
        wall_points = boundary_points['walls'].to(self.device)
        x_wall = wall_points[:, 0:1]
        y_wall = wall_points[:, 1:2]
        
        # Add time dimension for unsteady flow if needed
        if self.unsteady and wall_points.shape[1] > 2:
            t_wall = wall_points[:, 2:3]
            u_wall, v_wall, _ = self.uvp(x_wall, y_wall, t_wall)
        else:
            u_wall, v_wall, _ = self.uvp(x_wall, y_wall)
        
        bc_residuals['wall_u'] = u_wall
        bc_residuals['wall_v'] = v_wall
        
        return bc_residuals
    
    def get_adaptive_weights(self):
        """
        Get the current adaptive weights for loss terms
        
        Returns:
            Dictionary of weights for different loss terms
        """
        if not self.use_adaptive_weights:
            return {
                'momentum_x': 1.0,
                'momentum_y': 1.0,
                'continuity': 1.0,
                'bc': self.config.WEIGHT_BC,
                'data_u': self.config.WEIGHT_DATA_U,
                'data_v': self.config.WEIGHT_DATA_V,
                'data_p': self.config.WEIGHT_DATA_P
            }
        
        # Convert log weights to actual weights
        w_momentum_x = torch.exp(self.log_weight_pde_momentum_x)
        w_momentum_y = torch.exp(self.log_weight_pde_momentum_y)
        w_continuity = torch.exp(self.log_weight_pde_continuity)
        w_bc = torch.exp(self.log_weight_bc)
        w_data = torch.exp(self.log_weight_data)
        
        return {
            'momentum_x': w_momentum_x.item(),
            'momentum_y': w_momentum_y.item(),
            'continuity': w_continuity.item(),
            'bc': w_bc.item(),
            'data_u': w_data.item() * self.config.WEIGHT_DATA_U / self.config.WEIGHT_DATA_P,
            'data_v': w_data.item() * self.config.WEIGHT_DATA_V / self.config.WEIGHT_DATA_P,
            'data_p': w_data.item()
        }
    
    def get_inferred_viscosity_param(self):
        """
        Get the inferred viscosity parameter a
        """
        return self.a_param.item()
    
    def reinitialize_parameters(self, scale=0.1):
        """
        Reinitialize network parameters while keeping the viscosity parameter
        """
        # Store current viscosity parameter
        current_a = self.a_param.item()
        
        # Reinitialize network parameters
        for param in self.net.parameters():
            if len(param.shape) > 1:  # Weight matrices
                nn.init.xavier_normal_(param, gain=scale)
            else:  # Bias vectors
                nn.init.zeros_(param)
        
        # Restore viscosity parameter
        with torch.no_grad():
            self.a_param.copy_(torch.tensor([current_a], device=self.device))
    
    def save(self, filename):
        """
        Save the model to a file
        
        Args:
            filename: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'a_param': self.a_param.item(),
            'config': {
                'layers': self.config.PINN_LAYERS,
                'nu_base': self.nu_base,
                'x_min': self.x_min,
                'x_max': self.x_max,
                'y_min': self.y_min,
                'y_max': self.y_max,
                'unsteady': self.unsteady,
                'use_fourier_features': self.config.USE_FOURIER_FEATURES,
                'use_adaptive_weights': self.use_adaptive_weights
            }
        }, filename)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename, config=None):
        """
        Load a model from a file
        
        Args:
            filename: Path to the saved model
            config: Project configuration (optional)
            
        Returns:
            Loaded PINN model
        """
        if config is None:
            config = cfg
            
        checkpoint = torch.load(filename, map_location=config.DEVICE)
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print loaded parameters
        print(f"Model loaded from {filename}")
        print(f"Inferred viscosity parameter a: {checkpoint['a_param']:.6f}")
        
        return model

if __name__ == "__main__":
    print("="*70)
    print("Testing PINN model")
    print("="*70)
    
    # Create model
    model = PINN()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Initial viscosity parameter a: {model.get_inferred_viscosity_param():.6f}")
    
    # Test forward pass
    x = torch.linspace(cfg.X_MIN, cfg.X_MAX, 10, device=cfg.DEVICE)
    y = torch.linspace(cfg.Y_MIN, cfg.Y_MAX, 10, device=cfg.DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    
    u, v, p = model.uvp(x_flat, y_flat)
    print(f"Forward pass output shapes: u={u.shape}, v={v.shape}, p={p.shape}")
    
    # Test PDE residual
    residuals = model.pde_residual(x_flat[:5], y_flat[:5])  # Test with fewer points for speed
    for key, value in residuals.items():
        print(f"PDE residual {key}: shape={value.shape}, mean={value.abs().mean().item():.6f}")
    
    # Test viscosity function
    nu = model.get_viscosity(y_flat)
    print(f"Viscosity shape: {nu.shape}, range: [{nu.min().item():.6f}, {nu.max().item():.6f}]")
    
    # Test save and load
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    test_save_path = f"{cfg.OUTPUT_DIR}/test_model.pth"
    model.save(test_save_path)
    
    loaded_model = PINN.load(test_save_path)
    print(f"Loaded model parameter a: {loaded_model.get_inferred_viscosity_param():.6f}")
    
    # Clean up test file
    os.remove(test_save_path)
    
    print("\n" + "="*70)
    print("PINN model test complete")
    print("="*70)