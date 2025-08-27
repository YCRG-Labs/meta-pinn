"""
Integration tests for OperatorMetaPINN comparing operator-enhanced vs pure meta-learning.

This module tests the integration of neural operators with meta-learning and
validates the performance improvements from operator initialization.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any

from ml_research_pipeline.neural_operators.operator_meta_pinn import OperatorMetaPINN
from ml_research_pipeline.core.meta_pinn import MetaPINN
from ml_research_pipeline.config.model_config import MetaPINNConfig
from ml_research_pipeline.core.task_generator import FluidTaskGenerator


class TestOperatorMetaPINNIntegration:
    """Test suite for OperatorMetaPINN integration functionality"""
    
    @pytest.fixture
    def meta_pinn_config(self):
        """Create MetaPINN configuration for testing"""
        return MetaPINNConfig(
            input_dim=3,  # x, y, t
            output_dim=3,  # u, v, p
            hidden_layers=[64, 64, 64],
            activation="tanh",
            meta_lr=0.001,
            adapt_lr=0.01,
            adaptation_steps=5,
            physics_loss_weight=1.0
        )
    
    @pytest.fixture
    def fno_config(self):
        """Create FNO configuration for testing"""
        return {
            'modes1': 8,
            'modes2': 8,
            'width': 32,
            'input_channels': 3,
            'output_channels': 1,
            'n_layers': 2,
            'grid_size': (64, 64)
        }
    
    @pytest.fixture
    def deeponet_config(self):
        """Create DeepONet configuration for testing"""
        return {
            'branch_layers': [64, 64],
            'trunk_layers': [64, 64],
            'measurement_dim': 50,
            'coordinate_dim': 2,
            'latent_dim': 50,
            'output_dim': 1,
            'activation': 'tanh'
        }
    
    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing"""
        batch_size = 16
        n_support = 20
        n_query = 30
        
        # Generate sample coordinates and data
        support_coords = torch.rand(n_support, 3)  # x, y, t
        support_data = torch.rand(n_support, 3)    # u, v, p
        query_coords = torch.rand(n_query, 3)
        query_data = torch.rand(n_query, 3)
        
        # Sample measurements for operators
        measurements = torch.rand(50, 2)  # 50 measurement points with 2 values each
        
        # Sparse observations for FNO
        sparse_coords = torch.rand(10, 2)  # 10 sparse observation points
        sparse_values = torch.rand(10, 1)  # Values at sparse points
        grid_coords = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, 64),
            torch.linspace(0, 1, 64),
            indexing='ij'
        ), dim=-1)
        
        sparse_observations = {
            'coords': sparse_coords,
            'values': sparse_values,
            'grid_coords': grid_coords
        }
        
        task_info = {
            'viscosity_type': 'linear',
            'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.0},
            'reynolds': 100.0
        }
        
        return {
            'support_coords': support_coords,
            'support_data': support_data,
            'query_coords': query_coords,
            'query_data': query_data,
            'measurements': measurements,
            'sparse_observations': sparse_observations,
            'task_info': task_info
        }
    
    def test_operator_meta_pinn_initialization(self, meta_pinn_config, fno_config, deeponet_config):
        """Test OperatorMetaPINN initialization with different operator types"""
        
        # Test FNO only
        model_fno = OperatorMetaPINN(
            config=meta_pinn_config,
            operator_type='fno',
            fno_config=fno_config
        )
        assert model_fno.fno is not None
        assert model_fno.deeponet is None
        assert not hasattr(model_fno, 'fusion_network') or model_fno.fusion_network is None
        
        # Test DeepONet only
        model_deeponet = OperatorMetaPINN(
            config=meta_pinn_config,
            operator_type='deeponet',
            deeponet_config=deeponet_config
        )
        assert model_deeponet.fno is None
        assert model_deeponet.deeponet is not None
        
        # Test both operators
        model_both = OperatorMetaPINN(
            config=meta_pinn_config,
            operator_type='both',
            fno_config=fno_config,
            deeponet_config=deeponet_config
        )
        assert model_both.fno is not None
        assert model_both.deeponet is not None
        assert model_both.fusion_network is not None
    
    def test_operator_predictions(self, meta_pinn_config, fno_config, deeponet_config, sample_task):
        """Test operator prediction functionality"""
        
        model = OperatorMetaPINN(
            config=meta_pinn_config,
            operator_type='both',
            fno_config=fno_config,
            deeponet_config=deeponet_config
        )
        
        # Test operator predictions
        # Use 2D coordinates for DeepONet (it expects coordinate_dim=2)
        coords_2d = sample_task['query_coords'][:, :2].unsqueeze(0)  # Add batch dimension
        predictions = model.predict_with_operators(
            coords=coords_2d,
            measurements=sample_task['measurements'].unsqueeze(0),
            sparse_observations=sample_task['sparse_observations']
        )
        
        assert 'fno' in predictions
        assert 'deeponet' in predictions
        # Note: 'fused' may not be present if shapes don't match
        
        # Check shapes
        fno_pred = predictions['fno']
        deeponet_pred = predictions['deeponet']
        
        assert fno_pred.shape[0] == 1  # batch size
        assert deeponet_pred.shape[1] == coords_2d.shape[1]  # n_query points
    
    def test_operator_guided_initialization(self, meta_pinn_config, fno_config, deeponet_config, sample_task):
        """Test operator-guided parameter initialization"""
        
        model = OperatorMetaPINN(
            config=meta_pinn_config,
            operator_type='both',
            fno_config=fno_config,
            deeponet_config=deeponet_config
        )
        
        # Test initialization
        base_params = model.meta_pinn.clone_parameters()
        initialized_params = model.operator_guided_initialization(sample_task, base_params)
        
        # Check that we get parameters
        assert isinstance(initialized_params, dict)
        assert len(initialized_params) > 0
        
        # Check that parameters have correct shapes
        original_params = model.meta_pinn.clone_parameters()
        for name in original_params.keys():
            assert name in initialized_params
            assert initialized_params[name].shape == original_params[name].shape
        
        # Check that parameters are actually different (not just copied)
        param_differences = []
        for name in original_params.keys():
            diff = torch.norm(initialized_params[name] - original_params[name]).item()
            param_differences.append(diff)
        
        # At least some parameters should be different
        assert max(param_differences) > 1e-6
    
    def test_adaptation_with_operators(self, meta_pinn_config, fno_config, deeponet_config, sample_task):
        """Test adaptation process with operator initialization"""
        
        model = OperatorMetaPINN(
            config=meta_pinn_config,
            operator_type='both',
            fno_config=fno_config,
            deeponet_config=deeponet_config
        )
        
        # Test adaptation with operators
        adapted_params_with_ops = model.adapt_to_task(
            sample_task,
            adaptation_steps=3,
            use_operator_initialization=True
        )
        
        # Test adaptation without operators
        adapted_params_without_ops = model.adapt_to_task(
            sample_task,
            adaptation_steps=3,
            use_operator_initialization=False
        )
        
        # Check that both return valid parameters
        assert isinstance(adapted_params_with_ops, dict)
        assert isinstance(adapted_params_without_ops, dict)
        assert len(adapted_params_with_ops) == len(adapted_params_without_ops)
        
        # Check that the adaptations are different
        param_differences = []
        for name in adapted_params_with_ops.keys():
            diff = torch.norm(
                adapted_params_with_ops[name] - adapted_params_without_ops[name]
            ).item()
            param_differences.append(diff)
        
        # Adaptations should be different
        assert max(param_differences) > 1e-6
    
    def test_joint_training_loss(self, meta_pinn_config, fno_config, deeponet_config, sample_task):
        """Test joint training loss computation"""
        
        model = OperatorMetaPINN(
            config=meta_pinn_config,
            operator_type='both',
            fno_config=fno_config,
            deeponet_config=deeponet_config,
            joint_training=True
        )
        
        # Get operator predictions
        operator_predictions = model.predict_with_operators(
            coords=sample_task['support_coords'],
            measurements=sample_task['measurements'].unsqueeze(0),
            sparse_observations=sample_task['sparse_observations']
        )
        
        # Test joint loss computation
        adapted_params = model.meta_pinn.clone_parameters()
        losses = model.compute_joint_loss(sample_task, adapted_params, operator_predictions)
        
        assert 'meta_total_loss' in losses
        assert 'operator_loss' in losses
        assert 'total_joint_loss' in losses
        
        # Check that losses are valid tensors
        assert isinstance(losses['meta_total_loss'], torch.Tensor)
        assert isinstance(losses['operator_loss'], torch.Tensor)
        assert isinstance(losses['total_joint_loss'], torch.Tensor)
        
        # Check that losses are positive
        assert losses['meta_total_loss'].item() >= 0
        assert losses['operator_loss'].item() >= 0
        assert losses['total_joint_loss'].item() >= 0
    
    def test_meta_update_step(self, meta_pinn_config, fno_config, deeponet_config, sample_task):
        """Test meta-update step for joint training"""
        
        model = OperatorMetaPINN(
            config=meta_pinn_config,
            operator_type='both',
            fno_config=fno_config,
            deeponet_config=deeponet_config,
            joint_training=True
        )
        
        # Create optimizer
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Perform meta-update
        task_batch = [sample_task]
        loss_dict = model.meta_update(task_batch, meta_optimizer)
        
        # Check that loss dictionary is returned
        assert isinstance(loss_dict, dict)
        assert 'total_loss' in loss_dict
        assert 'meta_loss' in loss_dict
        assert 'operator_loss' in loss_dict
        
        # Check that parameters were updated
        params_changed = any(
            not torch.equal(initial, current) 
            for initial, current in zip(initial_params, model.parameters())
        )
        
        assert params_changed, "Model parameters should be updated"
    
    def test_adaptation_speed_evaluation(self, meta_pinn_config, fno_config, deeponet_config, sample_task):
        """Test adaptation speed evaluation functionality"""
        
        model = OperatorMetaPINN(
            config=meta_pinn_config,
            operator_type='fno',  # Use only FNO for simpler test
            fno_config=fno_config
        )
        
        # Evaluate adaptation speed
        results = model.evaluate_adaptation_speed(
            sample_task,
            max_steps=5,
            tolerance=1e-3
        )
        
        # Check results structure
        assert 'pure_meta_learning' in results
        assert 'operator_enhanced' in results
        assert 'improvement' in results
        
        # Check that results contain expected fields
        assert 'losses' in results['pure_meta_learning']
        assert 'convergence_steps' in results['pure_meta_learning']
        assert 'final_loss' in results['pure_meta_learning']
        
        assert 'losses' in results['operator_enhanced']
        assert 'convergence_steps' in results['operator_enhanced']
        assert 'final_loss' in results['operator_enhanced']
        
        assert 'speed_improvement' in results['improvement']
        assert 'final_loss_improvement' in results['improvement']
        
        # Check that results are reasonable
        assert results['pure_meta_learning']['convergence_steps'] > 0
        assert results['operator_enhanced']['convergence_steps'] > 0
        assert results['improvement']['speed_improvement'] > 0
        assert results['improvement']['final_loss_improvement'] > 0
    
    def test_training_mode_control(self, meta_pinn_config, fno_config, deeponet_config):
        """Test training mode control for different components"""
        
        model = OperatorMetaPINN(
            config=meta_pinn_config,
            operator_type='both',
            fno_config=fno_config,
            deeponet_config=deeponet_config
        )
        
        # Test setting training modes
        model.set_training_modes(meta_training=True, operator_training=False)
        
        # Check meta-learning parameters
        for param in model.meta_pinn.parameters():
            assert param.requires_grad == True
        
        # Check operator parameters
        for param in model.fno.parameters():
            assert param.requires_grad == False
        for param in model.deeponet.parameters():
            assert param.requires_grad == False
        
        # Test opposite setting
        model.set_training_modes(meta_training=False, operator_training=True)
        
        for param in model.meta_pinn.parameters():
            assert param.requires_grad == False
        for param in model.fno.parameters():
            assert param.requires_grad == True
        for param in model.deeponet.parameters():
            assert param.requires_grad == True
    
    def test_model_info_retrieval(self, meta_pinn_config, fno_config, deeponet_config):
        """Test model information retrieval"""
        
        model = OperatorMetaPINN(
            config=meta_pinn_config,
            operator_type='both',
            fno_config=fno_config,
            deeponet_config=deeponet_config
        )
        
        info = model.get_model_info()
        
        # Check required fields
        assert 'operator_type' in info
        assert 'joint_training' in info
        assert 'initialization_strategy' in info
        assert 'meta_pinn_params' in info
        assert 'total_params' in info
        assert 'fno_info' in info
        assert 'deeponet_info' in info
        
        # Check values
        assert info['operator_type'] == 'both'
        assert isinstance(info['meta_pinn_params'], int)
        assert isinstance(info['total_params'], int)
        assert info['total_params'] >= info['meta_pinn_params']
    
    def test_forward_pass_compatibility(self, meta_pinn_config, fno_config, deeponet_config, sample_task):
        """Test that forward pass is compatible with base MetaPINN"""
        
        # Create both models
        operator_model = OperatorMetaPINN(
            config=meta_pinn_config,
            operator_type='fno',
            fno_config=fno_config
        )
        
        base_model = MetaPINN(meta_pinn_config)
        
        # Test forward pass
        coords = sample_task['support_coords']
        
        # Both should produce outputs of same shape
        operator_output = operator_model.forward(coords)
        base_output = base_model.forward(coords)
        
        assert operator_output.shape == base_output.shape
        assert operator_output.shape == (coords.shape[0], meta_pinn_config.output_dim)
    
    @pytest.mark.parametrize("operator_type", ['fno', 'deeponet', 'both'])
    def test_different_operator_types(self, meta_pinn_config, fno_config, deeponet_config, sample_task, operator_type):
        """Test functionality with different operator types"""
        
        kwargs = {'config': meta_pinn_config, 'operator_type': operator_type}
        
        if operator_type in ['fno', 'both']:
            kwargs['fno_config'] = fno_config
        if operator_type in ['deeponet', 'both']:
            kwargs['deeponet_config'] = deeponet_config
        
        model = OperatorMetaPINN(**kwargs)
        
        # Test adaptation
        adapted_params = model.adapt_to_task(sample_task, adaptation_steps=2)
        assert isinstance(adapted_params, dict)
        assert len(adapted_params) > 0
        
        # Test forward pass
        output = model.forward(sample_task['support_coords'])
        expected_shape = (sample_task['support_coords'].shape[0], meta_pinn_config.output_dim)
        assert output.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__])