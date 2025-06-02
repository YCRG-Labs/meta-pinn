#!/usr/bin/env python3
"""
FastAPI Application for PINN Model Inference

This FastAPI application provides REST endpoints to run PINN model inference
and return data suitable for frontend plotting.

Usage:
    uvicorn app:app --reload --host 0.0.0.0 --port 5000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import os
import sys
import json
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import uuid
import asyncio
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import interactive.py functions
from interactive import (
    create_inference_config,
    load_trained_model,
    infer_3d_flow_field,
    infer_boundary_analysis,
    infer_centerline_analysis,
    infer_viscosity_profile,
    export_inference_summary,
    run_inference_session
)

# Initialize FastAPI app
app = FastAPI(
    title="PINN Inference API",
    description="API for running Physics-Informed Neural Network inference and generating plotting data",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default model path
DEFAULT_MODEL_PATH = "/home/brand/pinn_viscosity/backend/results/trained_model.pth"

# Pydantic models for API
class InferenceParameters(BaseModel):
    """Parameters for inference configuration"""
    reynolds_number: Optional[float] = Field(default=100, description="Reynolds number for the flow")
    nu_base_true: Optional[float] = Field(default=0.01, description="Base viscosity value")
    a_true: Optional[float] = Field(default=None, description="Reference viscosity parameter for comparison")
    u_max_inlet: Optional[float] = Field(default=1.0, description="Maximum inlet velocity")
    x_max: Optional[float] = Field(default=2.0, description="Domain width")
    y_max: Optional[float] = Field(default=1.0, description="Domain height")
    x_min: Optional[float] = Field(default=0.0, description="Domain x minimum")
    y_min: Optional[float] = Field(default=0.0, description="Domain y minimum")
    n_grid_x: Optional[int] = Field(default=100, description="Grid points in X direction")
    n_grid_y: Optional[int] = Field(default=50, description="Grid points in Y direction")
    n_time_slices: Optional[int] = Field(default=5, description="Number of time slices for unsteady flow")
    name: Optional[str] = Field(default="API Inference", description="Inference session name")

class InferenceRequest(BaseModel):
    """Request model for inference"""
    parameters: InferenceParameters
    model_path: Optional[str] = Field(default=DEFAULT_MODEL_PATH, description="Path to trained model")
    include_boundary: Optional[bool] = Field(default=True, description="Include boundary analysis")
    include_centerline: Optional[bool] = Field(default=True, description="Include centerline analysis")
    include_viscosity: Optional[bool] = Field(default=True, description="Include viscosity profile")

class MultiInferenceRequest(BaseModel):
    """Request model for multiple inference scenarios"""
    scenarios: List[InferenceParameters]
    model_path: Optional[str] = Field(default=DEFAULT_MODEL_PATH, description="Path to trained model")
    include_boundary: Optional[bool] = Field(default=True, description="Include boundary analysis")
    include_centerline: Optional[bool] = Field(default=True, description="Include centerline analysis")
    include_viscosity: Optional[bool] = Field(default=True, description="Include viscosity profile")

class FlowFieldData(BaseModel):
    """Flow field data for plotting"""
    x: List[float] = Field(description="X coordinates")
    y: List[float] = Field(description="Y coordinates")
    u_velocity: List[float] = Field(description="U velocity component")
    v_velocity: List[float] = Field(description="V velocity component")
    pressure: List[float] = Field(description="Pressure field")
    velocity_magnitude: List[float] = Field(description="Velocity magnitude")
    viscosity: List[float] = Field(description="Viscosity field")
    vorticity: List[float] = Field(description="Vorticity field")
    grid_shape: List[int] = Field(description="Grid shape [nx, ny]")
    learned_viscosity_param: float = Field(description="Learned viscosity parameter")

class BoundaryData(BaseModel):
    """Boundary analysis data"""
    x: List[float] = Field(description="X coordinates")
    y: List[float] = Field(description="Y coordinates")
    u_velocity: List[float] = Field(description="U velocity")
    v_velocity: List[float] = Field(description="V velocity")
    pressure: List[float] = Field(description="Pressure")
    boundary_type: List[str] = Field(description="Boundary type labels")

class CenterlineData(BaseModel):
    """Centerline analysis data"""
    x: List[float] = Field(description="X coordinates along centerline")
    u_velocity: List[float] = Field(description="U velocity along centerline")
    pressure: List[float] = Field(description="Pressure along centerline")
    velocity_magnitude: List[float] = Field(description="Velocity magnitude")
    viscosity: List[float] = Field(description="Viscosity along centerline")

class ViscosityProfileData(BaseModel):
    """Viscosity profile data"""
    y: List[float] = Field(description="Y coordinates")
    viscosity_learned: List[float] = Field(description="Learned viscosity profile")
    viscosity_reference: List[Optional[float]] = Field(description="Reference viscosity profile")
    absolute_error: List[Optional[float]] = Field(description="Absolute error")
    relative_error_percent: List[Optional[float]] = Field(description="Relative error percentage")

class InferenceResponse(BaseModel):
    """Response model for inference results"""
    success: bool = Field(description="Whether inference was successful")
    session_id: str = Field(description="Unique session identifier")
    learned_viscosity_param: float = Field(description="Learned viscosity parameter")
    total_points: int = Field(description="Total number of inference points")
    processing_time: float = Field(description="Processing time in seconds")
    flow_field: Optional[FlowFieldData] = Field(description="Flow field data")
    boundary_data: Optional[BoundaryData] = Field(description="Boundary analysis data")
    centerline_data: Optional[CenterlineData] = Field(description="Centerline analysis data")
    viscosity_profile: Optional[ViscosityProfileData] = Field(description="Viscosity profile data")
    model_info: Dict[str, Any] = Field(description="Model information")
    error_message: Optional[str] = Field(description="Error message if failed")

class MultiInferenceResponse(BaseModel):
    """Response model for multiple inference scenarios"""
    success: bool = Field(description="Whether all inferences were successful")
    session_id: str = Field(description="Unique session identifier")
    total_scenarios: int = Field(description="Total number of scenarios")
    scenarios: List[InferenceResponse] = Field(description="Individual scenario results")
    summary: Dict[str, Any] = Field(description="Summary statistics")
    error_message: Optional[str] = Field(description="Error message if failed")

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    success: bool = Field(description="Whether model loading was successful")
    model_path: str = Field(description="Path to model file")
    learned_viscosity_param: float = Field(description="Learned viscosity parameter")
    model_architecture: List[int] = Field(description="Neural network architecture")
    uses_fourier_features: bool = Field(description="Whether model uses Fourier features")
    uses_adaptive_weights: bool = Field(description="Whether model uses adaptive weights")
    model_exists: bool = Field(description="Whether model file exists")
    error_message: Optional[str] = Field(description="Error message if failed")

# Global storage for temporary inference data
inference_cache = {}

def process_csv_to_lists(df: pd.DataFrame) -> Dict[str, List]:
    """Convert DataFrame to dictionary of lists, handling NaN values"""
    result = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            result[column] = df[column].fillna('').tolist()
        else:
            # Replace NaN with None for JSON serialization
            result[column] = df[column].where(pd.notna(df[column]), None).tolist()
    return result

def create_temp_directory() -> str:
    """Create a temporary directory for inference results"""
    temp_dir = tempfile.mkdtemp(prefix="pinn_inference_")
    return temp_dir

def cleanup_temp_directory(temp_dir: str):
    """Clean up temporary directory"""
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PINN Inference API",
        "version": "1.0.0",
        "endpoints": {
            "/model/info": "Get model information",
            "/inference/single": "Run single inference",
            "/inference/multiple": "Run multiple inference scenarios",
            "/inference/flow-field": "Get flow field data only",
            "/health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(model_path: str = DEFAULT_MODEL_PATH):
    """Get information about the trained model"""
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            return ModelInfoResponse(
                success=False,
                model_path=model_path,
                learned_viscosity_param=0.0,
                model_architecture=[],
                uses_fourier_features=False,
                uses_adaptive_weights=False,
                model_exists=False,
                error_message=f"Model file not found: {model_path}"
            )
        
        # Create temporary config for model loading
        temp_config = create_inference_config()
        
        # Load model to get information
        model, config = load_trained_model(model_path, temp_config)
        
        return ModelInfoResponse(
            success=True,
            model_path=model_path,
            learned_viscosity_param=model.get_inferred_viscosity_param(),
            model_architecture=config.PINN_LAYERS,
            uses_fourier_features=config.USE_FOURIER_FEATURES,
            uses_adaptive_weights=config.USE_ADAPTIVE_WEIGHTS,
            model_exists=True
        )
        
    except Exception as e:
        return ModelInfoResponse(
            success=False,
            model_path=model_path,
            learned_viscosity_param=0.0,
            model_architecture=[],
            uses_fourier_features=False,
            uses_adaptive_weights=False,
            model_exists=os.path.exists(model_path),
            error_message=str(e)
        )

@app.post("/inference/single", response_model=InferenceResponse)
async def run_single_inference(request: InferenceRequest):
    """Run single inference session"""
    session_id = str(uuid.uuid4())
    temp_dir = None
    
    try:
        # Create temporary directory
        temp_dir = create_temp_directory()
        
        # Check if model exists
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
        
        # Create inference configuration
        inference_config = create_inference_config(
            reynolds_number=request.parameters.reynolds_number,
            nu_base_true=request.parameters.nu_base_true,
            a_true=request.parameters.a_true,
            u_max_inlet=request.parameters.u_max_inlet,
            x_max=request.parameters.x_max,
            y_max=request.parameters.y_max,
            x_min=request.parameters.x_min,
            y_min=request.parameters.y_min,
            n_grid_x=request.parameters.n_grid_x,
            n_grid_y=request.parameters.n_grid_y,
            n_time_slices=request.parameters.n_time_slices,
            name=request.parameters.name
        )
        
        # Load trained model
        model, inference_config = load_trained_model(request.model_path, inference_config)
        
        # Set output directory to temp directory
        inference_config.OUTPUT_DIR = temp_dir
        
        # Run flow field inference
        import time
        start_time = time.time()
        
        flow_results = infer_3d_flow_field(model, inference_config, temp_dir)
        
        # Process flow field data
        flow_df = pd.read_csv(os.path.join(temp_dir, 'inferred_flow_3d_complete.csv'))
        flow_data = process_csv_to_lists(flow_df)
        
        flow_field = FlowFieldData(
            x=flow_data['x'],
            y=flow_data['y'],
            u_velocity=flow_data['u_velocity'],
            v_velocity=flow_data['v_velocity'],
            pressure=flow_data['pressure'],
            velocity_magnitude=flow_data['velocity_magnitude'],
            viscosity=flow_data['viscosity'],
            vorticity=flow_data['vorticity'],
            grid_shape=flow_results['grid_size'],
            learned_viscosity_param=model.get_inferred_viscosity_param()
        )
        
        # Optional analyses
        boundary_data = None
        centerline_data = None
        viscosity_profile = None
        
        if request.include_boundary:
            boundary_df = infer_boundary_analysis(model, inference_config, temp_dir)
            boundary_dict = process_csv_to_lists(boundary_df)
            boundary_data = BoundaryData(
                x=boundary_dict['x'],
                y=boundary_dict['y'],
                u_velocity=boundary_dict['u_velocity'],
                v_velocity=boundary_dict['v_velocity'],
                pressure=boundary_dict['pressure'],
                boundary_type=boundary_dict['boundary_type']
            )
        
        if request.include_centerline:
            centerline_df = infer_centerline_analysis(model, inference_config, temp_dir)
            centerline_dict = process_csv_to_lists(centerline_df)
            centerline_data = CenterlineData(
                x=centerline_dict['x'],
                u_velocity=centerline_dict['u_velocity'],
                pressure=centerline_dict['pressure'],
                velocity_magnitude=centerline_dict['velocity_magnitude'],
                viscosity=centerline_dict['viscosity']
            )
        
        if request.include_viscosity:
            viscosity_df = infer_viscosity_profile(model, inference_config, temp_dir)
            viscosity_dict = process_csv_to_lists(viscosity_df)
            viscosity_profile = ViscosityProfileData(
                y=viscosity_dict['y'],
                viscosity_learned=viscosity_dict['viscosity_learned'],
                viscosity_reference=viscosity_dict['viscosity_reference'],
                absolute_error=viscosity_dict['absolute_error'],
                relative_error_percent=viscosity_dict['relative_error_percent']
            )
        
        processing_time = time.time() - start_time
        
        # Store results in cache for potential retrieval
        inference_cache[session_id] = {
            'temp_dir': temp_dir,
            'timestamp': datetime.now(),
            'results': flow_results
        }
        
        return InferenceResponse(
            success=True,
            session_id=session_id,
            learned_viscosity_param=model.get_inferred_viscosity_param(),
            total_points=flow_results['total_points'],
            processing_time=processing_time,
            flow_field=flow_field,
            boundary_data=boundary_data,
            centerline_data=centerline_data,
            viscosity_profile=viscosity_profile,
            model_info={
                'architecture': inference_config.PINN_LAYERS,
                'uses_fourier_features': inference_config.USE_FOURIER_FEATURES,
                'uses_adaptive_weights': inference_config.USE_ADAPTIVE_WEIGHTS,
                'reynolds_number': inference_config.REYNOLDS_NUMBER,
                'grid_resolution': f"{inference_config.N_GRID_X}x{inference_config.N_GRID_Y}"
            }
        )
        
    except Exception as e:
        # Clean up on error
        if temp_dir:
            cleanup_temp_directory(temp_dir)
        
        return InferenceResponse(
            success=False,
            session_id=session_id,
            learned_viscosity_param=0.0,
            total_points=0,
            processing_time=0.0,
            model_info={},
            error_message=str(e)
        )

@app.post("/inference/multiple", response_model=MultiInferenceResponse)
async def run_multiple_inference(request: MultiInferenceRequest):
    """Run multiple inference scenarios"""
    session_id = str(uuid.uuid4())
    
    try:
        # Check if model exists
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
        
        scenarios_results = []
        total_processing_time = 0.0
        total_points = 0
        
        for i, scenario_params in enumerate(request.scenarios):
            # Create individual inference request
            individual_request = InferenceRequest(
                parameters=scenario_params,
                model_path=request.model_path,
                include_boundary=request.include_boundary,
                include_centerline=request.include_centerline,
                include_viscosity=request.include_viscosity
            )
            
            # Run individual inference
            scenario_result = await run_single_inference(individual_request)
            scenarios_results.append(scenario_result)
            
            if scenario_result.success:
                total_processing_time += scenario_result.processing_time
                total_points += scenario_result.total_points
        
        # Calculate summary statistics
        successful_scenarios = [r for r in scenarios_results if r.success]
        failed_scenarios = [r for r in scenarios_results if not r.success]
        
        if successful_scenarios:
            avg_viscosity_param = np.mean([r.learned_viscosity_param for r in successful_scenarios])
            std_viscosity_param = np.std([r.learned_viscosity_param for r in successful_scenarios])
        else:
            avg_viscosity_param = 0.0
            std_viscosity_param = 0.0
        
        summary = {
            'total_scenarios': len(request.scenarios),
            'successful_scenarios': len(successful_scenarios),
            'failed_scenarios': len(failed_scenarios),
            'total_processing_time': total_processing_time,
            'total_points': total_points,
            'average_viscosity_param': float(avg_viscosity_param),
            'std_viscosity_param': float(std_viscosity_param),
            'viscosity_param_range': [
                float(min(r.learned_viscosity_param for r in successful_scenarios)) if successful_scenarios else 0.0,
                float(max(r.learned_viscosity_param for r in successful_scenarios)) if successful_scenarios else 0.0
            ]
        }
        
        return MultiInferenceResponse(
            success=len(successful_scenarios) > 0,
            session_id=session_id,
            total_scenarios=len(request.scenarios),
            scenarios=scenarios_results,
            summary=summary,
            error_message=f"{len(failed_scenarios)} scenarios failed" if failed_scenarios else None
        )
        
    except Exception as e:
        return MultiInferenceResponse(
            success=False,
            session_id=session_id,
            total_scenarios=len(request.scenarios),
            scenarios=[],
            summary={},
            error_message=str(e)
        )

@app.post("/inference/flow-field")
async def get_flow_field_only(request: InferenceRequest):
    """Get only flow field data for lightweight plotting"""
    temp_dir = None
    
    try:
        # Create temporary directory
        temp_dir = create_temp_directory()
        
        # Check if model exists
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")
        
        # Create inference configuration
        inference_config = create_inference_config(
            reynolds_number=request.parameters.reynolds_number,
            nu_base_true=request.parameters.nu_base_true,
            a_true=request.parameters.a_true,
            u_max_inlet=request.parameters.u_max_inlet,
            x_max=request.parameters.x_max,
            y_max=request.parameters.y_max,
            x_min=request.parameters.x_min,
            y_min=request.parameters.y_min,
            n_grid_x=request.parameters.n_grid_x,
            n_grid_y=request.parameters.n_grid_y,
            name=request.parameters.name
        )
        
        # Load trained model
        model, inference_config = load_trained_model(request.model_path, inference_config)
        
        # Set output directory to temp directory
        inference_config.OUTPUT_DIR = temp_dir
        
        # Run only flow field inference
        flow_results = infer_3d_flow_field(model, inference_config, temp_dir)
        
        # Read and return flow field data
        flow_df = pd.read_csv(os.path.join(temp_dir, 'inferred_flow_3d_complete.csv'))
        
        # Return raw data for frontend
        return {
            "success": True,
            "data": flow_df.to_dict('records'),
            "metadata": {
                "grid_shape": flow_results['grid_size'],
                "total_points": flow_results['total_points'],
                "learned_viscosity_param": model.get_inferred_viscosity_param()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_dir:
            cleanup_temp_directory(temp_dir)

@app.get("/inference/{session_id}/download/{file_type}")
async def download_inference_file(session_id: str, file_type: str):
    """Download specific inference result files"""
    if session_id not in inference_cache:
        raise HTTPException(status_code=404, detail="Session not found")
    
    cache_entry = inference_cache[session_id]
    temp_dir = cache_entry['temp_dir']
    
    file_map = {
        'flow_field': 'inferred_flow_3d_complete.csv',
        'boundary': 'inferred_boundary_analysis.csv',
        'centerline': 'inferred_centerline_analysis.csv',
        'viscosity': 'inferred_viscosity_profile.csv',
        'summary': 'inference_summary.json'
    }
    
    if file_type not in file_map:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file_type}")
    
    file_path = os.path.join(temp_dir, file_map[file_type])
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_type}")
    
    # Read and return file content
    try:
        if file_type == 'summary':
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            df = pd.read_csv(file_path)
            return {
                "filename": file_map[file_type],
                "data": df.to_dict('records'),
                "shape": list(df.shape)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@app.delete("/inference/{session_id}")
async def cleanup_inference_session(session_id: str):
    """Clean up inference session and temporary files"""
    if session_id not in inference_cache:
        raise HTTPException(status_code=404, detail="Session not found")
    
    cache_entry = inference_cache[session_id]
    temp_dir = cache_entry['temp_dir']
    
    # Clean up temporary directory
    cleanup_temp_directory(temp_dir)
    
    # Remove from cache
    del inference_cache[session_id]
    
    return {"message": f"Session {session_id} cleaned up successfully"}

@app.get("/inference/sessions")
async def list_active_sessions():
    """List all active inference sessions"""
    sessions = []
    for session_id, cache_entry in inference_cache.items():
        sessions.append({
            "session_id": session_id,
            "timestamp": cache_entry['timestamp'].isoformat(),
            "temp_dir": cache_entry['temp_dir']
        })
    
    return {
        "active_sessions": len(sessions),
        "sessions": sessions
    }

# Background task to clean up old sessions
@app.on_event("startup")
async def startup_event():
    """Clean up any existing temporary directories on startup"""
    print("PINN Inference API started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up all temporary directories on shutdown"""
    for session_id, cache_entry in inference_cache.items():
        cleanup_temp_directory(cache_entry['temp_dir'])
    inference_cache.clear()
    print("PINN Inference API shut down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)