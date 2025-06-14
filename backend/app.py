#!/usr/bin/env python3
"""
FastAPI Application for PINN Model Inference - Predefined Examples

This FastAPI application provides REST endpoints to serve pre-computed PINN model inference
results from 3 predefined scenarios for frontend 3D visualization.

The app generates 3 example scenarios on startup and serves the static data.

Usage:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
import torch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import interactive.py functions
from interactive import (
    create_inference_config,
    load_trained_model,
    run_inference_session
)

# Initialize FastAPI app
app = FastAPI(
    title="PINN Inference API - Predefined Examples",
    description="API for serving pre-computed Physics-Informed Neural Network inference results",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default model path
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "trained_model.pth")

# Data directory structure
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "examples")

# Define the 3 predefined scenarios
PREDEFINED_SCENARIOS = [
    {
        "id": "scenario_1_low_reynolds",
        "name": "Low Reynolds Number Flow",
        "description": "Low Reynolds number (Re=50) with fine viscosity variation",
        "parameters": {
            "reynolds_number": 50.0,
            "nu_base_true": 0.02,
            "a_true": 0.03,
            "u_max_inlet": 0.8,
            "x_max": 2.0,
            "y_max": 1.0,
            "x_min": 0.0,
            "y_min": 0.0,
            "n_grid_x": 120,
            "n_grid_y": 60,
            "n_time_slices": 1,
            "name": "Low_Re_Fine_Viscosity"
        }
    },
    {
        "id": "scenario_2_medium_reynolds",
        "name": "Medium Reynolds Number Flow",
        "description": "Medium Reynolds number (Re=100) with moderate viscosity variation",
        "parameters": {
            "reynolds_number": 100.0,
            "nu_base_true": 0.01,
            "a_true": 0.05,
            "u_max_inlet": 1.0,
            "x_max": 2.5,
            "y_max": 1.0,
            "x_min": 0.0,
            "y_min": 0.0,
            "n_grid_x": 150,
            "n_grid_y": 75,
            "n_time_slices": 1,
            "name": "Medium_Re_Moderate_Viscosity"
        }
    },
    {
        "id": "scenario_3_high_reynolds",
        "name": "High Reynolds Number Flow",
        "description": "High Reynolds number (Re=200) with strong viscosity variation",
        "parameters": {
            "reynolds_number": 200.0,
            "nu_base_true": 0.005,
            "a_true": 0.08,
            "u_max_inlet": 1.5,
            "x_max": 3.0,
            "y_max": 1.0,
            "x_min": 0.0,
            "y_min": 0.0,
            "n_grid_x": 180,
            "n_grid_y": 90,
            "n_time_slices": 1,
            "name": "High_Re_Strong_Viscosity"
        }
    },
    {
        "id": "scenario_4_very_high_reynolds",
        "name": "Very High Reynolds Number Flow",
        "description": "Very high Reynolds number (Re=200) with extreme viscosity variation",
        "parameters": {
            "reynolds_number": 200.0,
            "nu_base_true": 0.003,
            "a_true": 0.12,
            "u_max_inlet": 2.0,
            "x_max": 3.5,
            "y_max": 1.0,
            "x_min": 0.0,
            "y_min": 0.0,
            "n_grid_x": 200,
            "n_grid_y": 100,
            "n_time_slices": 1,
            "name": "Very_High_Re_Extreme_Viscosity"
        }
    }
]

# Pydantic models for API responses
class ScenarioMetadata(BaseModel):
    """Metadata for a scenario"""
    id: str = Field(description="Scenario identifier")
    name: str = Field(description="Scenario display name")
    description: str = Field(description="Scenario description")
    parameters: Dict[str, Any] = Field(description="Scenario parameters")
    learned_viscosity_param: Optional[float] = Field(description="Learned viscosity parameter")
    processing_time: Optional[float] = Field(description="Processing time in seconds")
    total_points: Optional[int] = Field(description="Total number of inference points")
    grid_shape: Optional[List[int]] = Field(description="Grid shape [nx, ny]")
    files_available: List[str] = Field(description="Available data files")
    generated_at: Optional[str] = Field(description="Generation timestamp")

class ScenariosListResponse(BaseModel):
    """Response for scenarios list"""
    success: bool = Field(description="Whether request was successful")
    total_scenarios: int = Field(description="Total number of scenarios")
    scenarios: List[ScenarioMetadata] = Field(description="List of scenarios")

class GenerationStatusResponse(BaseModel):
    """Response for generation status"""
    success: bool = Field(description="Whether generation was successful")
    status: str = Field(description="Generation status")
    scenarios_completed: int = Field(description="Number of scenarios completed")
    total_scenarios: int = Field(description="Total number of scenarios")
    current_scenario: Optional[str] = Field(description="Currently processing scenario")
    error_message: Optional[str] = Field(description="Error message if failed")

class DataFileResponse(BaseModel):
    """Response for data file content"""
    success: bool = Field(description="Whether request was successful")
    filename: str = Field(description="Filename")
    data_type: str = Field(description="Type of data")
    shape: Optional[List[int]] = Field(description="Data shape")
    columns: List[str] = Field(description="Column names")
    data: List[Dict[str, Any]] = Field(description="Data records")
    metadata: Optional[Dict[str, Any]] = Field(description="Additional metadata")

# Global status tracking
generation_status = {
    "is_generating": False,
    "completed_scenarios": 0,
    "current_scenario": None,
    "start_time": None,
    "error": None
}

def get_scenario_directory(scenario_id: str) -> str:
    """Get the directory path for a scenario"""
    return os.path.join(EXAMPLES_DIR, scenario_id)

def get_scenario_file_path(scenario_id: str, filename: str) -> str:
    """Get the full path for a scenario file"""
    scenario_dir = get_scenario_directory(scenario_id)
    # Check if file is in the inference subdirectory
    inference_dir = os.path.join(scenario_dir, f"inference_{scenario_id}")
    if os.path.exists(os.path.join(inference_dir, filename)):
        return os.path.join(inference_dir, filename)
    return os.path.join(scenario_dir, filename)

def scenario_exists(scenario_id: str) -> bool:
    """Check if a scenario directory and files exist"""
    scenario_dir = get_scenario_directory(scenario_id)
    if not os.path.exists(scenario_dir):
        return False
    
    # Check for required files
    required_files = [
        "inferred_flow_3d_complete.csv",
        "inferred_viscosity_profile.csv",
        "inference_summary.json",
        "metadata.json"
    ]
    
    for filename in required_files:
        if not os.path.exists(os.path.join(scenario_dir, filename)):
            return False
    
    return True

def get_available_files(scenario_id: str) -> List[str]:
    """Get list of available files for a scenario"""
    scenario_dir = get_scenario_directory(scenario_id)
    inference_dir = os.path.join(scenario_dir, f"inference_{scenario_id}")
    
    files = []
    # Check both directories
    for directory in [scenario_dir, inference_dir]:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith(('.csv', '.json')):
                    files.append(filename)
    
    return sorted(files)

def load_scenario_metadata(scenario_id: str) -> Optional[Dict[str, Any]]:
    """Load metadata for a scenario"""
    metadata_path = get_scenario_file_path(scenario_id, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata for {scenario_id}: {e}")
        return None

def save_scenario_metadata(scenario_id: str, metadata: Dict[str, Any]):
    """Save metadata for a scenario"""
    scenario_dir = get_scenario_directory(scenario_id)
    os.makedirs(scenario_dir, exist_ok=True)
    
    metadata_path = get_scenario_file_path(scenario_id, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

async def generate_scenario_data(scenario_config: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    """Generate data for a single scenario"""
    scenario_id = scenario_config["id"]
    scenario_name = scenario_config["name"]
    parameters = scenario_config["parameters"]
    
    print(f"Generating data for scenario: {scenario_name}")
    
    try:
        # Create inference configuration
        inference_config = create_inference_config(**parameters)
        
        # Load trained model
        model, inference_config = load_trained_model(model_path, inference_config)
        
        # Set output directory to scenario directory
        scenario_dir = get_scenario_directory(scenario_id)
        os.makedirs(scenario_dir, exist_ok=True)
        inference_config.OUTPUT_DIR = scenario_dir
        
        # Run inference session
        import time
        start_time = time.time()
        
        # Run comprehensive inference
        results = run_inference_session(model, inference_config, model_path, scenario_id)
        
        processing_time = time.time() - start_time
        
        # Create scenario metadata
        metadata = {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "description": scenario_config["description"],
            "parameters": parameters,
            "learned_viscosity_param": model.get_inferred_viscosity_param(),
            "processing_time": processing_time,
            "total_points": results['flow_results']['total_points'],
            "grid_shape": results['flow_results']['grid_size'],
            "files_available": get_available_files(scenario_id),
            "generated_at": datetime.now().isoformat(),
            "model_info": {
                "architecture": inference_config.PINN_LAYERS,
                "uses_fourier_features": inference_config.USE_FOURIER_FEATURES,
                "uses_adaptive_weights": inference_config.USE_ADAPTIVE_WEIGHTS,
                "reynolds_number": inference_config.REYNOLDS_NUMBER
            }
        }
        
        # Save metadata
        save_scenario_metadata(scenario_id, metadata)
        
        print(f"Completed scenario: {scenario_name} in {processing_time:.2f}s")
        
        return metadata
        
    except Exception as e:
        print(f"Error generating scenario {scenario_name}: {str(e)}")
        raise e

async def generate_all_scenarios():
    """Generate data for all predefined scenarios"""
    global generation_status
    
    if generation_status["is_generating"]:
        return
    
    generation_status["is_generating"] = True
    generation_status["completed_scenarios"] = 0
    generation_status["current_scenario"] = None
    generation_status["start_time"] = datetime.now()
    generation_status["error"] = None
    
    try:
        # Check if model exists
        if not os.path.exists(DEFAULT_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {DEFAULT_MODEL_PATH}")
        
        print(f"Starting generation of {len(PREDEFINED_SCENARIOS)} scenarios...")
        
        for i, scenario_config in enumerate(PREDEFINED_SCENARIOS):
            scenario_id = scenario_config["id"]
            generation_status["current_scenario"] = scenario_config["name"]
            
            # Skip if scenario already exists
            if scenario_exists(scenario_id):
                print(f"Scenario {scenario_id} already exists, skipping...")
                generation_status["completed_scenarios"] += 1
                continue
            
            # Generate scenario data
            await generate_scenario_data(scenario_config, DEFAULT_MODEL_PATH)
            generation_status["completed_scenarios"] += 1
        
        print("All scenarios generated successfully!")
        
    except Exception as e:
        generation_status["error"] = str(e)
        print(f"Error during generation: {e}")
    
    finally:
        generation_status["is_generating"] = False
        generation_status["current_scenario"] = None

@app.on_event("startup")
async def startup_event():
    """Generate scenarios on startup if they don't exist"""
    print("PINN Inference API started")
    
    # Check if any scenarios are missing
    missing_scenarios = []
    for scenario_config in PREDEFINED_SCENARIOS:
        if not scenario_exists(scenario_config["id"]):
            missing_scenarios.append(scenario_config["id"])
    
    if missing_scenarios:
        print(f"Missing scenarios: {missing_scenarios}")
        print("Generating scenarios in background...")
        # Generate scenarios in background
        asyncio.create_task(generate_all_scenarios())
    else:
        print("All scenarios already exist")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PINN Inference API - Predefined Examples",
        "version": "1.0.0",
        "total_scenarios": len(PREDEFINED_SCENARIOS),
        "endpoints": {
            "/scenarios": "List all available scenarios",
            "/scenarios/{scenario_id}": "Get specific scenario metadata",
            "/scenarios/{scenario_id}/data/{filename}": "Get scenario data file",
            "/scenarios/{scenario_id}/files": "List files for scenario",
            "/generate": "Trigger scenario generation",
            "/generation-status": "Check generation status",
            "/health": "Health check"
        },
        "data_structure": {
            "scenarios_directory": EXAMPLES_DIR,
            "scenario_files": [
                "inferred_flow_3d_complete.csv - Main flow field data",
                "inferred_boundary_analysis.csv - Boundary condition analysis",
                "inferred_centerline_analysis.csv - Centerline flow data",
                "inferred_viscosity_profile.csv - Viscosity inference results",
                "inference_summary.json - Summary of inference results",
                "metadata.json - Scenario metadata and parameters"
            ]
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/scenarios", response_model=ScenariosListResponse)
async def list_scenarios():
    """List all available scenarios with metadata"""
    scenarios = []
    
    for scenario_config in PREDEFINED_SCENARIOS:
        scenario_id = scenario_config["id"]
        
        # Load metadata if available
        metadata = load_scenario_metadata(scenario_id)
        
        if metadata:
            scenario_metadata = ScenarioMetadata(
                id=scenario_id,
                name=metadata["scenario_name"],
                description=metadata["description"],
                parameters=metadata["parameters"],
                learned_viscosity_param=metadata.get("learned_viscosity_param"),
                processing_time=metadata.get("processing_time"),
                total_points=metadata.get("total_points"),
                grid_shape=metadata.get("grid_shape"),
                files_available=metadata.get("files_available", []),
                generated_at=metadata.get("generated_at")
            )
        else:
            # Return basic info from predefined scenarios
            scenario_metadata = ScenarioMetadata(
                id=scenario_id,
                name=scenario_config["name"],
                description=scenario_config["description"],
                parameters=scenario_config["parameters"],
                files_available=get_available_files(scenario_id)
            )
        
        scenarios.append(scenario_metadata)
    
    return ScenariosListResponse(
        success=True,
        total_scenarios=len(scenarios),
        scenarios=scenarios
    )

@app.get("/scenarios/{scenario_id}")
async def get_scenario_metadata(scenario_id: str):
    """Get metadata for a specific scenario"""
    # Validate scenario ID
    valid_ids = [s["id"] for s in PREDEFINED_SCENARIOS]
    if scenario_id not in valid_ids:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_id}")
    
    metadata = load_scenario_metadata(scenario_id)
    
    if not metadata:
        # Return basic info if metadata not available
        scenario_config = next(s for s in PREDEFINED_SCENARIOS if s["id"] == scenario_id)
        return {
            "success": True,
            "scenario_id": scenario_id,
            "name": scenario_config["name"],
            "description": scenario_config["description"],
            "parameters": scenario_config["parameters"],
            "files_available": get_available_files(scenario_id),
            "generated": scenario_exists(scenario_id)
        }
    
    return {
        "success": True,
        **metadata
    }

@app.get("/scenarios/{scenario_id}/files")
async def list_scenario_files(scenario_id: str):
    """List all files available for a scenario"""
    # Validate scenario ID
    valid_ids = [s["id"] for s in PREDEFINED_SCENARIOS]
    if scenario_id not in valid_ids:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_id}")
    
    files = get_available_files(scenario_id)
    
    return {
        "success": True,
        "scenario_id": scenario_id,
        "files": files,
        "total_files": len(files)
    }

@app.get("/scenarios/{scenario_id}/data/{filename}")
async def get_scenario_data(scenario_id: str, filename: str):
    """Get data file content for a scenario"""
    # Validate scenario ID
    valid_ids = [s["id"] for s in PREDEFINED_SCENARIOS]
    if scenario_id not in valid_ids:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_id}")
    
    file_path = get_scenario_file_path(scenario_id, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    try:
        if filename.endswith('.csv'):
            # Return CSV data as JSON
            df = pd.read_csv(file_path)
            
            return DataFileResponse(
                success=True,
                filename=filename,
                data_type="csv",
                shape=list(df.shape),
                columns=df.columns.tolist(),
                data=df.to_dict('records'),
                metadata={
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
                }
            )
            
        elif filename.endswith('.json'):
            # Return JSON data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            return {
                "success": True,
                "filename": filename,
                "data_type": "json",
                "data": data
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@app.get("/scenarios/{scenario_id}/download/{filename}")
async def download_scenario_file(scenario_id: str, filename: str):
    """Download a scenario file directly"""
    # Validate scenario ID
    valid_ids = [s["id"] for s in PREDEFINED_SCENARIOS]
    if scenario_id not in valid_ids:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_id}")
    
    file_path = get_scenario_file_path(scenario_id, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    return FileResponse(
        file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@app.post("/generate", response_model=GenerationStatusResponse)
async def trigger_generation():
    """Trigger generation of all scenarios"""
    global generation_status
    
    if generation_status["is_generating"]:
        return GenerationStatusResponse(
            success=False,
            status="already_generating",
            scenarios_completed=generation_status["completed_scenarios"],
            total_scenarios=len(PREDEFINED_SCENARIOS),
            current_scenario=generation_status["current_scenario"],
            error_message="Generation already in progress"
        )
    
    # Start generation in background
    asyncio.create_task(generate_all_scenarios())
    
    return GenerationStatusResponse(
        success=True,
        status="started",
        scenarios_completed=0,
        total_scenarios=len(PREDEFINED_SCENARIOS),
        current_scenario=None
    )

@app.get("/generation-status", response_model=GenerationStatusResponse)
async def get_generation_status():
    """Get current generation status"""
    global generation_status
    
    if generation_status["is_generating"]:
        status = "generating"
    elif generation_status["error"]:
        status = "error"
    elif generation_status["completed_scenarios"] == len(PREDEFINED_SCENARIOS):
        status = "completed"
    else:
        status = "idle"
    
    return GenerationStatusResponse(
        success=True,
        status=status,
        scenarios_completed=generation_status["completed_scenarios"],
        total_scenarios=len(PREDEFINED_SCENARIOS),
        current_scenario=generation_status["current_scenario"],
        error_message=generation_status["error"]
    )

@app.delete("/scenarios/{scenario_id}")
async def delete_scenario(scenario_id: str):
    """Delete a scenario and all its data"""
    # Validate scenario ID
    valid_ids = [s["id"] for s in PREDEFINED_SCENARIOS]
    if scenario_id not in valid_ids:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_id}")
    
    scenario_dir = get_scenario_directory(scenario_id)
    
    if os.path.exists(scenario_dir):
        shutil.rmtree(scenario_dir)
    
    return {
        "success": True,
        "message": f"Scenario {scenario_id} deleted successfully"
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    print("PINN Inference API shut down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)