# PINN Viscosity Inference API Documentation

## Overview
The PINN (Physics-Informed Neural Network) Viscosity Inference API provides endpoints for running flow field simulations and retrieving data for visualization. This document outlines the available endpoints and how to interact with them.

## Base URL
```
http://localhost:5000
```

## Available Endpoints

### 1. Health Check
```http
GET /health
```
Returns the API health status and current timestamp.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-03-21T10:00:00.000Z"
}
```

### 2. Model Information
```http
GET /model/info
```
Retrieves information about the trained model.

**Query Parameters:**
- `model_path` (optional): Path to the trained model file (default: `/home/brand/pinn_viscosity/backend/results/trained_model.pth`)

**Response:**
```json
{
    "success": true,
    "model_path": "/path/to/model.pth",
    "learned_viscosity_param": 0.05,
    "model_architecture": [64, 128, 64],
    "uses_fourier_features": true,
    "uses_adaptive_weights": true,
    "model_exists": true
}
```

### 3. Single Inference
```http
POST /inference/single
```
Runs a single inference session with specified parameters.

**Request Body:**
```json
{
    "parameters": {
        "reynolds_number": 100,
        "nu_base_true": 0.01,
        "a_true": 0.05,
        "u_max_inlet": 1.0,
        "x_max": 2.0,
        "y_max": 1.0,
        "x_min": 0.0,
        "y_min": 0.0,
        "n_grid_x": 100,
        "n_grid_y": 50,
        "n_time_slices": 5,
        "name": "API Inference"
    },
    "model_path": "/path/to/model.pth",
    "include_boundary": true,
    "include_centerline": true,
    "include_viscosity": true
}
```

**Response:**
```json
{
    "success": true,
    "session_id": "uuid-string",
    "learned_viscosity_param": 0.05,
    "total_points": 5000,
    "processing_time": 1.23,
    "flow_field": {
        "x": [...],
        "y": [...],
        "u_velocity": [...],
        "v_velocity": [...],
        "pressure": [...],
        "velocity_magnitude": [...],
        "viscosity": [...],
        "vorticity": [...],
        "grid_shape": [100, 50],
        "learned_viscosity_param": 0.05
    },
    "boundary_data": {
        "x": [...],
        "y": [...],
        "u_velocity": [...],
        "v_velocity": [...],
        "pressure": [...],
        "boundary_type": [...]
    },
    "centerline_data": {
        "x": [...],
        "u_velocity": [...],
        "pressure": [...],
        "velocity_magnitude": [...],
        "viscosity": [...]
    },
    "viscosity_profile": {
        "y": [...],
        "viscosity_learned": [...],
        "viscosity_reference": [...],
        "absolute_error": [...],
        "relative_error_percent": [...]
    },
    "model_info": {
        "architecture": [64, 128, 64],
        "uses_fourier_features": true,
        "uses_adaptive_weights": true,
        "reynolds_number": 100,
        "grid_resolution": "100x50"
    }
}
```

### 4. Multiple Inference Scenarios
```http
POST /inference/multiple
```
Runs multiple inference scenarios with different parameters.

**Request Body:**
```json
{
    "model_path": "/path/to/model.pth",
    "scenarios": [
        {
            "reynolds_number": 10,
            "nu_base_true": 0.05,
            "a_true": 0.02,
            "u_max_inlet": 0.5,
            "n_grid_x": 150,
            "n_grid_y": 75,
            "name": "Low_Re_HighRes"
        },
        {
            "reynolds_number": 100,
            "nu_base_true": 0.01,
            "a_true": 0.05,
            "u_max_inlet": 1.0,
            "n_grid_x": 200,
            "n_grid_y": 100,
            "name": "Medium_Re_HighRes"
        }
    ],
    "include_boundary": true,
    "include_centerline": true,
    "include_viscosity": true
}
```

**Response:**
```json
{
    "success": true,
    "session_id": "uuid-string",
    "scenarios_results": [...],
    "summary": {
        "total_scenarios": 2,
        "successful_scenarios": 2,
        "failed_scenarios": 0,
        "avg_viscosity_param": 0.035,
        "std_viscosity_param": 0.015,
        "total_processing_time": 2.46,
        "total_points": 10000
    }
}
```

## Data Visualization

The API returns data in a format suitable for various types of visualizations:

1. **Flow Field Visualization**
   - Use `flow_field` data to create 2D or 3D velocity field plots
   - Plot pressure contours using `pressure` data
   - Visualize vorticity using `vorticity` data

2. **Boundary Analysis**
   - Use `boundary_data` to plot velocity profiles at domain boundaries
   - Analyze pressure distribution along boundaries

3. **Centerline Analysis**
   - Use `centerline_data` to plot velocity and pressure along the centerline
   - Compare velocity magnitude and viscosity profiles

4. **Viscosity Profile**
   - Use `viscosity_profile` to compare learned vs. reference viscosity
   - Plot error metrics (absolute and relative)

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `200 OK`: Successful request
- `404 Not Found`: Model file not found
- `500 Internal Server Error`: Server-side error

Error responses include:
```json
{
    "success": false,
    "error_message": "Detailed error description"
}
```

## Best Practices

1. **Session Management**
   - Store the `session_id` for future reference
   - Sessions are automatically cleaned up after a period of inactivity

2. **Parameter Selection**
   - Start with moderate grid resolutions (e.g., 100x50)
   - Adjust Reynolds number based on flow characteristics
   - Use appropriate viscosity parameters for your fluid

3. **Performance Considerations**
   - Higher grid resolutions increase processing time
   - Multiple scenarios can be run in parallel
   - Consider caching results for frequently used parameters

## Example Usage

```javascript
// Example using fetch API
async function runInference() {
    const response = await fetch('http://localhost:5000/inference/single', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            parameters: {
                reynolds_number: 100,
                nu_base_true: 0.01,
                n_grid_x: 100,
                n_grid_y: 50
            }
        })
    });
    
    const data = await response.json();
    // Process and visualize the data
}
``` 