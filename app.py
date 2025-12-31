"""
FastAPI Backend for Diabetes Prediction
Serves predictions from Logistic Regression and Decision Tree models
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from typing import Dict, List


# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="ML API for diabetes classification using Logistic Regression and Decision Tree",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Input data model
class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    Glucose: float = Field(..., ge=0, le=300, description="Plasma glucose concentration")
    BloodPressure: float = Field(..., ge=0, le=200, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(..., ge=0, le=1000, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=0, le=100, description="Body mass index (weight in kg/(height in m)^2)")
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3, description="Diabetes pedigree function")
    Age: int = Field(..., ge=1, le=120, description="Age in years")

    class Config:
        schema_extra = {
            "example": {
                "Pregnancies": 6,
                "Glucose": 148,
                "BloodPressure": 72,
                "SkinThickness": 35,
                "Insulin": 0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50
            }
        }


# Response model
class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: float
    model_name: str


class CombinedPredictionResponse(BaseModel):
    logistic_regression: PredictionResponse
    decision_tree: PredictionResponse
    features: Dict[str, float]


# Global variables for models
lr_model = None
dt_model = None
scaler = None
feature_names = None


def load_models():
    """
    Load trained models and preprocessing objects
    """
    global lr_model, dt_model, scaler, feature_names
    
    models_dir = 'saved_models'
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(
            f"Models directory '{models_dir}' not found. Please train the models first by running train_models.py"
        )
    
    try:
        lr_model = joblib.load(f'{models_dir}/logistic_regression_model.pkl')
        dt_model = joblib.load(f'{models_dir}/decision_tree_model.pkl')
        scaler = joblib.load(f'{models_dir}/scaler.pkl')
        feature_names = joblib.load(f'{models_dir}/feature_names.pkl')
        print("Models loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"Error loading models: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """
    Load models on startup
    """
    load_models()


@app.get("/")
async def root():
    """
    Root endpoint - serves the web UI
    """
    return FileResponse('static/index.html')


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "models_loaded": all([lr_model, dt_model, scaler, feature_names])
    }


@app.get("/api/features")
async def get_features():
    """
    Get list of required features
    """
    if feature_names is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    return {
        "features": feature_names,
        "count": len(feature_names)
    }


def prepare_input(data: DiabetesInput) -> np.ndarray:
    """
    Prepare input data for prediction
    """
    # Convert input to array in the correct order
    features = [
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]
    
    # Scale the features
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    
    return features_scaled


@app.post("/api/predict/logistic-regression", response_model=PredictionResponse)
async def predict_logistic_regression(data: DiabetesInput):
    """
    Predict using Logistic Regression model
    """
    try:
        features_scaled = prepare_input(data)
        
        # Make prediction
        prediction = lr_model.predict(features_scaled)[0]
        probability = lr_model.predict_proba(features_scaled)[0][1]
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label="Diabetic" if prediction == 1 else "Non-Diabetic",
            probability=float(probability),
            model_name="Logistic Regression"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/predict/decision-tree", response_model=PredictionResponse)
async def predict_decision_tree(data: DiabetesInput):
    """
    Predict using Decision Tree model
    """
    try:
        features_scaled = prepare_input(data)
        
        # Make prediction
        prediction = dt_model.predict(features_scaled)[0]
        probability = dt_model.predict_proba(features_scaled)[0][1]
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label="Diabetic" if prediction == 1 else "Non-Diabetic",
            probability=float(probability),
            model_name="Decision Tree"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/predict/both", response_model=CombinedPredictionResponse)
async def predict_both(data: DiabetesInput):
    """
    Predict using both models
    """
    try:
        features_scaled = prepare_input(data)
        
        # Logistic Regression prediction
        lr_prediction = lr_model.predict(features_scaled)[0]
        lr_probability = lr_model.predict_proba(features_scaled)[0][1]
        
        # Decision Tree prediction
        dt_prediction = dt_model.predict(features_scaled)[0]
        dt_probability = dt_model.predict_proba(features_scaled)[0][1]
        
        return CombinedPredictionResponse(
            logistic_regression=PredictionResponse(
                prediction=int(lr_prediction),
                prediction_label="Diabetic" if lr_prediction == 1 else "Non-Diabetic",
                probability=float(lr_probability),
                model_name="Logistic Regression"
            ),
            decision_tree=PredictionResponse(
                prediction=int(dt_prediction),
                prediction_label="Diabetic" if dt_prediction == 1 else "Non-Diabetic",
                probability=float(dt_probability),
                model_name="Decision Tree"
            ),
            features=data.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Mount static files directory for serving the web UI
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass  # Static directory might not exist yet


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
