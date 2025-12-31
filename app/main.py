from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import os
import numpy as np
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load trained models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT_MODEL_PATH = os.path.join(BASE_DIR, "model_training", "model_bundle", "decision_tree_model.joblib")
LOG_MODEL_PATH = os.path.join(BASE_DIR, "model_training", "model_bundle", "logistic_model.joblib")

decision_tree_model = joblib.load(DT_MODEL_PATH)
logistic_model = joblib.load(LOG_MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/logistic", response_class=HTMLResponse)
async def predict_logistic(
    request: Request,
    pregnancies: float = Form(...),
    glucose: float = Form(...),
    blood_pressure: float = Form(...),
    skin_thickness: float = Form(...),
    insulin: float = Form(...),
    bmi: float = Form(...),
    diabetes_pedigree: float = Form(...),
    age: float = Form(...)
):
    # Convert to DataFrame
    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, age
    ]], columns=[
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ])

    # Get prediction and probability
    prediction = logistic_model.predict(input_data)[0]
    probability = logistic_model.predict_proba(input_data)[0]
    
    # Get probability for positive class (diabetes)
    diabetes_probability = probability[1] * 100
    no_diabetes_probability = probability[0] * 100
    
    result = "Positive (Likely Diabetic)" if prediction == 1 else "Negative (Not Diabetic)"
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "model_used": "Logistic Regression",
        "diabetes_prob": f"{diabetes_probability:.2f}",
        "no_diabetes_prob": f"{no_diabetes_probability:.2f}"
    })

@app.post("/predict/decision_tree", response_class=HTMLResponse)
async def predict_decision_tree(
    request: Request,
    pregnancies: float = Form(...),
    glucose: float = Form(...),
    blood_pressure: float = Form(...),
    skin_thickness: float = Form(...),
    insulin: float = Form(...),
    bmi: float = Form(...),
    diabetes_pedigree: float = Form(...),
    age: float = Form(...)
):
    # Convert to DataFrame
    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, age
    ]], columns=[
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ])

    # Get prediction and probability
    prediction = decision_tree_model.predict(input_data)[0]
    probability = decision_tree_model.predict_proba(input_data)[0]
    
    # Get probability for positive class (diabetes)
    diabetes_probability = probability[1] * 100
    no_diabetes_probability = probability[0] * 100
    
    result = "Positive (Likely Diabetic)" if prediction == 1 else "Negative (Not Diabetic)"
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "model_used": "Decision Tree",
        "diabetes_prob": f"{diabetes_probability:.2f}",
        "no_diabetes_prob": f"{no_diabetes_probability:.2f}"
    })