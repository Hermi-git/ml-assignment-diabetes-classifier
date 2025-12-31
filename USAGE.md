# Quick Start Guide

## Overview
This project implements a complete ML system for diabetes classification with:
- Two ML models: Logistic Regression and Decision Tree
- FastAPI REST API backend
- Interactive web interface

## Setup (3 steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Models
```bash
python train_models.py
```

This generates:
- `saved_models/logistic_regression_model.pkl`
- `saved_models/decision_tree_model.pkl`
- `saved_models/scaler.pkl`
- `saved_models/feature_names.pkl`

### 3. Start the Server
```bash
python app.py
```

Server runs at: http://localhost:8000

## Using the Web UI

1. Open http://localhost:8000 in your browser
2. Fill in patient information (or click "Load Sample")
3. Click "Predict" to get results from both models
4. View side-by-side comparison and risk assessment

## Using the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Predictions
```bash
curl -X POST http://localhost:8000/api/predict/both \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

### API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| Pregnancies | Number of times pregnant | 0-20 |
| Glucose | Plasma glucose concentration (mg/dL) | 0-300 |
| BloodPressure | Diastolic blood pressure (mm Hg) | 0-200 |
| SkinThickness | Triceps skin fold thickness (mm) | 0-100 |
| Insulin | 2-Hour serum insulin (mu U/ml) | 0-1000 |
| BMI | Body mass index (kg/m²) | 0-100 |
| DiabetesPedigreeFunction | Diabetes pedigree function | 0-3 |
| Age | Age in years | 1-120 |

## Model Performance

**Logistic Regression**
- Accuracy: ~83%
- Good for linear patterns
- Fast predictions

**Decision Tree**
- Accuracy: ~86%
- Captures non-linear patterns
- Interpretable rules

## Troubleshooting

**"Models not loaded"**
- Run `python train_models.py` first

**"Port 8000 already in use"**
- Change port: `uvicorn app:app --port 8001`

**API connection error**
- Ensure server is running
- Check firewall settings

## Development

### Project Structure
```
ml-assignment-diabetes-classifier/
├── train_models.py       # ML training pipeline
├── app.py               # FastAPI backend
├── requirements.txt     # Dependencies
├── static/              # Web UI
│   ├── index.html
│   ├── style.css
│   └── script.js
└── saved_models/        # Trained models (generated)
```

### Running Tests
```bash
# Test model training
python train_models.py

# Test API endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/predict/both -H "Content-Type: application/json" -d @test_data.json
```

## Notes

⚠️ **For Educational Use Only**
This system is for learning purposes and should not be used for medical diagnosis.
