# ML Assignment - Diabetes Classifier

A comprehensive machine learning project for diabetes classification using Logistic Regression and Decision Tree models, with a FastAPI backend and interactive web UI.

## 🎯 Project Overview

This project implements a diabetes prediction system that:
- Trains two ML models: **Logistic Regression** and **Decision Tree**
- Provides a **FastAPI** backend for serving predictions
- Includes an **interactive web UI** for easy patient data input and result visualization
- Compares predictions from both models with confidence scores

## 🏗️ Project Structure

```
ml-assignment-diabetes-classifier/
├── train_models.py          # ML model training pipeline
├── app.py                   # FastAPI backend application
├── requirements.txt         # Python dependencies
├── saved_models/           # Directory for trained models (auto-generated)
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── static/                 # Web UI files
    ├── index.html          # Main HTML page
    ├── style.css           # Styling
    └── script.js           # JavaScript logic
```

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hermi-git/ml-assignment-diabetes-classifier.git
   cd ml-assignment-diabetes-classifier
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎓 Usage

### Step 1: Train the Models

Run the training script to train both Logistic Regression and Decision Tree models:

```bash
python train_models.py
```

This will:
- Generate a synthetic diabetes dataset
- Train both models
- Evaluate their performance
- Save the trained models to the `saved_models/` directory

**Expected Output:**
```
Loading diabetes data...
Dataset shape: (768, 9)
Class distribution:
...
Logistic Regression Accuracy: ~0.75-0.85
Decision Tree Accuracy: ~0.70-0.80
```

### Step 2: Start the FastAPI Server

Launch the backend server:

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

### Step 3: Access the Web UI

Open your browser and navigate to:

```
http://localhost:8000
```

You can now:
1. Enter patient information in the form
2. Click "Predict" to get predictions from both models
3. View and compare the results
4. Use "Load Sample" to test with pre-filled data

## 📊 Features

### Input Parameters

The system accepts the following patient features:

1. **Pregnancies**: Number of times pregnant (0-20)
2. **Glucose**: Plasma glucose concentration (0-300 mg/dL)
3. **Blood Pressure**: Diastolic blood pressure (0-200 mm Hg)
4. **Skin Thickness**: Triceps skin fold thickness (0-100 mm)
5. **Insulin**: 2-Hour serum insulin (0-1000 mu U/ml)
6. **BMI**: Body mass index (0-100 kg/m²)
7. **Diabetes Pedigree Function**: Diabetes pedigree function (0-3)
8. **Age**: Age in years (1-120)

### API Endpoints

- `GET /`: Serves the web UI
- `GET /health`: Health check endpoint
- `GET /api/features`: Get list of required features
- `POST /api/predict/logistic-regression`: Predict using Logistic Regression
- `POST /api/predict/decision-tree`: Predict using Decision Tree
- `POST /api/predict/both`: Get predictions from both models

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 Example API Request

```bash
curl -X POST "http://localhost:8000/api/predict/both" \
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

## 🎨 Web UI Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Validation**: Input validation for all fields
- **Visual Comparison**: Side-by-side model comparison
- **Probability Visualization**: Progress bars showing confidence levels
- **Risk Assessment**: Combined risk interpretation from both models
- **Sample Data**: Quick testing with pre-filled sample data

## 🔬 Model Information

### Logistic Regression
- Linear model for binary classification
- Good interpretability
- Works well with linearly separable data

### Decision Tree
- Non-linear model
- Captures complex patterns
- Good for understanding feature importance

## 📝 Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

The project follows PEP 8 style guidelines.

## ⚠️ Disclaimer

This is an educational project for demonstration purposes only. The predictions made by this system should **NOT** be used for actual medical diagnosis. Always consult with qualified healthcare professionals for medical advice.

## 📄 License

This project is for educational purposes.

## 👤 Author

Created as part of an ML assignment demonstrating end-to-end machine learning system development.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 🌟 Acknowledgments

- Scikit-learn for ML algorithms
- FastAPI for the modern web framework
- The diabetes research community for datasets and insights