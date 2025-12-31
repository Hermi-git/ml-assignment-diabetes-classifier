"""
ML Training Module for Diabetes Classification
Trains both Logistic Regression and Decision Tree models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


def load_diabetes_data():
    """
    Load the Pima Indians Diabetes Database
    Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    """
    # Using the classic Pima Indians Diabetes dataset
    # Since we don't have external data access, we'll create a synthetic dataset based on known characteristics
    np.random.seed(42)
    n_samples = 768
    
    # Generate synthetic data based on Pima Indians Diabetes dataset characteristics
    data = {
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.randint(0, 200, n_samples),
        'BloodPressure': np.random.randint(0, 122, n_samples),
        'SkinThickness': np.random.randint(0, 99, n_samples),
        'Insulin': np.random.randint(0, 846, n_samples),
        'BMI': np.random.uniform(0, 67.1, n_samples),
        'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
        'Age': np.random.randint(21, 81, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on risk factors
    # Higher glucose, BMI, age, and pregnancies increase diabetes risk
    risk_score = (
        (df['Glucose'] > 120).astype(int) * 2 +
        (df['BMI'] > 30).astype(int) * 2 +
        (df['Age'] > 40).astype(int) +
        (df['Pregnancies'] > 5).astype(int) +
        (df['BloodPressure'] > 80).astype(int)
    )
    
    # Add controlled randomness to create more realistic class distribution
    # Noise threshold: probability of random positive case regardless of risk score
    NOISE_THRESHOLD = 0.7  # 30% chance of random noise affecting classification
    noise = np.random.random(n_samples)
    df['Outcome'] = ((risk_score >= 3) | (noise > NOISE_THRESHOLD)).astype(int)
    
    return df


def preprocess_data(df):
    """
    Preprocess the diabetes dataset
    """
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model
    """
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    """
    Train Decision Tree model
    """
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy


def save_models(lr_model, dt_model, scaler, feature_names):
    """
    Save trained models and preprocessing objects
    """
    os.makedirs('saved_models', exist_ok=True)
    
    joblib.dump(lr_model, 'saved_models/logistic_regression_model.pkl')
    joblib.dump(dt_model, 'saved_models/decision_tree_model.pkl')
    joblib.dump(scaler, 'saved_models/scaler.pkl')
    joblib.dump(feature_names, 'saved_models/feature_names.pkl')
    
    print("\nModels saved successfully!")


def main():
    """
    Main training pipeline
    """
    print("Loading diabetes data...")
    df = load_diabetes_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Outcome'].value_counts()}")
    
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    print("\nTraining Logistic Regression model...")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_accuracy = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    print("\nTraining Decision Tree model...")
    dt_model = train_decision_tree(X_train, y_train)
    dt_accuracy = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    
    print("\nSaving models...")
    save_models(lr_model, dt_model, scaler, feature_names)
    
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
