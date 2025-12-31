// Risk assessment thresholds based on diabetes probability
// These thresholds are used for educational demonstration purposes
const RISK_THRESHOLD_LOW = 30;    // Below 30%: Low risk
const RISK_THRESHOLD_HIGH = 60;   // Above 60%: High risk, 30-60%: Moderate risk

// API base URL
const API_BASE_URL = '';

// Sample data for testing
const sampleData = {
    Pregnancies: 6,
    Glucose: 148,
    BloodPressure: 72,
    SkinThickness: 35,
    Insulin: 0,
    BMI: 33.6,
    DiabetesPedigreeFunction: 0.627,
    Age: 50
};

// Form submission handler
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    await makePrediction();
});

// Make prediction
async function makePrediction() {
    // Get form data
    const formData = {
        Pregnancies: parseInt(document.getElementById('pregnancies').value),
        Glucose: parseFloat(document.getElementById('glucose').value),
        BloodPressure: parseFloat(document.getElementById('bloodPressure').value),
        SkinThickness: parseFloat(document.getElementById('skinThickness').value),
        Insulin: parseFloat(document.getElementById('insulin').value),
        BMI: parseFloat(document.getElementById('bmi').value),
        DiabetesPedigreeFunction: parseFloat(document.getElementById('diabetesPedigree').value),
        Age: parseInt(document.getElementById('age').value)
    };

    // Show loading spinner
    showLoading();
    hideError();
    hideResults();

    try {
        // Call API endpoint for both predictions
        const response = await fetch(`${API_BASE_URL}/api/predict/both`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        
        // Hide loading
        hideLoading();
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        hideLoading();
        showError(`Error making prediction: ${error.message}`);
        console.error('Prediction error:', error);
    }
}

// Display prediction results
function displayResults(data) {
    // Logistic Regression results
    const lrPrediction = data.logistic_regression.prediction_label;
    const lrProbability = (data.logistic_regression.probability * 100).toFixed(2);
    const lrIsDiabetic = data.logistic_regression.prediction === 1;

    document.getElementById('lrPrediction').textContent = lrPrediction;
    document.getElementById('lrPrediction').className = 
        `prediction-label ${lrIsDiabetic ? 'prediction-diabetic' : 'prediction-non-diabetic'}`;
    document.getElementById('lrProbability').textContent = `Probability: ${lrProbability}%`;
    document.getElementById('lrProbabilityBar').style.width = `${lrProbability}%`;

    // Decision Tree results
    const dtPrediction = data.decision_tree.prediction_label;
    const dtProbability = (data.decision_tree.probability * 100).toFixed(2);
    const dtIsDiabetic = data.decision_tree.prediction === 1;

    document.getElementById('dtPrediction').textContent = dtPrediction;
    document.getElementById('dtPrediction').className = 
        `prediction-label ${dtIsDiabetic ? 'prediction-diabetic' : 'prediction-non-diabetic'}`;
    document.getElementById('dtProbability').textContent = `Probability: ${dtProbability}%`;
    document.getElementById('dtProbabilityBar').style.width = `${dtProbability}%`;

    // Comparison text
    let comparisonText = '';
    if (lrIsDiabetic === dtIsDiabetic) {
        comparisonText = `<strong>Both models agree:</strong> The patient is classified as <strong>${lrPrediction}</strong>. `;
        comparisonText += `Logistic Regression shows ${lrProbability}% confidence, while Decision Tree shows ${dtProbability}% confidence.`;
    } else {
        comparisonText = `<strong>Models disagree:</strong> Logistic Regression predicts <strong>${lrPrediction}</strong> (${lrProbability}% confidence), `;
        comparisonText += `while Decision Tree predicts <strong>${dtPrediction}</strong> (${dtProbability}% confidence). `;
        comparisonText += `Consider consulting with a medical professional for accurate diagnosis.`;
    }

    // Add risk interpretation
    const avgProbability = (parseFloat(lrProbability) + parseFloat(dtProbability)) / 2;
    comparisonText += '<br><br><strong>Risk Assessment:</strong> ';
    if (avgProbability < RISK_THRESHOLD_LOW) {
        comparisonText += 'Low risk of diabetes based on provided information.';
    } else if (avgProbability < RISK_THRESHOLD_HIGH) {
        comparisonText += 'Moderate risk of diabetes. Regular monitoring recommended.';
    } else {
        comparisonText += 'High risk of diabetes. Medical consultation strongly recommended.';
    }

    document.getElementById('comparisonText').innerHTML = comparisonText;

    // Show results section
    showResults();
}

// UI helper functions
function showLoading() {
    document.getElementById('loadingSpinner').style.display = 'block';
}

function hideLoading() {
    document.getElementById('loadingSpinner').style.display = 'none';
}

function showResults() {
    document.getElementById('resultsSection').style.display = 'block';
    // Smooth scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function hideResults() {
    document.getElementById('resultsSection').style.display = 'none';
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    errorDiv.scrollIntoView({ behavior: 'smooth' });
}

function hideError() {
    document.getElementById('errorMessage').style.display = 'none';
}

// Reset form to default values
function resetForm() {
    document.getElementById('predictionForm').reset();
    hideResults();
    hideError();
}

// Load sample data
function loadSample() {
    document.getElementById('pregnancies').value = sampleData.Pregnancies;
    document.getElementById('glucose').value = sampleData.Glucose;
    document.getElementById('bloodPressure').value = sampleData.BloodPressure;
    document.getElementById('skinThickness').value = sampleData.SkinThickness;
    document.getElementById('insulin').value = sampleData.Insulin;
    document.getElementById('bmi').value = sampleData.BMI;
    document.getElementById('diabetesPedigree').value = sampleData.DiabetesPedigreeFunction;
    document.getElementById('age').value = sampleData.Age;
    
    hideResults();
    hideError();
}

// Check API health on page load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (!data.models_loaded) {
            showError('Warning: Models are not loaded. Please train the models first by running train_models.py');
        }
    } catch (error) {
        showError('Warning: Cannot connect to the API server. Please ensure the backend is running.');
    }
});
