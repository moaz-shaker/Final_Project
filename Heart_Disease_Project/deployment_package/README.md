
# Heart Disease Prediction - Deployment Package

This package contains all necessary files to deploy the Heart Disease Prediction system.

## Files Included:
- `models/complete_pipeline.pkl`: Complete ML pipeline (preprocessing + model)
- `models/feature_names.pkl`: List of selected feature names
- `models/predictor.py`: Standalone prediction class
- `models/model_info.pkl`: Model information and metadata
- `models/model_info.txt`: Human-readable model information
- `requirements.txt`: Python dependencies
- `README.md`: Project documentation

## Quick Start:

### 1. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 2. Use the Predictor:
```python
from models.predictor import HeartDiseasePredictor

# Initialize predictor
predictor = HeartDiseasePredictor()

# Make prediction
sample_data = {
    'age': 50,
    'sex': 1,
    'cp': 0,
    'trestbps': 120,
    'chol': 200,
    'fbs': 0,
    'restecg': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.0,
    'slope': 1,
    'ca': 0,
    'thal': 0
}

result = predictor.predict(sample_data)
print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['probability']:.2%}")
```

### 3. Deploy with Streamlit:
```bash
streamlit run ui/app.py
```

## Model Performance:
- Best Model: [Model Name]
- Accuracy: [Accuracy Score]
- F1-Score: [F1 Score]

## Support:
For questions or issues, please refer to the main project documentation.
