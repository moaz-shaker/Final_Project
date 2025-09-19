
import joblib
import pandas as pd
import numpy as np

class HeartDiseasePredictor:
    """Heart Disease Prediction Class"""
    
    def __init__(self, model_path='models/complete_pipeline.pkl', 
                 feature_names_path='models/feature_names.pkl'):
        """Initialize the predictor"""
        self.pipeline = joblib.load(model_path)
        self.feature_names = joblib.load(feature_names_path)
    
    def predict(self, data):
        """Make prediction on new data"""
        if isinstance(data, dict):
            # Convert dictionary to DataFrame
            data = pd.DataFrame([data])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select only the required features
        data_selected = data[self.feature_names]
        
        # Make prediction
        prediction = self.pipeline.predict(data_selected)[0]
        probability = self.pipeline.predict_proba(data_selected)[0]
        
        return {
            'prediction': int(prediction),
            'probability': float(probability[1]),  # Probability of heart disease
            'risk_level': 'HIGH RISK' if prediction == 1 else 'LOW RISK'
        }
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.pipeline.named_steps['model'], 'feature_importances_'):
            importance = self.pipeline.named_steps['model'].feature_importances_
            return dict(zip(self.feature_names, importance))
        return None

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Example data
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
    
    # Make prediction
    result = predictor.predict(sample_data)
    print(f"Prediction: {result['risk_level']}")
    print(f"Probability: {result['probability']:.2%}")
