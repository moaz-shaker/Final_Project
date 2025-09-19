#!/usr/bin/env python3
"""
Model Export Script for Heart Disease Prediction System
This script exports all trained models and creates a deployment package.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

def create_model_pipeline():
    """Create a complete model pipeline for deployment"""
    
    # Load the best model and preprocessing objects
    try:
        # Load preprocessing objects
        scaler = joblib.load('models/scaler.pkl')
        selected_features = joblib.load('models/selected_features.pkl')
        
        # Load the best model
        best_model = joblib.load('models/final_best_model.pkl')
        
        print("✅ Successfully loaded preprocessing objects and best model")
        
        # Create a complete pipeline
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', best_model)
        ])
        
        # Save the complete pipeline
        joblib.dump(pipeline, 'models/complete_pipeline.pkl')
        joblib.dump(selected_features, 'models/feature_names.pkl')
        
        print("✅ Complete pipeline saved successfully")
        
        return pipeline, selected_features
        
    except Exception as e:
        print(f"❌ Error creating pipeline: {str(e)}")
        return None, None

def create_prediction_function():
    """Create a standalone prediction function"""
    
    prediction_code = '''
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
'''
    
    # Save the prediction function
    with open('models/predictor.py', 'w') as f:
        f.write(prediction_code)
    
    print("✅ Standalone prediction function created")

def create_model_info():
    """Create model information file"""
    
    try:
        # Load model comparison results
        model_comparison = joblib.load('results/model_comparison.pkl')
        best_model_idx = model_comparison['F1-Score'].idxmax()
        best_model_info = model_comparison.loc[best_model_idx]
        
        # Create model info dictionary
        model_info = {
            'best_model': best_model_info['Model'],
            'performance': {
                'accuracy': float(best_model_info['Accuracy']),
                'precision': float(best_model_info['Precision']),
                'recall': float(best_model_info['Recall']),
                'f1_score': float(best_model_info['F1-Score']),
                'auc': float(best_model_info['AUC']) if best_model_info['AUC'] != 'N/A' else None
            },
            'dataset_info': {
                'name': 'UCI Heart Disease Dataset',
                'samples': 303,
                'features': 13,
                'selected_features': 8,
                'target': 'Binary classification (0=No disease, 1=Disease)'
            },
            'feature_names': joblib.load('models/selected_features.pkl'),
            'model_type': 'Supervised Learning - Classification',
            'algorithms_used': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM'],
            'preprocessing': ['StandardScaler', 'Feature Selection'],
            'created_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save model info
        joblib.dump(model_info, 'models/model_info.pkl')
        
        # Create human-readable info file
        with open('models/model_info.txt', 'w') as f:
            f.write("Heart Disease Prediction Model Information\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Best Model: {model_info['best_model']}\n")
            f.write(f"Model Type: {model_info['model_type']}\n")
            f.write(f"Created: {model_info['created_date']}\n\n")
            
            f.write("Performance Metrics:\n")
            for metric, value in model_info['performance'].items():
                if value is not None:
                    f.write(f"  {metric.title()}: {value:.4f}\n")
            
            f.write(f"\nDataset Information:\n")
            for key, value in model_info['dataset_info'].items():
                f.write(f"  {key.title()}: {value}\n")
            
            f.write(f"\nSelected Features:\n")
            for i, feature in enumerate(model_info['feature_names'], 1):
                f.write(f"  {i}. {feature}\n")
            
            f.write(f"\nAlgorithms Used:\n")
            for i, algo in enumerate(model_info['algorithms_used'], 1):
                f.write(f"  {i}. {algo}\n")
        
        print("✅ Model information file created")
        
    except Exception as e:
        print(f"❌ Error creating model info: {str(e)}")

def create_deployment_package():
    """Create a deployment package with all necessary files"""
    
    # Create deployment directory
    os.makedirs('deployment_package', exist_ok=True)
    
    # Files to copy
    files_to_copy = [
        'models/complete_pipeline.pkl',
        'models/feature_names.pkl',
        'models/predictor.py',
        'models/model_info.pkl',
        'models/model_info.txt',
        'requirements.txt',
        'README.md'
    ]
    
    # Copy files
    for file_path in files_to_copy:
        if os.path.exists(file_path):
            # Create directory structure in deployment package
            dest_dir = os.path.join('deployment_package', os.path.dirname(file_path))
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy file
            import shutil
            shutil.copy2(file_path, os.path.join('deployment_package', file_path))
            print(f"✅ Copied {file_path}")
        else:
            print(f"⚠️  File not found: {file_path}")
    
    # Create deployment README
    deployment_readme = '''
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
'''
    
    with open('deployment_package/README.md', 'w') as f:
        f.write(deployment_readme)
    
    print("✅ Deployment package created successfully")

def main():
    """Main function to export all models"""
    print("🚀 Starting model export process...")
    print("=" * 50)
    
    # Create model pipeline
    pipeline, features = create_model_pipeline()
    
    if pipeline is not None:
        # Create prediction function
        create_prediction_function()
        
        # Create model info
        create_model_info()
        
        # Create deployment package
        create_deployment_package()
        
        print("\n✅ Model export completed successfully!")
        print("📦 Deployment package created in 'deployment_package/' directory")
        print("🔧 All models and utilities are ready for deployment")
        
    else:
        print("\n❌ Model export failed. Please check your model files.")

if __name__ == "__main__":
    main()
