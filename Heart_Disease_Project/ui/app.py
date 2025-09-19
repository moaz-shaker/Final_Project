import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e74c3c;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .feature-input {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_data
def load_models():
    """Load all trained models and preprocessing objects"""
    try:
        # Load preprocessing objects
        scaler = joblib.load('models/scaler.pkl')
        selected_features = joblib.load('models/selected_features.pkl')
        
        # Load models
        models = {
            'Logistic Regression': joblib.load('models/logistic_regression_model.pkl'),
            'Decision Tree': joblib.load('models/decision_tree_model.pkl'),
            'Random Forest': joblib.load('models/random_forest_model.pkl'),
            'SVM': joblib.load('models/svm_model.pkl')
        }
        
        # Load best model
        best_model = joblib.load('models/final_best_model.pkl')
        
        return scaler, selected_features, models, best_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

@st.cache_data
def load_results():
    """Load evaluation results"""
    try:
        model_comparison = joblib.load('results/model_comparison.pkl')
        return model_comparison
    except:
        return None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # Load models
    scaler, selected_features, models, best_model = load_models()
    model_comparison = load_results()
    
    if scaler is None:
        st.error("Unable to load models. Please ensure all model files are present.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Home",
        "🔮 Prediction",
        "📊 Model Performance",
        "📈 Data Analysis",
        "ℹ️ About"
    ])
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Prediction":
        show_prediction_page(scaler, selected_features, models, best_model)
    elif page == "📊 Model Performance":
        show_performance_page(model_comparison)
    elif page == "📈 Data Analysis":
        show_analysis_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Display home page"""
    st.markdown("""
    ## Welcome to the Heart Disease Prediction System
    
    This application uses machine learning to predict the risk of heart disease based on various health parameters.
    
    ### Features:
    - **Real-time Prediction**: Input your health data and get instant predictions
    - **Multiple Models**: Compare predictions from different ML algorithms
    - **Performance Analysis**: View detailed model performance metrics
    - **Data Visualization**: Explore heart disease patterns and trends
    
    ### How to Use:
    1. Navigate to the **Prediction** page
    2. Input your health parameters in the sidebar
    3. Click "Predict" to get your heart disease risk assessment
    4. View model performance and analysis in other sections
    
    ### Health Parameters:
    - **Age**: Your age in years
    - **Sex**: Gender (0 = Female, 1 = Male)
    - **Chest Pain Type**: Type of chest pain experienced
    - **Resting Blood Pressure**: Blood pressure at rest
    - **Cholesterol**: Serum cholesterol level
    - **Fasting Blood Sugar**: Blood sugar level after fasting
    - **Resting ECG**: Electrocardiographic results
    - **Max Heart Rate**: Maximum heart rate achieved
    - **Exercise Induced Angina**: Chest pain during exercise
    - **ST Depression**: ST depression induced by exercise
    - **Slope**: Slope of peak exercise ST segment
    - **Number of Vessels**: Number of major vessels colored by fluoroscopy
    - **Thal**: Thalassemia type
    """)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Models Trained", "4", "Logistic Regression, Decision Tree, Random Forest, SVM")
    
    with col2:
        st.metric("Features Used", "8", "Selected from 13 original features")
    
    with col3:
        st.metric("Accuracy", "~85%", "Best model performance")
    
    with col4:
        st.metric("Dataset Size", "303", "UCI Heart Disease Dataset")

def show_prediction_page(scaler, selected_features, models, best_model):
    """Display prediction page"""
    st.header("🔮 Heart Disease Risk Prediction")
    
    # Create two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Feature inputs
        feature_inputs = {}
        
        # Age
        feature_inputs['age'] = st.slider("Age", 29, 77, 50)
        
        # Sex
        sex_options = {0: "Female", 1: "Male"}
        sex_selected = st.selectbox("Sex", options=list(sex_options.keys()), 
                                  format_func=lambda x: sex_options[x])
        feature_inputs['sex'] = sex_selected
        
        # Chest Pain Type
        cp_options = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
        cp_selected = st.selectbox("Chest Pain Type", options=list(cp_options.keys()),
                                 format_func=lambda x: cp_options[x])
        feature_inputs['cp'] = cp_selected
        
        # Resting Blood Pressure
        feature_inputs['trestbps'] = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 120)
        
        # Cholesterol
        feature_inputs['chol'] = st.slider("Serum Cholesterol (mg/dl)", 126, 564, 200)
        
        # Fasting Blood Sugar
        fbs_options = {0: "≤ 120 mg/dl", 1: "> 120 mg/dl"}
        fbs_selected = st.selectbox("Fasting Blood Sugar", options=list(fbs_options.keys()),
                                  format_func=lambda x: fbs_options[x])
        feature_inputs['fbs'] = fbs_selected
        
        # Resting ECG
        restecg_options = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
        restecg_selected = st.selectbox("Resting ECG", options=list(restecg_options.keys()),
                                      format_func=lambda x: restecg_options[x])
        feature_inputs['restecg'] = restecg_selected
        
        # Max Heart Rate
        feature_inputs['thalach'] = st.slider("Max Heart Rate Achieved", 71, 202, 150)
        
        # Exercise Induced Angina
        exang_options = {0: "No", 1: "Yes"}
        exang_selected = st.selectbox("Exercise Induced Angina", options=list(exang_options.keys()),
                                    format_func=lambda x: exang_options[x])
        feature_inputs['exang'] = exang_selected
        
        # ST Depression
        feature_inputs['oldpeak'] = st.slider("ST Depression", 0.0, 6.2, 1.0, 0.1)
        
        # Slope
        slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
        slope_selected = st.selectbox("Slope", options=list(slope_options.keys()),
                                    format_func=lambda x: slope_options[x])
        feature_inputs['slope'] = slope_selected
        
        # Number of Vessels
        ca_options = {0: "0", 1: "1", 2: "2", 3: "3"}
        ca_selected = st.selectbox("Number of Major Vessels", options=list(ca_options.keys()),
                                 format_func=lambda x: ca_options[x])
        feature_inputs['ca'] = ca_selected
        
        # Thal
        thal_options = {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}
        thal_selected = st.selectbox("Thalassemia", options=list(thal_options.keys()),
                                   format_func=lambda x: thal_options[x])
        feature_inputs['thal'] = thal_selected
        
        # Predict button
        if st.button("🔮 Predict Heart Disease Risk", type="primary"):
            make_prediction(feature_inputs, scaler, selected_features, models, best_model)
    
    with col2:
        st.subheader("Prediction Results")
        st.info("👈 Please input your health parameters and click 'Predict' to see results.")

def make_prediction(feature_inputs, scaler, selected_features, models, best_model):
    """Make prediction and display results"""
    # Create DataFrame from inputs
    input_df = pd.DataFrame([feature_inputs])
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)
    
    # Select only the features used in training
    input_selected = input_scaled_df[selected_features]
    
    # Make predictions with all models
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        pred = model.predict(input_selected)[0]
        prob = model.predict_proba(input_selected)[0] if hasattr(model, 'predict_proba') else None
        predictions[name] = pred
        probabilities[name] = prob
    
    # Best model prediction
    best_pred = best_model.predict(input_selected)[0]
    best_prob = best_model.predict_proba(input_selected)[0] if hasattr(best_model, 'predict_proba') else None
    
    # Display results
    st.subheader("🎯 Prediction Results")
    
    # Main prediction card
    if best_pred == 1:
        st.markdown("""
        <div class="prediction-card">
            <h2>⚠️ HIGH RISK</h2>
            <p>Based on your health parameters, there is a high risk of heart disease.</p>
            <p><strong>Please consult with a healthcare professional.</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="prediction-card">
            <h2>✅ LOW RISK</h2>
            <p>Based on your health parameters, there is a low risk of heart disease.</p>
            <p><strong>Continue maintaining a healthy lifestyle!</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model predictions comparison
    st.subheader("📊 Model Predictions Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Individual Model Predictions:**")
        for name, pred in predictions.items():
            risk_level = "HIGH RISK" if pred == 1 else "LOW RISK"
            color = "🔴" if pred == 1 else "🟢"
            st.write(f"{color} **{name}**: {risk_level}")
    
    with col2:
        st.write("**Prediction Probabilities:**")
        for name, prob in probabilities.items():
            if prob is not None:
                risk_prob = prob[1] * 100  # Probability of heart disease
                st.write(f"**{name}**: {risk_prob:.1f}% risk")
    
    # Best model details
    st.subheader("🏆 Best Model Prediction")
    if best_prob is not None:
        risk_probability = best_prob[1] * 100
        confidence = max(best_prob) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Probability", f"{risk_probability:.1f}%")
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")
        with col3:
            st.metric("Prediction", "HIGH RISK" if best_pred == 1 else "LOW RISK")
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        st.subheader("📈 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title="Feature Importance in Prediction")
        st.plotly_chart(fig, use_container_width=True)

def show_performance_page(model_comparison):
    """Display model performance page"""
    st.header("📊 Model Performance Analysis")
    
    if model_comparison is None:
        st.error("Model comparison data not available.")
        return
    
    # Model comparison table
    st.subheader("Model Performance Comparison")
    st.dataframe(model_comparison.round(4), use_container_width=True)
    
    # Performance metrics visualization
    st.subheader("Performance Metrics Visualization")
    
    # Create performance charts
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=model_comparison['Model'],
            y=model_comparison[metric],
            text=model_comparison[metric].round(3),
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model highlight
    best_model_idx = model_comparison['F1-Score'].idxmax()
    best_model = model_comparison.loc[best_model_idx]
    
    st.subheader("🏆 Best Performing Model")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", best_model['Model'])
    with col2:
        st.metric("F1-Score", f"{best_model['F1-Score']:.4f}")
    with col3:
        st.metric("Accuracy", f"{best_model['Accuracy']:.4f}")
    with col4:
        st.metric("AUC", f"{best_model['AUC']:.4f}")

def show_analysis_page():
    """Display data analysis page"""
    st.header("📈 Heart Disease Data Analysis")
    
    # Sample data visualization
    st.subheader("Dataset Overview")
    
    # Create sample data for visualization
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Age': np.random.normal(54, 9, 100),
        'Cholesterol': np.random.normal(246, 51, 100),
        'Blood Pressure': np.random.normal(131, 17, 100),
        'Heart Rate': np.random.normal(149, 23, 100),
        'Heart Disease': np.random.choice([0, 1], 100, p=[0.46, 0.54])
    })
    
    # Age distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(sample_data, x='Age', color='Heart Disease',
                          title="Age Distribution by Heart Disease Status",
                          color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(sample_data, x='Heart Disease', y='Cholesterol',
                    title="Cholesterol Levels by Heart Disease Status",
                    color='Heart Disease',
                    color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlation Analysis")
    
    # Create correlation matrix
    corr_data = sample_data.corr()
    
    fig = px.imshow(corr_data, 
                   text_auto=True,
                   aspect="auto",
                   title="Feature Correlation Matrix",
                   color_continuous_scale='RdBu_r')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors analysis
    st.subheader("Key Risk Factors")
    
    risk_factors = [
        "High cholesterol levels (>240 mg/dl)",
        "High blood pressure (>140/90 mm Hg)",
        "Age over 65 years",
        "Male gender",
        "Family history of heart disease",
        "Smoking",
        "Diabetes",
        "Physical inactivity"
    ]
    
    for i, factor in enumerate(risk_factors, 1):
        st.write(f"{i}. {factor}")

def show_about_page():
    """Display about page"""
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## Heart Disease Prediction System
    
    This application is built using machine learning techniques to predict the risk of heart disease 
    based on various health parameters. The system uses the UCI Heart Disease dataset and implements 
    multiple classification algorithms for accurate predictions.
    
    ### Technology Stack:
    - **Frontend**: Streamlit
    - **Machine Learning**: Scikit-learn
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn, Plotly
    
    ### Models Implemented:
    1. **Logistic Regression**: Linear classification model
    2. **Decision Tree**: Tree-based classification
    3. **Random Forest**: Ensemble method with multiple trees
    4. **Support Vector Machine (SVM)**: Kernel-based classification
    
    ### Features:
    - Real-time prediction interface
    - Multiple model comparison
    - Performance metrics visualization
    - Data analysis and insights
    - Responsive web interface
    
    ### Dataset Information:
    - **Source**: UCI Machine Learning Repository
    - **Dataset ID**: 45 (Heart Disease)
    - **Samples**: 303
    - **Features**: 13 (reduced to 8 after feature selection)
    - **Target**: Binary classification (0 = No disease, 1 = Disease)
    
    ### Disclaimer:
    This application is for educational and research purposes only. 
    It should not be used as a substitute for professional medical advice, 
    diagnosis, or treatment. Always consult with qualified healthcare 
    professionals for medical concerns.
    
    ### Contact:
    For questions or feedback, please contact the development team.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>© 2024 Heart Disease Prediction System | Built with ❤️ and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
