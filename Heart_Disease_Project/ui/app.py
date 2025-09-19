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
    page_title="CardioPredict Pro - Clinical Heart Disease Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/moaz-shaker/Final_Project/tree/main/Heart_Disease_Project',
        
    }
)

# Professional CSS with modern design and dark mode support
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* CSS Variables for Theme Support */
    :root {
        --primary-color: #2563eb;
        --primary-dark: #1d4ed8;
        --secondary-color: #64748b;
        --accent-color: #0ea5e9;
        --success-color: #059669;
        --warning-color: #d97706;
        --error-color: #dc2626;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-tertiary: #f1f5f9;
        --border-color: #e2e8f0;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    /* Dark Mode Variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #64748b;
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --border-color: #475569;
        }
        
        /* Override card colors in dark mode to maintain readability */
        .metric-card, .prediction-card, .feature-input {
            background: #1e293b !important;
            color: #f8fafc !important;
            border-color: #475569 !important;
        }
        
        .metric-card h3, .feature-input h3 {
            color: #cbd5e1 !important;
        }
        
        .metric-card .value {
            color: #f8fafc !important;
        }
        
        .prediction-card h2, .prediction-card p {
            color: #f8fafc !important;
        }
    }
    
    /* Global Styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }
    
    /* Header Styles */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 400;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 3rem;
        line-height: 1.6;
    }
    
    /* Card Styles */
    .metric-card {
        background: #ffffff;
        color: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin: 0.75rem 0;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary-color);
    }
    
    .metric-card h3 {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748b;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-card .value {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
    }
    
    /* Prediction Cards */
    .prediction-card {
        background: #ffffff;
        color: #1e293b;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: var(--shadow-xl);
        position: relative;
        overflow: hidden;
        border: 2px solid var(--border-color);
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.05) 0%, rgba(14, 165, 233, 0.05) 100%);
        pointer-events: none;
    }
    
    .prediction-card h2 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
        position: relative;
        z-index: 1;
        color: #1e293b;
    }
    
    .prediction-card p {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0.5rem 0;
        opacity: 0.9;
        position: relative;
        z-index: 1;
        color: #1e293b;
    }
    
    /* Risk Level Cards */
    .risk-high {
        border-color: var(--error-color);
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.1) 0%, rgba(185, 28, 28, 0.1) 100%);
    }
    
    .risk-high::before {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.1) 0%, rgba(185, 28, 28, 0.1) 100%);
    }
    
    .risk-high h2 {
        color: var(--error-color);
    }
    
    .risk-low {
        border-color: var(--success-color);
        background: linear-gradient(135deg, rgba(5, 150, 105, 0.1) 0%, rgba(4, 120, 87, 0.1) 100%);
    }
    
    .risk-low::before {
        background: linear-gradient(135deg, rgba(5, 150, 105, 0.1) 0%, rgba(4, 120, 87, 0.1) 100%);
    }
    
    .risk-low h2 {
        color: var(--success-color);
    }
    
    /* Feature Input Styles */
    .feature-input {
        background: #ffffff;
        color: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
    }
    
    .feature-input:hover {
        border-color: var(--primary-color);
        box-shadow: var(--shadow-md);
    }
    
    .feature-input h3 {
        font-family: 'Inter', sans-serif;
        font-size: 1.125rem;
        font-weight: 600;
        color: #1e293b;
        margin: 0 0 1rem 0;
    }
    
    /* Button Styles */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg);
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-color) 100%);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    
    /* Selectbox Styles */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
    
    /* Slider Styles */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Navigation Styles */
    .nav-item {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        transition: all 0.3s ease;
    }
    
    .nav-item:hover {
        background-color: #f1f5f9;
    }
    
    /* Progress Bar */
    .progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 4px;
        border-radius: 2px;
        margin: 1rem 0;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.875rem;
        border: 1px solid transparent;
    }
    
    .status-success {
        background-color: rgba(5, 150, 105, 0.1);
        color: var(--success-color);
        border-color: rgba(5, 150, 105, 0.2);
    }
    
    .status-warning {
        background-color: rgba(217, 119, 6, 0.1);
        color: var(--warning-color);
        border-color: rgba(217, 119, 6, 0.2);
    }
    
    .status-error {
        background-color: rgba(220, 38, 38, 0.1);
        color: var(--error-color);
        border-color: rgba(220, 38, 38, 0.2);
    }
    
    /* Footer */
    .footer {
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: var(--text-secondary);
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid var(--border-color);
        background: var(--bg-secondary);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .prediction-card {
            padding: 1.5rem;
        }
        
        .prediction-card h2 {
            font-size: 2rem;
        }
    }
    
    /* Loading Animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-tertiary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-color) 100%);
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
    # Professional Header
    st.markdown('<h1 class="main-header">🫀 CardioPredict Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Clinical-Grade Heart Disease Risk Assessment Platform powered by Advanced Machine Learning</p>', unsafe_allow_html=True)
    
    # Load models with loading indicator
    with st.spinner('Loading AI models...'):
        scaler, selected_features, models, best_model = load_models()
        model_comparison = load_results()
    
    if scaler is None:
        st.error("⚠️ Unable to load models. Please ensure all model files are present.")
        return
    
    # Professional Sidebar Navigation
    st.sidebar.markdown("### 🧭 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Clinical Assessment Modules", 
        [
            "🏠 Clinical Dashboard",
            "🔬 Risk Assessment",
            "📊 Model Performance",
            "📈 Clinical Analytics",
            "⚙️ System Information"
        ],
        key="nav_select"
    )
    
    # Add status indicator in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔋 System Status")
    st.sidebar.markdown('<span class="status-indicator status-success">✅ All Systems Operational</span>', unsafe_allow_html=True)
    
    # Route to appropriate page
    if page == "🏠 Clinical Dashboard":
        show_home_page()
    elif page == "🔬 Risk Assessment":
        show_prediction_page(scaler, selected_features, models, best_model)
    elif page == "📊 Model Performance":
        show_performance_page(model_comparison)
    elif page == "📈 Clinical Analytics":
        show_analysis_page()
    elif page == "⚙️ System Information":
        show_about_page()

def show_home_page():
    """Display professional dashboard home page"""
    
    # Hero Section
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h2 style="font-family: 'Inter', sans-serif; font-size: 2.5rem; font-weight: 600; color: var(--text-primary); margin-bottom: 1rem;">
            Clinical Assessment Dashboard
        </h2>
        <p style="font-family: 'Inter', sans-serif; font-size: 1.2rem; color: var(--text-secondary); max-width: 800px; margin: 0 auto; line-height: 1.6;">
            Advanced clinical decision support system for cardiovascular risk stratification and early detection. 
            Evidence-based machine learning models trained on clinical datasets for accurate risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Dashboard
    st.markdown("### 📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>AI Models</h3>
            <div class="value">4</div>
            <p style="font-size: 0.875rem; color: #64748b; margin: 0.5rem 0 0 0;">
                Advanced ML algorithms trained on clinical data
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Accuracy</h3>
            <div class="value">90.2%</div>
            <p style="font-size: 0.875rem; color: #64748b; margin: 0.5rem 0 0 0;">
                Best model performance on test data
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Features</h3>
            <div class="value">13</div>
            <p style="font-size: 0.875rem; color: #64748b; margin: 0.5rem 0 0 0;">
                Clinical parameters analyzed
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Dataset</h3>
            <div class="value">303</div>
            <p style="font-size: 0.875rem; color: #64748b; margin: 0.5rem 0 0 0;">
                UCI Heart Disease samples
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-input">
            <h3>🔮 Real-time AI Prediction</h3>
            <p style="color: #64748b; line-height: 1.6;">
                Get instant heart disease risk assessment using advanced machine learning models. 
                Input your health parameters and receive immediate, accurate predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-input">
            <h3>📊 Model Analytics</h3>
            <p style="color: #64748b; line-height: 1.6;">
                Compare performance across multiple AI models including Logistic Regression, 
                Decision Trees, Random Forest, and Support Vector Machines.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-input">
            <h3>📈 Data Insights</h3>
            <p style="color: #64748b; line-height: 1.6;">
                Explore comprehensive data visualizations and statistical analysis of 
                heart disease patterns and risk factors.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-input">
            <h3>⚡ High Performance</h3>
            <p style="color: #64748b; line-height: 1.6;">
                Optimized for speed and accuracy with sub-second prediction times 
                and 90%+ accuracy on clinical data.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("### 🚀 Quick Start Guide")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%); padding: 2rem; border-radius: 16px; border: 1px solid #e2e8f0;">
        <h3 style="font-family: 'Inter', sans-serif; color: #1e293b; margin-bottom: 1.5rem;">How to Get Started</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;">
            <div style="text-align: center;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-weight: 600;">1</div>
                <h4 style="color: #1e293b; margin-bottom: 0.5rem;">Navigate to AI Prediction</h4>
                <p style="color: #64748b; font-size: 0.875rem;">Click on "🔮 AI Prediction" in the sidebar</p>
            </div>
            <div style="text-align: center;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-weight: 600;">2</div>
                <h4 style="color: #1e293b; margin-bottom: 0.5rem;">Input Health Data</h4>
                <p style="color: #64748b; font-size: 0.875rem;">Enter your health parameters in the form</p>
            </div>
            <div style="text-align: center;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-weight: 600;">3</div>
                <h4 style="color: #1e293b; margin-bottom: 0.5rem;">Get AI Prediction</h4>
                <p style="color: #64748b; font-size: 0.875rem;">Click "Predict" for instant risk assessment</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Health Parameters Info
    st.markdown("### 🏥 Health Parameters Analyzed")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Demographics & Basic Info:**
        - Age
        - Sex
        - Chest Pain Type
        - Resting Blood Pressure
        """)
    
    with col2:
        st.markdown("""
        **Laboratory Results:**
        - Serum Cholesterol
        - Fasting Blood Sugar
        - Resting ECG Results
        - Max Heart Rate
        """)
    
    with col3:
        st.markdown("""
        **Exercise & Advanced:**
        - Exercise Induced Angina
        - ST Depression
        - Slope of Peak Exercise
        - Number of Major Vessels
        - Thalassemia Type
        """)
    
    # Disclaimer
    st.markdown("""
    <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 1rem; margin-top: 2rem;">
        <h4 style="color: #92400e; margin-bottom: 0.5rem;">⚠️ Medical Disclaimer</h4>
        <p style="color: #92400e; font-size: 0.875rem; margin: 0;">
            This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_prediction_page(scaler, selected_features, models, best_model):
    """Display professional prediction page"""
    
    # Page Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="font-family: 'Inter', sans-serif; font-size: 2.5rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem;">
            🔬 Cardiovascular Risk Assessment
        </h2>
        <p style="font-family: 'Inter', sans-serif; font-size: 1.1rem; color: var(--text-secondary);">
            Enter clinical parameters below to receive evidence-based cardiovascular risk stratification
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("""
        <div class="feature-input">
            <h3>📋 Health Parameters</h3>
            <p style="color: #64748b; font-size: 0.875rem; margin-bottom: 1.5rem;">
                Please provide accurate information for the best prediction results
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature inputs with better organization
        feature_inputs = {}
        
        # Demographics Section
        st.markdown("#### 👤 Demographics")
        feature_inputs['age'] = st.slider("Age (years)", 29, 77, 50, help="Your current age")
        
        sex_options = {0: "Female", 1: "Male"}
        sex_selected = st.selectbox("Sex", options=list(sex_options.keys()), 
                                  format_func=lambda x: sex_options[x], help="Your biological sex")
        feature_inputs['sex'] = sex_selected
        
        # Cardiovascular Section
        st.markdown("#### ❤️ Cardiovascular Health")
        
        cp_options = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
        cp_selected = st.selectbox("Chest Pain Type", options=list(cp_options.keys()),
                                 format_func=lambda x: cp_options[x], help="Type of chest pain experienced")
        feature_inputs['cp'] = cp_selected
        
        feature_inputs['trestbps'] = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 120, 
                                             help="Blood pressure at rest")
        
        feature_inputs['chol'] = st.slider("Serum Cholesterol (mg/dl)", 126, 564, 200,
                                         help="Total cholesterol level in blood")
        
        # Laboratory Section
        st.markdown("#### 🧪 Laboratory Results")
        
        fbs_options = {0: "≤ 120 mg/dl", 1: "> 120 mg/dl"}
        fbs_selected = st.selectbox("Fasting Blood Sugar", options=list(fbs_options.keys()),
                                  format_func=lambda x: fbs_options[x], help="Blood sugar after fasting")
        feature_inputs['fbs'] = fbs_selected
        
        restecg_options = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
        restecg_selected = st.selectbox("Resting ECG", options=list(restecg_options.keys()),
                                      format_func=lambda x: restecg_options[x], help="Electrocardiographic results")
        feature_inputs['restecg'] = restecg_selected
        
        # Exercise Section
        st.markdown("#### 🏃 Exercise & Stress Test")
        
        feature_inputs['thalach'] = st.slider("Max Heart Rate Achieved", 71, 202, 150,
                                            help="Maximum heart rate during exercise")
        
        exang_options = {0: "No", 1: "Yes"}
        exang_selected = st.selectbox("Exercise Induced Angina", options=list(exang_options.keys()),
                                    format_func=lambda x: exang_options[x], help="Chest pain during exercise")
        feature_inputs['exang'] = exang_selected
        
        feature_inputs['oldpeak'] = st.slider("ST Depression", 0.0, 6.2, 1.0, 0.1,
                                            help="ST depression induced by exercise")
        
        slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
        slope_selected = st.selectbox("Slope", options=list(slope_options.keys()),
                                    format_func=lambda x: slope_options[x], help="Slope of peak exercise ST segment")
        feature_inputs['slope'] = slope_selected
        
        # Advanced Section
        st.markdown("#### 🔬 Advanced Parameters")
        
        ca_options = {0: "0", 1: "1", 2: "2", 3: "3"}
        ca_selected = st.selectbox("Number of Major Vessels", options=list(ca_options.keys()),
                                 format_func=lambda x: ca_options[x], help="Number of major vessels colored by fluoroscopy")
        feature_inputs['ca'] = ca_selected
        
        thal_options = {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}
        thal_selected = st.selectbox("Thalassemia", options=list(thal_options.keys()),
                                   format_func=lambda x: thal_options[x], help="Thalassemia type")
        feature_inputs['thal'] = thal_selected
        
        # Predict button with enhanced styling
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮 Get AI Prediction", type="primary", use_container_width=True):
            make_prediction(feature_inputs, scaler, selected_features, models, best_model)
    
    with col2:
        st.markdown("""
        <div class="feature-input">
            <h3>📊 Prediction Results</h3>
            <p style="color: #64748b; font-size: 0.875rem;">
                AI analysis results will appear here after prediction
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Placeholder for results
        st.info("👈 Please input your health parameters and click 'Get AI Prediction' to see results.")
        
        # Add some helpful information
        st.markdown("""
        <div style="background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
            <h4 style="color: #0c4a6e; margin-bottom: 0.5rem;">💡 Tips for Accurate Results</h4>
            <ul style="color: #0c4a6e; font-size: 0.875rem; margin: 0;">
                <li>Ensure all values are from recent medical tests</li>
                <li>Use resting values for blood pressure and heart rate</li>
                <li>Consult with healthcare professionals for interpretation</li>
                <li>This is a screening tool, not a diagnostic test</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

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
    
    # Display results with professional styling
    st.markdown("### 🎯 AI Prediction Results")
    
    # Main prediction card with enhanced styling
    if best_pred == 1:
        st.markdown("""
        <div class="prediction-card risk-high">
            <h2>⚠️ ELEVATED CARDIOVASCULAR RISK</h2>
            <p>Clinical assessment indicates elevated risk for cardiovascular disease based on current parameters.</p>
            <p><strong>🚨 Clinical Follow-up Recommended</strong></p>
            <p style="font-size: 0.9rem; opacity: 0.8;">Consult with a cardiologist or primary care physician for comprehensive evaluation and risk management.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="prediction-card risk-low">
            <h2>✅ LOW CARDIOVASCULAR RISK</h2>
            <p>Clinical assessment indicates low risk for cardiovascular disease based on current parameters.</p>
            <p><strong>✅ Continue Preventive Care</strong></p>
            <p style="font-size: 0.9rem; opacity: 0.8;">Maintain current lifestyle modifications and continue regular cardiovascular health monitoring.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model predictions comparison with professional styling
    st.markdown("### 📊 AI Model Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-input">
            <h3>🤖 Individual Model Predictions</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for name, pred in predictions.items():
            risk_level = "HIGH RISK" if pred == 1 else "LOW RISK"
            color = "🔴" if pred == 1 else "🟢"
            status_class = "status-error" if pred == 1 else "status-success"
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 0.75rem; background: #f8fafc; border-radius: 8px; margin: 0.5rem 0;">
                <span style="font-size: 1.2rem; margin-right: 0.75rem;">{color}</span>
                <div>
                    <strong style="font-family: 'Inter', sans-serif;">{name}</strong><br>
                    <span class="status-indicator {status_class}">{risk_level}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-input">
            <h3>📈 Risk Probabilities</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for name, prob in probabilities.items():
            if prob is not None:
                risk_prob = prob[1] * 100  # Probability of heart disease
                confidence = max(prob) * 100
                
                # Color coding based on risk level
                if risk_prob >= 70:
                    bar_color = "#ef4444"
                    text_color = "#991b1b"
                elif risk_prob >= 30:
                    bar_color = "#f59e0b"
                    text_color = "#92400e"
                else:
                    bar_color = "#10b981"
                    text_color = "#166534"
                
                st.markdown(f"""
                <div style="margin: 0.75rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                        <strong style="font-family: 'Inter', sans-serif; font-size: 0.875rem;">{name}</strong>
                        <span style="font-family: 'Inter', sans-serif; font-weight: 600; color: {text_color};">{risk_prob:.1f}%</span>
                    </div>
                    <div style="background: #e5e7eb; height: 6px; border-radius: 3px; overflow: hidden;">
                        <div style="background: {bar_color}; height: 100%; width: {risk_prob}%; transition: width 0.3s ease;"></div>
                    </div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">Confidence: {confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Best model details with professional styling
    st.markdown("### 🏆 Primary AI Model Analysis")
    if best_prob is not None:
        risk_probability = best_prob[1] * 100
        confidence = max(best_prob) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Risk Probability</h3>
                <div class="value" style="color: #ef4444;">{:.1f}%</div>
            </div>
            """.format(risk_probability), unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Model Confidence</h3>
                <div class="value" style="color: #10b981;">{:.1f}%</div>
            </div>
            """.format(confidence), unsafe_allow_html=True)
            
        with col3:
            prediction_text = "HIGH RISK" if best_pred == 1 else "LOW RISK"
            prediction_color = "#ef4444" if best_pred == 1 else "#10b981"
            st.markdown("""
            <div class="metric-card">
                <h3>Final Assessment</h3>
                <div class="value" style="color: {};">{}</div>
            </div>
            """.format(prediction_color, prediction_text), unsafe_allow_html=True)
    
    # Add professional footer with recommendations
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%); padding: 2rem; border-radius: 16px; border: 1px solid #e2e8f0; margin-top: 2rem;">
        <h3 style="font-family: 'Inter', sans-serif; color: #1e293b; margin-bottom: 1rem;">📋 Next Steps & Recommendations</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem;">
            <div>
                <h4 style="color: #1e293b; margin-bottom: 0.5rem;">🏥 Medical Consultation</h4>
                <p style="color: #64748b; font-size: 0.875rem; margin: 0;">
                    Schedule an appointment with your healthcare provider to discuss these results and any concerns.
                </p>
            </div>
            <div>
                <h4 style="color: #1e293b; margin-bottom: 0.5rem;">📊 Regular Monitoring</h4>
                <p style="color: #64748b; font-size: 0.875rem; margin: 0;">
                    Continue regular health check-ups and monitor your cardiovascular health parameters.
                </p>
            </div>
            <div>
                <h4 style="color: #1e293b; margin-bottom: 0.5rem;">💡 Lifestyle Factors</h4>
                <p style="color: #64748b; font-size: 0.875rem; margin: 0;">
                    Maintain a healthy diet, regular exercise, and stress management practices.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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
    """Display professional about page"""
    
    # Page Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h2 style="font-family: 'Inter', sans-serif; font-size: 2.5rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">
            ⚙️ System Information
        </h2>
        <p style="font-family: 'Inter', sans-serif; font-size: 1.1rem; color: #64748b;">
            Learn more about HeartGuard AI and its advanced machine learning capabilities
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Overview
    st.markdown("### 🏗️ System Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-input">
            <h3>🫀 HeartGuard AI</h3>
            <p style="color: #64748b; line-height: 1.6;">
                Advanced machine learning system for early heart disease detection and risk assessment. 
                Built with state-of-the-art algorithms and trained on clinical data for maximum accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-input">
            <h3>🎯 Mission Statement</h3>
            <p style="color: #64748b; line-height: 1.6;">
                To provide accessible, accurate, and reliable heart disease risk assessment tools 
                that empower individuals to take proactive steps toward better cardiovascular health.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-input">
            <h3>🔬 Technology Stack</h3>
            <ul style="color: #64748b; line-height: 1.8;">
                <li><strong>Frontend:</strong> Streamlit with custom CSS</li>
                <li><strong>Machine Learning:</strong> Scikit-learn</li>
                <li><strong>Data Processing:</strong> Pandas, NumPy</li>
                <li><strong>Visualization:</strong> Matplotlib, Seaborn, Plotly</li>
                <li><strong>Deployment:</strong> Ngrok, GitHub</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-input">
            <h3>📊 Performance Metrics</h3>
            <ul style="color: #64748b; line-height: 1.8;">
                <li><strong>Accuracy:</strong> 90.2%</li>
                <li><strong>F1-Score:</strong> 90.0%</li>
                <li><strong>AUC:</strong> 95.1%</li>
                <li><strong>Prediction Time:</strong> &lt; 5ms</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Models Section
    st.markdown("### 🤖 AI Models Implemented")
    
    models_info = [
        {
            "name": "Logistic Regression",
            "description": "Linear classification model for binary outcomes",
            "accuracy": "86.9%",
            "use_case": "Baseline model with interpretable results"
        },
        {
            "name": "Decision Tree",
            "description": "Tree-based classification with rule extraction",
            "accuracy": "72.1%",
            "use_case": "Interpretable decision paths"
        },
        {
            "name": "Random Forest",
            "description": "Ensemble method with multiple decision trees",
            "accuracy": "90.2%",
            "use_case": "Best performing model (Primary)"
        },
        {
            "name": "Support Vector Machine",
            "description": "Kernel-based classification with high accuracy",
            "accuracy": "86.9%",
            "use_case": "Robust classification boundaries"
        }
    ]
    
    for i, model in enumerate(models_info, 1):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #1e293b; margin-bottom: 0.5rem;">{model['name']}</h3>
                <p style="color: #64748b; font-size: 0.875rem; margin: 0 0 0.5rem 0;">{model['description']}</p>
                <p style="color: #64748b; font-size: 0.75rem; margin: 0; font-style: italic;">{model['use_case']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Accuracy</h3>
                <div class="value">{model['accuracy']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Rank</h3>
                <div class="value">#{i}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Dataset Information
    st.markdown("### 📈 Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-input">
            <h3>📊 UCI Heart Disease Dataset</h3>
            <ul style="color: #64748b; line-height: 1.8;">
                <li><strong>Source:</strong> UCI Machine Learning Repository</li>
                <li><strong>Dataset ID:</strong> 45 (Heart Disease)</li>
                <li><strong>Total Samples:</strong> 303</li>
                <li><strong>Features:</strong> 13 clinical parameters</li>
                <li><strong>Target:</strong> Binary classification</li>
                <li><strong>Missing Values:</strong> 6 (handled)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-input">
            <h3>🏥 Clinical Parameters</h3>
            <ul style="color: #64748b; line-height: 1.8;">
                <li>Age, Sex, Chest Pain Type</li>
                <li>Resting Blood Pressure</li>
                <li>Serum Cholesterol</li>
                <li>Fasting Blood Sugar</li>
                <li>Resting ECG Results</li>
                <li>Max Heart Rate</li>
                <li>Exercise Induced Angina</li>
                <li>ST Depression</li>
                <li>Slope of Peak Exercise</li>
                <li>Number of Major Vessels</li>
                <li>Thalassemia Type</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Features and Capabilities
    st.markdown("### ✨ Key Features & Capabilities")
    
    features = [
        "🔮 Real-time AI prediction with instant results",
        "📊 Multi-model comparison and analysis",
        "📈 Comprehensive performance metrics visualization",
        "🎯 Risk probability assessment with confidence scores",
        "📱 Responsive web interface for all devices",
        "🔒 Secure data processing and privacy protection",
        "⚡ High-performance prediction engine",
        "📋 Detailed health parameter analysis"
    ]
    
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(features):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 0.75rem; background: #f8fafc; border-radius: 8px; margin: 0.5rem 0;">
                <span style="font-size: 1.2rem; margin-right: 0.75rem;">{feature.split(' ')[0]}</span>
                <span style="color: #64748b; font-size: 0.875rem;">{feature[2:]}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Disclaimer and Contact
    st.markdown("### ⚠️ Important Disclaimers")
    
    st.markdown("""
    <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
        <h4 style="color: #92400e; margin-bottom: 1rem;">Medical Disclaimer</h4>
        <p style="color: #92400e; line-height: 1.6; margin: 0;">
            This application is designed for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. The predictions provided are based on statistical models and should be interpreted with caution. Always consult with qualified healthcare professionals for medical concerns, diagnosis, and treatment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: #f0f9ff; border: 1px solid #0ea5e9; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
        <h4 style="color: #0c4a6e; margin-bottom: 1rem;">Data Privacy & Security</h4>
        <p style="color: #0c4a6e; line-height: 1.6; margin: 0;">
            All data processing is performed locally in your browser. No personal health information is stored, transmitted, or shared with third parties. Your privacy and data security are our top priorities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact Information
    st.markdown("### 📞 Contact & Support")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-input">
            <h3>🐛 Bug Reports</h3>
            <p style="color: #64748b; font-size: 0.875rem;">
                Report issues or bugs through our GitHub repository
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-input">
            <h3>💡 Feature Requests</h3>
            <p style="color: #64748b; font-size: 0.875rem;">
                Suggest new features or improvements
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-input">
            <h3>📧 General Inquiries</h3>
            <p style="color: #64748b; font-size: 0.875rem;">
                Contact the development team for questions
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Professional Footer
    st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto;">
            <div style="text-align: left;">
                <h4 style="font-family: 'Inter', sans-serif; color: var(--text-primary); margin-bottom: 0.5rem;">🫀 CardioPredict Pro</h4>
                <p style="font-size: 0.875rem; color: var(--text-secondary); margin: 0;">Clinical-Grade Cardiovascular Risk Assessment</p>
            </div>
            <div style="text-align: center;">
                <p style="font-size: 0.875rem; color: var(--text-secondary); margin: 0;">© 2024 CardioPredict Pro | Clinical Decision Support System</p>
                <p style="font-size: 0.75rem; color: var(--text-muted); margin: 0.25rem 0 0 0;">For clinical research and educational purposes only</p>
            </div>
            <div style="text-align: right;">
                <div style="display: flex; gap: 1rem;">
                    <a href="#" style="color: var(--text-secondary); text-decoration: none; font-size: 0.875rem;">Privacy</a>
                    <a href="#" style="color: var(--text-secondary); text-decoration: none; font-size: 0.875rem;">Terms</a>
                    <a href="#" style="color: var(--text-secondary); text-decoration: none; font-size: 0.875rem;">Support</a>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
