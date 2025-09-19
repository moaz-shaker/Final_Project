# Heart Disease Prediction - Machine Learning Pipeline

A comprehensive machine learning project for predicting heart disease using the UCI Heart Disease dataset.

## Project Overview

This project implements a complete ML pipeline including:
- Data preprocessing and cleaning
- Dimensionality reduction (PCA)
- Feature selection
- Supervised learning (Classification)
- Unsupervised learning (Clustering)
- Hyperparameter tuning
- Model deployment with Streamlit UI

## Dataset

The project uses the Heart Disease UCI dataset (ID: 45) which contains 14 attributes for heart disease prediction.

## Project Structure

```
Heart_Disease_Project/
│── data/                   # Dataset files
│── notebooks/              # Jupyter notebooks for each step
│── models/                 # Trained model files
│── ui/                     # Streamlit application
│── deployment/             # Deployment configurations
│── results/                # Evaluation results
│── README.md
│── requirements.txt
│── .gitignore
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Heart_Disease_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

1. **Data Preprocessing**: `notebooks/01_data_preprocessing.ipynb`
2. **PCA Analysis**: `notebooks/02_pca_analysis.ipynb`
3. **Feature Selection**: `notebooks/03_feature_selection.ipynb`
4. **Supervised Learning**: `notebooks/04_supervised_learning.ipynb`
5. **Unsupervised Learning**: `notebooks/05_unsupervised_learning.ipynb`
6. **Hyperparameter Tuning**: `notebooks/06_hyperparameter_tuning.ipynb`

### Running the Streamlit App

```bash
streamlit run ui/app.py
```

### Deployment with Ngrok

1. Install Ngrok
2. Run the Streamlit app locally
3. In another terminal, run:
```bash
ngrok http 8501
```

## Models Implemented

### Supervised Learning
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

### Unsupervised Learning
- K-Means Clustering
- Hierarchical Clustering

## Performance Metrics
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve & AUC Score

## Results
Model performance results and visualizations are saved in the `results/` directory.
