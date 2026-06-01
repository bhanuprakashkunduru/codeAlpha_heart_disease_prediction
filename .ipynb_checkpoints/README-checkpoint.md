# Heart Disease Prediction Using Machine Learning

## Project Overview

This project focuses on predicting the likelihood of heart disease using patient medical data and machine learning classification algorithms.

The goal is to analyze various health-related attributes and build predictive models that can assist in identifying individuals at risk of heart disease.

---

## Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection can help healthcare professionals provide timely treatment and preventive care.

This project uses machine learning techniques to predict whether a patient is likely to have heart disease based on medical attributes.

---

## Dataset Information

The dataset contains patient health information such as:

- Cholesterol Level
- Fasting Blood Sugar (FBS)
- Thalassemia (Thal)
- Resting Blood Pressure (Trestbps)
- Maximum Heart Rate Achieved (Thalach)
- Number of Major Vessels (CA)
- Exercise Induced Angina (Exang)

### Target Variable

- Presence of Heart Disease
- No Heart Disease

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Jupyter Notebook

---

## Data Preprocessing

The following preprocessing steps were performed:

- Data Loading
- Feature Selection
- Train-Test Split
- Feature Scaling using StandardScaler
- Data Preparation for Model Training

---

## Machine Learning Models Implemented

### 1. Support Vector Machine (SVM)

Support Vector Machine was used to classify patients into heart disease and non-heart disease categories.

### 2. Random Forest Classifier

Random Forest was used as an ensemble learning method to improve prediction performance.

### 3. Logistic Regression

Logistic Regression was used as a baseline classification algorithm after feature scaling.

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy Score
- Classification Report
- Precision
- Recall
- F1-Score

---

## Project Workflow

1. Import Libraries
2. Load Dataset
3. Feature Selection
4. Train-Test Split
5. Feature Scaling
6. Train Machine Learning Models
7. Evaluate Model Performance
8. Compare Results

---

## Features Used

The following features were used for prediction:

- Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Thalassemia (thal)
- Resting Blood Pressure (trestbps)
- Maximum Heart Rate (thalach)
- Number of Major Vessels (ca)
- Exercise Induced Angina (exang)

---

## Results

Multiple classification models were trained and evaluated to predict heart disease.

Models Compared:

- Support Vector Machine (SVM)
- Random Forest Classifier
- Logistic Regression

Performance was measured using classification metrics to identify the most effective model.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
```

Move to the project folder:

```bash
cd heart-disease-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
task_4.ipynb
```

Run all cells to reproduce the results.

---

## Repository Structure

```
Heart-Disease-Prediction/
│
├── task_4.ipynb
├── heart.csv
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Future Improvements

- Hyperparameter Tuning
- Cross Validation
- Feature Engineering
- XGBoost Implementation
- Model Deployment using Streamlit
- Medical Risk Dashboard

---

## Learning Outcomes

This project demonstrates:

- Data Preprocessing
- Feature Scaling
- Classification Algorithms
- Model Evaluation
- Healthcare Data Analysis

---

## Author

Bhanu Prakash

Machine Learning Enthusiast | Data Science Learner