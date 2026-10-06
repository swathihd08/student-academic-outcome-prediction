# 🎓 Student Academic Outcome Prediction

## 1. Problem Statement

This project predicts a student's final academic outcome as:

- Dropout
- Enrolled
- Graduate

using information available at the time of student enrollment.

This is a multiclass classification problem.

## 2. Dataset

Dataset:
UCI Machine Learning Repository - Predict Students' Dropout and Academic Success

Dataset ID: 697

The supplied dataset contains 4,424 student records.

The dataset is semicolon-delimited.

## 3. Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 4. Data Preparation

The dataset was inspected for:

- Number of records
- Data types
- Missing values
- Duplicate records
- Target distribution

No data-row values were modified.

## 5. Leakage Prevention

The prediction is intended to be made at student enrollment time.

Therefore, the following first- and second-semester performance variables were excluded:

- Curricular units 1st sem (credited)
- Curricular units 1st sem (enrolled)
- Curricular units 1st sem (evaluations)
- Curricular units 1st sem (approved)
- Curricular units 1st sem (grade)
- Curricular units 1st sem (without evaluations)
- Curricular units 2nd sem (credited)
- Curricular units 2nd sem (enrolled)
- Curricular units 2nd sem (evaluations)
- Curricular units 2nd sem (approved)
- Curricular units 2nd sem (grade)
- Curricular units 2nd sem (without evaluations)

These variables are unavailable at enrollment time and could introduce temporal data leakage.

## 6. Train/Test Strategy

An 80/20 stratified train/test split was used.

Random seed: 42.

The test set was kept separate from model training and model selection.

## 7. Models

Two models were compared:

1. Logistic Regression
2. Random Forest

Class weighting was used to help address class imbalance.

## 8. Evaluation

Models were evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Per-class Precision
- Per-class Recall
- Per-class F1
- Confusion Matrix

Macro F1 was emphasized because the target classes are imbalanced.

## 9. Model Selection

The final model was selected based primarily on Macro F1 and balanced class-level performance rather than accuracy alone.

See:

`outputs/model_comparison.csv`

for the actual results produced by the implementation.

## 10. Feature Interpretation

Feature importance was examined to identify variables that were useful for prediction.

Feature importance represents predictive association and does not prove causation.

## 11. Interactive Demo

A Streamlit application is included for demonstrating predictions using the trained model.

Run:

```bash
streamlit run app.py