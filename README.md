# Student Academic Outcome Prediction

An AI/ML project that predicts a student's academic outcome at enrollment time as:

- **Dropout**
- **Enrolled**
- **Graduate**

The project uses machine learning with a leakage-safe preprocessing and evaluation pipeline based on the UCI Student Performance dataset.

---

## 🚀 Live Demo

🔗 **[Try the Deployed Student Academic Outcome Prediction App](https://student-academic-outcome-prediction-app-url.streamlit.app/)**

The application is deployed using **Streamlit Community Cloud**.

You can directly open the link, enter student information available at enrollment time, and obtain a predicted academic outcome along with prediction probabilities.

---

## 📌 Project Objective

The objective of this project is to predict whether a student will:

1. Dropout
2. Remain Enrolled
3. Graduate

using only information that would be available **at the time of student enrollment**.

This makes the prediction useful for early identification of students who may require additional academic support.

---

## 📊 Dataset

The project uses the **Predict Students' Dropout and Academic Success** dataset from the UCI Machine Learning Repository.

### Dataset Statistics

- **Total records:** 4,424
- **Total columns:** 37
- **Target column:** `Target`
- **Classes:** Dropout, Enrolled, Graduate
- **Missing values:** 0
- **Duplicate rows:** 0

### Target Distribution

| Outcome | Number of Students | Percentage |
|---|---:|---:|
| Graduate | 2,209 | 49.93% |
| Dropout | 1,421 | 32.12% |
| Enrolled | 794 | 17.95% |

### Dataset Source

UCI Machine Learning Repository:

**Predict Students' Dropout and Academic Success**

https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

---

## ⚠️ Data Leakage Prevention

A major consideration in this project is **temporal data leakage**.

The original dataset contains student performance information from the first and second semesters. These variables would not be available at the time of enrollment.

Therefore, the following semester-performance features were excluded:

### First Semester

- Curricular units 1st sem (credited)
- Curricular units 1st sem (enrolled)
- Curricular units 1st sem (evaluations)
- Curricular units 1st sem (approved)
- Curricular units 1st sem (grade)
- Curricular units 1st sem (without evaluations)

### Second Semester

- Curricular units 2nd sem (credited)
- Curricular units 2nd sem (enrolled)
- Curricular units 2nd sem (evaluations)
- Curricular units 2nd sem (approved)
- Curricular units 2nd sem (grade)
- Curricular units 2nd sem (without evaluations)

After removing these 12 leakage-prone variables, the model uses **24 enrollment-time predictors**.

This ensures that the prediction task matches the intended real-world scenario: predicting outcomes using information available when a student enrolls.

---

## 🧠 Machine Learning Workflow

The project follows the following workflow:

```text
Dataset
   ↓
Data Validation
   ↓
Target Distribution Analysis
   ↓
Remove Temporal Leakage Features
   ↓
Train/Test Split
   ↓
Preprocessing
   ├── Numerical Features
   └── Categorical Features
   ↓
Model Training
   ├── Logistic Regression
   └── Random Forest
   ↓
Model Evaluation
   ├── Accuracy
   ├── Precision
   ├── Recall
   ├── Macro F1
   └── Confusion Matrix
   ↓
Model Comparison
   ↓
Final Model Selection
   ↓
Model Saving
   ↓
Streamlit Deployment