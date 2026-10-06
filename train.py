import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. SETTINGS
# ============================================================

RANDOM_STATE = 42

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("STUDENT ACADEMIC OUTCOME PREDICTION")
print("=" * 60)

df = pd.read_csv("data/data.csv", sep=";")

print("\nDataset shape:", df.shape)

# ============================================================
# 3. BASIC DATA INSPECTION
# ============================================================

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Missing values ---")
print(df.isnull().sum().sum())

print("\n--- Duplicate rows ---")
print(df.duplicated().sum())

print("\n--- Target distribution ---")
print(df["Target"].value_counts())
print(df["Target"].value_counts(normalize=True) * 100)

# ============================================================
# 4. TARGET VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Target")
plt.title("Student Academic Outcome Distribution")
plt.xlabel("Academic Outcome")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("outputs/target_distribution.png", dpi=150)
plt.close()

# ============================================================
# 5. REMOVE LEAKAGE FEATURES
# ============================================================

leakage_features = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)"
]

print("\n--- Leakage Prevention ---")
print("Removing:", len(leakage_features), "semester-performance features")

X = df.drop(columns=["Target"] + leakage_features)
y = df["Target"]

print("Features remaining:", X.shape[1])

# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

# Continuous numerical variables
numerical_features = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]

# Everything else is treated as encoded categorical data
categorical_features = [
    col for col in X.columns
    if col not in numerical_features
]

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical/encoded-categorical features:")
print(categorical_features)

# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ============================================================
# 8. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ============================================================
# 9. MODEL 1 - LOGISTIC REGRESSION
# ============================================================

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

# ============================================================
# 10. MODEL 2 - RANDOM FOREST
# ============================================================

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    ))
])

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}

# ============================================================
# 11. TRAIN + EVALUATE
# ============================================================

results = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    results[name] = {
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1
    }

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro Precision:{precision:.4f}")
    print(f"Macro Recall:   {recall:.4f}")
    print(f"Macro F1:       {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y),
        yticklabels=np.unique(y)
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        f"outputs/confusion_matrix_{filename}.png",
        dpi=150
    )
    plt.close()

# ============================================================
# 12. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results).T

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)

results_df.to_csv(
    "outputs/model_comparison.csv"
)

# ============================================================
# 13. SELECT FINAL MODEL USING MACRO F1
# ============================================================

best_model_name = results_df["Macro F1"].idxmax()

print("\nFinal model selected:", best_model_name)

final_model = models[best_model_name]

# ============================================================
# 14. FEATURE IMPORTANCE / COEFFICIENTS
# ============================================================

final_classifier = final_model.named_steps["classifier"]
final_preprocessor = final_model.named_steps["preprocessor"]

feature_names = final_preprocessor.get_feature_names_out()

if best_model_name == "Random Forest":

    importance = pd.Series(
        final_classifier.feature_importances_,
        index=feature_names
    )

else:

    # Logistic Regression has coefficients for each class.
    # Use mean absolute coefficient across classes.
    importance = pd.Series(
        np.mean(
            np.abs(final_classifier.coef_),
            axis=0
        ),
        index=feature_names
    )

top_features = importance.sort_values(
    ascending=False
).head(15)

print("\nTop predictive features:")
print(top_features)

plt.figure(figsize=(9, 6))
top_features.sort_values().plot(kind="barh")
plt.title("Top Predictive Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(
    "outputs/feature_importance.png",
    dpi=150
)
plt.close()

# ============================================================
# 15. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    final_model,
    "models/student_outcome_model.joblib"
)

# Save metadata for app
metadata = {
    "model_name": best_model_name,
    "features": list(X.columns),
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "leakage_features_removed": leakage_features
}

joblib.dump(
    metadata,
    "models/metadata.joblib"
)

# ============================================================
# 16. SAMPLE PREDICTIONS
# ============================================================

sample = X_test.iloc[:5]

sample_predictions = final_model.predict(sample)

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for i, prediction in enumerate(sample_predictions):
    print(f"Student {i + 1}: {prediction}")

print("\nTraining complete!")
print("Model saved to: models/student_outcome_model.joblib")
print("Outputs saved to: outputs/")