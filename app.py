import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Load model and metadata
# -------------------------------------------------

model = joblib.load("models/student_outcome_model.joblib")
metadata = joblib.load("models/metadata.joblib")

features = metadata["features"]

st.set_page_config(
    page_title="Student Academic Outcome Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Academic Outcome Predictor")

st.write(
    """
    Predict a student's academic outcome using information
    available at the time of enrollment.
    """
)

st.info(
    "The model predicts Dropout, Enrolled, or Graduate. "
    "Semester-performance information is intentionally excluded "
    "to prevent temporal data leakage."
)

st.header("Student Information")

# -------------------------------------------------
# Create input fields
# -------------------------------------------------

input_data = {}

for feature in features:

    if feature in metadata["numerical_features"]:

        default_value = float(
            pd.read_csv(
                "data/data.csv",
                sep=";"
            )[feature].median()
        )

        input_data[feature] = st.number_input(
            feature,
            value=default_value
        )

    else:

        df = pd.read_csv(
            "data/data.csv",
            sep=";"
        )

        options = sorted(
            df[feature].dropna().unique().tolist()
        )

        input_data[feature] = st.selectbox(
            feature,
            options
        )

# -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("🔮 Predict Academic Outcome"):

    input_df = pd.DataFrame(
        [input_data]
    )

    prediction = model.predict(input_df)[0]

    st.success(
        f"### Predicted Outcome: {prediction}"
    )

    # Probability
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            input_df
        )[0]

        classes = model.classes_

        probability_df = pd.DataFrame({
            "Outcome": classes,
            "Probability": probabilities
        })

        st.subheader("Prediction Probabilities")

        st.bar_chart(
            probability_df.set_index("Outcome")
        )

        for outcome, probability in zip(
            classes,
            probabilities
        ):
            st.write(
                f"**{outcome}:** "
                f"{probability * 100:.2f}%"
            )

st.divider()

st.caption(
    "Note: Model predictions represent statistical associations "
    "learned from historical data and do not establish causation."
)