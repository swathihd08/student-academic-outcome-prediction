import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Academic Outcome Predictor",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/student_outcome_model.joblib"
DATA_PATH = "data/data.csv"


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = joblib.load(MODEL_PATH)

data = pd.read_csv(
    DATA_PATH,
    sep=";"
)


# ============================================================
# NORMALIZE DATASET COLUMN NAMES
# ============================================================

def normalize_column_name(name):
    return (
        str(name)
        .replace("\ufeff", "")
        .strip()
    )


data.columns = [
    normalize_column_name(col)
    for col in data.columns
]


# ============================================================
# HUMAN-READABLE LABELS
# ============================================================

MARITAL_STATUS = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto union",
    6: "Legally separated",
}


APPLICATION_MODE = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99 - item b2",
    27: "Ordinance No. 533-A/99 - item b3",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "International change of institution/course",
}


COURSE = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service - Evening",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management - Evening",
}


PREVIOUS_QUALIFICATION = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year - not completed",
    10: "11th year - not completed",
    12: "Other - 11th year",
    14: "10th year",
    15: "10th year - not completed",
    19: "Basic education 3rd cycle",
    38: "Basic education 2nd cycle",
    39: "Technological specialization course",
    40: "Higher education - degree",
    42: "Professional higher technical course",
    43: "Higher education - master's",
}


NATIONALITY = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldova",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian",
}


QUALIFICATION_LABELS = {
    1: "Secondary Education",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year - Not Completed",
    10: "11th Year - Not Completed",
    11: "7th Year - Old",
    12: "Other - 11th Year",
    13: "2nd Year Complementary High School",
    14: "10th Year",
    18: "General Commerce Course",
    19: "Basic Education 3rd Cycle",
    20: "Complementary High School Course",
    22: "Technical-professional Course",
    25: "Complementary High School - Not Concluded",
    26: "7th Year of Schooling",
    27: "2nd Cycle General High School",
    29: "9th Year - Not Completed",
    30: "8th Year of Schooling",
    31: "General Administration and Commerce",
    33: "Supplementary Accounting and Administration",
    34: "Unknown",
    35: "Can't read or write",
    36: "Can read without 4th year",
    37: "Basic Education 1st Cycle",
    38: "Basic Education 2nd Cycle",
    39: "Technological Specialization Course",
    40: "Higher Education - Degree",
    41: "Specialized Higher Studies Course",
    42: "Professional Higher Technical Course",
    43: "Higher Education - Master",
    44: "Higher Education - Doctorate",
}


OCCUPATION_LABELS = {
    0: "Student",
    1: "Legislative / Executive Representatives and Directors",
    2: "Specialists in Intellectual and Scientific Activities",
    3: "Intermediate Level Technicians and Professionals",
    4: "Administrative Staff",
    5: "Personal Services, Security and Sales Workers",
    6: "Farmers and Skilled Agriculture Workers",
    7: "Skilled Industry, Construction and Craft Workers",
    8: "Installation and Machine Operators",
    9: "Unskilled Workers",
    10: "Armed Forces Professions",
    90: "Other Situation",
    99: "Not Specified",
    101: "Armed Forces Officers",
    102: "Armed Forces Sergeants",
    103: "Other Armed Forces Personnel",
    112: "Administrative and Commercial Directors",
    114: "Hotel / Catering / Trade Directors",
    121: "Physical Science / Mathematics / Engineering Specialists",
    122: "Health Professionals",
    123: "Teachers",
    124: "Finance / Accounting / Administrative Specialists",
    125: "ICT Specialists",
    131: "Science and Engineering Technicians",
    132: "Intermediate Health Technicians",
    134: "Legal / Social / Sports / Cultural Technicians",
    135: "ICT Technicians",
    141: "Office Workers and Secretaries",
    143: "Accounting / Statistical / Financial Operators",
    144: "Administrative Support Staff",
    151: "Personal Service Workers",
    152: "Sellers",
    153: "Personal Care Workers",
    154: "Protection and Security Personnel",
    161: "Market-Oriented Farmers",
    163: "Subsistence Farmers / Fishermen / Hunters",
    171: "Skilled Construction Workers",
    172: "Skilled Metallurgy / Metalworking Workers",
    173: "Printing / Precision / Jewelry / Artisan Workers",
    174: "Skilled Electricity / Electronics Workers",
    175: "Food Processing / Woodworking / Clothing Workers",
    181: "Fixed Plant and Machine Operators",
    182: "Assembly Workers",
    183: "Vehicle Drivers / Mobile Equipment Operators",
    191: "Cleaning Workers",
    192: "Unskilled Agriculture Workers",
    193: "Unskilled Industry / Construction Workers",
    194: "Meal Preparation Assistants",
    195: "Street Vendors / Service Providers",
}


YES_NO = {
    0: "No",
    1: "Yes",
}


GENDER = {
    0: "Female",
    1: "Male",
}


ATTENDANCE = {
    0: "Evening",
    1: "Daytime",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_code(value):
    return int(float(value))


def get_available_codes(column):
    values = []

    for value in data[column].unique():
        try:
            code = clean_code(value)

            if code not in values:
                values.append(code)

        except (ValueError, TypeError):
            pass

    return sorted(values)


def select_coded(label, mapping, column):
    """
    Displays readable labels but returns original numeric code.
    """

    codes = get_available_codes(column)

    if not codes:
        st.error(f"No valid values found for {label}.")
        st.stop()

    labels = []

    for code in codes:
        labels.append(
            mapping.get(code, f"Other category")
        )

    selected_label = st.selectbox(
        label,
        labels
    )

    reverse_mapping = {
        mapping.get(code, "Other category"): code
        for code in codes
    }

    return reverse_mapping[selected_label]


def normalize_for_matching(name):
    """
    Normalize names only for matching.

    This is important because the original trained model may
    contain a trailing space or tab in a feature name.
    """

    return (
        str(name)
        .replace("\ufeff", "")
        .strip()
    )


def align_input_to_model(input_data):
    """
    IMPORTANT FIX.

    The original trained model contains its original feature names.
    One of those names contains a hidden tab:

        Daytime/evening attendance\\t

    The UI uses the clean name:

        Daytime/evening attendance

    This function matches the names after normalization and then
    renames the columns back to EXACTLY what the trained model expects.
    """

    expected_columns = getattr(
        model,
        "feature_names_in_",
        None
    )

    # If feature names are unavailable, return normal input.
    if expected_columns is None:
        return input_data

    expected_columns = list(expected_columns)

    normalized_expected = {}

    for expected in expected_columns:
        normalized_expected[
            normalize_for_matching(expected)
        ] = expected

    rename_map = {}

    for column in input_data.columns:

        normalized = normalize_for_matching(column)

        if normalized in normalized_expected:
            rename_map[column] = normalized_expected[normalized]

    aligned = input_data.rename(
        columns=rename_map
    )

    # Make sure every model feature exists.
    missing = [
        column
        for column in expected_columns
        if column not in aligned.columns
    ]

    if missing:
        raise ValueError(
            f"Model input columns are missing: {missing}"
        )

    # Return columns in exactly the same order used during training.
    aligned = aligned[
        expected_columns
    ]

    return aligned


# ============================================================
# TITLE
# ============================================================

st.title(
    "🎓 Student Academic Outcome Predictor"
)

st.write(
    "Predict a student's academic outcome using information "
    "available at the time of enrollment."
)

st.info(
    "The model predicts Dropout, Enrolled, or Graduate. "
    "Semester-performance information is intentionally excluded "
    "to prevent temporal data leakage."
)


# ============================================================
# STUDENT INFORMATION
# ============================================================

st.header("Student Information")

st.caption(
    "Choose the student's information below. "
    "The application uses readable labels while automatically "
    "converting them to the original dataset codes."
)


# ============================================================
# APPLICATION / ACADEMIC INFORMATION
# ============================================================

marital = select_coded(
    "Marital Status",
    MARITAL_STATUS,
    "Marital status"
)


application_mode = select_coded(
    "Application Mode",
    APPLICATION_MODE,
    "Application mode"
)


application_order = st.selectbox(
    "Application Order",
    list(range(10)),
    format_func=lambda x:
        "First choice"
        if x == 0
        else f"Choice {x + 1}"
)


course = select_coded(
    "Course",
    COURSE,
    "Course"
)


attendance = select_coded(
    "Daytime / Evening Attendance",
    ATTENDANCE,
    "Daytime/evening attendance"
)


previous_qualification = select_coded(
    "Previous Qualification",
    PREVIOUS_QUALIFICATION,
    "Previous qualification"
)


previous_grade = st.number_input(
    "Previous Qualification Grade",
    min_value=0.0,
    max_value=200.0,
    value=float(
        data["Previous qualification (grade)"].median()
    ),
    step=0.1
)


nationality = select_coded(
    "Nationality",
    NATIONALITY,
    "Nacionality"
)


mother_qualification = select_coded(
    "Mother's Qualification",
    QUALIFICATION_LABELS,
    "Mother's qualification"
)


father_qualification = select_coded(
    "Father's Qualification",
    QUALIFICATION_LABELS,
    "Father's qualification"
)


mother_occupation = select_coded(
    "Mother's Occupation",
    OCCUPATION_LABELS,
    "Mother's occupation"
)


father_occupation = select_coded(
    "Father's Occupation",
    OCCUPATION_LABELS,
    "Father's occupation"
)


admission_grade = st.number_input(
    "Admission Grade",
    min_value=0.0,
    max_value=200.0,
    value=float(
        data["Admission grade"].median()
    ),
    step=0.1
)


# ============================================================
# STUDENT STATUS
# ============================================================

displaced = select_coded(
    "Displaced Student",
    YES_NO,
    "Displaced"
)


special_needs = select_coded(
    "Educational Special Needs",
    YES_NO,
    "Educational special needs"
)


debtor = select_coded(
    "Debtor",
    YES_NO,
    "Debtor"
)


tuition = select_coded(
    "Tuition Fees Up To Date",
    YES_NO,
    "Tuition fees up to date"
)


gender = select_coded(
    "Gender",
    GENDER,
    "Gender"
)


scholarship = select_coded(
    "Scholarship Holder",
    YES_NO,
    "Scholarship holder"
)


age = st.number_input(
    "Age at Enrollment",
    min_value=15.0,
    max_value=100.0,
    value=float(
        data["Age at enrollment"].median()
    ),
    step=1.0
)


international = select_coded(
    "International Student",
    YES_NO,
    "International"
)


# ============================================================
# ECONOMIC INFORMATION
# ============================================================

unemployment = st.number_input(
    "Unemployment Rate",
    value=float(
        data["Unemployment rate"].median()
    ),
    step=0.1
)


inflation = st.number_input(
    "Inflation Rate",
    value=float(
        data["Inflation rate"].median()
    ),
    step=0.1
)


gdp = st.number_input(
    "GDP",
    value=float(
        data["GDP"].median()
    ),
    step=0.01
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Academic Outcome",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Validate grades
    # --------------------------------------------------------

    if not 0 <= previous_grade <= 200:

        st.error(
            "Previous Qualification Grade must be between 0 and 200."
        )

        st.stop()


    if not 0 <= admission_grade <= 200:

        st.error(
            "Admission Grade must be between 0 and 200."
        )

        st.stop()


    # --------------------------------------------------------
    # Create raw input
    # --------------------------------------------------------

    input_data = pd.DataFrame([

        {
            "Marital status": marital,

            "Application mode": application_mode,

            "Application order": application_order,

            "Course": course,

            "Daytime/evening attendance": attendance,

            "Previous qualification": previous_qualification,

            "Previous qualification (grade)": previous_grade,

            "Nacionality": nationality,

            "Mother's qualification": mother_qualification,

            "Father's qualification": father_qualification,

            "Mother's occupation": mother_occupation,

            "Father's occupation": father_occupation,

            "Admission grade": admission_grade,

            "Displaced": displaced,

            "Educational special needs": special_needs,

            "Debtor": debtor,

            "Tuition fees up to date": tuition,

            "Gender": gender,

            "Scholarship holder": scholarship,

            "Age at enrollment": age,

            "International": international,

            "Unemployment rate": unemployment,

            "Inflation rate": inflation,

            "GDP": gdp,
        }

    ])


    # --------------------------------------------------------
    # Align columns with trained model
    # --------------------------------------------------------

    try:

        input_data = align_input_to_model(
            input_data
        )

    except Exception as error:

        st.error(
            "The model input could not be prepared."
        )

        st.exception(error)

        st.stop()


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            input_data
        )[0]

        outcome = str(prediction)


        # ----------------------------------------------------
        # Outcome display
        # ----------------------------------------------------

        if outcome == "Dropout":

            st.error(
                f"## 🎓 Predicted Outcome: {outcome}"
            )

        elif outcome == "Graduate":

            st.success(
                f"## 🎓 Predicted Outcome: {outcome}"
            )

        else:

            st.warning(
                f"## 🎓 Predicted Outcome: {outcome}"
            )


        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_data
            )[0]

            classes = model.classes_


            probability_df = pd.DataFrame(
                {
                    "Outcome": classes,
                    "Probability": probabilities
                }
            )


            st.subheader(
                "Prediction Probabilities"
            )


            # ------------------------------------------------
            # Chart
            # ------------------------------------------------

            fig, ax = plt.subplots(
                figsize=(9, 4)
            )


            bars = ax.bar(
                probability_df["Outcome"],
                probability_df["Probability"]
            )


            ax.set_ylim(
                0,
                1
            )


            ax.set_ylabel(
                "Probability"
            )


            ax.set_xlabel(
                "Academic Outcome"
            )


            ax.set_title(
                "Model Prediction Probabilities"
            )


            # Add percentage labels
            for bar, probability in zip(
                bars,
                probabilities
            ):

                ax.text(
                    bar.get_x()
                    + bar.get_width() / 2,

                    probability + 0.02,

                    f"{probability * 100:.1f}%",

                    ha="center",

                    fontweight="bold"
                )


            st.pyplot(
                fig
            )


            plt.close(
                fig
            )


            # ------------------------------------------------
            # Probability cards
            # ------------------------------------------------

            columns = st.columns(
                len(classes)
            )


            for column, outcome_name, probability in zip(
                columns,
                classes,
                probabilities
            ):

                with column:

                    st.metric(
                        str(outcome_name),

                        f"{probability * 100:.2f}%"
                    )


        # ----------------------------------------------------
        # Submitted information
        # ----------------------------------------------------

        with st.expander(
            "🔎 View Submitted Student Information"
        ):

            readable_values = {

                "Marital Status":
                    MARITAL_STATUS.get(
                        marital,
                        "Other"
                    ),

                "Application Mode":
                    APPLICATION_MODE.get(
                        application_mode,
                        "Other"
                    ),

                "Application Order":
                    (
                        "First choice"
                        if application_order == 0
                        else f"Choice {application_order + 1}"
                    ),

                "Course":
                    COURSE.get(
                        course,
                        "Other"
                    ),

                "Attendance":
                    ATTENDANCE.get(
                        attendance,
                        "Other"
                    ),

                "Previous Qualification":
                    PREVIOUS_QUALIFICATION.get(
                        previous_qualification,
                        "Other"
                    ),

                "Previous Qualification Grade":
                    previous_grade,

                "Nationality":
                    NATIONALITY.get(
                        nationality,
                        "Other"
                    ),

                "Mother's Qualification":
                    QUALIFICATION_LABELS.get(
                        mother_qualification,
                        "Other"
                    ),

                "Father's Qualification":
                    QUALIFICATION_LABELS.get(
                        father_qualification,
                        "Other"
                    ),

                "Mother's Occupation":
                    OCCUPATION_LABELS.get(
                        mother_occupation,
                        "Other occupation"
                    ),

                "Father's Occupation":
                    OCCUPATION_LABELS.get(
                        father_occupation,
                        "Other occupation"
                    ),

                "Admission Grade":
                    admission_grade,

                "Displaced":
                    YES_NO.get(
                        displaced,
                        "Other"
                    ),

                "Educational Special Needs":
                    YES_NO.get(
                        special_needs,
                        "Other"
                    ),

                "Debtor":
                    YES_NO.get(
                        debtor,
                        "Other"
                    ),

                "Tuition Fees Up To Date":
                    YES_NO.get(
                        tuition,
                        "Other"
                    ),

                "Gender":
                    GENDER.get(
                        gender,
                        "Other"
                    ),

                "Scholarship Holder":
                    YES_NO.get(
                        scholarship,
                        "Other"
                    ),

                "Age at Enrollment":
                    age,

                "International Student":
                    YES_NO.get(
                        international,
                        "Other"
                    ),

                "Unemployment Rate":
                    unemployment,

                "Inflation Rate":
                    inflation,

                "GDP":
                    gdp,
            }


            summary = pd.DataFrame(
                readable_values.items(),
                columns=[
                    "Field",
                    "Selected Value"
                ]
            )


            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )


    except Exception as error:

        st.error(
            "Prediction could not be completed."
        )

        st.exception(
            error
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚠️ Model predictions represent statistical associations "
    "learned from historical data and do not establish causation."
)

st.caption(
    "Semester-performance variables are intentionally excluded "
    "to prevent temporal data leakage."
)