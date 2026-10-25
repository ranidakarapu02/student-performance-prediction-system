"""
Student Performance Prediction System - Streamlit Dashboard
============================================================
An interactive web application for predicting a student's final marks
based on attendance, study hours, and previous marks.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "student_model.pkl")
PLOTS_DIR = os.path.join(BASE_DIR, "eda_plots")

# ---------------------------------------------------------
# LOAD MODEL & DATA (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

model_artifact = load_trained_model()
df_students = load_dataset()

# Custom CSS for clean UI styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border-left: 4px solid #2563EB;
    }
    .prediction-box {
        background: #EFF6FF;
        border: 2px solid #3B82F6;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .score-badge {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR: USER INPUTS
# ---------------------------------------------------------
st.sidebar.header("🎓 Student Input Parameters")
st.sidebar.write("Adjust the sliders below to predict performance:")

input_attendance = st.sidebar.slider(
    "Attendance Percentage (%)",
    min_value=0.0,
    max_value=100.0,
    value=85.0,
    step=0.5,
    help="Overall class attendance percentage (0% to 100%)"
)

input_study_hours = st.sidebar.slider(
    "Weekly Study Hours",
    min_value=0.0,
    max_value=40.0,
    value=15.0,
    step=0.5,
    help="Hours per week spent on self-study outside classroom hours"
)

input_prev_marks = st.sidebar.slider(
    "Previous Exam Marks (/100)",
    min_value=0.0,
    max_value=100.0,
    value=72.0,
    step=0.5,
    help="Score achieved in the previous examination (0 to 100)"
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Quick Tip:** Increasing study hours and maintaining higher attendance "
    "consistently raises predicted scores."
)

# ---------------------------------------------------------
# MAIN INTERFACE TABS
# ---------------------------------------------------------
st.markdown('<div class="main-title">🎓 Student Performance Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A beginner-friendly Machine Learning application to forecast academic performance based on attendance, study habits, and previous results.</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Live Prediction",
    "📊 Exploratory Data Analysis (EDA)",
    "📁 Dataset Explorer",
    "🧠 Model Architecture & Explanation"
])

# ---------------------------------------------------------
# TAB 1: LIVE PREDICTION
# ---------------------------------------------------------
with tab1:
    col_pred, col_details = st.columns([1.2, 1.0])

    with col_pred:
        st.subheader("Predict Student's Final Marks")

        if model_artifact is None:
            st.error("⚠️ Model file not found! Please run `python train_model.py` first to train the model.")
        else:
            model = model_artifact["model"]
            model_name = model_artifact.get("model_name", "Trained ML Model")
            metrics = model_artifact.get("metrics", {})

            # Prepare feature input
            user_features = pd.DataFrame([{
                "attendance": input_attendance,
                "study_hours": input_study_hours,
                "previous_marks": input_prev_marks
            }])

            # Predict score
            raw_prediction = model.predict(user_features)[0]
            prediction = round(float(np.clip(raw_prediction, 0.0, 100.0)), 1)

            # Determine performance grade/badge
            if prediction >= 85:
                status_color = "green"
                status_text = "🌟 Outstanding / Distinction"
                feedback = "Excellent performance! Keep up the consistent study routine and high attendance."
            elif prediction >= 70:
                status_color = "blue"
                status_text = "👍 First Class / Good"
                feedback = "Solid standing with good comprehension. Increasing study hours can push this to top tier."
            elif prediction >= 50:
                status_color = "orange"
                status_text = "⚠️ Average / Pass"
                feedback = "Moderate performance. Focus on improving attendance and dedicated weekly study hours."
            else:
                status_color = "red"
                status_text = "🚨 At Risk / Needs Immediate Attention"
                feedback = "High risk of academic struggle. Recommend intervention, tutoring, and attendance tracking."

            st.markdown(f"""
            <div class="prediction-box">
                <span style="font-size: 1.1rem; color: #374151; font-weight: 600;">Forecasted Final Score</span>
                <div class="score-badge">{prediction} <span style="font-size: 1.5rem; color: #6B7280;">/ 100</span></div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {status_color}; margin-top: 5px;">
                    {status_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.info(f"**Recommendation:** {feedback}")

    with col_details:
        st.subheader("Student Feature Summary")
        summary_df = pd.DataFrame({
            "Feature": ["Attendance (%)", "Weekly Study (Hours)", "Previous Score (/100)"],
            "Student Value": [f"{input_attendance}%", f"{input_study_hours} hrs", f"{input_prev_marks} / 100"],
            "Dataset Average": [
                f"{df_students['attendance'].mean():.1f}%" if df_students is not None else "78.0%",
                f"{df_students['study_hours'].mean():.1f} hrs" if df_students is not None else "14.0 hrs",
                f"{df_students['previous_marks'].mean():.1f} / 100" if df_students is not None else "65.0 / 100"
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.subheader("Model Performance Metrics (Tested)")
        if model_artifact:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("R² Score", f"{metrics.get('r2_score', 0):.3f}")
            with m2:
                st.metric("MAE", f"{metrics.get('mae', 0):.2f} pts")
            with m3:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.2f} pts")

# ---------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------
with tab2:
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write(
        "EDA helps us uncover underlying patterns, correlations, and distributions "
        "before training machine learning algorithms."
    )

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        st.markdown("**1. Correlation Heatmap**")
        st.caption("Shows how strongly each factor correlates with Final Marks.")
        heatmap_img = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
        if os.path.exists(heatmap_img):
            st.image(heatmap_img, use_container_width=True)
        else:
            st.warning("Heatmap not generated yet. Run `python train_model.py`.")

    with eda_col2:
        st.markdown("**2. Actual vs. Predicted Marks**")
        st.caption("Dots along the red dashed line indicate near-perfect predictions on test data.")
        pred_img = os.path.join(PLOTS_DIR, "actual_vs_predicted.png")
        if os.path.exists(pred_img):
            st.image(pred_img, use_container_width=True)
        else:
            st.warning("Prediction plot not generated yet. Run `python train_model.py`.")

    st.markdown("**3. Feature Distributions**")
    dist_img = os.path.join(PLOTS_DIR, "feature_distributions.png")
    if os.path.exists(dist_img):
        st.image(dist_img, use_container_width=True)
    else:
        st.warning("Distribution plot not generated yet. Run `python train_model.py`.")

# ---------------------------------------------------------
# TAB 3: DATASET EXPLORER
# ---------------------------------------------------------
with tab3:
    st.subheader("Sample Student Dataset")
    if df_students is not None:
        st.write(f"The dataset contains **{len(df_students)}** records with attendance, study hours, previous marks, and final marks.")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            min_score = st.slider("Filter by Minimum Final Marks", 0, 100, 40)
        with col_f2:
            min_att = st.slider("Filter by Minimum Attendance (%)", 0, 100, 50)

        filtered_df = df_students[
            (df_students["final_marks"] >= min_score) &
            (df_students["attendance"] >= min_att)
        ]
        st.dataframe(filtered_df, use_container_width=True)

        st.subheader("Dataset Summary Statistics")
        st.dataframe(df_students.describe().round(2), use_container_width=True)
    else:
        st.warning("Dataset not found at data/student_data.csv. Run `python train_model.py` to generate it.")

# ---------------------------------------------------------
# TAB 4: MODEL ARCHITECTURE & EXPLANATION
# ---------------------------------------------------------
with tab4:
    st.subheader("How Does the Prediction Model Work?")
    st.markdown("""
    ### 1. The Machine Learning Pipeline
    - **Step 1: Data Collection & Simulation:** Synthetic records mimicking real classroom dynamics.
    - **Step 2: Cleaning & Preprocessing:** Handling null values via median imputation and removing duplicates.
    - **Step 3: Feature & Target Split:**
      - Features ($X$): `attendance`, `study_hours`, `previous_marks`
      - Target ($y$): `final_marks`
    - **Step 4: Train-Test Split:** 80% used for model training, 20% reserved for unbiased testing.
    - **Step 5: Regression Modeling:** Testing Linear Regression & Random Forest Regressor to discover optimal coefficients.
    - **Step 6: Evaluation:** Measuring accuracy via $R^2$ (coefficient of determination), Mean Absolute Error (MAE), and RMSE.

    ### 2. Evaluation Metrics Explained
    - **$R^2$ Score (Coefficient of Determination):** Measures the proportion of variance in final marks explained by our inputs (closer to 1.0 is better).
    - **MAE (Mean Absolute Error):** The average difference in marks between predicted scores and actual scores.
    - **RMSE (Root Mean Squared Error):** Penalizes larger prediction mistakes more heavily.
    """)

    if model_artifact and "all_results" in model_artifact:
        st.subheader("Comparison of Tested Algorithms")
        results = model_artifact["all_results"]
        comparison_df = pd.DataFrame([
            {
                "Model": "Linear Regression",
                "R² Score": f"{results['linear_regression']['r2']:.4f}",
                "MAE (Marks)": f"{results['linear_regression']['mae']:.2f}",
                "RMSE (Marks)": f"{results['linear_regression']['rmse']:.2f}"
            },
            {
                "Model": "Random Forest Regressor",
                "R² Score": f"{results['random_forest']['r2']:.4f}",
                "MAE (Marks)": f"{results['random_forest']['mae']:.2f}",
                "RMSE (Marks)": f"{results['random_forest']['rmse']:.2f}"
            }
        ])
        st.table(comparison_df)
