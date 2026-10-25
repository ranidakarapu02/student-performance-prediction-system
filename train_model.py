"""
Student Performance Prediction System - Model Training & EDA Script
===================================================================
This script handles:
1. Synthetic student dataset generation
2. Data cleaning & preprocessing
3. Exploratory Data Analysis (EDA) and visualization generation
4. Model training (Linear Regression & Random Forest)
5. Model evaluation on test data (MAE, RMSE, R2 Score)
6. Saving the trained model for the Streamlit web dashboard
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Ensure output directories exist
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "eda_plots")

for d in [DATA_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "student_data.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "student_model.pkl")

# ---------------------------------------------------------
# STEP 1: CREATE SAMPLE DATASET
# ---------------------------------------------------------
def generate_sample_data(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic student performance dataset with:
    - attendance: Percentage of classes attended (50% to 100%)
    - study_hours: Weekly study hours outside class (2 to 35 hours)
    - previous_marks: Earlier exam score (35 to 98 out of 100)
    - final_marks: Target exam score (0 to 100)
    """
    np.random.seed(random_state)
    
    # 1. Attendance percentage (mean=78%, std=12)
    attendance = np.random.normal(loc=78, scale=12, size=n_samples)
    attendance = np.clip(attendance, 45.0, 100.0).round(1)

    # 2. Weekly study hours (mean=14 hours, std=6)
    study_hours = np.random.normal(loc=14, scale=6, size=n_samples)
    study_hours = np.clip(study_hours, 2.0, 36.0).round(1)

    # 3. Previous exam marks (mean=65, std=15)
    previous_marks = np.random.normal(loc=65, scale=15, size=n_samples)
    previous_marks = np.clip(previous_marks, 30.0, 99.0).round(1)

    # 4. Realistic formula for Final Marks with random human variation (noise)
    # Weights reflect strong influence of previous foundation, study effort, and regular attendance
    noise = np.random.normal(loc=0, scale=3.5, size=n_samples)
    final_marks = (
        0.48 * previous_marks +
        0.28 * attendance +
        0.65 * study_hours +
        noise
    )
    final_marks = np.clip(final_marks, 0.0, 100.0).round(1)

    df = pd.DataFrame({
        "attendance": attendance,
        "study_hours": study_hours,
        "previous_marks": previous_marks,
        "final_marks": final_marks
    })

    # Introduce a small number of realistic missing values (5 rows) to demonstrate data cleaning
    missing_indices = np.random.choice(df.index, size=5, replace=False)
    df.loc[missing_indices[:2], "study_hours"] = np.nan
    df.loc[missing_indices[2:], "attendance"] = np.nan

    # Add a couple of duplicate rows to demonstrate deduplication
    duplicates = df.iloc[:3].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df

# ---------------------------------------------------------
# STEP 2: DATA CLEANING & EDA
# ---------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning:
    - Removes duplicate rows
    - Imputes missing values using feature medians
    - Validates data boundaries
    """
    print("--- [Data Cleaning] ---")
    initial_shape = df.shape
    print(f"Initial dataset shape: {initial_shape}")

    # Check and remove duplicates
    dupes_count = df.duplicated().sum()
    if dupes_count > 0:
        print(f"Removing {dupes_count} duplicate row(s)...")
        df = df.drop_duplicates().reset_index(drop=True)

    # Check and impute missing values
    null_counts = df.isnull().sum()
    print("Missing values per column:")
    for col, count in null_counts.items():
        print(f"  - {col}: {count} missing")
        if count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"    -> Imputed {col} missing values with median: {median_val:.2f}")

    # Check constraints
    df["attendance"] = df["attendance"].clip(0.0, 100.0)
    df["study_hours"] = df["study_hours"].clip(0.0, 60.0)
    df["previous_marks"] = df["previous_marks"].clip(0.0, 100.0)
    df["final_marks"] = df["final_marks"].clip(0.0, 100.0)

    print(f"Cleaned dataset shape: {df.shape}")
    return df

def perform_eda(df: pd.DataFrame):
    """
    Computes statistical summaries and produces exploratory visualization charts.
    """
    print("\n--- [Exploratory Data Analysis (EDA)] ---")
    print("Descriptive Statistics:")
    print(df.describe().round(2))

    # Set plot aesthetic style
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Distribution Plots
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Feature Distributions (Student Dataset)", fontsize=14, fontweight="bold")

    features = [
        ("attendance", "Attendance (%)", "royalblue"),
        ("study_hours", "Weekly Study Hours", "teal"),
        ("previous_marks", "Previous Marks (/100)", "darkorange"),
        ("final_marks", "Final Marks (/100)", "crimson")
    ]

    for ax, (col, label, color) in zip(axes.flatten(), features):
        sns.histplot(df[col], kde=True, color=color, ax=ax, bins=25)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    plt.tight_layout()
    dist_plot_path = os.path.join(PLOTS_DIR, "feature_distributions.png")
    plt.savefig(dist_plot_path, dpi=200)
    plt.close()
    print(f"Saved distributions plot to: {dist_plot_path}")

    # 2. Correlation Heatmap
    plt.figure(figsize=(7, 5))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".3f", linewidths=1.0, cbar=True)
    plt.title("Correlation Matrix: Student Features vs Final Marks", fontsize=12, fontweight="bold")
    plt.tight_layout()
    corr_plot_path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(corr_plot_path, dpi=200)
    plt.close()
    print(f"Saved correlation heatmap to: {corr_plot_path}")

# ---------------------------------------------------------
# STEP 3 & 4: MODEL TRAINING & EVALUATION
# ---------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame):
    """
    Splits data into train and test sets, trains Linear Regression and Random Forest models,
    measures real performance metrics, and saves the best model.
    """
    print("\n--- [Model Training & Evaluation] ---")

    X = df[["attendance", "study_hours", "previous_marks"]]
    y = df["final_marks"]

    # 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 1. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)

    print("\n[Model 1: Linear Regression]")
    print(f"  - R² Score:                  {lr_r2:.4f}")
    print(f"  - Mean Absolute Error (MAE): {lr_mae:.2f} marks")
    print(f"  - Root Mean Squared Error:   {lr_rmse:.2f} marks")
    print("  Learned Formula:")
    coef = lr_model.coef_
    intercept = lr_model.intercept_
    print(f"    Predicted Score = {intercept:.2f} + ({coef[0]:.2f} × Attendance) + ({coef[1]:.2f} × Study Hours) + ({coef[2]:.2f} × Prev Marks)")

    # 2. Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)

    print("\n[Model 2: Random Forest Regressor]")
    print(f"  - R² Score:                  {rf_r2:.4f}")
    print(f"  - Mean Absolute Error (MAE): {rf_mae:.2f} marks")
    print(f"  - Root Mean Squared Error:   {rf_rmse:.2f} marks")

    # Select the model with highest R² score
    if lr_r2 >= rf_r2:
        best_model = lr_model
        best_name = "Linear Regression"
        best_r2 = lr_r2
        best_mae = lr_mae
        best_rmse = lr_rmse
        best_preds = lr_preds
    else:
        best_model = rf_model
        best_name = "Random Forest Regressor"
        best_r2 = rf_r2
        best_mae = rf_mae
        best_rmse = rf_rmse
        best_preds = rf_preds

    print(f"\nBest Performing Model: {best_name} (R² = {best_r2:.4f})")

    # 3. Generate Actual vs. Predicted Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, best_preds, color="dodgerblue", alpha=0.7, edgecolors="k", label="Test Predictions")
    min_val = min(y_test.min(), best_preds.min())
    max_val = max(y_test.max(), best_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Perfect Fit (y = x)")
    plt.xlabel("Actual Final Marks", fontsize=11)
    plt.ylabel("Predicted Final Marks", fontsize=11)
    plt.title(f"Actual vs. Predicted Marks ({best_name})", fontsize=12, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    pred_plot_path = os.path.join(PLOTS_DIR, "actual_vs_predicted.png")
    plt.savefig(pred_plot_path, dpi=200)
    plt.close()
    print(f"Saved actual vs. predicted plot to: {pred_plot_path}")

    # 4. Save best model and metadata
    metadata = {
        "model": best_model,
        "model_name": best_name,
        "features": ["attendance", "study_hours", "previous_marks"],
        "metrics": {
            "r2_score": float(best_r2),
            "mae": float(best_mae),
            "rmse": float(best_rmse)
        },
        "all_results": {
            "linear_regression": {"r2": float(lr_r2), "mae": float(lr_mae), "rmse": float(lr_rmse)},
            "random_forest": {"r2": float(rf_r2), "mae": float(rf_mae), "rmse": float(rf_rmse)}
        }
    }
    joblib.dump(metadata, MODEL_PATH)
    print(f"Model successfully saved to: {MODEL_PATH}")

    return metadata

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("==================================================")
    print("  STUDENT PERFORMANCE PREDICTION SYSTEM: TRAINING  ")
    print("==================================================")
    
    # 1. Generate & Save raw dataset
    df_raw = generate_sample_data(n_samples=500, random_state=42)
    df_raw.to_csv(CSV_PATH, index=False)
    print(f"Sample dataset saved to: {CSV_PATH}")

    # 2. Clean data
    df_clean = clean_data(df_raw)

    # 3. Exploratory Data Analysis
    perform_eda(df_clean)

    # 4. Train, evaluate and save model
    train_and_evaluate(df_clean)
    print("\nTraining workflow completed successfully!")

if __name__ == "__main__":
    main()
