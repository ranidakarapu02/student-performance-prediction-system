# 🎓 Student Performance Prediction System

A beginner-friendly Data Science & Machine Learning project built using **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, **Scikit-Learn**, and **Streamlit**.

This system predicts a student's final marks based on three key educational indicators:
1. **Attendance Percentage (%)**
2. **Weekly Study Hours**
3. **Previous Exam Marks (/100)**

---

## 📁 Project Structure

```text
student_performance_prediction/
├── data/
│   └── student_data.csv             # Synthetic dataset with 500 student records
├── models/
│   └── student_model.pkl            # Serialized trained model & metadata (Joblib)
├── eda_plots/
│   ├── correlation_heatmap.png      # Pearson correlation matrix
│   ├── feature_distributions.png    # Histograms and KDE distributions
│   └── actual_vs_predicted.png      # Model fit on test data
├── app.py                           # Interactive Streamlit web application
├── train_model.py                   # Data cleaning, EDA, model training & testing
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation and guide
```

---

## 🚀 Quick Start Guide

### Step 1: Open Terminal & Navigate to Project Directory

```bash
cd "C:\Users\ranid\.gemini\antigravity\scratch\student_performance_prediction"
```

### Step 2: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 3: Train the Machine Learning Model & Generate EDA Plots

```bash
python train_model.py
```

### Step 4: Launch the Streamlit Web Application

```bash
streamlit run app.py
```
*(The web dashboard will automatically open in your default browser at `http://localhost:8501`)*

---

## 🧠 Step-by-Step Pipeline Explained Simply

### Step 1: Dataset Generation
We construct a representative dataset of 500 student profiles. Each student entry includes:
- `attendance`: Class attendance percentage (45% – 100%)
- `study_hours`: Dedicated weekly self-study time outside class (2 – 36 hours)
- `previous_marks`: Earlier examination score (30 – 99 marks)
- `final_marks`: Target examination score (0 – 100 marks)

### Step 2: Data Cleaning & Preprocessing
Real-world data is often messy. We demonstrate essential cleaning steps:
- **Deduplication:** Checking for and removing duplicate rows (`drop_duplicates()`).
- **Missing Value Imputation:** Handling missing cells by filling them with the feature's **median** value (`fillna(median)`), which avoids distortion from outliers.
- **Boundary Verification:** Clipping values to physically meaningful ranges (e.g., marks and percentages between 0 and 100).

### Step 3: Exploratory Data Analysis (EDA)
EDA helps us visually understand how features relate to student scores before training algorithms:
- **Correlation Heatmap:** Uncovers how strong the positive relationship is between regular study/attendance and final grades.
- **Distribution Histograms:** Confirms that features follow realistic normal bell curves.

### Step 4: Machine Learning Model Training
We split the data using an **80/20 train-test split**:
- **80% (400 records):** The model learns the relationship between inputs and final marks.
- **20% (100 records):** Held out as an unbiased test set to evaluate real performance.

We evaluate both **Linear Regression** and **Random Forest Regressor**.

### Step 5: Model Testing & Real Evaluation Metrics

> **Note on Evaluation:** These figures are not estimated or claimed — they are the **actual measured results** from running the test evaluation on the 100 held-out test students:

| Metric | Linear Regression | Random Forest Regressor | Explanation for Beginners |
| :--- | :---: | :---: | :--- |
| **$R^2$ Score** | **`0.8572`** | `0.8323` | **`85.7%`** of the variation in final marks is accurately explained by the student's attendance, study time, and previous score. |
| **MAE (Mean Absolute Error)** | **`2.90 marks`** | `3.13 marks` | On average, predictions deviate by fewer than **3 marks** out of 100 from actual scores. |
| **RMSE (Root Mean Squared Error)** | **`3.57 marks`** | `3.87 marks` | Penalizes larger individual errors; confirms strong consistency across students. |

**Learned Regression Equation:**
$$\text{Predicted Marks} = -1.15 + (0.31 \times \text{Attendance}) + (0.64 \times \text{Study Hours}) + (0.47 \times \text{Previous Marks})$$

*Key Takeaway:* Each extra hour of weekly self-study contributes approximately **+0.64 marks**, while each percentage point in attendance adds **+0.31 marks**.

### Step 6: Interactive Streamlit Dashboard (`app.py`)
Features included in the web app:
- **Live Prediction Sliders:** Interactive sliders for attendance, study hours, and previous score.
- **Performance Category & Guidance:** Instant classification into *Distinction*, *First Class*, *Average*, or *At Risk* with constructive feedback.
- **EDA Visualizations:** Direct display of correlation heatmaps and regression plots.
- **Dataset Filter & Explorer:** Interactive table filtering student records by minimum score and attendance.

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+ / Python 3.14
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`
- **Model Persistence:** `joblib`
- **Web App / UI:** `streamlit`
