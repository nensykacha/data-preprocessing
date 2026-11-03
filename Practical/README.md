# Customer Purchase Propensity: End-to-End Data Cleaning & Feature Engineering Pipeline

A comprehensive, production-ready Data Science pipeline developed for analyzing customer behavior, engineering predictors, and preprocessing raw transactional/demographic data for machine learning models.

This project rigorously adheres to a 10-step practical curriculum covering Exploratory Data Analysis (EDA), Missing Value Imputation, Outlier Handling, Feature Engineering, Transformation, Scaling, and Exporting.

---

## 📌 Project Overview

Understanding customer purchase propensity is crucial for targeted marketing, churn reduction, and revenue optimization. Raw real-world datasets often contain missing values, extreme outliers, unformatted dates, and mixed-data features.

This repository provides an automated and modular Python data pipeline built with **Pandas**, **NumPy**, and **Scikit-Learn** to transform noisy customer records into an ML-ready dataset.

---

## 🗂️ Dataset Architecture (`customers.csv`)

The raw dataset contains 1,000 customer records with demographic, transactional, and engagement attributes:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `customer_id` | Categorical / String | Unique alphanumeric identifier (e.g., `CUST-1001`) |
| `age` | Numerical | Age of the customer in years |
| `gender` | Categorical | Demographic gender classification |
| `city` | Categorical | Customer location |
| `income` | Numerical | Annual household income in USD |
| `membership_tier` | Categorical | Subscription level (`Standard`, `Silver`, `Gold`, `Platinum`) |
| `total_purchases` | Numerical | Total number of historical purchases |
| `satisfaction_score` | Numerical | Customer feedback score (1.0 to 5.0) |
| `signup_date` | Date String | Account creation date |
| `last_purchase_date` | Date String | Timestamp of the most recent purchase |

---

## 🛠️ Pipeline Workflow (Steps 1–10)

The pipeline executes sequentially in a structured modular approach:

*   **Step 1: Environment Setup & Library Imports**
    *   Configures logging, visualization defaults (Emerald Green palette `#064E3B`, `#10B981`), and loads essential libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).
*   **Step 2: Data Loading & Initial Quality Assessment**
    *   Inspects dataset dimensions (`df.shape`), column types (`df.info()`), and missing value distributions (`df.isnull().sum()`).
*   **Step 3: Exploratory Data Analysis (EDA)**
    *   Generates distribution plots, correlation heatmaps, and statistical summaries (`df.describe()`) to understand underlying patterns.
*   **Step 4: Handling Missing Data**
    *   *SimpleImputer:* Median imputation for numerical features; Most Frequent (Mode) for categorical attributes.
    *   *Missing Indicator & Random Sampling:* Preserves missingness signals while maintaining statistical distributions.
    *   *KNNImputer:* Distance-based multivariate imputation ($k=5$).
    *   *MICE Algorithm (IterativeImputer):* Multivariate chained regression imputation for correlated attributes.
    *   *Complete Case Analysis:* Evaluated as a comparative baseline.
*   **Step 5: Outlier Detection & Handling**
    *   *Z-Score Method:* Filters extreme records ($|Z| > 3$).
    *   *IQR Method:* Calculates bounds ($Q1 - 1.5 \times IQR$ to $Q3 + 1.5 \times IQR$).
    *   *Winsorization:* Caps extreme tail values at the 1st and 99th percentiles to avoid dropping rows unnecessarily.
*   **Step 6: Processing Date/Time & Mixed-Type Variables**
    *   Parses string dates (`signup_date`, `last_purchase_date`) to native datetime formats.
    *   Derives temporal features: `days_since_last_purchase`, `days_since_signup`, `signup_year`, and `signup_month`.
    *   Handles mixed-type identifier strings (`customer_id`) by regex-extracting clean numerical IDs.
*   **Step 7: Categorical Data Encoding & Binning**
    *   *Label Encoding:* Applied to binary features (e.g., `gender`).
    *   *One-Hot Encoding:* Applied to nominal attributes (e.g., `city`).
    *   *Ordinal Encoding:* Preserves hierarchical order for `membership_tier` (`Standard` $\rightarrow$ `Silver` $\rightarrow$ `Gold` $\rightarrow$ `Platinum`).
    *   *Feature Binning:* Groups continuous income into discrete categories (`Low`, `Medium`, `High`) and ordinal-encodes them safely handling `NaN` placeholders.
*   **Step 8: Feature Scaling & ColumnTransformer**
    *   Applies concurrent feature scaling techniques via Scikit-Learn's `ColumnTransformer`:
        *   `StandardScaler` ($\mu=0, \sigma=1$)
        *   `MinMaxScaler` (Range $[0, 1]$)
        *   `RobustScaler` (Outlier-resistant scaling)
        *   `MaxAbsScaler` & `Normalizer`
*   **Step 9: Feature Construction & Mathematical Transformations**
    *   *Interaction Features:* Constructs unit rates such as `purchase_per_day = total_purchases / days_since_signup`.
    *   *FunctionTransformer:* Applies logarithmic (`log1p`) and square-root transformations to reduce skewness.
    *   *PowerTransformer:* Applies Yeo-Johnson transformations for Gaussian distribution alignment.
    *   *Binarization:* Constructs binary flags like `frequent_buyer` based on median activity thresholds.
*   **Step 10: Exporting Final Output & Pipeline Summary**
    *   Exports the fully cleaned, transformed, and feature-engineered dataset to `processed_customer_data.csv`.
    *   Generates a structural summary report detailing data quality resolution and feature engineering choices.

---
