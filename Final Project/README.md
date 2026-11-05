# 📊 Holistic Data Preparation & Feature Engineering Pipeline

A comprehensive end-to-end data preparation, cleaning, feature engineering, and transformation pipeline built using Python, Pandas, NumPy, and Scikit-Learn. This project processes synthetic customer credit risk data to prepare it for machine learning classification models.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Key Project Phases](#-key-project-phases)
   * [Part A: Conceptual Foundation](#part-a-conceptual-foundation)
   * [Part B: Data Acquisition](#part-b-data-acquisition)
   * [Part C: Data Understanding & Cleaning](#part-c-data-understanding--cleaning)
   * [Part D: Outlier Handling](#part-d-outlier-handling)
   * [Part E: Feature Engineering & Discretization](#part-e-feature-engineering--discretization)
   * [Part F: Feature Scaling](#part-f-feature-scaling)
   * [Part G: Feature Construction & Mathematical Transformations](#part-g-feature-construction--mathematical-transformations)
   * [Part H: Pipeline Integration & Final Deliverables](#part-h-pipeline-integration--final-deliverables)
4. [Installation & Setup](#-installation--setup)
5. [Usage](#-usage)
6. [Summary Report Metrics](#-summary-report-metrics)

---

## 🛠️ Project Overview

This notebook demonstrates the complete data engineering workflow required before training Machine Learning models:

- **Data Generation:** Synthetic credit dataset creation with realistic anomalies.
- **Multi-source Ingestion:** Handling CSV, JSON, SQL, and REST APIs.
- **Missing Data Imputation:** Using Median, Mode, KNN, and Iterative techniques.
- **Outlier Mitigation:** Applying Z-score, IQR capping, Percentiles, and Winsorization.
- **Categorical Encoding:** Utilizing Ordinal, One-Hot, and Label encoding strategies.
- **Skewness Correction:** Using Log, Yeo-Johnson, and Box-Cox power transformations.
- **Pipeline Automation:** Building reproducible Scikit-Learn `ColumnTransformer` models.

---

## 📋 Dataset Description

The primary dataset (`customer_credit_risk_dataset.csv`) contains **1,000 customer records** across **15 initial fields**:

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `customer_id` | Integer | Unique identifier for each customer |
| `age` | Float | Age of customer (contains NaNs for imputation) |
| `gender` | Categorical | Male / Female / Other (contains missing values) |
| `region` | Categorical | Geographic region (North, South, East, West) |
| `education_level` | Ordinal | Primary, Secondary, Graduate, Post-Graduate |
| `employment_type` | Categorical | Salaried, Self-Employed, Unemployed |
| `annual_income` | Float (₹) | Customer annual income (contains outliers & NaNs) |
| `loan_amount` | Float (₹) | Requested loan amount (exponential distribution) |
| `loan_purpose` | Categorical | Home, Car, Education, Business, Other |
| `credit_score` | Float | Credit score range (300 – 850) |
| `repayment_history` | Integer | Missed payments count in last 12 months |
| `transaction_count` | Integer | Total banking transactions in last 6 months |
| `spending_ratio` | Float (%) | Monthly spending ratio |
| `join_date` | Date String | Account opening date |
| **`default_flag`** | Binary | Target Variable ($0 = \text{No Default}, 1 = \text{Default}$) |

---

## 🚀 Key Project Phases

### Part A: Conceptual Foundation
- **Data Analysis & ML Framing:** Formulated problem framing for loan default prediction as a binary classification task.
- **Tensors:** Demonstrations of tensor ranks (0-D Scalar, 1-D Vector, 2-D Matrix, 3-D Tensors) using NumPy arrays.

### Part B: Data Acquisition
Integrates customer datasets from four heterogeneous data sources:
1. **CSV:** Main transactions data.
2. **JSON:** Customer demographics metadata.
3. **SQLite Database:** Credit repayment history.
4. **REST API:** Macroeconomic indicators using `requests`.

### Part C: Data Understanding & Cleaning
- **EDA & Profiling:** Detailed statistics using Pandas (`.info()`, `.describe()`, missingness checks).
- **Imputation Techniques:**
  - `SimpleImputer(strategy='median')` for skewed features (`age`, `annual_income`, `credit_score`).
  - `SimpleImputer(strategy='most_frequent')` for categorical columns (`gender`, `employment_type`).
  - Multivariate `KNNImputer(n_neighbors=5)` for spatial numerical imputation.

### Part D: Outlier Handling
Detected and neutralized extreme financial outliers:
- **Z-Score Method:** Filtered extreme values where $|Z| > 3$.
- **IQR Method:** Capped values outside $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$.
- **Percentile Capping:** Clipped values at 1st and 99th percentiles.
- **Winsorization:** Upper 2% boundary capping using `scipy.stats.mstats.winsorize`.

### Part E: Feature Engineering & Discretization
- **Date-Time Extraction:** Extracted `join_year`, `join_month`, `join_day`, and `join_weekday`.
- **Categorical Encoding:**
  - Ordinal Mapping: `education_level` $\rightarrow \{1, 2, 3, 4\}$
  - Label Encoding: `gender`
  - One-Hot Encoding: `region`, `loan_purpose`, `employment_type`
- **Binning & Discretization:**
  - Fixed Range Binning: `annual_income` (Low, Medium, High).
  - Binarization: `high_credit_flag` ($1$ if `credit_score` $> 700$, else $0$).
  - Quantile Binning (`pd.qcut`): `transaction_count` split into Q1-Q4 buckets.

### Part F: Feature Scaling
Applied distance-preserving numerical transformations:
1. **Standardization (`StandardScaler`):** Rescaled data to $\mu = 0, \sigma = 1$.
2. **Min-Max Scaling (`MinMaxScaler`):** Normalized data into $[0, 1]$ bounds.
3. **Robust Scaling (`RobustScaler`):** Outlier-resistant scaling using Median and IQR.
4. **MaxAbs Scaling (`MaxAbsScaler`):** Preserved sparsity scaling.

### Part G: Feature Construction & Mathematical Transformations
- **Constructed Metrics:**
  - Debt-to-Income Ratio ($\text{DTI} = \frac{\text{loan\_amount}}{\text{annual\_income} + \epsilon}$)
  - Average Monthly Transactions ($\frac{\text{transaction\_count}}{6.0}$)
- **Mathematical Transformations:**
  - Log Transformation ($\log(1+x)$) via `FunctionTransformer`.
  - Power Transformations using **Yeo-Johnson** and **Box-Cox**.
- **Scikit-Learn Pipeline:** Combined multi-type transformers using `ColumnTransformer`.

### Part H: Pipeline Integration & Final Deliverables
- Executed the complete end-to-end data pipeline script.
- Exported the finalized dataset to `final_cleaned_customer_credit_risk.csv`.

---
🎬 Explanation Video Link :

[https://drive.google.com/file/d/1USK90ZWCfnDVKpLU6YtELG5Aw8k17aMB/view?usp=sharing]
