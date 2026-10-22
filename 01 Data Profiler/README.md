# 📊 Customer Purchase Behavior Analysis & Churn Prediction

## 📌 Project Overview
This project focuses on **Data Preprocessing, Multi-Source Data Integration, Exploratory Data Analysis (EDA), and Data Profiling** using Python in a Jupyter Notebook environment. The target objective is to analyze customer purchasing patterns and prepare the dataset for predicting customer churn[cite: 1, 2].

🛠️ Tech Stack & Environment
Language: Python 3.14
Development Environment: VS Code Jupyter Notebook
Data Processing: Pandas, NumPy
Data Ingestion & APIs: SQLite3, Requests
Visualization: Matplotlib, Seaborn
Automated Data Profiling: Sweetviz

🚀 Key Modules & Project Execution
1. Fundamentals & ML Framing
Formulated a supervised classification problem to predict customer churn based on behavioral metrics.

Demonstrated multi-dimensional tensor operations (1D, 2D, 3D arrays) using NumPy.

2. Multi-Source Data Acquisition
Integrated data across four heterogeneous formats:

CSV: Raw purchase logs.

JSON: Customer demographic information.

SQLite Database: SQL queries executed via sqlite3 to fetch customer tiers.

REST API: Mock API requests using requests to supplement data.

3. Data Cleaning & Preprocessing
Handled missing values, duplicates, and invalid data entries.

Performed type casting, feature scaling, and feature engineering for analysis readiness.

4. Exploratory Data Analysis (EDA)
Univariate Analysis: Analyzed distributions of Age, Income, and Total Purchases.

Bivariate Analysis: Investigated relationships such as Gender vs. Purchases and Income vs. Churn.

Multivariate Analysis: Evaluated feature correlations using Seaborn heatmaps and pair plots.

5. Automated Data Profiling
Generated customer_profiling_report.html using Sweetviz for complete statistics, missing value rates, and associations.

Technical Note on Tooling: Sweetviz was chosen as the profiling library to ensure native compatibility with Python 3.14 while delivering equivalent interactive HTML reporting capabilities.

⚙️ How to Run
Clone or download this repository.

Install required libraries:

Bash
pip install pandas numpy matplotlib seaborn sweetviz
Open customer_analysis.ipynb in VS Code / Jupyter Notebook and execute all cells sequentially.

Open customer_profiling_report.html in any browser to inspect the automated profiling report.
---

## 📁 Repository & Project Structure
```text
data-preprocessing/
│
├── customer_purchases.csv         # Main dataset containing purchase history
├── customer_data.json             # Supplementary customer demographic data
├── company_database.db            # SQLite database with membership records
├── customer_analysis.ipynb        # Main Jupyter Notebook with complete analysis code
├── customer_profiling_report.html # Interactive Sweetviz Data Profiling Report
└── README.md                      # Project documentation file
