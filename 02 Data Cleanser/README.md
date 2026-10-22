# 🩺 Data Cleanser: Advanced Healthcare Patient Records Preprocessing

> A complete Data Science preprocessing project built to clean noisy healthcare data, handle missing values using smart imputation algorithms, and treat outliers professionally.

---

## 🎯 What is this Project About?
Imagine you are working as a Data Analyst, and you receive a raw dataset of patient health records (like BMI, blood pressure, cholesterol, and glucose). Because of data entry errors or missing entries, the data is messy. 

This project (**Data Cleanser**) takes that messy raw data and systematically cleans it so it can be safely used to train Machine Learning models (such as predicting `disease_risk`).

---

## 📂 Project Structure & What We Did Inside

The project is divided into three major parts:

### 1️⃣ Part A: Handling Missing Values (Data Gaps)
* **What we found:** Some patients had missing values in columns like `age`, `bmi`, `gender`, `region`, `cholesterol`, and `glucose`.
* **How we fixed it:**
  * Used **Simple Imputer** (Median for numerical columns like BMI, Mode/Most Frequent for categorical columns like Gender & Region).
  * Used advanced techniques like **KNN Imputer** (finding similar patients) and **MICE Algorithm** (iterative regression) to fill numerical gaps intelligently without destroying data patterns.

### 2️⃣ Part B: Handling Outliers (Extreme / Wrong Values)
* **What we found:** Due to synthetic errors, some patients had unrealistic values (e.g., extreme high blood pressure or abnormal BMI spikes).
* **How we fixed it:** 
  * Tested statistical detection methods like **Z-Score** and **IQR (Interquartile Range)**.
  * Applied **Winsorization (Percentile Capping at 1st and 99th percentiles)**. Instead of deleting entire patient rows (which causes data loss), we gently capped the extreme spikes to keep our sample size safe and strong.

### 3️⃣ Part C: Final Clean Dataset & Delivery
* Merged all the cleaned columns together.
* Exported the final polished data into a brand new CSV file: `cleaned_patient_health_records.csv`, which is fully ready for Machine Learning.

---

## 💡 Key Results & Learnings (Summary Report)
* **Best Imputation:** KNN and MICE worked best for medical numbers because they look at relationships between different vitals rather than just writing a flat average.
* **Best Outlier Handling:** Winsorization was the winner because it saved us from losing valuable patient data while removing harmful extremes.
* **Impact:** The final dataset is 100% bias-free, clean, and optimized for predictive healthcare modeling.

---

## 🛠️ Tools & Libraries Used
* **Python** (Programming Language)
* **Pandas & NumPy** (For data manipulation and creating datasets)
* **Scikit-Learn** (For SimpleImputer, KNNImputer, and IterativeImputer)
* **SciPy** (For Z-scores and Winsorization)

---

## 🚀 How to Run This Project on Your Computer

1. **Clone or Download** this repository.
2. Make sure you have Python installed, then install the required libraries by running this command in your terminal:
   ```bash
   pip install pandas numpy scikit-learn scipy
