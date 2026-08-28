# hospital_readmission
hospital-readmission-ai

 # 🏥 Hospital Readmission AI - Ultra Enterprise Healthcare Edition

An enterprise-grade, interactive Clinical Decision Support System (CDSS) built with Python and Streamlit. This application leverages machine learning to predict 30-day hospital readmission risks, analyze clinical datasets, and support discharge planning for healthcare and operational teams.

---

## 🌟 Key Features

* **Secure Authentication:** Built-in login and registration portal with secure password hashing.
* **Synthetic Healthcare Data Generator:** Instantly generate or upload custom patient datasets simulating emergency, urgent, and elective admissions.
* **Automated Data Cleaning & Feature Engineering:** Automatically handles missing values, removes duplicates, and generates critical clinical risk indicators (e.g., medication burden, high LOS risk, clinical complexity scores)[cite: 1].
* **Exploratory Data Analysis (EDA):** Interactive Plotly visualizations exploring readmission trends by diagnosis category, age group, and clinical factors[cite: 1].
* **Multi-Model Machine Learning Training:** Train and benchmark 5 different classification algorithms:
  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Gradient Boosting
  * Support Vector Machine (SVM)[cite: 1]
* **Comprehensive Evaluation Suite:** Review performance metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC) and confusion matrices[cite: 1].
* **Patient Risk Prediction & Real-Time Simulation:** Input individual patient profiles or run batch simulations to calculate immediate 30-day readmission probabilities and risk categories (Low, Medium, High)[cite: 1].
* **Enterprise Reporting:** Export original datasets, cleaned datasets, and model performance summaries directly to CSV[cite: 1].

---

## 📁 Project Structure

```text
hospital-readmission-ai/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies and package versions
├── users.json          # User authentication store
└── README.md           # Project documentation
