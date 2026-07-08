## 🌐 Live Demo

🔗 [Open Streamlit App](https://credit-risk-prediction-aysenur.streamlit.app/)

# 🏦 Credit Risk Prediction Using Machine Learning

## 📖 Project Overview

This project presents a complete end-to-end machine learning pipeline for predicting loan default risk using a real-world credit risk dataset.

The objective is to assist financial institutions in identifying high-risk loan applicants by developing accurate, interpretable, and deployment-ready predictive models.

Unlike a traditional machine learning notebook, this project covers the entire data science lifecycle—from raw data exploration to production-ready model deployment.

---

# 🎯 Business Problem

Loan default prediction is one of the most important challenges faced by financial institutions.

Incorrect lending decisions may lead to:

- Financial losses
- Increased credit risk
- Poor portfolio quality
- Reduced operational efficiency

The goal of this project is to develop a machine learning solution capable of supporting intelligent credit approval decisions.

---

# 📊 Dataset

The project uses a publicly available banking credit risk dataset containing customer demographic information, financial characteristics, and loan-related variables.

Target Variable:

**loan_status**

- 0 → Non-default
- 1 → Default

---

# 🔬 Project Workflow

The project follows a complete machine learning pipeline:

- 📥 Data Understanding
- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Cleaning
- ⚙️ Feature Engineering
- 🔄 Feature Encoding
- 📈 Feature Scaling
- 🤖 Machine Learning Model Development
- 📊 Model Evaluation
- 🏆 Model Comparison
- 💼 Business Evaluation
- 🚀 Deployment Preparation

---

# 🤖 Machine Learning Models

The following classification algorithms were implemented and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

---

# 📈 Evaluation Metrics

Each model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Additional evaluation techniques include:

- Confusion Matrix
- ROC Curve
- Precision–Recall Curve
- Learning Curve
- Cross Validation
- Feature Importance
- Threshold Analysis

---

# 🏆 Final Model

After a comprehensive comparison of all developed models,

**CatBoost** was selected as the final production-ready model.

Reasons:

- Highest Accuracy
- Highest Precision
- Highest F1 Score
- Excellent ROC-AUC
- Strong business applicability
- Deployment readiness

---

# 🚀 Deployment

The final CatBoost model was prepared for deployment by:

- Saving the trained model (.pkl)
- Saving feature configuration
- Generating probability predictions
- Assigning business-oriented risk categories
- Creating deployment-ready outputs

Future deployment options include:

- Streamlit Dashboard
- FastAPI
- Flask REST API

---

# 📂 Project Structure

```

Credit-Risk-Prediction/

│

├── data/

│ └── credit_risk_dataset.csv

│

├── notebook/

│ └── Credit_Risk_Prediction_End_to_End.ipynb

│

├── models/

│ ├── final_catboost_credit_risk_model.pkl

│ └── model_features.pkl

│

├── images/

├── requirements.txt

├── README.md

└── app.py

```

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- Joblib
- Streamlit

---

# 📌 Future Improvements

Potential future developments include:

- Explainable AI (SHAP)
- Hyperparameter Optimization
- Cloud Deployment
- Real-Time Prediction API
- Automatic Model Retraining
- Customer Behavioral Analytics
- Macroeconomic Indicators

---

# 👩‍💻 Author

**Ayşenur Cingöz**

Statistics Student

Interested in:

- Data Science
- Machine Learning
- Credit Risk Analytics
- Banking Analytics
- Artificial Intelligence

---

⭐ If you found this project interesting, feel free to explore the notebook and share your feedback!
