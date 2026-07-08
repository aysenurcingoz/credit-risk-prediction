import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from catboost import Pool


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon="🏦",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f7fb;
    }
    .stApp, .stMarkdown, p, span, div, label {
        color: #0f172a;
    }

        .hero, .hero h1, .hero p {
        color: white !important;
    }
        section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
  }
        section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a, #1e3a8a, #2563eb);
        padding: 32px;
        border-radius: 22px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.18);
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 18px;
        color: #dbeafe;
    }

    .risk-low {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        color: #166534;
        padding: 22px;
        border-radius: 18px;
        font-size: 24px;
        font-weight: 800;
        text-align: center;
        border: 1px solid #86efac;
    }

    .risk-medium {
        background: linear-gradient(135deg, #fef9c3, #fde68a);
        color: #854d0e;
        padding: 22px;
        border-radius: 18px;
        font-size: 24px;
        font-weight: 800;
        text-align: center;
        border: 1px solid #facc15;
    }

    .risk-high {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #991b1b;
        padding: 22px;
        border-radius: 18px;
        font-size: 24px;
        font-weight: 800;
        text-align: center;
        border: 1px solid #f87171;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 20px;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_artifacts():
    possible_model_paths = [
        "Models/final_catboost_credit_risk_model.pkl",
        "models/final_catboost_credit_risk_model.pkl"
    ]

    possible_feature_paths = [
        "Models/model_features.pkl",
        "models/model_features.pkl"
    ]

    model_path = next((p for p in possible_model_paths if os.path.exists(p)), None)
    feature_path = next((p for p in possible_feature_paths if os.path.exists(p)), None)

    if model_path is None or feature_path is None:
        st.error("Model files could not be found. Please check the Models folder.")
        st.stop()

    model = joblib.load(model_path)
    features = joblib.load(feature_path)

    return model, features


model, model_features = load_artifacts()


# =========================================================
# STATIC PROJECT RESULTS
# =========================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost"
    ],
    "Accuracy": [0.8679, 0.8894, 0.9320, 0.9356, 0.9349, 0.9365],
    "Precision": [0.7689, 0.7367, 0.9588, 0.9726, 0.9708, 0.9837],
    "Recall": [0.5640, 0.7672, 0.7194, 0.7250, 0.7236, 0.7208],
    "F1 Score": [0.6507, 0.7516, 0.8220, 0.8308, 0.8292, 0.8320],
    "ROC-AUC": [0.8693, 0.8453, 0.9323, 0.9497, 0.9500, 0.9381]
})

comparison["Average Score"] = comparison[
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
].mean(axis=1)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def classify_risk(probability):
    if probability < 0.30:
        return "Low Risk", "🟢", "risk-low", "APPROVE", "Loan can be considered for approval."
    elif probability < 0.60:
        return "Medium Risk", "🟡", "risk-medium", "MANUAL REVIEW", "Manual review is recommended."
    else:
        return "High Risk", "🔴", "risk-high", "STRICT REVIEW", "Loan application requires strict risk review."


def prepare_input(data):
    encoded = pd.get_dummies(data, drop_first=True)
    encoded = encoded.reindex(columns=model_features, fill_value=0)
    return encoded


def create_gauge(probability):
    fig, ax = plt.subplots(figsize=(7, 3.6))

    value = probability * 100

    colors = ["#22c55e", "#facc15", "#ef4444"]
    sizes = [30, 30, 40]

    ax.pie(
        sizes,
        startangle=180,
        counterclock=False,
        colors=colors,
        wedgeprops={"width": 0.32, "edgecolor": "white"}
    )

    angle = 180 - (value / 100) * 180
    x = 0.75 * np.cos(np.deg2rad(angle))
    y = 0.75 * np.sin(np.deg2rad(angle))

    ax.plot([0, x], [0, y], color="#0f172a", linewidth=4)
    ax.scatter([0], [0], color="#0f172a", s=80)

    ax.text(0, -0.15, f"{value:.2f}%", ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(-0.95, -0.18, "0%", ha="center", fontsize=10)
    ax.text(0.95, -0.18, "100%", ha="center", fontsize=10)
    ax.text(0, -0.42, "Default Probability", ha="center", fontsize=12, color="#475569")

    ax.set_aspect("equal")
    ax.axis("off")

    return fig


def create_local_shap_explanation(input_encoded):
    pool = Pool(input_encoded)

    shap_values = model.get_feature_importance(
        data=pool,
        type="ShapValues"
    )

    shap_contributions = shap_values[0, :-1]

    shap_df = pd.DataFrame({
        "Feature": model_features,
        "SHAP Value": shap_contributions,
        "Absolute Impact": np.abs(shap_contributions),
        "Feature Value": input_encoded.iloc[0].values
    })

    shap_df = shap_df.sort_values(
        by="Absolute Impact",
        ascending=False
    )

    return shap_df


def generate_report(input_data, probability, prediction, risk_category, decision, recommendation):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
CREDIT RISK PREDICTION REPORT
Generated at: {now}

FINAL MODEL
Model: CatBoost Classifier

PREDICTION RESULT
Predicted Class: {int(prediction)}
Default Probability: {probability:.2%}
Risk Category: {risk_category}
Business Decision: {decision}
Recommendation: {recommendation}

APPLICANT INFORMATION
{input_data.to_string(index=False)}

MODEL PERFORMANCE
Accuracy: 93.65%
Precision: 98.37%
F1 Score: 83.20%
ROC-AUC: 93.81%

NOTE
This report is generated for decision-support purposes.
Final lending decisions should also consider institutional policies,
credit officer review, and regulatory requirements.
"""
    return report


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🏦 Credit Risk System")
st.sidebar.markdown("**Machine Learning Decision Support Tool**")
st.sidebar.divider()

st.sidebar.markdown(
    """
    **Final Model:** CatBoost  
    **Accuracy:** 93.65%  
    **Precision:** 98.37%  
    **F1 Score:** 83.20%  
    """
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Created by:**  
    Ayşenur Cingöz  

    **Focus Areas:**  
    Data Science · Machine Learning · Credit Risk Analytics
    """
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🏦 Credit Risk Prediction System</h1>
        <p>
        Professional machine learning dashboard for loan default prediction,
        model comparison, risk segmentation, explainability, and deployment-ready credit analytics.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏦 Prediction",
    "📊 Model Performance",
    "📈 Model Comparison",
    "🔍 Feature Importance",
    "📋 Project Summary"
])


# =========================================================
# TAB 1 — PREDICTION
# =========================================================

with tab1:
    st.subheader("🏦 Applicant Credit Risk Prediction")

    st.markdown(
        """
        Enter applicant and loan information below.  
        The system estimates default probability, assigns a risk category,
        provides a business recommendation, and explains the prediction using SHAP values.
        """
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 👤 Applicant Information")

        person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Annual Income", min_value=0, value=50000)
        person_emp_length = st.number_input("Employment Length", min_value=0.0, max_value=50.0, value=5.0)

        person_home_ownership = st.selectbox(
            "Home Ownership",
            ["RENT", "OWN", "MORTGAGE", "OTHER"]
        )

        cb_person_default_on_file = st.selectbox(
            "Previous Default on File",
            ["N", "Y"]
        )

        cb_person_cred_hist_length = st.number_input(
            "Credit History Length",
            min_value=0,
            max_value=50,
            value=5
        )

    with col_right:
        st.markdown("### 💳 Loan Information")

        loan_intent = st.selectbox(
            "Loan Intent",
            [
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION"
            ]
        )

        loan_grade = st.selectbox(
            "Loan Grade",
            ["A", "B", "C", "D", "E", "F", "G"]
        )

        loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=12.0)
        loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.20)

    input_data = pd.DataFrame({
        "person_age": [person_age],
        "person_income": [person_income],
        "person_home_ownership": [person_home_ownership],
        "person_emp_length": [person_emp_length],
        "loan_intent": [loan_intent],
        "loan_grade": [loan_grade],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_default_on_file": [cb_person_default_on_file],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length]
    })

    input_encoded = prepare_input(input_data)

    st.divider()

    if st.button("🔍 Predict Credit Risk", use_container_width=True):
        probability = model.predict_proba(input_encoded)[0][1]
        prediction = model.predict(input_encoded)[0]

        risk_category, risk_icon, risk_class, decision, recommendation = classify_risk(probability)

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Default Probability", f"{probability:.2%}")
        k2.metric("Predicted Class", int(prediction))
        k3.metric("Risk Category", risk_category)
        k4.metric("Business Decision", decision)

        st.markdown(
            f"""
            <div class="{risk_class}">
                {risk_icon} {risk_category}<br>
                <span style="font-size:16px;">{recommendation}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.pyplot(create_gauge(probability))

        st.subheader("📋 Applicant Input Summary")
        st.dataframe(input_data, use_container_width=True)

        st.subheader("🔎 SHAP Explainability")

        st.write(
            """
            SHAP values explain how each feature influenced this individual prediction.  
            Positive SHAP values increase the predicted default risk, while negative SHAP values decrease it.
            """
        )

        try:
            shap_df = create_local_shap_explanation(input_encoded)

            st.dataframe(
                shap_df.head(10),
                use_container_width=True
            )

            top_shap = shap_df.head(10).sort_values("SHAP Value")

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                data=top_shap,
                x="SHAP Value",
                y="Feature",
                ax=ax
            )

            ax.axvline(0, color="black", linewidth=1)
            ax.set_title("Top 10 Local SHAP Contributions")
            ax.grid(alpha=0.25)

            st.pyplot(fig)

        except Exception as e:
            st.warning("SHAP explanation could not be generated for this prediction.")
            st.write(e)

        report = generate_report(
            input_data,
            probability,
            prediction,
            risk_category,
            decision,
            recommendation
        )

        st.download_button(
            label="📄 Download Risk Report",
            data=report,
            file_name="credit_risk_prediction_report.txt",
            mime="text/plain",
            use_container_width=True
        )

    else:
        st.info("Enter applicant information and click **Predict Credit Risk**.")


# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================

with tab2:
    st.subheader("📊 Final CatBoost Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", "93.65%")
    c2.metric("Precision", "98.37%")
    c3.metric("F1 Score", "83.20%")
    c4.metric("ROC-AUC", "93.81%")

    st.markdown(
        """
        CatBoost was selected as the final model because it achieved the strongest
        overall balance between predictive performance, classification reliability,
        and business applicability.
        """
    )

    catboost_metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        "Score": [0.9365, 0.9837, 0.7208, 0.8320, 0.9381]
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=catboost_metrics, x="Metric", y="Score", ax=ax)
    ax.set_ylim(0.6, 1.0)
    ax.set_title("CatBoost Performance Metrics")
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    st.dataframe(catboost_metrics, use_container_width=True)


# =========================================================
# TAB 3 — MODEL COMPARISON
# =========================================================

with tab3:
    st.subheader("📈 Machine Learning Model Comparison")

    st.markdown(
        """
        Six machine learning models were trained and evaluated using the same dataset
        and the same evaluation framework.
        """
    )

    st.dataframe(
        comparison.style.background_gradient(
            cmap="YlGn",
            subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Average Score"]
        ).format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "Average Score": "{:.4f}"
        }),
        use_container_width=True
    )

    st.markdown("### Overall Metric Comparison")

    comparison_long = comparison.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score"
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=comparison_long, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Overall Performance Comparison")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    st.pyplot(fig)

    st.markdown("### Average Score Ranking")

    ranking = comparison.sort_values("Average Score", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=ranking, x="Average Score", y="Model", ax=ax)
    ax.set_title("Average Performance Score by Model")
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    st.markdown("### Performance Heatmap")

    heatmap_data = comparison.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Performance Heatmap")
    st.pyplot(fig)


# =========================================================
# TAB 4 — FEATURE IMPORTANCE
# =========================================================

with tab4:
    st.subheader("🔍 CatBoost Feature Importance")

    try:
        feature_importance = pd.DataFrame({
            "Feature": model_features,
            "Importance": model.get_feature_importance()
        })

        feature_importance = feature_importance.sort_values(
            by="Importance",
            ascending=False
        )

        st.dataframe(feature_importance.head(15), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(
            data=feature_importance.head(15),
            x="Importance",
            y="Feature",
            ax=ax
        )
        ax.set_title("Top 15 CatBoost Feature Importance")
        ax.grid(alpha=0.25)
        st.pyplot(fig)

        st.markdown(
            """
            Feature importance indicates which variables contributed most strongly
            to the final CatBoost model. In this project, financial capacity,
            loan burden, loan grade, and housing-related variables played major roles
            in default risk prediction.
            """
        )

    except Exception as e:
        st.warning("Feature importance could not be displayed.")
        st.write(e)


# =========================================================
# TAB 5 — PROJECT SUMMARY
# =========================================================

with tab5:
    st.subheader("📋 Project Summary")

    project_summary = pd.DataFrame({
        "Project Component": [
            "Dataset",
            "Problem Type",
            "Final Model",
            "Models Compared",
            "Deployment Status",
            "Business Use Case"
        ],
        "Description": [
            "Credit Risk Dataset",
            "Binary Classification",
            "CatBoost Classifier",
            "6 Machine Learning Models",
            "Deployment Ready",
            "Loan Default Risk Assessment"
        ]
    })

    st.dataframe(project_summary, use_container_width=True)

    st.markdown(
        """
        This project presents a complete end-to-end machine learning workflow for
        credit risk prediction, including data preprocessing, feature engineering,
        model development, model comparison, business interpretation, and deployment preparation.
        """
    )

    st.success(
        """
        Final Recommendation: CatBoost is selected as the final model due to its strong
        balance between predictive performance, precision, F1 Score, and business applicability.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Developed as an end-to-end machine learning credit risk project.
    </div>
    """,
    unsafe_allow_html=True
)