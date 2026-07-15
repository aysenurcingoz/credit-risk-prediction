import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon="🏦",
    layout="wide"
)


# =========================================================
# DESIGN
# =========================================================

st.markdown("""
<style>

/* =========================
   GENEL SAYFA
========================= */

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #F4F7FB !important;
    color: #0F172A !important;
}

[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background-color: #F4F7FB !important;
}

/* Genel metinler */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6 {
    color: #0F172A;
}

/* Markdown metinleri */
[data-testid="stMarkdownContainer"] {
    color: #0F172A !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #0F172A !important;
}


/* =========================
   SIDEBAR
========================= */

section[data-testid="stSidebar"] {
    background-color: #E8F1FA !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span:not([data-baseweb="tag"]),
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #0F172A !important;
}


/* =========================
   INPUT ALANLARI
========================= */

input,
textarea {
    color: #0F172A !important;
    background-color: #FFFFFF !important;
    -webkit-text-fill-color: #0F172A !important;
}

/* Number input */
[data-testid="stNumberInput"] input {
    color: #0F172A !important;
    background-color: #FFFFFF !important;
    -webkit-text-fill-color: #0F172A !important;
}

/* Text input */
[data-testid="stTextInput"] input {
    color: #0F172A !important;
    background-color: #FFFFFF !important;
    -webkit-text-fill-color: #0F172A !important;
}

/* Selectbox ve multiselect */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

/* Açılan selectbox menüsü */
ul[role="listbox"],
li[role="option"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}

/* Slider etiketleri */
[data-testid="stSlider"] {
    color: #0F172A !important;
}


/* =========================
   METRIC VE BİLGİ KUTULARI
========================= */

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 14px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #0F172A !important;
}

/* Alert kutuları */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: inherit !important;
}


/* =========================
   HERO ALANI
========================= */

.hero {
    background: linear-gradient(
        135deg,
        #0F172A,
        #1E3A8A,
        #2563EB
    );
    padding: 32px;
    border-radius: 22px;
    margin-bottom: 25px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.22);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 38px;
    font-weight: 800;
    line-height: 1.2;
    margin: 0 0 10px 0;
}

.hero p {
    color: #DBEAFE !important;
    font-size: 17px;
    line-height: 1.6;
    margin: 0;
}


/* =========================
   SONUÇ KUTULARI
========================= */

.result-low,
.result-medium,
.result-high {
    padding: 24px;
    border-radius: 18px;
    text-align: center;
    font-size: 24px;
    font-weight: 800;
}

.result-low,
.result-low * {
    background-color: #DCFCE7;
    color: #166534 !important;
}

.result-low {
    border: 1px solid #86EFAC;
}

.result-medium,
.result-medium * {
    background-color: #FEF3C7;
    color: #92400E !important;
}

.result-medium {
    border: 1px solid #FACC15;
}

.result-high,
.result-high * {
    background-color: #FEE2E2;
    color: #991B1B !important;
}

.result-high {
    border: 1px solid #FCA5A5;
}


/* =========================
   BUTONLAR
========================= */

.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border-radius: 12px;
    border: none;
    font-weight: 700;
    padding: 0.7rem 1rem;
}

.stButton > button *,
[data-testid="stFormSubmitButton"] > button * {
    color: #FFFFFF !important;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #1D4ED8 !important;
    color: #FFFFFF !important;
    border: none;
}

.stButton > button:focus,
[data-testid="stFormSubmitButton"] > button:focus {
    color: #FFFFFF !important;
    border: none;
    box-shadow: 0 0 0 0.2rem rgba(37, 99, 235, 0.25);
}


/* =========================
   TABLOLAR
========================= */

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}


/* =========================
   MOBİL UYUM
========================= */

@media (max-width: 768px) {
    .hero {
        padding: 22px;
        border-radius: 16px;
    }

    .hero h1 {
        font-size: 28px;
    }

    .hero p {
        font-size: 15px;
    }

    .result-low,
    .result-medium,
    .result-high {
        font-size: 20px;
        padding: 18px;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_artifacts():
    model_paths = [
        "Models/final_catboost_credit_risk_model.pkl",
        "models/final_catboost_credit_risk_model.pkl",
        "final_catboost_credit_risk_model.pkl"
    ]

    feature_paths = [
        "Models/model_features.pkl",
        "models/model_features.pkl",
        "model_features.pkl"
    ]

    model_path = next((p for p in model_paths if os.path.exists(p)), None)
    feature_path = next((p for p in feature_paths if os.path.exists(p)), None)

    if model_path is None:
        st.error("Model bulunamadı. final_catboost_credit_risk_model.pkl dosyasını kontrol et.")
        st.stop()

    if feature_path is None:
        st.error("Feature dosyası bulunamadı. model_features.pkl dosyasını kontrol et.")
        st.stop()

    model = joblib.load(model_path)
    model_features = joblib.load(feature_path)

    return model, list(model_features)


model, model_features = load_artifacts()


# =========================================================
# FUNCTIONS
# =========================================================

raw_features = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length"
]


def prepare_input(input_df):
    input_df = input_df.copy()

    input_df["loan_percent_income"] = (
        input_df["loan_amnt"] / input_df["person_income"]
    )

    input_df["loan_percent_income"] = input_df["loan_percent_income"].replace(
        [np.inf, -np.inf], 0
    ).fillna(0)

    input_df["loan_percent_income"] = input_df["loan_percent_income"].round(4)

    # Eğer model raw kolonlarla eğitildiyse
    if set(model_features) == set(raw_features):
        return input_df[model_features]

    # Eğer model one-hot encoded kolonlarla eğitildiyse
    encoded = pd.get_dummies(input_df, drop_first=False)

    encoded = encoded.reindex(columns=model_features, fill_value=0)

    return encoded


def risk_classification(probability):
    if probability < 0.30:
        return "Low Risk", "APPROVE", "result-low", "🟢"
    elif probability < 0.60:
        return "Medium Risk", "MANUAL REVIEW", "result-medium", "🟡"
    else:
        return "High Risk", "STRICT REVIEW", "result-high", "🔴"


def create_gauge(probability):
    fig, ax = plt.subplots(figsize=(4.8, 2.8))

    value = probability * 100

    sizes = [30, 30, 40]
    colors = ["#22C55E", "#FACC15", "#EF4444"]

    ax.pie(
        sizes,
        startangle=180,
        counterclock=False,
        colors=colors,
        wedgeprops={"width": 0.28, "edgecolor": "white"}
    )

    angle = 180 - (value / 100) * 180
    x = 0.68 * np.cos(np.deg2rad(angle))
    y = 0.68 * np.sin(np.deg2rad(angle))

    ax.plot([0, x], [0, y], color="#0F172A", linewidth=3)
    ax.scatter([0], [0], color="#0F172A", s=45)

    ax.text(0, -0.08, f"{value:.2f}%", ha="center", fontsize=18, fontweight="bold")
    ax.text(0, -0.33, "Default Probability", ha="center", fontsize=10)

    ax.set_aspect("equal")
    ax.axis("off")

    return fig


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🏦 Credit Risk System")
st.sidebar.markdown("Final Model: **CatBoost**")
st.sidebar.markdown("Accuracy: **93.65%**")
st.sidebar.markdown("Precision: **98.37%**")
st.sidebar.markdown("F1 Score: **83.20%**")
st.sidebar.divider()
st.sidebar.markdown("Created by **Ayşenur Cingöz**")


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🏦 Credit Risk Prediction System</h1>
    <p>
    Professional machine learning dashboard for loan default prediction,
    risk segmentation, model comparison, and credit risk analytics.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🏦 Prediction",
    "📊 Model Performance",
    "📈 Model Comparison",
    "📋 Project Summary"
])


# =========================================================
# TAB 1
# =========================================================

with tab1:
    st.subheader("Applicant Credit Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Applicant Information")

        person_age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        person_income = st.number_input("Annual Income", min_value=1000, max_value=1000000, value=50000, step=1000)
        person_emp_length = st.number_input("Employment Length", min_value=0.0, max_value=50.0, value=5.0, step=0.5)

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
            value=5,
            step=1
        )

    with col2:
        st.markdown("### 💳 Loan Information")

        loan_intent = st.selectbox(
            "Loan Intent",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )

        loan_grade = st.selectbox(
            "Loan Grade",
            ["A", "B", "C", "D", "E", "F", "G"]
        )

        loan_amnt = st.number_input("Loan Amount", min_value=100, max_value=1000000, value=10000, step=500)
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=60.0, value=12.0, step=0.1)

        loan_percent_income = loan_amnt / person_income
        st.metric("Loan Percent Income", f"{loan_percent_income:.2%}")

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
        probability = float(model.predict_proba(input_encoded)[0][1])
        prediction = int(model.predict(input_encoded)[0])

        risk_category, decision, css_class, icon = risk_classification(probability)

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Default Probability", f"{probability:.2%}")
        m2.metric("Predicted Class", prediction)
        m3.metric("Risk Category", risk_category)
        m4.metric("Decision", decision)

        st.markdown(
            f"""
            <div class="{css_class}">
                {icon} {risk_category}<br>
                <span style="font-size:15px;">Business Decision: {decision}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        g1, g2 = st.columns([1, 1.4])

        with g1:
            st.pyplot(create_gauge(probability), use_container_width=False)

        with g2:
            st.markdown("### Applicant Input")
            st.dataframe(input_data, use_container_width=True)

        with st.expander("Model input kontrol tablosu"):
            st.write("Buradaki değerler değişiyorsa model gerçekten farklı input alıyor.")
            st.dataframe(input_encoded, use_container_width=True)

        report = f"""
CREDIT RISK PREDICTION REPORT
Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Predicted Class: {prediction}
Default Probability: {probability:.2%}
Risk Category: {risk_category}
Decision: {decision}

Applicant Information:
{input_data.to_string(index=False)}
"""

        st.download_button(
            "📄 Download Risk Report",
            data=report,
            file_name="credit_risk_prediction_report.txt",
            mime="text/plain",
            use_container_width=True
        )

    else:
        st.info("Bilgileri gir ve tahmin almak için butona bas.")


# =========================================================
# TAB 2
# =========================================================

with tab2:
    st.subheader("Final CatBoost Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", "93.65%")
    c2.metric("Precision", "98.37%")
    c3.metric("F1 Score", "83.20%")
    c4.metric("ROC-AUC", "93.81%")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        "Score": [0.9365, 0.9837, 0.7208, 0.8320, 0.9381]
    })

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    sns.barplot(data=metrics, x="Metric", y="Score", ax=ax)
    ax.set_ylim(0.6, 1.0)
    ax.set_title("CatBoost Performance Metrics", fontsize=11)
    ax.grid(alpha=0.25)

    st.pyplot(fig, use_container_width=False)
    st.dataframe(metrics, use_container_width=True)


# =========================================================
# TAB 3
# =========================================================

with tab3:
    st.subheader("Machine Learning Model Comparison")

    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
        "Accuracy": [0.8679, 0.8894, 0.9320, 0.9356, 0.9349, 0.9365],
        "Precision": [0.7689, 0.7367, 0.9588, 0.9726, 0.9708, 0.9837],
        "Recall": [0.5640, 0.7672, 0.7194, 0.7250, 0.7236, 0.7208],
        "F1 Score": [0.6507, 0.7516, 0.8220, 0.8308, 0.8292, 0.8320],
        "ROC-AUC": [0.8693, 0.8453, 0.9323, 0.9497, 0.9500, 0.9381]
    })

    comparison["Average Score"] = comparison[
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ].mean(axis=1)

    st.dataframe(comparison, use_container_width=True)

    comparison_long = comparison.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score"
    )

    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(data=comparison_long, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Overall Model Comparison", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    st.pyplot(fig, use_container_width=False)


# =========================================================
# TAB 4
# =========================================================

with tab4:
    st.subheader("Project Summary")

    summary = pd.DataFrame({
        "Project Component": [
            "Dataset",
            "Problem Type",
            "Final Model",
            "Deployment",
            "Business Use Case"
        ],
        "Description": [
            "Credit Risk Dataset",
            "Binary Classification",
            "CatBoost Classifier",
            "Streamlit Dashboard",
            "Loan Default Risk Assessment"
        ]
    })

    st.dataframe(summary, use_container_width=True)

    st.success("This project presents an end-to-end machine learning dashboard for credit risk prediction.")