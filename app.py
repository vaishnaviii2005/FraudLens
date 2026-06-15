"""
🛡️ Extensive Insurance Claims Fraud Detection Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------------------
# Page Configuration & Styling
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Fraud Detection Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

css = """
<style>
div[data-testid="stMetricValue"] {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ff4b4b;
}
div[data-testid="stMetricLabel"] {
    font-size: 1rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-container {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #334155;
    margin-bottom: 20px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
}
h1 {
    background: linear-gradient(135deg, #ff4b4b 0%, #f43f5e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    margin-bottom: 5px !important;
}
.sub-header {
    color: #9ca3af;
    font-size: 1.1rem;
    margin-bottom: 25px;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 24px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #e11d48;
    transform: translateY(-2px);
    box-shadow: 0 10px 20px -10px rgba(225, 29, 72, 0.5);
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ----------------------------------------------------------------
# Load model artifacts
# ----------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(CURRENT_DIR, "models")):
    BASE_DIR = CURRENT_DIR
else:
    BASE_DIR = os.path.dirname(CURRENT_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")

required_files = [
    "fraud_model.pkl",
    "scaler.pkl",
    "model_columns.pkl",
    "needs_scaling.pkl",
    "model_name.pkl",
]
missing_files = []

if not os.path.exists(MODELS_DIR):
    missing_files.append(f"models/ directory (searched at: {MODELS_DIR})")
else:
    for rf in required_files:
        if not os.path.exists(os.path.join(MODELS_DIR, rf)):
            missing_files.append(rf)

if missing_files:
    st.error("### ❌ Missing Model Artifacts")
    st.markdown(
        "The dashboard cannot load because required ML model files are missing.\n\n"
        f"**Searched Directory:** `{MODELS_DIR}`\n\n"
        f"**Missing:** {', '.join(f'`{m}`' for m in missing_files)}\n\n"
        "**Fix:**\n"
        "1. Make sure `models/` exists locally and contains all `.pkl` files.\n"
        "2. Check `.gitignore` — it may be excluding `*.pkl`.\n"
        "3. Force-add and push:\n"
        "```bash\n"
        "git add -f models/*.pkl\n"
        "git commit -m 'force add model pickles'\n"
        "git push\n"
        "```"
    )
    st.stop()


@st.cache_resource
def load_model_artifacts():
    model         = joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl"))
    scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    model_columns = joblib.load(os.path.join(MODELS_DIR, "model_columns.pkl"))
    needs_scaling = joblib.load(os.path.join(MODELS_DIR, "needs_scaling.pkl"))
    model_name    = joblib.load(os.path.join(MODELS_DIR, "model_name.pkl"))

    try:
        model_metrics = joblib.load(os.path.join(MODELS_DIR, "model_metrics.pkl"))
    except Exception:
        model_metrics = None

    try:
        feature_importances = joblib.load(os.path.join(MODELS_DIR, "feature_importances.pkl"))
    except Exception:
        feature_importances = None

    return model, scaler, model_columns, needs_scaling, model_name, model_metrics, feature_importances


model, scaler, model_columns, needs_scaling, model_name, model_metrics, feature_importances = load_model_artifacts()

# ----------------------------------------------------------------
# Shared Preprocessing & Prediction Logic
# ----------------------------------------------------------------
def predict_dataframe(df_input):
    df_cleaned = df_input.copy()

    for col in ["hospital_charges", "annual_premium"]:
        if col in df_cleaned.columns:
            default = 2000.0 if col == "hospital_charges" else 1200.0
            df_cleaned[col] = df_cleaned[col].fillna(default)

    if "previous_claims" in df_cleaned.columns:
        df_cleaned["previous_claims"] = df_cleaned["previous_claims"].fillna(0)

    df_cleaned["claim_to_premium_ratio"]  = df_cleaned["claim_amount"] / (df_cleaned["annual_premium"] + 1)
    df_cleaned["charges_to_claim_ratio"]  = df_cleaned["hospital_charges"] / (df_cleaned["claim_amount"] + 1)
    df_cleaned["high_previous_claims"]    = (df_cleaned["previous_claims"] >= 3).astype(int)

    encoded_df = pd.DataFrame(index=df_cleaned.index)

    num_cols = [
        "claim_amount", "age", "hospital_charges", "annual_premium",
        "previous_claims", "claim_to_premium_ratio",
        "charges_to_claim_ratio", "high_previous_claims",
    ]
    for col in num_cols:
        if col in df_cleaned.columns:
            encoded_df[col] = df_cleaned[col]

    for col in model_columns:
        if col.startswith("policy_type_"):
            val = col.replace("policy_type_", "")
            encoded_df[col] = (df_cleaned["policy_type"] == val).astype(int) if "policy_type" in df_cleaned.columns else 0
        elif col.startswith("claim_history_"):
            val = col.replace("claim_history_", "")
            encoded_df[col] = (df_cleaned["claim_history"] == val).astype(int) if "claim_history" in df_cleaned.columns else 0

    for col in model_columns:
        if col not in encoded_df.columns:
            encoded_df[col] = 0

    encoded_df = encoded_df[model_columns]

    if needs_scaling:
        probs = model.predict_proba(scaler.transform(encoded_df))[:, 1]
    else:
        probs = model.predict_proba(encoded_df)[:, 1]

    return probs, encoded_df

# ----------------------------------------------------------------
# Header
# ----------------------------------------------------------------
st.title("🛡️ Insurance Claims Fraud Detection Platform")
st.markdown(
    "<p class='sub-header'>A machine learning engine monitoring risk, "
    "analyzing patterns, and predicting claims fraud.</p>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Claim Predictor",
    "📁 Batch Claim Predictor",
    "📊 Model Analytics & EDA",
])

# ================================================================
# Tab 1 — Single Claim Predictor
# ================================================================
with tab1:
    st.subheader("Evaluate a New Claim")

    with st.form("single_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            claim_amount    = st.number_input("Claim Amount ($)",         min_value=0.0,  value=5000.0,  step=100.0)
            age             = st.number_input("Age of Policyholder",      min_value=18,   max_value=100, value=35)
            policy_type     = st.selectbox("Policy Type",                 ["Basic", "Premium", "Comprehensive"])
            annual_premium  = st.number_input("Annual Premium ($)",       min_value=0.0,  value=1200.0,  step=50.0)

        with col2:
            hospital_charges = st.number_input("Hospital Charges ($)",   min_value=0.0,  value=2000.0,  step=100.0)
            claim_history    = st.selectbox("Claim History",             ["No_History", "Minor", "Major"])
            previous_claims  = st.number_input("Number of Previous Claims", min_value=0, value=0,       step=1)

        submit_btn = st.form_submit_button("🔍 Run Fraud Assessment")

    if submit_btn:
        input_dict = {
            "claim_amount":    claim_amount,
            "age":             age,
            "policy_type":     policy_type,
            "annual_premium":  annual_premium,
            "hospital_charges": hospital_charges,
            "claim_history":   claim_history,
            "previous_claims": previous_claims,
        }

        probs, encoded_df = predict_dataframe(pd.DataFrame([input_dict]))
        prob           = probs[0]
        fraud_percent  = round(prob * 100, 2)
        classification = "High Risk" if prob > 0.5 else "Low Risk"
        colour         = "#ef4444" if classification == "High Risk" else "#10b981"

        st.write("### Assessment Results")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(
                f"<div class='metric-container'>"
                f"<p style='margin:0;color:#9ca3af;font-size:0.9rem;text-transform:uppercase;letter-spacing:0.05em;'>FRAUD RISK SCORE</p>"
                f"<p style='margin:0;font-size:2.8rem;font-weight:800;color:{colour};'>{fraud_percent}%</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f"<div class='metric-container'>"
                f"<p style='margin:0;color:#9ca3af;font-size:0.9rem;text-transform:uppercase;letter-spacing:0.05em;'>RISK CLASSIFICATION</p>"
                f"<p style='margin:0;font-size:2.8rem;font-weight:800;color:{colour};'>{classification}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.progress(min(int(fraud_percent), 100))

        st.write("#### Key Risk Indicators Identified:")
        indicators = []

        if claim_amount / (annual_premium + 1) > 5:
            indicators.append("🚨 **Extreme Claim-to-Premium Ratio** — claim is over 5× the annual premium.")
        if hospital_charges / (claim_amount + 1) > 1.2:
            indicators.append("🚨 **Inflated Hospital Charges** — charges significantly exceed the claim amount.")
        if previous_claims >= 3:
            indicators.append("🚨 **High Frequency Claimant** — 3 or more previous claims on record.")
        if claim_history == "Major":
            indicators.append("🚨 **Major Prior History** — history of major previous claims.")

        if not indicators:
            st.success("✅ No anomalous risk indicators found for this claim.")
        else:
            for ind in indicators:
                st.markdown(ind)

        with st.expander("Show engineered features sent to the model"):
            st.dataframe(encoded_df.T.rename(columns={0: "Feature Value"}), use_container_width=True)

# ================================================================
# Tab 2 — Batch Claim Predictor
# ================================================================
with tab2:
    st.subheader("Process Batch Claims")

    sample_df = pd.DataFrame({
        "claim_amount":    [12500.00, 3200.50, 22000.00, 4100.00],
        "age":             [28, 62, 45, 19],
        "policy_type":     ["Basic", "Premium", "Comprehensive", "Basic"],
        "hospital_charges": [9500.00, 3100.00, 24000.00, 3800.00],
        "annual_premium":  [800.00, 1600.00, 2600.00, 850.00],
        "claim_history":   ["No_History", "Minor", "Major", "No_History"],
        "previous_claims": [0, 1, 4, 0],
    })

    st.download_button(
        label="📥 Download CSV Template",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="claims_batch_template.csv",
        mime="text/csv",
    )

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Claims CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            required_cols = ["claim_amount", "age", "policy_type", "hospital_charges",
                             "annual_premium", "claim_history", "previous_claims"]
            missing_cols = [c for c in required_cols if c not in df_batch.columns]

            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                probs, _ = predict_dataframe(df_batch)
                df_batch["Fraud Risk (%)"] = np.round(probs * 100, 2)
                df_batch["Risk Level"]     = np.where(probs > 0.5, "High Risk", "Low Risk")

                total_claims     = len(df_batch)
                high_risk_count  = (df_batch["Risk Level"] == "High Risk").sum()
                high_risk_pct    = round((high_risk_count / total_claims) * 100, 1)
                potential_saving = df_batch[df_batch["Risk Level"] == "High Risk"]["claim_amount"].sum()

                st.write("### Batch Analysis Summary")
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.metric("Total Claims Checked", total_claims)
                with sc2:
                    st.metric("High-Risk Claims Flagged", f"{high_risk_count} ({high_risk_pct}%)")
                with sc3:
                    st.metric("Potential Fraud Value", f"${potential_saving:,.2f}")

                st.write("### Processed Claims Table")

                high_risk_idx = df_batch[df_batch["Risk Level"] == "High Risk"].index

                def highlight_high_risk(row):
                    colour = "background-color: rgba(239, 68, 68, 0.2)" if row.name in high_risk_idx else ""
                    return [colour] * len(row)

                st.dataframe(
                    df_batch.style.apply(highlight_high_risk, axis=1),
                    use_container_width=True,
                )

                st.download_button(
                    label="📥 Export Report with Predictions",
                    data=df_batch.to_csv(index=False).encode("utf-8"),
                    file_name="claims_fraud_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error parsing file: {e}")

# ================================================================
# Tab 3 — Model Analytics & EDA
# ================================================================
with tab3:
    st.subheader("Model Validation & Training Analytics")
    st.write(f"Current Model: **{model_name}**")

    col_metrics, col_plot = st.columns([2, 3])

    with col_metrics:
        if model_metrics is not None:
            st.write("#### Cross-Validation Model Metrics")
            metrics_df = pd.DataFrame(model_metrics).T
            display_cols = ["roc_auc", "accuracy", "f1", "precision", "recall"]
            metrics_df = metrics_df[display_cols].rename(columns={
                "roc_auc":   "ROC AUC",
                "accuracy":  "Accuracy",
                "f1":        "F1 Score",
                "precision": "Precision",
                "recall":    "Recall",
            })
            st.dataframe(
                metrics_df.style.highlight_max(axis=0, color="rgba(255, 75, 75, 0.2)"),
                use_container_width=True,
            )

        if feature_importances is not None:
            st.write("#### Feature Importances")
            selected_model_name = st.selectbox(
                "Select Model to Inspect", list(feature_importances.keys())
            )
            imp_series = pd.Series(feature_importances[selected_model_name]).sort_values(ascending=True)
            st.bar_chart(imp_series)

    with col_plot:
        st.write("#### ROC Validation Curves")
        roc_path = os.path.join(PLOTS_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)

    st.markdown("---")
    st.subheader("Exploratory Data Analysis (EDA) Charts")

    eda_col1, eda_col2, eda_col3 = st.columns(3)

    with eda_col1:
        p = os.path.join(PLOTS_DIR, "claim_amount_distribution.png")
        if os.path.exists(p):
            st.image(p, caption="Distribution of Claim Amounts", use_container_width=True)

    with eda_col2:
        p = os.path.join(PLOTS_DIR, "fraud_vs_nonfraud.png")
        if os.path.exists(p):
            st.image(p, caption="Target Distribution (Fraud vs Normal)", use_container_width=True)

    with eda_col3:
        p = os.path.join(PLOTS_DIR, "age_distribution.png")
        if os.path.exists(p):
            st.image(p, caption="Age Distribution of Claimants", use_container_width=True)
