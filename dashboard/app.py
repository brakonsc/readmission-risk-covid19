# =============================================================
# COVID-19 Readmission Risk Dashboard
# Care Manager Decision Support Tool
#
# Architecture:
#   Tab 1 — Worklist: flagged patients sorted by risk score
#   Tab 2 — Patient Detail: individual risk factor breakdown
#   Tab 3 — Model Performance: metrics for QI/informatics audience
#
# Clinical framing:
#   Threshold = 0.61 (50% precision, 81% recall)
#   "Flag patients whose 30-day readmission risk exceeds 61%"
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Readmission Risk | Care Management",
    page_icon="🏥",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────
# dashboard/app.py is one level below project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# ── Load artifacts ────────────────────────────────────────────
@st.cache_resource  # cache so model loads once, not on every interaction
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "readmission_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(MODELS_DIR, "model_config.json")) as f:
        config = json.load(f)
    return model, scaler, config

@st.cache_data  # cache data separately — can reload without reloading model
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "model_dataset.csv"))
    patient_ids = pd.read_csv(os.path.join(DATA_DIR, "patient_ids.csv"))
    return df, patient_ids

model, scaler, config = load_artifacts()
df, patient_ids = load_data()

THRESHOLD = config["operating_threshold"]  # 0.61
FEATURES = config["features"]              # ordered feature list

# ── Score all patients ────────────────────────────────────────
# In production this would score only today's discharges
# Here we score the full dataset for demonstration
X = df[FEATURES]
X_scaled = scaler.transform(X)
risk_scores = model.predict_proba(X_scaled)[:, 1]
flagged = (risk_scores >= THRESHOLD).astype(int)

# Build display dataframe
display_df = pd.DataFrame({
    "patient_id": patient_ids["PATIENT"].values,
    "risk_score": (risk_scores * 100).round(1),
    "flagged": flagged,
    "age": df["age_at_admission"].round(0).astype(int),
    "icu_admission": df["icu_admission"],
    "length_of_stay": df["length_of_stay"].round(1),
    "active_conditions": df["active_conditions"],
    "active_medications": df["active_medications"],
    "prior_emergency": df["prior_emergency"],
    "has_hypertension": df["has_hypertension"],
    "has_diabetes": df["has_diabetes"],
    "has_obesity": df["has_obesity"],
    "has_chronic_resp": df["has_chronic_resp"],
    "actual_readmit": df["readmitted_30d"]
})

# ── Header ────────────────────────────────────────────────────
st.title("🏥 COVID-19 Readmission Risk Dashboard")
st.caption(
    "Care management decision support — "
    f"Operating threshold: {THRESHOLD:.0%} | "
    "Model: Logistic Regression | "
    "Cohort: COVID-19 inpatient discharges (Synthea, 2020)"
)

# ── Summary metrics ───────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Patients", len(display_df))
col2.metric("Flagged High-Risk",
            display_df["flagged"].sum(),
            f"{display_df['flagged'].mean()*100:.1f}% of cohort")
col3.metric("ICU Admissions",
            display_df["icu_admission"].sum(),
            f"{display_df['icu_admission'].mean()*100:.1f}% of cohort")
col4.metric("Avg Risk Score",
            f"{display_df['risk_score'].mean():.1f}%",
            "Population baseline")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📋 Care Manager Worklist",
    "🔍 Patient Risk Detail",
    "📊 Model Performance"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — CARE MANAGER WORKLIST
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("High-Risk Discharge Worklist")
    st.caption(
        f"Patients with predicted 30-day readmission risk ≥ {THRESHOLD:.0%}. "
        "Sorted by risk score — highest priority first."
    )

    # Filter controls
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        show_all = st.toggle("Show all patients (including low-risk)", value=False)
    with col_b:
        icu_only = st.toggle("ICU patients only", value=False)
    with col_c:
        min_risk = st.slider("Minimum risk score (%)", 0, 100, 0)

    # Apply filters
    worklist = display_df.copy()
    if not show_all:
        worklist = worklist[worklist["flagged"] == 1]
    if icu_only:
        worklist = worklist[worklist["icu_admission"] == 1]
    worklist = worklist[worklist["risk_score"] >= min_risk]
    worklist = worklist.sort_values("risk_score", ascending=False)

    # Risk tier labels
    def risk_tier(score):
        if score >= 80:
            return "🔴 Critical"
        elif score >= 61:
            return "🟡 High"
        else:
            return "🟢 Low"

    worklist["Risk Tier"] = worklist["risk_score"].apply(risk_tier)

    # Display table
    display_cols = {
        "patient_id": "Patient ID",
        "risk_score": "Risk Score (%)",
        "Risk Tier": "Risk Tier",
        "age": "Age",
        "icu_admission": "ICU",
        "length_of_stay": "LOS (days)",
        "active_conditions": "Active Conditions",
        "active_medications": "Active Meds",
        "prior_emergency": "Prior ED Visits"
    }

    worklist_display = worklist[list(display_cols.keys())].rename(columns=display_cols)
    worklist_display["ICU"] = worklist_display["ICU"].map({1: "Yes", 0: "No"})

    st.dataframe(
        worklist_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk Score (%)": st.column_config.ProgressColumn(
                "Risk Score (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            )
        }
    )

    st.caption(
        f"Showing {len(worklist)} patients | "
        f"Critical (≥80%): {(worklist['risk_score']>=80).sum()} | "
        f"High (61-79%): {((worklist['risk_score']>=61) & (worklist['risk_score']<80)).sum()}"
    )

    # Risk score distribution chart
    st.subheader("Risk Score Distribution")
    fig_dist = px.histogram(
        display_df,
        x="risk_score",
        color="flagged",
        color_discrete_map={0: "#2196F3", 1: "#F44336"},
        labels={
            "risk_score": "Predicted Risk Score (%)",
            "flagged": "Flagged"
        },
        title="Distribution of Predicted Readmission Risk",
        barmode="overlay",
        opacity=0.75,
        nbins=40
    )
    fig_dist.add_vline(
        x=THRESHOLD * 100,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Threshold: {THRESHOLD:.0%}",
        annotation_position="top right"
    )
    fig_dist.update_layout(height=350)
    st.plotly_chart(fig_dist, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — PATIENT RISK DETAIL
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Patient Risk Factor Detail")

    # Patient selector — default to highest risk patient
    flagged_patients = display_df[display_df["flagged"] == 1].sort_values(
        "risk_score", ascending=False
    )
    patient_options = flagged_patients["patient_id"].tolist()

    if not patient_options:
        st.warning("No flagged patients to display.")
    else:
        selected_id = st.selectbox(
            "Select a flagged patient to review:",
            options=patient_options,
            index=0
        )

        patient = display_df[display_df["patient_id"] == selected_id].iloc[0]

        # Patient header
        risk_color = "#F44336" if patient["risk_score"] >= 80 else "#FF9800"
        st.markdown(
            f"### Patient Risk Score: "
            f"<span style='color:{risk_color}; font-size:2em'>"
            f"{patient['risk_score']:.1f}%</span>",
            unsafe_allow_html=True
        )

        # Key clinical facts
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Age", f"{patient['age']} yrs")
        col2.metric("ICU Admission", "Yes" if patient["icu_admission"] else "No")
        col3.metric("Length of Stay", f"{patient['length_of_stay']} days")
        col4.metric("Prior ED Visits", int(patient["prior_emergency"]))

        st.divider()

        # Risk factor waterfall chart
        # Show which features push risk up (positive coef) or down (negative)
        coef_df = pd.DataFrame({
            "feature": FEATURES,
            "coefficient": model.coef_[0]
        })

        # Get scaled feature values for this patient
        patient_features = df[df.index == display_df[
            display_df["patient_id"] == selected_id
        ].index[0]][FEATURES]
        patient_scaled = scaler.transform(patient_features)

        # Contribution = coefficient × scaled feature value
        contributions = (coef_df["coefficient"].values * patient_scaled[0])
        contrib_df = pd.DataFrame({
            "Feature": FEATURES,
            "Contribution": contributions
        }).sort_values("Contribution", key=abs, ascending=True)

        # Take top 10 by absolute contribution
        contrib_df = contrib_df.tail(10)
        contrib_df["Direction"] = contrib_df["Contribution"].apply(
            lambda x: "Increases Risk" if x > 0 else "Reduces Risk"
        )

        fig_contrib = px.bar(
            contrib_df,
            x="Contribution",
            y="Feature",
            color="Direction",
            color_discrete_map={
                "Increases Risk": "#F44336",
                "Reduces Risk": "#4CAF50"
            },
            orientation="h",
            title="Top Risk Factor Contributions for This Patient",
            labels={"Contribution": "Risk Contribution (log-odds units)"}
        )
        fig_contrib.add_vline(x=0, line_color="black", line_width=1)
        fig_contrib.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_contrib, use_container_width=True)

        # Comorbidity flags
        st.subheader("Active Comorbidities at Discharge")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hypertension",
                  "✅ Yes" if patient["has_hypertension"] else "❌ No")
        c2.metric("Diabetes/Prediabetes",
                  "✅ Yes" if patient["has_diabetes"] else "❌ No")
        c3.metric("Obesity",
                  "✅ Yes" if patient["has_obesity"] else "❌ No")
        c4.metric("Chronic Respiratory",
                  "✅ Yes" if patient["has_chronic_resp"] else "❌ No")

        st.caption(
            f"Total active conditions: {int(patient['active_conditions'])} | "
            f"Active medications: {int(patient['active_medications'])}"
        )

# ════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance Summary")
    st.caption(
        "Intended audience: Clinical informatics, quality improvement, "
        "and model governance teams."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Cross-Validation Performance (5-fold)")
        cv_data = {
            "Metric": ["Recall", "Precision", "F1", "AUC-ROC", "AUC-PR"],
            "Mean": [0.817, 0.406, 0.539, 0.854, 0.590],
            "Std": [0.049, 0.063, 0.056, 0.024, 0.082]
        }
        cv_df = pd.DataFrame(cv_data)
        cv_df["95% CI"] = cv_df.apply(
            lambda r: f"{r['Mean']-2*r['Std']:.3f} – {r['Mean']+2*r['Std']:.3f}",
            axis=1
        )
        st.dataframe(cv_df, use_container_width=True, hide_index=True)

        st.markdown("#### Operating Threshold Rationale")
        st.info(
            f"**Threshold: {THRESHOLD:.0%}**\n\n"
            "Selected based on operational capacity constraint (≤20% flag rate) "
            "and clinically meaningful precision floor (≥50%).\n\n"
            "**Clinical rationale:** A missed readmission (false negative) carries "
            "substantially higher cost — clinical, financial, and patient experience — "
            "than an unnecessary follow-up call (false positive). "
            "This threshold catches 81% of readmissions at a 1:2 "
            "signal-to-noise ratio (1 true readmission per 2 calls made)."
        )

    with col2:
        st.markdown("#### Confusion Matrix (Test Set, n=333)")
        cm_data = [[267, 34], [6, 26]]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=["Predicted: No Readmit", "Predicted: Readmit"],
            y=["Actual: No Readmit", "Actual: Readmit"],
            colorscale="Blues",
            text=[[str(v) for v in row] for row in cm_data],
            texttemplate="%{text}",
            textfont={"size": 20},
            showscale=False
        ))
        fig_cm.update_layout(
            title="Confusion Matrix at Threshold 0.61",
            height=300,
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("#### Cohort Summary")
        cohort_info = {
            "Dataset": "Synthea COVID-19 (synthetic)",
            "Population": "10,000 patients → 1,662 adult COVID inpatient discharges",
            "Date range": "January – April 2020 (first US COVID wave)",
            "Positive rate": "9.51% (158/1,662 readmitted within 30 days)",
            "Model": "Logistic Regression, class_weight='balanced'",
            "Features": "23 features across 5 clinical domains"
        }
        for k, v in cohort_info.items():
            st.markdown(f"**{k}:** {v}")

    st.divider()
    st.markdown("#### ⚠️ Fairness & Limitations")
    st.warning(
        "**Race/Ethnicity:** After controlling for clinical severity features, "
        "race shows modest independent effects (OR 0.94–1.10). "
        "Raw subgroup differences appear largely explained by severity and "
        "utilization patterns, not race itself. "
        "A formal equity audit is recommended before any real-world deployment.\n\n"
        "**Data:** This model was trained on synthetic data (Synthea). "
        "Performance on real EHR data will differ and requires external validation.\n\n"
        "**Native American subgroup:** n=6 — too small for reliable inference. "
        "Subgroup metrics for this group should not be reported."
    )