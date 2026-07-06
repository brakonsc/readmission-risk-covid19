# COVID-19 30-Day Readmission Risk — Clinical Decision Support Pipeline

> An end-to-end clinical informatics project: EHR data modeling, cohort construction,
> ML risk scoring, and a care manager decision support dashboard — built for the
> population health and value-based care context.

---

## The Clinical Problem

Under the **Hospital Readmissions Reduction Program (HRRP)**, CMS penalizes hospitals
for excess 30-day readmissions with Medicare payment reductions of up to 3%. For a
mid-size health system, that represents millions in annual revenue exposure.

The standard intervention — proactive care management outreach at discharge — is
effective but resource-constrained. Care management teams cannot follow up with every
patient. The clinical question this project addresses:

> **Which patients, at the moment of discharge, are most likely to return within
> 30 days — and which risk factors are driving that prediction?**

A reliable answer lets care managers concentrate limited capacity on the patients
who need them most, reducing readmissions, improving outcomes, and protecting
hospital revenue under value-based reimbursement.

---

## Project Architecture

```
readmission-risk/
│
├── data/
│   ├── 10k_synthea_covid19_csv/     # Raw Synthea EHR export (16 tables)
│   └── processed/
│       ├── model_dataset.csv        # Feature matrix (1,662 × 26)
│       └── patient_ids.csv          # Patient IDs for traceability
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Cohort construction & feature engineering
│   └── 02_modeling.ipynb            # ML pipeline, evaluation, threshold tuning
│
├── dashboard/
│   └── app.py                       # Streamlit care manager dashboard
│
├── models/
│   ├── readmission_model.joblib     # Trained logistic regression
│   ├── scaler.joblib                # Fitted StandardScaler
│   └── model_config.json            # Threshold, features, clinical rationale
│
└── notes/
    └── 00_project_reference.md      # Living reference: decisions, concepts, glossary
```

---

## Dataset

**Source:** Synthea COVID-19 10,000 Patient Sample (synthetic EHR)
**Tables used:** patients, encounters, conditions, medications (of 16 available)
**Cohort window:** January – April 2020 (first US COVID wave)

Synthea generates realistic multi-table EHR exports — patients, encounters,
conditions, medications, observations, procedures — mirroring the structure of
real clinical data systems (Epic, Oracle Health, Cerner). Working with this
format requires the same join logic, temporal reasoning, and data quality handling
as production clinical analytics work.

---

## Methodology

### Cohort Construction

A critical distinction from notebook-style ML: this project constructs a
**clinical cohort**, not just a filtered dataset. Each decision is documented
with clinical rationale.

| Step | Decision | Rationale |
|------|----------|-----------|
| Identify COVID encounters | Keyword match across DESCRIPTION + REASONDESCRIPTION | REASONCODE 78% missing — real EHR data limitation |
| Index encounter | First inpatient COVID admission per patient | Anchors analysis to initial disease event |
| Exclude: died during admission | Cannot be readmitted if never discharged | Standard readmission study design |
| Exclude: age < 18 | Pediatric readmission drivers not generalizable | 104 patients removed |
| **Final cohort** | **1,662 adult COVID inpatient discharges** | |

**Cohort funnel:**
```
COVID inpatient encounters    2,376
Unique patients               1,867
Died during admission          -101
Pediatric (age < 18)           -104
──────────────────────────────────
Final adult cohort            1,662
```

### Target Variable

There is no "readmitted" column in EHR data. The 30-day readmission label is
**derived** from raw encounter timestamps — identifying any inpatient or emergency
encounter starting within 30 days of index discharge.

This is called **cohort construction** and is standard clinical informatics work.

**Positive rate: 9.51%** (158/1,662) — consistent with published COVID-19
30-day readmission rates from the 2020 literature.

### Feature Engineering

25 features across 5 clinical domains, constructed with strict **temporal
leakage prevention**: every feature uses only information available at the
moment of discharge.

| Domain | Features | Clinical Rationale |
|--------|----------|-------------------|
| Demographics | Age, gender, race, marital status, financial coverage | Social determinants of health; age is independent readmission risk factor |
| Encounter context | ICU admission flag, length of stay, discharge hour, weekend discharge | Severity proxies; weekend discharge → reduced follow-up access |
| Prior utilization | Prior inpatient/ED/outpatient counts (180-day lookback) | Past utilization is one of the most validated readmission predictors |
| Comorbidity burden | Active condition count + flags for HTN, obesity, diabetes, chronic respiratory | Known COVID severity amplifiers; active at discharge |
| Medication burden | Active medication count | Polypharmacy correlates with comorbidity complexity and readmission risk |

**Methodological note:** Initial feature set included `prior_total` (sum of
prior utilization components) and `polypharmacy` (derived from `active_medications`).
These were identified as multicollinear and removed. Model performance was
unaffected; coefficient interpretability improved substantially. This is documented
as a methodological finding, not corrected silently.

### Pre-Modeling Equity Analysis

Subgroup readmission rates were examined **before modeling**, consistent with
emerging standards for clinical AI fairness review.

| Group | Readmission Rate | N |
|-------|-----------------|---|
| Female | 10.3% | 904 |
| Male | 8.6% | 758 |
| Asian | 14.9% | 114 |
| Black | 8.7% | 150 |
| Native | 16.7% | **6 — unreliable** |
| White | 9.1% | 1,392 |

After controlling for clinical severity features (ICU status, utilization,
comorbidities), race shows modest independent odds ratios (0.94–1.10),
suggesting the raw subgroup differences are largely explained by clinical
severity patterns. A formal equity audit is recommended before any
real-world deployment.

---

## Model

**Algorithm:** Logistic Regression with `class_weight='balanced'`

**Why logistic regression first:**
- Interpretable coefficients — explainable to clinicians and model governance committees
- Deployable as an odds ratio table — standard in clinical risk scoring literature
- Strong baseline — most published 30-day readmission models ARE logistic regressions
- Fast to validate, easy to audit

**Why `class_weight='balanced'` over SMOTE:**
With only 126 positive training cases and 23 features, SMOTE would generate
roughly as many synthetic patients as real ones. Class weighting is more
conservative and clinically defensible: "I penalized the model for missing
real readmissions" vs. "I fabricated patient data."

### Performance

**5-Fold Stratified Cross-Validation** (reported; single-split confirmed consistent):

| Metric | Mean | Std | Clinical Meaning |
|--------|------|-----|-----------------|
| Recall | 0.817 | 0.049 | 82% of readmissions caught |
| Precision | 0.406 | 0.063 | ~1 in 2.5 flagged = true readmission |
| AUC-ROC | 0.854 | 0.024 | Strong discrimination (published models: 0.65–0.75) |
| AUC-PR | 0.590 | 0.082 | More honest metric for imbalanced data |

CV confirmed the single-split result was not a lucky draw (std ≤ 0.05 across
all primary metrics).

### Key Predictors (Odds Ratios)

| Feature | OR | Clinical Interpretation |
|---------|-----|------------------------|
| ICU admission | **6.84** | Dominant predictor — severe disease marker |
| Prior emergency visits | 1.38 | Classic utilization signal; most validated readmission predictor |
| Single marital status | 1.24 | Social support deficit at discharge |
| Age at admission | 1.19 | Expected direction; modest independent effect |
| Active medications | 0.52 | Protective — hypothesis: reflects better-managed chronic disease |

**Headline finding:**
> ICU admission during the index hospitalization was the dominant predictor of
> 30-day readmission (OR ≈ 6.8), with prior emergency utilization as a secondary
> signal (OR ≈ 1.4) — both consistent with established readmission literature.

---

## Threshold Selection

**Operating threshold: 0.61** (61% predicted probability)

Threshold selection is framed as a **clinical operations decision**, not a
mathematical optimization. The constraint: care management teams have finite
capacity. Flagging 94% of patients (what recall ≥ 0.90 produced algorithmically)
creates alert fatigue and is operationally indefensible.

| Metric | Default (0.50) | **Selected (0.61)** |
|--------|---------------|---------------------|
| Flag rate | 19.2% | **15.6%** |
| Recall | 81.2% | **81.2%** |
| Precision | 40.6% | **50.0%** |
| Caught | 26/32 | **26/32** |
| False alarms | 38 | **26** |

**Clinical rationale:** At threshold 0.61, for every 2 care management calls
made, 1 reaches a patient who will be readmitted without intervention. This
1:2 signal-to-noise ratio is operationally sustainable for a 2–3 FTE
care management team and clinically meaningful.

**Business case:**
> Deployed across 1,000 monthly COVID-related discharges: ~156 flagged,
> ~78 readmissions intercepted. At $15,000–$25,000 per prevented readmission
> (CMS cost + HRRP penalty avoidance), potential monthly revenue protection
> of **$1.2M–$2.0M**.

---

## Dashboard

A Streamlit care manager decision support application with three audience-specific views:

**Tab 1 — Care Manager Worklist**
Flagged patients sorted by risk score with risk tier labels (🔴 Critical ≥80%,
🟡 High 61–79%). Filterable by ICU status and minimum risk score. Designed for
a 7am morning huddle workflow — zero training required to act on it.

**Tab 2 — Patient Risk Detail**
Per-patient risk factor contribution chart showing which features drive the
individual score up or down. Answers the care manager's real question: not just
*"is this patient high-risk?"* but *"why, and what should I do about it?"*

**Tab 3 — Model Performance**
Confusion matrix, cross-validation metrics, cohort summary, and fairness
considerations. Audience: clinical informatics directors, CMOs, model governance
committees.

**Launch:**
```bash
python -m streamlit run dashboard/app.py
```

---

## Limitations & Future Work

| Limitation | Impact | Mitigation Path |
|------------|--------|-----------------|
| Synthetic data (Synthea) | Model not validated on real EHR data | External validation on de-identified EHR data; MIMIC-IV is next project |
| Small positive class (n=158) | Wide confidence intervals on metrics | Larger cohort; cross-institutional validation |
| Single institution/cohort | Generalizability unknown | Multi-site validation |
| No lab values or vitals | Key severity signals missing | Extend to observations.csv (O2 sat, BMI, BP) |
| Race as feature | Equity risk if race proxies structural disparities | Formal equity audit; consider removing from production model |
| ZIP code excluded (46.5% missing) | SDOH signal lost | SDOH enrichment via Census data by zip |
| Static model | No drift monitoring | MLOps layer: monthly retraining trigger if recall < 0.75 |

**Next project:** MIMIC-IV ICU readmission risk — real de-identified data,
richer clinical signal (labs, vitals, medications), PhysioNet credentialing
demonstrating data use agreement experience relevant to real clinical deployments.

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Data processing | Python, Pandas, NumPy |
| Machine learning | Scikit-learn (LogisticRegression, StandardScaler, StratifiedKFold) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model serialization | Joblib |
| Data format | CSV (Synthea EHR export) |

---

## Clinical Background

This project was built by an Adult-Gerontology Acute Care NP with 10+ years in
neurosurgical, cardiothoracic, and surgical ICU settings. The clinical framing —
cohort construction, temporal leakage prevention, threshold selection as an
operational capacity problem, and fairness analysis before modeling — reflects
how these problems actually present in health system analytics and population
health programs, not just how they appear in academic ML papers.

---

## References

- Strack et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates.
  *BioMed Research International.* https://doi.org/10.1155/2014/781670
- CMS Hospital Readmissions Reduction Program (HRRP):
  https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp
- Synthea COVID-19 Dataset:
  https://synthea.mitre.org/downloads
- Wijnberge et al. (2020). Prediction models for clinical readmission:
  systematic review. *BMJ Open.*