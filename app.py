"""
app.py
------
Loan Approval Prediction — Streamlit Web Application
Deployment for: loan_approval_project

Project structure:
    loan-prediction-app/
    ├── app.py                  ← this file  (streamlit run app.py)
    ├── train_model.py          ← run once to generate loan_model.pkl
    ├── loan_model.pkl          ← generated artifact
    ├── data/
    │   └── loan-train.csv
    ├── src/
    │   ├── __init__.py
    │   ├── preprocessing.py
    │   ├── model.py
    │   └── visualize.py
    └── requirements.txt
"""

import pickle, sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — warm editorial theme ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  { font-family: 'Outfit', sans-serif; }
#MainMenu, footer, header   { visibility: hidden; }
.stApp                      { background: #faf8f4; color: #1c1917; }

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"]   { background: #1c1917 !important; border-right: none; }
[data-testid="stSidebar"] * { color: #a8a29e !important; }
[data-testid="stSidebar"] label {
    font-size: 0.7rem !important; text-transform: uppercase;
    letter-spacing: 1.4px; color: #57534e !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #292524 !important; border-color: #3d3935 !important;
    color: #e7e5e4 !important; border-radius: 8px !important; }
[data-testid="stSidebar"] input {
    background: #292524 !important; border-color: #3d3935 !important;
    color: #e7e5e4 !important; border-radius: 8px !important; }

/* ── Predict button ──────────────────────────────── */
div[data-testid="stSidebar"] .stButton > button {
    background: #d97706 !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
    font-size: 0.95rem !important; width: 100% !important;
    padding: 0.7rem !important; margin-top: 1rem !important;
    letter-spacing: 0.5px !important; }
div[data-testid="stSidebar"] .stButton > button:hover {
    background: #b45309 !important; }

/* ── Header ──────────────────────────────────────── */
.header {
    background: #1c1917;
    padding: 2rem 2.5rem 1.8rem;
    margin: -3rem -2rem 1.8rem;
    display: flex; align-items: flex-end; justify-content: space-between;
    gap: 1rem; flex-wrap: wrap; }
.header-left h1 {
    font-family: 'Cormorant Garamond', serif; font-size: 2.4rem;
    font-weight: 700; color: #fafaf9; margin: 0 0 0.2rem; line-height: 1.1; }
.header-left h1 span { color: #d97706; }
.header-left p {
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    color: #44403c; margin: 0; letter-spacing: 1.5px; }
.header-right { display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: center; }
.stat-pill {
    background: #292524; border: 1px solid #3d3935; border-radius: 20px;
    padding: 0.3rem 0.9rem; font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #78716c; white-space: nowrap; }
.stat-pill b { color: #d97706; }

/* ── KPI strip ───────────────────────────────────── */
.kpi-row   { display:flex; gap:0.8rem; margin-bottom:1.4rem; flex-wrap:wrap; }
.kpi       { flex:1; min-width:115px; background:#fff; border:1px solid #e7e5e4;
             border-top:3px solid #1c1917; padding:0.95rem 1.1rem; border-radius:6px; }
.kpi-label { font-family:'JetBrains Mono',monospace; font-size:0.58rem;
             text-transform:uppercase; letter-spacing:1.3px; color:#a8a29e; margin-bottom:0.3rem; }
.kpi-val   { font-family:'Cormorant Garamond',serif; font-size:1.9rem;
             font-weight:700; color:#1c1917; line-height:1; }
.kpi-val.warn { color:#dc2626; }
.kpi-val.good { color:#15803d; }

/* ── Result cards ────────────────────────────────── */
.result { border-radius:8px; padding:1.8rem 2rem; text-align:center; margin-bottom:1rem; }
.approved { background:#f0fdf4; border:1px solid #bbf7d0; border-left:7px solid #16a34a; }
.rejected { background:#fef2f2; border:1px solid #fecaca; border-left:7px solid #dc2626; }
.result .icon { font-size:2.2rem; }
.result h2 { font-family:'Cormorant Garamond',serif; font-size:2.1rem;
             font-weight:700; margin:0.25rem 0 0.4rem; letter-spacing:0.5px; }
.approved h2 { color:#15803d; }
.rejected h2 { color:#dc2626; }
.result p  { font-size:0.84rem; color:#78716c; margin:0; }

/* ── Prob bars ───────────────────────────────────── */
.prow     { margin:0.45rem 0 0.85rem; }
.pmeta    { display:flex; justify-content:space-between;
            font-family:'JetBrains Mono',monospace; font-size:0.7rem;
            color:#a8a29e; margin-bottom:0.22rem; }
.pouter   { height:8px; background:#f5f5f4; border-radius:4px; overflow:hidden; }
.pgreen   { background:linear-gradient(90deg,#15803d,#4ade80); height:100%; border-radius:4px; }
.pred     { background:linear-gradient(90deg,#b91c1c,#f87171); height:100%; border-radius:4px; }

/* ── Misc ────────────────────────────────────────── */
.sec  { font-family:'JetBrains Mono',monospace; font-size:0.6rem; text-transform:uppercase;
        letter-spacing:2px; color:#a8a29e; border-bottom:1px solid #e7e5e4;
        padding-bottom:0.32rem; margin:1.4rem 0 0.85rem; }
.chip { display:inline-block; background:#f5f5f4; border:1px solid #e7e5e4;
        border-radius:4px; padding:0.18rem 0.6rem; margin:0.14rem;
        font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#78716c; }
.chip.ok  { background:#f0fdf4; border-color:#bbf7d0; color:#15803d; }
.chip.bad { background:#fef2f2; border-color:#fecaca; color:#dc2626; }
.idle { background:#fff; border:1px dashed #d6d3d1; border-radius:8px;
        padding:3rem 2rem; text-align:center; }
.idle p { font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#d6d3d1; margin:0; }
.side-sec { font-family:'JetBrains Mono',monospace; font-size:0.58rem;
            text-transform:uppercase; letter-spacing:1.5px; color:#292524 !important;
            border-bottom:1px solid #3d3935; padding-bottom:0.26rem;
            margin:1.1rem 0 0.55rem; }
</style>
""", unsafe_allow_html=True)


# ── Load model artifact ───────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifact():
    with open("loan_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    art      = load_artifact()
    MODEL_OK = True
except FileNotFoundError:
    MODEL_OK = False

if MODEL_OK:
    rf       = art["model"]
    FEATURES = art["features"]
    ENC_MAP  = art["enc_map"]
    IMPUTE   = art["impute"]
    CV_ACC   = art["cv_accuracy"]
    CV_AUC   = art["cv_auc"]
    CV_F1    = art["cv_f1"]
    FI_DF    = art["feat_importance"]


# ── Inference helper ──────────────────────────────────────
def build_row(raw: dict) -> pd.DataFrame:
    """
    Replicates preprocessing.py's preprocess() for a single new
    applicant row — same steps, same column order as training.
    """
    app_inc   = raw["ApplicantIncome"]
    co_inc    = raw["CoapplicantIncome"]
    loan_amt  = raw["LoanAmount"]
    loan_term = raw["Loan_Amount_Term"]

    total_income     = app_inc + co_inc
    loan_amount_log  = np.log1p(loan_amt)
    total_income_log = np.log1p(total_income)
    emi              = loan_amt / loan_term if loan_term else 0
    balance_income   = total_income - (emi * 1000)

    dep_str = str(raw["Dependents"]).replace("3+", "3")
    try:
        dep_int = int(dep_str)
    except ValueError:
        dep_int = 0

    row = {
        "Gender":           ENC_MAP["Gender"].get(raw["Gender"], 0),
        "Married":          ENC_MAP["Married"].get(raw["Married"], 0),
        "Dependents":       dep_int,
        "Education":        ENC_MAP["Education"].get(raw["Education"], 0),
        "Self_Employed":    ENC_MAP["Self_Employed"].get(raw["Self_Employed"], 0),
        "ApplicantIncome":  app_inc,
        "CoapplicantIncome":co_inc,
        "LoanAmount":       loan_amt,
        "Loan_Amount_Term": loan_term,
        "Credit_History":   float(raw["Credit_History"]),
        "Property_Area":    ENC_MAP["Property_Area"].get(raw["Property_Area"], 0),
        "TotalIncomeLog":   total_income_log,
        "LoanAmountLog":    loan_amount_log,
        "EMI":              emi,
        "BalanceIncome":    balance_income,
    }
    return pd.DataFrame([row])[FEATURES]


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.3rem'>
        <div style='font-family:Cormorant Garamond,serif;font-size:1.6rem;
                    font-weight:700;color:#fafaf9;'>🏦 LoanSense</div>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.58rem;
                    color:#292524;letter-spacing:1.5px;margin-top:0.1rem;'>
            APPLICANT FORM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='side-sec'>Personal</div>", unsafe_allow_html=True)
    gender     = st.selectbox("Gender",        ["Male", "Female"])
    married    = st.selectbox("Married",       ["Yes", "No"])
    dependents = st.selectbox("Dependents",    ["0", "1", "2", "3+"])
    education  = st.selectbox("Education",     ["Graduate", "Not Graduate"])
    self_emp   = st.selectbox("Self Employed", ["No", "Yes"])

    st.markdown("<div class='side-sec'>Financial</div>", unsafe_allow_html=True)
    app_income  = st.number_input("Applicant Income ($/mo)",    0, 100_000, 5_000, 500)
    co_income   = st.number_input("Co-applicant Income ($/mo)", 0,  60_000,     0, 500)
    loan_amount = st.number_input("Loan Amount ($ thousands)",  1,     700,   128,   5)
    loan_term   = st.selectbox("Loan Term (months)",
                               [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)
    credit_hist = st.selectbox("Credit History", ["Good (1.0)", "Poor (0.0)"])

    st.markdown("<div class='side-sec'>Property</div>", unsafe_allow_html=True)
    prop_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    predict_btn = st.button("⚡  PREDICT ELIGIBILITY")


# ── Derived KPI values ────────────────────────────────────
total_inc = app_income + co_income
emi_val   = round((loan_amount * 1000) / loan_term, 1) if loan_term else 0
dti_val   = round((loan_amount * 1000) / (total_inc * 12 + 1) * 100, 1) if total_inc else 0
bal_val   = round(total_inc - emi_val, 0)
ch_num    = 1.0 if credit_hist.startswith("Good") else 0.0

raw_input = {
    "Gender": gender, "Married": married, "Dependents": dependents,
    "Education": education, "Self_Employed": self_emp,
    "ApplicantIncome": app_income, "CoapplicantIncome": co_income,
    "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
    "Credit_History": ch_num, "Property_Area": prop_area,
}


# ── Header ────────────────────────────────────────────────
st.markdown(f"""
<div class='header'>
  <div class='header-left'>
    <h1>Loan Approval <span>Predictor</span></h1>
    <p>RANDOM FOREST · SCIKIT-LEARN · STREAMLIT · 614 TRAINING ROWS</p>
  </div>
  <div class='header-right'>
    <span class='stat-pill'>CV Acc <b>{CV_ACC if MODEL_OK else '—'}</b></span>
    <span class='stat-pill'>ROC-AUC <b>{CV_AUC if MODEL_OK else '—'}</b></span>
    <span class='stat-pill'>F1 <b>{CV_F1 if MODEL_OK else '—'}</b></span>
    <span class='stat-pill'>n_estimators <b>300</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

if not MODEL_OK:
    st.error("""
    **`loan_model.pkl` not found.**  Run the training script first:
    ```bash
    python train_model.py
    ```
    """)
    st.stop()

# ── KPI strip ─────────────────────────────────────────────
st.markdown(f"""
<div class='kpi-row'>
  <div class='kpi'>
    <div class='kpi-label'>Total Income / mo</div>
    <div class='kpi-val'>${total_inc:,}</div>
  </div>
  <div class='kpi'>
    <div class='kpi-label'>Monthly EMI</div>
    <div class='kpi-val'>${emi_val:,.0f}</div>
  </div>
  <div class='kpi'>
    <div class='kpi-label'>Debt-to-Income</div>
    <div class='kpi-val {"warn" if dti_val > 40 else "good"}'>{dti_val}%</div>
  </div>
  <div class='kpi'>
    <div class='kpi-label'>Balance After EMI</div>
    <div class='kpi-val {"warn" if bal_val < 0 else "good"}'>${bal_val:,.0f}</div>
  </div>
  <div class='kpi'>
    <div class='kpi-label'>Credit History</div>
    <div class='kpi-val {"good" if ch_num == 1 else "warn"}'>
        {"✓ GOOD" if ch_num == 1 else "✗ POOR"}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Two-column body ───────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

# ═══ LEFT — prediction ═══════════════════════════════════
with left:
    st.markdown("<div class='sec'>Prediction Result</div>", unsafe_allow_html=True)

    if predict_btn:
        X_row  = build_row(raw_input)
        pred   = rf.predict(X_row)[0]
        proba  = rf.predict_proba(X_row)[0]
        p_app  = round(proba[1] * 100, 1)
        p_rej  = round(proba[0] * 100, 1)

        if pred == 1:
            st.markdown(f"""
            <div class='result approved'>
              <div class='icon'>✅</div>
              <h2>LOAN APPROVED</h2>
              <p>Applicant profile meets the model's lending criteria.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result rejected'>
              <div class='icon'>❌</div>
              <h2>LOAN REJECTED</h2>
              <p>Profile does not meet current lending criteria.</p>
            </div>""", unsafe_allow_html=True)

        # Probability bars
        st.markdown(f"""
        <div class='prow'>
          <div class='pmeta'><span>Approval probability</span><span>{p_app}%</span></div>
          <div class='pouter'><div class='pgreen' style='width:{p_app}%'></div></div>
        </div>
        <div class='prow'>
          <div class='pmeta'><span>Rejection probability</span><span>{p_rej}%</span></div>
          <div class='pouter'><div class='pred' style='width:{p_rej}%'></div></div>
        </div>
        """, unsafe_allow_html=True)

        # Profile chips
        st.markdown("<div class='sec'>Applicant Profile</div>", unsafe_allow_html=True)
        edu_c = "ok"  if education == "Graduate" else ""
        ch_c  = "ok"  if ch_num == 1             else "bad"
        dt_c  = "bad" if dti_val > 40            else "ok"
        st.markdown(f"""
        <div>
          <span class='chip'>{gender}</span>
          <span class='chip'>{married} · Dep:{dependents}</span>
          <span class='chip {edu_c}'>{education}</span>
          <span class='chip {ch_c}'>Credit: {"Good" if ch_num==1 else "Poor"}</span>
          <span class='chip'>{prop_area}</span>
          <span class='chip {dt_c}'>DTI {dti_val}%</span>
          <span class='chip'>Term {loan_term}mo</span>
          <span class='chip'>${loan_amount}K loan</span>
        </div>
        """, unsafe_allow_html=True)

        # Engineered feature values
        st.markdown("<div class='sec'>Engineered Features Sent to Model</div>",
                    unsafe_allow_html=True)
        eng = pd.DataFrame({
            "Feature": ["TotalIncomeLog", "LoanAmountLog", "EMI", "BalanceIncome"],
            "Value": [
                round(float(X_row["TotalIncomeLog"].iloc[0]),  4),
                round(float(X_row["LoanAmountLog"].iloc[0]),   4),
                round(float(X_row["EMI"].iloc[0]),             4),
                round(float(X_row["BalanceIncome"].iloc[0]),   2),
            ],
            "Formula": [
                "log1p(Applicant + Co-applicant Income)",
                "log1p(LoanAmount)",
                "LoanAmount / Loan_Amount_Term",
                "TotalIncome − (EMI × 1000)",
            ],
        })
        st.dataframe(eng, use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div class='idle'>
          <p>Complete the sidebar form<br>and click
          <strong style='color:#d97706'>⚡ PREDICT ELIGIBILITY</strong></p>
        </div>
        """, unsafe_allow_html=True)


# ═══ RIGHT — feature importance chart ════════════════════
with right:
    st.markdown("<div class='sec'>Feature Importances</div>", unsafe_allow_html=True)

    fi_sorted = FI_DF.sort_values("Importance")
    engineered_feats = {"TotalIncomeLog", "LoanAmountLog", "EMI", "BalanceIncome"}
    colors = [
        "#d97706" if f in engineered_feats else "#d6d3d1"
        for f in fi_sorted["Feature"]
    ]

    fig = go.Figure(go.Bar(
        x=fi_sorted["Importance"],
        y=fi_sorted["Feature"],
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.3f}" for v in fi_sorted["Importance"]],
        textposition="outside",
        textfont=dict(size=9, color="#a8a29e", family="JetBrains Mono"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#78716c", family="Outfit"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(size=10, family="JetBrains Mono", color="#78716c")),
        margin=dict(l=0, r=55, t=5, b=5),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div style='display:flex;gap:1.2rem;margin-top:-0.4rem;margin-bottom:0.6rem;'>
      <span style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#d97706;'>
        ■ Engineered feature</span>
      <span style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#d6d3d1;'>
        ■ Original feature</span>
    </div>
    """, unsafe_allow_html=True)

    top3 = fi_sorted.tail(3)["Feature"].values[::-1]
    st.markdown(
        "<div class='sec'>Top-3 Decision Drivers</div>" +
        "".join([f"<span class='chip ok'>#{i+1} {f}</span>"
                 for i, f in enumerate(top3)]),
        unsafe_allow_html=True
    )


# ── Tabs ──────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊 Dataset Explorer", "🔬 Model Pipeline", "📥 Batch Predict"])

# ── Tab 1: Dataset ────────────────────────────────────────
with tab1:
    try:
        raw = pd.read_csv("data/loan-train.csv")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",         raw.shape[0])
        c2.metric("Features",     raw.shape[1] - 2)
        approved_pct = round(raw["Loan_Status"].value_counts(normalize=True).get("Y", 0) * 100)
        c3.metric("Approval Rate", f"{approved_pct}%")
        c4.metric("Missing Cells", int(raw.isnull().sum().sum()))

        st.markdown("**Sample rows**")
        st.dataframe(raw.head(8), use_container_width=True, hide_index=True)

        miss = raw.isnull().sum()
        miss = miss[miss > 0].reset_index()
        miss.columns = ["Column", "Missing"]
        if not miss.empty:
            st.markdown("**Columns with missing values**")
            st.dataframe(miss, use_container_width=True, hide_index=True)

        st.markdown("**Target class distribution**")
        vc = raw["Loan_Status"].value_counts().reset_index()
        vc.columns = ["Status", "Count"]
        vc["Label"] = vc["Status"].map({"Y": "Approved", "N": "Rejected"})
        fig_cls = go.Figure(go.Bar(
            x=vc["Label"], y=vc["Count"],
            marker_color=["#15803d", "#dc2626"],
            text=vc["Count"], textposition="outside",
        ))
        fig_cls.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=260,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(t=20, b=10, l=0, r=0),
        )
        st.plotly_chart(fig_cls, use_container_width=True)
    except FileNotFoundError:
        st.warning("`data/loan-train.csv` not found.")

# ── Tab 2: Pipeline ───────────────────────────────────────
with tab2:
    st.markdown("**`src/model.py` — `build_model()` configuration**")
    st.code("""RandomForestClassifier(
    n_estimators      = 300,
    max_depth         = 8,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    max_features      = 'sqrt',
    class_weight      = 'balanced',
    random_state      = 42,
    n_jobs            = -1,
)""", language="python")

    st.markdown("**`src/preprocessing.py` — `preprocess()` feature engineering**")
    st.code("""# Applied to raw inputs at both training AND inference
TotalIncome    = ApplicantIncome + CoapplicantIncome
LoanAmountLog  = log1p(LoanAmount)
TotalIncomeLog = log1p(TotalIncome)
EMI            = LoanAmount / Loan_Amount_Term
BalanceIncome  = TotalIncome - (EMI * 1000)""", language="python")

    st.markdown("**Encoding map** *(reconstructed from training data and stored in `loan_model.pkl`)*")
    enc_rows = [{"Column": col, "Mapping": str(m)} for col, m in ENC_MAP.items()]
    st.table(pd.DataFrame(enc_rows))

    st.markdown("**Feature column order sent to `rf.predict()`**")
    st.code(str(FEATURES), language="python")

    c1, c2, c3 = st.columns(3)
    c1.metric("CV Accuracy",  CV_ACC)
    c2.metric("CV ROC-AUC",   CV_AUC)
    c3.metric("CV F1-Score",  CV_F1)

# ── Tab 3: Batch predict ──────────────────────────────────
with tab3:
    st.markdown("""
    Upload a CSV with the same columns as `data/loan-train.csv`
    *(without `Loan_Status`)*. The app predicts each row and
    lets you download the results.
    """)
    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            batch.drop(columns=["Loan_ID", "Loan_Status", "Unnamed: 0"],
                       errors="ignore", inplace=True)
            st.write(f"Uploaded: **{len(batch)} rows**")

            rows, preds_out = [], []
            for _, r in batch.iterrows():
                raw_r = {
                    "Gender":            r.get("Gender",            IMPUTE["Gender"]),
                    "Married":           r.get("Married",           IMPUTE["Married"]),
                    "Dependents":        str(r.get("Dependents",    "0")),
                    "Education":         r.get("Education",         "Graduate"),
                    "Self_Employed":     r.get("Self_Employed",     IMPUTE["Self_Employed"]),
                    "ApplicantIncome":   r.get("ApplicantIncome",   5000),
                    "CoapplicantIncome": r.get("CoapplicantIncome", 0),
                    "LoanAmount":        r.get("LoanAmount",        IMPUTE["LoanAmount"]),
                    "Loan_Amount_Term":  r.get("Loan_Amount_Term",  IMPUTE["Loan_Amount_Term"]),
                    "Credit_History":    r.get("Credit_History",    IMPUTE["Credit_History"]),
                    "Property_Area":     r.get("Property_Area",     "Urban"),
                }
                X_b  = build_row(raw_r)
                pred = rf.predict(X_b)[0]
                prob = rf.predict_proba(X_b)[0][1]
                preds_out.append({
                    "Prediction":    "Approved" if pred == 1 else "Rejected",
                    "Approval Prob": f"{prob*100:.1f}%",
                })

            out = pd.concat([batch.reset_index(drop=True),
                             pd.DataFrame(preds_out)], axis=1)
            st.dataframe(out, use_container_width=True)
            st.download_button("⬇️  Download Predictions CSV",
                               out.to_csv(index=False),
                               "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")


# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:2.5rem;padding-top:1rem;border-top:2px solid #1c1917;
            display:flex;justify-content:space-between;flex-wrap:wrap;gap:0.4rem;'>
  <span style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#d6d3d1;'>
    LoanSense · Random Forest · src/preprocessing.py · src/model.py · Streamlit
  </span>
  <span style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#d6d3d1;'>
    Pipeline: impute → feature engineer → label encode → RF(300 trees) → predict
  </span>
</div>
""", unsafe_allow_html=True)
