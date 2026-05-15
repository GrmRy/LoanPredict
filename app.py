import streamlit as st
import joblib
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSight · Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #0D0F14;
    --surface:   #13161E;
    --surface2:  #1A1E29;
    --border:    #252A38;
    --accent:    #4F8EF7;
    --accent2:   #7B61FF;
    --success:   #22C55E;
    --danger:    #EF4444;
    --warning:   #F59E0B;
    --text:      #E8EAF0;
    --muted:     #6B7280;
    --radius:    14px;
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 960px !important; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    margin-bottom: 2rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(79,142,247,0.15), rgba(123,97,255,0.15));
    border: 1px solid rgba(79,142,247,0.3);
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 400;
    line-height: 1.15;
    margin: 0 0 1rem;
    background: linear-gradient(135deg, #E8EAF0 30%, #8B9BB4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ── Section card ── */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    margin-bottom: 1.25rem;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    font-weight: 400;
    color: var(--text);
    margin: 0 0 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-subtitle {
    font-size: 0.82rem;
    color: var(--muted);
    margin: 0 0 1.5rem;
    font-weight: 300;
}
.section-divider {
    height: 1px;
    background: var(--border);
    margin: 0 0 1.5rem;
}

/* ── Input labels ── */
label, .stSlider label, [data-testid="stNumberInput"] label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #9BA3B4 !important;
    letter-spacing: 0.02em !important;
    margin-bottom: 0.3rem !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: white !important;
    border: 2px solid var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.25) !important;
}

/* ── Number input ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 0.85rem !important;
    transition: border-color 0.2s;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.15) !important;
    outline: none !important;
}

/* ── Submit button ── */
[data-testid="stFormSubmitButton"] button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
    margin-top: 0.5rem !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stFormSubmitButton"] button:active {
    transform: translateY(0) !important;
}

/* ── Result card ── */
.result-card {
    border-radius: var(--radius);
    padding: 2rem 2.2rem;
    margin-top: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
    animation: fadeUp 0.45s cubic-bezier(0.16,1,0.3,1) both;
}
.result-card.paid {
    background: linear-gradient(135deg, rgba(34,197,94,0.08), rgba(34,197,94,0.03));
    border: 1px solid rgba(34,197,94,0.25);
}
.result-card.risk {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(239,68,68,0.03));
    border: 1px solid rgba(239,68,68,0.25);
}
.result-icon {
    font-size: 2.4rem;
    flex-shrink: 0;
}
.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
    opacity: 0.6;
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    font-weight: 400;
    margin: 0 0 0.3rem;
}
.result-title.paid  { color: var(--success); }
.result-title.risk  { color: var(--danger); }
.result-desc {
    font-size: 0.88rem;
    color: var(--muted);
    font-weight: 300;
    line-height: 1.5;
}

/* ── Confidence bar ── */
.conf-wrap { margin-top: 1.2rem; }
.conf-label {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 0.4rem;
    font-weight: 400;
}
.conf-bar-bg {
    height: 6px;
    background: var(--border);
    border-radius: 999px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s cubic-bezier(0.16,1,0.3,1);
}
.conf-bar-fill.paid { background: linear-gradient(90deg, #22C55E, #4ADE80); }
.conf-bar-fill.risk { background: linear-gradient(90deg, #EF4444, #F87171); }

/* ── Info pills ── */
.pill-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}
.pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.75rem;
    color: var(--muted);
    font-weight: 400;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    font-size: 0.78rem;
    color: var(--muted);
    font-weight: 300;
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("loan_classifier.pkl")
    scaler = joblib.load("std_scaler.bin")
    return model, scaler

model, scaler = load_artifacts()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI · Credit Risk Assessment</div>
    <h1>Will This Loan<br><em>Be Repaid?</em></h1>
    <p>Enter the applicant's financial profile below to get an instant prediction powered by a Random Forest model trained on real lending data.</p>
</div>
""", unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("loan_form"):

    # ── Section 1: Financial Profile ──
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📊 Financial Profile</div>
        <div class="section-subtitle">Basic income and loan terms</div>
        <div class="section-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        annual_inc = st.number_input(
            "Annual Income (USD)",
            min_value=1_000.0, max_value=5_000_000.0,
            value=55_000.0, step=1_000.0,
            help="Gross annual income of the applicant."
        )
    with c2:
        installment = st.number_input(
            "Monthly Installment (USD)",
            min_value=10.0, max_value=2_000.0,
            value=350.0, step=10.0,
            help="Fixed monthly payment amount."
        )
    with c3:
        int_rate = st.slider(
            "Interest Rate (%)",
            min_value=5.0, max_value=25.0,
            value=12.0, step=0.1,
            help="Annual interest rate on the loan."
        )

    c4, c5 = st.columns(2)
    with c4:
        dti = st.slider(
            "Debt-to-Income Ratio",
            min_value=0.0, max_value=40.0,
            value=14.0, step=0.1,
            help="Total monthly debt payments / gross monthly income."
        )
    with c5:
        fico = st.slider(
            "FICO Credit Score",
            min_value=300, max_value=850,
            value=700, step=1,
            help="Borrower's FICO credit score (higher = better)."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Credit History ──
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📋 Credit History</div>
        <div class="section-subtitle">Revolving credit and past delinquencies</div>
        <div class="section-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    c6, c7, c8 = st.columns(3)
    with c6:
        revol_bal = st.number_input(
            "Revolving Balance (USD)",
            min_value=0.0, max_value=500_000.0,
            value=8_000.0, step=500.0,
            help="Current outstanding revolving credit balance."
        )
    with c7:
        revol_util = st.slider(
            "Revolving Utilization (%)",
            min_value=0.0, max_value=100.0,
            value=45.0, step=1.0,
            help="Percentage of revolving credit currently used."
        )
    with c8:
        inq_last_6mths = st.slider(
            "Credit Inquiries (Last 6 mo.)",
            min_value=0, max_value=10,
            value=1, step=1,
            help="Number of hard credit inquiries in the last 6 months."
        )

    c9, c10 = st.columns(2)
    with c9:
        delinq_2yrs = st.slider(
            "Delinquencies (Last 2 yrs)",
            min_value=0, max_value=10,
            value=0, step=1,
            help="Number of 30+ day past-due incidents in the last 2 years."
        )
    with c10:
        pub_rec = st.slider(
            "Public Derogatory Records",
            min_value=0, max_value=5,
            value=0, step=1,
            help="Number of derogatory public records (bankruptcies, liens, etc.)."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("⚡  Predict Loan Repayment")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    # Derived / engineered features
    log_annual_inc              = np.log(annual_inc)
    installment_to_income_ratio = installment / log_annual_inc
    credit_history              = (delinq_2yrs + pub_rec) / max(fico, 1)

    # Feature order must match training
    int_rate_decimal = int_rate / 100
    features = [
        int_rate, installment, log_annual_inc, dti, fico,
        revol_bal, revol_util, inq_last_6mths, delinq_2yrs,
        pub_rec, credit_history, installment_to_income_ratio,
    ]

    input_array  = np.array([features])
    scaled_array = scaler.transform(input_array)
    prediction   = model.predict(scaled_array)[0]
    proba        = model.predict_proba(scaled_array)[0]

    # 0 = Fully Paid, 1 = Not Fully Paid
    if prediction == 0:
        conf    = proba[0] * 100
        cls     = "paid"
        icon    = "✅"
        title   = "Likely to Repay"
        desc    = (
            f"The model predicts this borrower will <strong>fully repay</strong> the loan "
            f"with <strong>{conf:.0f}% confidence</strong>. "
            "The applicant's credit profile appears healthy."
        )
    else:
        conf    = proba[1] * 100
        cls     = "risk"
        icon    = "⚠️"
        title   = "High Default Risk"
        desc    = (
            f"The model predicts this borrower is <strong>at risk of default</strong> "
            f"with <strong>{conf:.0f}% confidence</strong>. "
            "Review the applicant's debt levels and credit history carefully."
        )

    st.markdown(f"""
    <div class="result-card {cls}">
        <div class="result-icon">{icon}</div>
        <div style="flex:1">
            <div class="result-label">Prediction Result</div>
            <div class="result-title {cls}">{title}</div>
            <div class="result-desc">{desc}</div>
            <div class="conf-wrap">
                <div class="conf-label">Model confidence — {conf:.1f}%</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill {cls}" style="width:{conf:.1f}%"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick insight pills ──
    flags = []
    if fico < 650:       flags.append("🔴 Low FICO score")
    if dti > 20:         flags.append("🔴 High DTI ratio")
    if int_rate > 18:    flags.append("🟡 High interest rate")
    if delinq_2yrs > 1:  flags.append("🔴 Recent delinquencies")
    if pub_rec > 0:      flags.append("🔴 Derogatory records")
    if revol_util > 75:  flags.append("🟡 High credit utilization")
    if inq_last_6mths > 3: flags.append("🟡 Multiple recent inquiries")
    if not flags:        flags.append("🟢 No major risk flags detected")

    pills_html = "".join(f'<span class="pill">{f}</span>' for f in flags)
    st.markdown(f"""
    <div style="margin-top:1.2rem">
        <div style="font-size:0.78rem;color:#6B7280;margin-bottom:0.6rem;font-weight:500;">
            RISK FLAGS DETECTED
        </div>
        <div class="pill-row">{pills_html}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    For demonstration purposes only. Not financial advice.<br>
</div>
""", unsafe_allow_html=True)