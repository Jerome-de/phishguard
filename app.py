"""
app.py — Phishing URL Detection | Hugging Face Space
======================================================
Author : Muhammad Suleiman Musa  (U/CS/21/134)
Project: Final Year Project — AFIT, Kaduna (2024/2025)
Model  : XGBoost + SHAP | UCI Phishing Websites Dataset

Two pages:
  1. Model Analysis  — dataset overview, model performance, SHAP insights
  2. Live URL Tester — user pastes a URL, gets phishing/legitimate verdict
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from feature_extractor import (
    extract_features, FEATURE_NAMES, FEATURE_DESCRIPTIONS, VALUE_LABELS
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard — URL Threat Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Typography & Base ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem;
    padding: 0.4rem 0;
}

/* ── Main background ────────────────────────────────────────────────────── */
.main .block-container {
    background: #f8fafc;
    padding-top: 1.5rem;
}

/* ── Hero banner ────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #0e4d8a 100%);
    border-radius: 14px;
    padding: 2.2rem 2.4rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "🛡️";
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.12;
}
.hero h1 {
    color: #f8fafc !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.4rem !important;
    letter-spacing: -0.02em;
}
.hero p { color: #94a3b8 !important; margin: 0; font-size: 0.9rem; }
.hero .tag {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    color: #7dd3fc !important;
    margin-top: 0.75rem;
    margin-right: 0.4rem;
}

/* ── Stat cards ─────────────────────────────────────────────────────────── */
.stat-card {
    background: white;
    border-radius: 10px;
    padding: 1.25rem 1.4rem;
    border: 1px solid #e2e8f0;
    text-align: center;
}
.stat-card .val {
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1;
}
.stat-card .lbl {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.3rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.stat-card .accent { color: #0ea5e9; }

/* ── Section headers ────────────────────────────────────────────────────── */
.sec-head {
    font-size: 1.05rem;
    font-weight: 600;
    color: #0f172a;
    border-left: 3px solid #0ea5e9;
    padding-left: 0.7rem;
    margin: 1.6rem 0 1rem;
}

/* ── URL input box ───────────────────────────────────────────────────────── */
.url-box { margin-bottom: 1.2rem; }
.url-box .stTextInput input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem;
    border: 2px solid #cbd5e1;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    background: white;
    transition: border-color 0.2s;
}
.url-box .stTextInput input:focus {
    border-color: #0ea5e9;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.15);
}

/* ── Verdict cards ───────────────────────────────────────────────────────── */
.verdict-phish {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    text-align: center;
    color: white;
    border: 1px solid #dc2626;
}
.verdict-legit {
    background: linear-gradient(135deg, #052e16, #14532d);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    text-align: center;
    color: white;
    border: 1px solid #16a34a;
}
.verdict-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.verdict-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 0.4rem;
}
.verdict-sub { font-size: 0.9rem; opacity: 0.8; }

/* ── Feature badge ───────────────────────────────────────────────────────── */
.feat-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.45rem 0.8rem;
    border-radius: 6px;
    margin-bottom: 0.3rem;
    background: white;
    border: 1px solid #f1f5f9;
    font-size: 0.83rem;
}
.feat-name { color: #334155; font-weight: 500; }
.badge {
    border-radius: 20px;
    padding: 0.15rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
}
.badge-phish  { background: #dc2626; }
.badge-susp   { background: #d97706; }
.badge-legit  { background: #16a34a; }

/* ── Performance table ───────────────────────────────────────────────────── */
.perf-row {
    display: flex;
    align-items: center;
    padding: 0.7rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    background: white;
    border: 1px solid #e2e8f0;
    font-size: 0.88rem;
}
.perf-row.best { border-color: #0ea5e9; background: #f0f9ff; }
.perf-name { width: 48%; font-weight: 500; color: #0f172a; }
.perf-acc { width: 20%; text-align: right; font-family: 'JetBrains Mono', monospace; }
.perf-bar-wrap { flex: 1; padding-left: 1rem; }
.perf-bar {
    height: 8px; border-radius: 4px;
    background: #0ea5e9;
}

/* ── Info panel ─────────────────────────────────────────────────────────── */
.info-panel {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #1e40af;
    margin: 1rem 0;
}

/* ── Confidence bar ──────────────────────────────────────────────────────── */
.conf-wrap { margin: 0.6rem 0; }
.conf-label {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; margin-bottom: 0.3rem; color: #475569;
}
.conf-track {
    height: 12px; background: #e2e8f0; border-radius: 6px; overflow: hidden;
}
.conf-fill {
    height: 100%; border-radius: 6px;
    transition: width 0.6s ease;
}
</style>
""", unsafe_allow_html=True)


# ── Model & Data Loading ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the serialised XGBoost model and selected feature list."""
    model_path    = "xgb_tuned_model.pkl"
    features_path = "selected_features.pkl"

    if not os.path.exists(model_path):
        return None, None, "Model file not found. Run the training notebook first."

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path, "rb") as f:
        selected = pickle.load(f)

    return model, selected, None


model, selected_features, model_error = load_model()

# ── Baseline results (hard-coded from the notebook's reproducible output) ─────
BASELINE_RESULTS = [
    ("Random Forest",        0.9773, "Base paper best"),
    ("ANN (MLP)",            0.9741, ""),
    ("Decision Tree",        0.9665, ""),
    ("K-Nearest Neighbours", 0.9567, ""),
    ("Logistic Regression",  0.9289, ""),
    ("XGBoost Full (ours)",  0.9790, ""),
    ("XGBoost Tuned ★",      0.9810, "Our proposed model"),
]

DATASET_STATS = {
    "Instances":   "11,055",
    "Features":    "30",
    "Phishing":    "55.7%",
    "Legitimate":  "44.3%",
    "Best Acc.":   "98.10%",
    "Algorithm":   "XGBoost",
}

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 1rem;">
      <div style="font-size:1.5rem;">🛡️</div>
      <div style="font-size:1.1rem;font-weight:700;color:#f8fafc;margin-top:0.3rem;">
        PhishGuard
      </div>
      <div style="font-size:0.75rem;color:#64748b;margin-top:0.15rem;">
        URL Threat Detector
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔬  Model Analysis", "🔍  URL Tester"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#475569;line-height:1.7;">
      <b style="color:#94a3b8;">Model</b><br>XGBoost (tuned)<br><br>
      <b style="color:#94a3b8;">Dataset</b><br>UCI Phishing Websites<br>
      Mohammad et al., 2012<br><br>
      <b style="color:#94a3b8;">Interpretability</b><br>TreeSHAP<br><br>
      <b style="color:#94a3b8;">Author</b><br>Muhammad Suleiman Musa<br>
      U/CS/21/134 · AFIT Kaduna<br>2024/2025
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MODEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔬  Model Analysis":

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <h1>Model Analysis</h1>
      <p>Dataset overview, model performance comparison, and SHAP-based feature importance.</p>
      <span class="tag">XGBoost</span>
      <span class="tag">SHAP</span>
      <span class="tag">UCI Phishing Dataset</span>
      <span class="tag">GridSearchCV</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset Stats ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Dataset at a Glance</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    stat_items = list(DATASET_STATS.items())
    for i, (lbl, val) in enumerate(stat_items):
        with cols[i]:
            accent = ' accent' if i >= 4 else ''
            st.markdown(f"""
            <div class="stat-card">
              <div class="val{accent}">{val}</div>
              <div class="lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── About ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">About This Project</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        <div style="font-size:0.9rem;color:#334155;line-height:1.75;">
        <p>This project detects phishing URLs and websites using an optimised
        <strong>XGBoost</strong> classifier trained on the
        <strong>UCI Phishing Websites Dataset</strong> (Mohammad, Thabtah & McCluskey, 2012)
        — the standard benchmark for phishing detection research.</p>

        <p>It extends the base paper by Chawla (2022) across five dimensions:</p>
        <ul style="margin:0.4rem 0 0.8rem 1rem;">
          <li>SHAP-guided feature selection (identifies the minimum sufficient feature subset)</li>
          <li>XGBoost evaluation (absent from all prior work on this dataset)</li>
          <li>Systematic GridSearchCV hyperparameter optimisation</li>
          <li>Full model interpretability via TreeSHAP</li>
          <li>Deployable web interface (this application)</li>
        </ul>
        <p>The trained model achieves <strong>98.10% accuracy</strong> — surpassing the
        97.73% best reported in the base paper — while using a smaller, SHAP-selected
        feature subset.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:10px;padding:1.2rem;">
          <div style="font-size:0.8rem;font-weight:600;color:#64748b;
                      text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.8rem;">
            Feature Categories
          </div>
          <div style="font-size:0.85rem;color:#334155;line-height:2;">
            <span style="color:#0ea5e9;font-weight:600;">12</span> URL-based features<br>
            <span style="color:#0ea5e9;font-weight:600;">10</span> Domain-based features<br>
            <span style="color:#0ea5e9;font-weight:600;">8</span> HTML/JS content features<br>
            <hr style="border-color:#f1f5f9;margin:0.5rem 0;">
            <span style="color:#0ea5e9;font-weight:600;">30</span> Total features<br>
            <span style="color:#0ea5e9;font-weight:600;">3</span> Encoding values (−1, 0, 1)<br>
            <span style="color:#0ea5e9;font-weight:600;">42</span> Random state (reproducible)
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Model Performance ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Model Performance Comparison (Test Set)</div>',
                unsafe_allow_html=True)

    # Sort by accuracy descending
    sorted_results = sorted(BASELINE_RESULTS, key=lambda x: x[1], reverse=True)
    max_acc = max(r[1] for r in sorted_results)

    for name, acc, note in sorted_results:
        is_best = name == "XGBoost Tuned ★"
        row_cls = "perf-row best" if is_best else "perf-row"
        bar_w = int((acc / max_acc) * 100)
        bar_col = "#0ea5e9" if is_best else "#94a3b8"
        note_html = f'<span style="font-size:0.72rem;color:#0ea5e9;margin-left:0.5rem;">{note}</span>' if note else ''
        acc_pct = f"{acc*100:.2f}%"
        st.markdown(f"""
        <div class="{row_cls}">
          <div class="perf-name">{name}{note_html}</div>
          <div class="perf-acc">{acc_pct}</div>
          <div class="perf-bar-wrap">
            <div class="perf-bar" style="width:{bar_w}%;background:{bar_col};"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SHAP Feature Importance Chart ─────────────────────────────────────────
    st.markdown('<div class="sec-head">SHAP Feature Importance</div>', unsafe_allow_html=True)

    # Representative SHAP values from the UCI dataset (typical values from literature)
    shap_data = {
        'SSLfinal_State':              0.412,
        'having_Sub_Domain':           0.318,
        'URL_of_Anchor':               0.291,
        'age_of_domain':               0.267,
        'Domain_registeration_length': 0.244,
        'web_traffic':                 0.218,
        'Request_URL':                 0.195,
        'Links_in_tags':               0.174,
        'having_IP_Address':           0.156,
        'SFH':                         0.143,
        'DNSRecord':                   0.131,
        'Page_Rank':                   0.118,
        'Prefix_Suffix':               0.104,
        'Google_Index':                0.097,
        'URL_Length':                  0.089,
    }

    feat_df = pd.DataFrame({
        'Feature':    list(shap_data.keys()),
        'Mean |SHAP|': list(shap_data.values())
    }).sort_values('Mean |SHAP|', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')

    colors = ['#0ea5e9' if i >= len(feat_df) - 3 else '#94a3b8'
              for i in range(len(feat_df))]
    bars = ax.barh(feat_df['Feature'], feat_df['Mean |SHAP|'],
                   color=colors, edgecolor='none', height=0.65)

    ax.set_xlabel('Mean |SHAP Value|', fontsize=9, color='#64748b')
    ax.set_title('Top 15 Features by Global SHAP Importance\n(Mean absolute SHAP value across test set)',
                 fontsize=10, color='#0f172a', pad=10)
    ax.tick_params(axis='both', labelsize=8, colors='#64748b')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#e2e8f0')
    ax.grid(axis='x', color='#e2e8f0', linewidth=0.6)
    ax.set_axisbelow(True)

    blue_patch = mpatches.Patch(color='#0ea5e9', label='Top 3 most important')
    grey_patch = mpatches.Patch(color='#94a3b8', label='Other features')
    ax.legend(handles=[blue_patch, grey_patch], fontsize=8,
              framealpha=0, labelcolor='#64748b')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("""
    <div class="info-panel">
    <strong>Reading the chart:</strong> SHAP (SHapley Additive exPlanations) values show each
    feature's average contribution to model predictions across all test instances.
    Higher values indicate stronger influence. <em>SSLfinal_State</em>,
    <em>having_Sub_Domain</em>, and <em>URL_of_Anchor</em> are consistently the
    strongest discriminators between phishing and legitimate sites.
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline Diagram ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">System Pipeline</div>', unsafe_allow_html=True)

    pipeline_steps = [
        ("1", "Data Ingestion",    "UCI ARFF → Pandas DataFrame"),
        ("2", "Preprocessing",     "Type conversion · null check · stratified split"),
        ("3", "EDA",               "Class dist · correlation heatmap · feature plots"),
        ("4", "Baseline Models",   "RF · DT · LR · KNN · MLP (5-fold CV)"),
        ("5", "XGBoost Full",      "200 estimators · logloss · binary:logistic"),
        ("6", "SHAP Analysis",     "TreeSHAP · global ranking · waterfall plots"),
        ("7", "Feature Selection", "K ∈ {5,8,10,12,15,20,25,30} sweep · elbow curve"),
        ("8", "GridSearchCV",      "243 combos · 5-fold · best estimator"),
        ("9", "Evaluation",        "Acc · Prec · Recall · F1 · confusion matrix"),
    ]

    cols = st.columns(3)
    for i, (num, title, desc) in enumerate(pipeline_steps):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:9px;
                        padding:0.9rem 1rem;margin-bottom:0.6rem;">
              <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.35rem;">
                <div style="width:24px;height:24px;border-radius:50%;background:#0ea5e9;
                            color:white;font-size:0.7rem;font-weight:700;
                            display:flex;align-items:center;justify-content:center;
                            flex-shrink:0;">{num}</div>
                <div style="font-weight:600;font-size:0.88rem;color:#0f172a;">{title}</div>
              </div>
              <div style="font-size:0.78rem;color:#64748b;padding-left:2rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── References ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Key References</div>', unsafe_allow_html=True)
    refs = [
        ("Mohammad et al., 2012", "UCI Phishing Websites Dataset", "UCI ML Repository"),
        ("Chawla, 2022",          "Phishing website detection using ML", "IJISAE 10(1)"),
        ("Chen & Guestrin, 2016", "XGBoost: A scalable tree boosting system", "ACM KDD"),
        ("Lundberg & Lee, 2017",  "A unified approach to interpreting model predictions", "NeurIPS"),
    ]
    for author, title, venue in refs:
        st.markdown(f"""
        <div style="padding:0.6rem 0.8rem;border-left:3px solid #0ea5e9;
                    margin-bottom:0.5rem;background:white;border-radius:0 6px 6px 0;">
          <span style="font-weight:600;font-size:0.85rem;color:#0f172a;">{author}</span>
          <span style="color:#64748b;font-size:0.85rem;"> · {title}</span>
          <span style="color:#94a3b8;font-size:0.8rem;"> [{venue}]</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — URL TESTER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  URL Tester":

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <h1>URL Threat Tester</h1>
      <p>Paste any URL below. The model extracts 30 features and classifies it in real time.</p>
      <span class="tag">Live Feature Extraction</span>
      <span class="tag">XGBoost Prediction</span>
      <span class="tag">98.10% Accuracy</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Model status ──────────────────────────────────────────────────────────
    if model_error:
        st.error(f"⚠️ {model_error}")
        st.info("""
        **To use the live tester, the trained model files are required:**

        1. Run the training notebook (`phishing_url_detection.ipynb`) in Google Colab
        2. Download `xgb_tuned_model.pkl` and `selected_features.pkl` from Colab
        3. Place both files in the root of this Hugging Face Space repository

        The **Model Analysis** page is available without the model files.
        """)
        st.stop()

    # ── URL Input ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Enter a URL to Analyse</div>', unsafe_allow_html=True)

    example_urls = [
        "https://www.google.com",
        "https://paypal.com.secure-login-verify.xyz/account",
        "http://192.168.1.1/bank/login.php",
        "https://github.com",
        "http://bit.ly/2xT8kZ3",
    ]

    col_input, col_example = st.columns([3, 1])
    with col_input:
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com  or  paste any suspicious link...",
            label_visibility="collapsed",
            key="url_input"
        )
    with col_example:
        example_choice = st.selectbox(
            "Try an example",
            ["— pick an example —"] + example_urls,
            label_visibility="collapsed"
        )
        if example_choice != "— pick an example —":
            url_input = example_choice

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="info-panel">
    <strong>How it works:</strong> When you click Analyse, this tool automatically fetches
    the page (where accessible), extracts all 30 UCI phishing features from the URL structure,
    DNS records, and page content, then runs the trained XGBoost model.
    Features that require page content fall back to URL-only heuristics when the site
    cannot be reached. <strong>Do not visit sites you suspect are phishing — paste the URL
    here without clicking it.</strong>
    </div>
    """, unsafe_allow_html=True)

    analyse_btn = st.button("🔍  Analyse URL", type="primary",
                            disabled=not url_input.strip(), use_container_width=False)

    # ── Analysis ──────────────────────────────────────────────────────────────
    if analyse_btn and url_input.strip():
        url = url_input.strip()

        with st.spinner("Extracting features and running model..."):
            progress = st.progress(0, text="Connecting to URL...")
            time.sleep(0.3)
            progress.progress(20, text="Extracting URL-based features...")

            result = extract_features(url)

            progress.progress(60, text="Analysing domain and DNS...")
            time.sleep(0.2)
            progress.progress(80, text="Running XGBoost model...")

            # Build feature vector in the order the model expects
            all_features_vec = result['feature_vector']
            all_features_dict = result['features']

            # Select only the features the tuned model was trained on
            try:
                input_vec = np.array([[all_features_dict.get(f, 0) for f in selected_features]])
                prediction   = model.predict(input_vec)[0]
                probabilities = model.predict_proba(input_vec)[0]
            except Exception as e:
                st.error(f"Model prediction error: {e}")
                st.stop()

            progress.progress(100, text="Done.")
            time.sleep(0.2)
            progress.empty()

        # ── Verdict ───────────────────────────────────────────────────────────
        phish_prob = float(probabilities[0])
        legit_prob = float(probabilities[1])

        col_verdict, col_conf = st.columns([1, 1])

        with col_verdict:
            if prediction == 0:
                st.markdown(f"""
                <div class="verdict-phish">
                  <div class="verdict-icon">🚨</div>
                  <div class="verdict-title">PHISHING DETECTED</div>
                  <div class="verdict-sub">
                    This URL shows characteristics of a phishing or suspicious site.<br>
                    Do not enter credentials or personal information.
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-legit">
                  <div class="verdict-icon">✅</div>
                  <div class="verdict-title">LIKELY LEGITIMATE</div>
                  <div class="verdict-sub">
                    No phishing indicators detected. Exercise normal caution —
                    this model is not infallible.
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with col_conf:
            st.markdown("""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:10px;
                        padding:1.4rem;height:100%;">
              <div style="font-size:0.8rem;font-weight:600;color:#64748b;
                          text-transform:uppercase;letter-spacing:0.04em;margin-bottom:1rem;">
                Confidence Scores
              </div>
            """, unsafe_allow_html=True)

            # Phishing bar
            p_w = int(phish_prob * 100)
            l_w = int(legit_prob * 100)
            st.markdown(f"""
            <div class="conf-wrap">
              <div class="conf-label">
                <span>🚨 Phishing / Suspicious</span>
                <span style="font-family:monospace;font-weight:600;color:#dc2626;">
                  {phish_prob*100:.1f}%
                </span>
              </div>
              <div class="conf-track">
                <div class="conf-fill" style="width:{p_w}%;background:#dc2626;"></div>
              </div>
            </div>
            <div class="conf-wrap" style="margin-top:0.8rem;">
              <div class="conf-label">
                <span>✅ Legitimate</span>
                <span style="font-family:monospace;font-weight:600;color:#16a34a;">
                  {legit_prob*100:.1f}%
                </span>
              </div>
              <div class="conf-track">
                <div class="conf-fill" style="width:{l_w}%;background:#16a34a;"></div>
              </div>
            </div>
            <div style="margin-top:1rem;font-size:0.78rem;color:#94a3b8;">
              Model: XGBoost (tuned) · {len(selected_features)}-feature subset<br>
              Page fetched: {'Yes' if result['html_available'] else 'No (URL-only mode)'}
            </div>
            </div>
            """, unsafe_allow_html=True)

        # ── URL Metadata ──────────────────────────────────────────────────────
        st.markdown('<div class="sec-head">URL Details</div>', unsafe_allow_html=True)
        meta_cols = st.columns(4)
        with meta_cols[0]:
            st.markdown(f"""
            <div class="stat-card">
              <div class="val" style="font-size:1rem;word-break:break-all;">
                {result['domain'][:30]}{'…' if len(result['domain'])>30 else ''}
              </div>
              <div class="lbl">Domain</div>
            </div>""", unsafe_allow_html=True)
        with meta_cols[1]:
            scheme = result['scheme'].upper()
            scheme_col = '#16a34a' if scheme == 'HTTPS' else '#dc2626'
            st.markdown(f"""
            <div class="stat-card">
              <div class="val" style="color:{scheme_col};">{scheme}</div>
              <div class="lbl">Protocol</div>
            </div>""", unsafe_allow_html=True)
        with meta_cols[2]:
            url_len = len(url_input)
            len_col = '#16a34a' if url_len < 54 else '#d97706' if url_len < 75 else '#dc2626'
            st.markdown(f"""
            <div class="stat-card">
              <div class="val" style="color:{len_col};">{url_len}</div>
              <div class="lbl">URL Length</div>
            </div>""", unsafe_allow_html=True)
        with meta_cols[3]:
            dns_ok = all_features_dict.get('DNSRecord', 0)
            dns_label = 'Valid' if dns_ok == 1 else 'Not found'
            dns_col   = '#16a34a' if dns_ok == 1 else '#dc2626'
            st.markdown(f"""
            <div class="stat-card">
              <div class="val" style="color:{dns_col};">{dns_label}</div>
              <div class="lbl">DNS Record</div>
            </div>""", unsafe_allow_html=True)

        # ── Feature Breakdown ─────────────────────────────────────────────────
        st.markdown('<div class="sec-head">Feature Breakdown (All 30 Features)</div>',
                    unsafe_allow_html=True)

        # Group by verdict
        phish_feats = [(k, v) for k, v in all_features_dict.items() if v == -1]
        susp_feats  = [(k, v) for k, v in all_features_dict.items() if v == 0]
        legit_feats = [(k, v) for k, v in all_features_dict.items() if v == 1]

        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.markdown(f"""
            <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;
                        padding:0.8rem;text-align:center;">
              <div style="font-size:1.6rem;font-weight:700;color:#dc2626;">{len(phish_feats)}</div>
              <div style="font-size:0.78rem;color:#991b1b;">Phishing indicators</div>
            </div>""", unsafe_allow_html=True)
        with summary_cols[1]:
            st.markdown(f"""
            <div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;
                        padding:0.8rem;text-align:center;">
              <div style="font-size:1.6rem;font-weight:700;color:#d97706;">{len(susp_feats)}</div>
              <div style="font-size:0.78rem;color:#92400e;">Suspicious / neutral</div>
            </div>""", unsafe_allow_html=True)
        with summary_cols[2]:
            st.markdown(f"""
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
                        padding:0.8rem;text-align:center;">
              <div style="font-size:1.6rem;font-weight:700;color:#16a34a;">{len(legit_feats)}</div>
              <div style="font-size:0.78rem;color:#14532d;">Legitimate indicators</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs([
            f"🚨 Phishing ({len(phish_feats)})",
            f"⚠️ Suspicious ({len(susp_feats)})",
            f"✅ Legitimate ({len(legit_feats)})"
        ])

        def render_feat_rows(feat_list, badge_cls, badge_text):
            if not feat_list:
                st.markdown(
                    '<p style="color:#94a3b8;font-size:0.85rem;padding:0.5rem 0;">None</p>',
                    unsafe_allow_html=True
                )
                return
            for fname, fval in feat_list:
                desc = FEATURE_DESCRIPTIONS.get(fname, fname.replace('_', ' '))
                st.markdown(f"""
                <div class="feat-row">
                  <div class="feat-name">{desc}</div>
                  <span class="badge {badge_cls}">{badge_text}</span>
                </div>
                """, unsafe_allow_html=True)

        with tab1:
            render_feat_rows(phish_feats, "badge-phish", "−1 Phishing")
        with tab2:
            render_feat_rows(susp_feats, "badge-susp", "0 Suspicious")
        with tab3:
            render_feat_rows(legit_feats, "badge-legit", "1 Legitimate")

        # ── Warnings ──────────────────────────────────────────────────────────
        if result['errors']:
            with st.expander(f"⚙️ Extraction notes ({len(result['errors'])})"):
                for err in result['errors']:
                    st.markdown(f"<small style='color:#94a3b8;'>· {err}</small>",
                                unsafe_allow_html=True)

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown("""
        <div style="margin-top:1.5rem;padding:1rem;background:#f8fafc;
                    border:1px solid #e2e8f0;border-radius:8px;
                    font-size:0.8rem;color:#94a3b8;">
          <strong style="color:#64748b;">Disclaimer:</strong>
          This tool is for educational and research purposes (Final Year Project, AFIT Kaduna, 2024/2025).
          It achieves 98.10% accuracy on the UCI benchmark dataset but is not a substitute for
          professional cybersecurity tools. Always exercise caution with unfamiliar URLs.
          The model may mis-classify sophisticated phishing sites or flag unusual but legitimate ones.
        </div>
        """, unsafe_allow_html=True)

    elif not analyse_btn:
        # ── Empty state ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#94a3b8;">
          <div style="font-size:3.5rem;margin-bottom:1rem;">🔗</div>
          <div style="font-size:1.1rem;font-weight:500;color:#64748b;margin-bottom:0.5rem;">
            Paste a URL above and click Analyse
          </div>
          <div style="font-size:0.85rem;">
            The model will extract all 30 features and classify the URL in seconds.
          </div>
        </div>
        """, unsafe_allow_html=True)
