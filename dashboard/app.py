"""
FinSight AI — Streamlit Dashboard (Main Entry Point)
======================================================
Multi-module financial intelligence dashboard for FinSight AI.

Run with:
    streamlit run dashboard/app.py

Author  : FinSight AI Team
Version : 1.1.0 (bugfix: sys import, CSS readability)
"""

# ── CRITICAL: import sys and pathlib FIRST — needed by all module pages ────────
import sys
import pathlib
import importlib.util
import traceback

# ── Add BOTH src/ and dashboard/ to sys.path ─────────────────────────────────
_DASHBOARD_DIR = str(pathlib.Path(__file__).resolve().parent)
_SRC_DIR       = str(pathlib.Path(__file__).resolve().parent.parent / "src")
for _d in [_DASHBOARD_DIR, _SRC_DIR]:
    if _d not in sys.path:
        sys.path.insert(0, _d)


def _load_page(module_name: str):
    """Load a dashboard page by module name from the page_modules/ subfolder.
    Uses importlib spec-loading so it works regardless of cwd.
    Shows a readable error instead of a blank page if something fails.
    NOTE: Folder is named 'page_modules/' (not 'pages/') to prevent
    Streamlit from auto-discovering them as MPA navigation pages.
    """
    page_path = pathlib.Path(__file__).resolve().parent / "page_modules" / f"{module_name}.py"
    try:
        spec   = importlib.util.spec_from_file_location(module_name, page_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.render()
    except FileNotFoundError:
        import streamlit as _st
        _st.error(f"⚠️ Page file not found: `{page_path}`")
    except Exception:
        import streamlit as _st
        _st.error(f"⚠️ Error loading **{module_name}**. See details below.")
        _st.code(traceback.format_exc(), language="python")


import streamlit as st

# ── Page configuration — must be FIRST Streamlit call ────────────────────────
st.set_page_config(
    page_title    = "FinSight AI | Financial Intelligence Platform",
    page_icon     = "₹",
    layout        = "wide",
    initial_sidebar_state = "expanded",
    menu_items    = {
        "Get Help"    : None,
        "Report a bug": None,
        "About"       : "FinSight AI v1.1 — Enterprise Financial Intelligence Platform",
    }
)

# ── Brand Colour Palette ──────────────────────────────────────────────────────
PALETTE = {
    "background" : "#EEE9DF",
    "surface"    : "#C9C1B1",
    "dark_base"  : "#2C3B4D",
    "accent"     : "#FFB162",
    "highlight"  : "#A35139",
    "deep_dark"  : "#CD5C5C",
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Root variables ──────────────────────────────────────────────────────── */
:root {{
    --primary-color: {PALETTE["accent"]};
    --background-color: {PALETTE["background"]};
    --secondary-background-color: {PALETTE["surface"]};
    --text-color: {PALETTE["deep_dark"]};
}}

/* ── App background ──────────────────────────────────────────────────────── */
.stApp {{
    background-color: {PALETTE["background"]};
    color: {PALETTE["deep_dark"]};
}}

/* ── READABILITY: all generic text ──────────────────────────────────────── */
p, span, div, li, label {{
    color: {PALETTE["deep_dark"]};
}}

/* ── READABILITY: widget labels ─────────────────────────────────────────── */
[data-testid="stWidgetLabel"] > label,
[data-testid="stWidgetLabel"] p,
.stSlider > label,
.stSelectbox > label,
.stNumberInput > label,
.stTextArea > label,
.stRadio > label {{
    color: {PALETTE["deep_dark"]} !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}}

/* ── READABILITY: caption / help text ───────────────────────────────────── */
[data-testid="stCaptionContainer"] p,
.stCaption {{
    color: {PALETTE["dark_base"]} !important;
    font-size: 0.82rem !important;
}}

/* ── READABILITY: selectbox dropdown text ───────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div {{
    background-color: {PALETTE["dark_base"]};
    color: {PALETTE["background"]} !important;
    border-radius: 8px;
    border: 1px solid {PALETTE["surface"]};
}}
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] p,
.stSelectbox [data-baseweb="select"] div {{
    color: {PALETTE["background"]} !important;
}}

/* ── READABILITY: number input ──────────────────────────────────────────── */
.stNumberInput input {{
    background-color: {PALETTE["dark_base"]};
    color: {PALETTE["background"]} !important;
    border-radius: 8px;
    border: 1px solid {PALETTE["surface"]};
    font-size: 1rem;
}}
.stNumberInput [data-testid="stNumberInputContainer"] {{
    background-color: {PALETTE["dark_base"]};
    border-radius: 8px;
}}

/* ── READABILITY: text area ─────────────────────────────────────────────── */
.stTextArea textarea {{
    background-color: {PALETTE["dark_base"]};
    color: {PALETTE["background"]} !important;
    border-radius: 8px;
    border: 1px solid {PALETTE["surface"]};
    font-size: 0.95rem;
}}

/* ── READABILITY: slider value labels ───────────────────────────────────── */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
.stSlider p {{
    color: {PALETTE["deep_dark"]} !important;
}}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {PALETTE["dark_base"]};
    border-right: 2px solid {PALETTE["accent"]};
}}
[data-testid="stSidebar"] * {{
    color: {PALETTE["background"]} !important;
}}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] > label {{
    color: {PALETTE["background"]} !important;
}}

/* ── Metric cards ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background-color: {PALETTE["surface"]};
    border-radius: 12px;
    padding: 1rem;
    border-left: 4px solid {PALETTE["accent"]};
}}
[data-testid="stMetricLabel"] {{ color: {PALETTE["deep_dark"]} !important; font-weight: 600; }}
[data-testid="stMetricValue"] {{ color: {PALETTE["dark_base"]} !important; font-size: 1.8rem; }}
[data-testid="stMetricDelta"] {{ font-size: 0.8rem; }}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {{
    background-color: {PALETTE["accent"]};
    color: {PALETTE["deep_dark"]};
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-size: 1rem;
    transition: background-color 0.2s ease, transform 0.1s ease;
}}
.stButton > button:hover {{
    background-color: {PALETTE["highlight"]};
    color: {PALETTE["background"]};
    transform: translateY(-1px);
}}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab"] {{
    background: {PALETTE["surface"]};
    border-radius: 8px 8px 0 0;
    color: {PALETTE["deep_dark"]} !important;
    font-weight: 600;
    padding: 0.5rem 1rem;
}}
.stTabs [aria-selected="true"] {{
    background: {PALETTE["dark_base"]} !important;
    color: {PALETTE["background"]} !important;
}}
.stTabs [data-baseweb="tab"] p {{
    color: inherit !important;
}}

/* ── DataFrames ───────────────────────────────────────────────────────────── */
.stDataFrame {{ border-radius: 8px; }}

/* ── Typography ───────────────────────────────────────────────────────────── */
h1, h2, h3, h4 {{ color: {PALETTE["deep_dark"]} !important; }}
hr {{ border-color: {PALETTE["surface"]}; margin: 1rem 0; }}

/* ── Alerts / info boxes ──────────────────────────────────────────────────── */
.stAlert {{ border-radius: 8px; }}

/* ── Cards ────────────────────────────────────────────────────────────────── */
.finsight-card {{
    background-color: {PALETTE["surface"]};
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid {PALETTE["accent"]};
}}
.finsight-header {{
    background: linear-gradient(135deg, {PALETTE["dark_base"]}, {PALETTE["deep_dark"]});
    color: {PALETTE["background"]};
    padding: 2rem 2.5rem;
    border-radius: 14px;
    margin-bottom: 1.8rem;
    border-left: 5px solid {PALETTE["accent"]};
    box-shadow: 0 4px 24px rgba(27,38,50,0.35);
}}

/* ── Radio buttons in sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] p {{
    color: {PALETTE["background"]} !important;
}}
[data-testid="stSidebar"] .stRadio label span {{
    color: {PALETTE["background"]} !important;
}}

/* ── HIDE Streamlit toolbar: Deploy button + ⋮ menu + footer ─────────────── */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{
    display: none !important;
}}
#MainMenu {{
    visibility: hidden !important;
    display: none !important;
}}
footer {{
    visibility: hidden !important;
    display: none !important;
}}
header[data-testid="stHeader"] {{
    background: transparent !important;
    height: 0rem !important;
    min-height: 0rem !important;
    padding: 0 !important;
}}

/* ── Sidebar Brand Block ──────────────────────────────────────────────────── */
.finsight-brand {{
    text-align: center;
    padding: 1.5rem 0.5rem 1rem 0.5rem;
    border-bottom: 1px solid rgba(255,177,98,0.3);
    margin-bottom: 0.6rem;
}}
.finsight-brand-icon {{
    font-size: 4rem;
    line-height: 1;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 0 10px rgba(255,177,98,0.55));
    display: block;
}}
.finsight-brand-name {{
    font-size: 2.1rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.5px !important;
    color: {PALETTE["accent"]} !important;
    margin: 0 0 0 0 !important;
    text-shadow: 0 0 22px rgba(255,177,98,0.5) !important;
    line-height: 1.15 !important;
}}
.finsight-brand-bar {{
    width: 54px;
    height: 3px;
    background: linear-gradient(90deg, {PALETTE["highlight"]}, {PALETTE["accent"]});
    border-radius: 2px;
    margin: 0.55rem auto 0.5rem auto;
}}
.finsight-brand-sub {{
    font-size: 0.68rem !important;
    color: {PALETTE["surface"]} !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}}
</style>
""", unsafe_allow_html=True)




# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="finsight-brand">
        <div class="finsight-brand-icon">₹</div>
        <h1 class="finsight-brand-name">FinSight AI</h1>
        <div class="finsight-brand-bar"></div>
        <p class="finsight-brand-sub">Financial Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<p style='color:{PALETTE['accent']}; font-weight:700; font-size:0.8rem;'>SELECT MODULE</p>",
                unsafe_allow_html=True)

    selected_module = st.radio(
        label   = "",
        options = [
            "🏠  Overview",
            "📊  Module 1 — Credit Risk",
            "👥  Module 2 — Customer Segments",
            "📰  Module 3 — News Sentiment",
            "📈  Module 4 — Forecasting",
        ],
        index   = 0,
        label_visibility = "collapsed"
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="color:{PALETTE['surface']}; font-size:0.72rem; text-align:center;">
        <b style="color:{PALETTE['accent']};">FinSight AI v1.1</b><br>
        Enterprise Analytics Suite<br>
        Indian BFSI Context<br><br>
        <i>Powered by scikit-learn,<br>XGBoost, Prophet &amp; VADER</i>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE ROUTER
# ─────────────────────────────────────────────────────────────────────────────

if "Overview" in selected_module:
    # ── OVERVIEW PAGE ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="finsight-header">
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.6rem;">
            <span style="font-size:3.5rem; filter:drop-shadow(0 0 12px rgba(255,177,98,0.6));">₹</span>
            <div>
                <h1 style="background:linear-gradient(90deg, #FFB162, #FFECD0);
                           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                           background-clip:text;
                           margin:0; font-size:3rem; font-weight:900;
                           letter-spacing:0.5px; line-height:1.1;
                           filter:drop-shadow(0 0 16px rgba(255,177,98,0.55));">FinSight AI</h1>
                <p style="color:#EEE9DF; margin:0; font-size:1.05rem; font-weight:500;">
                    Multi-Dimensional Financial Intelligence Platform
                </p>
            </div>
        </div>
        <div style="width:80px; height:3px;
                    background:linear-gradient(90deg,{PALETTE['highlight']},{PALETTE['accent']});
                    border-radius:2px; margin:0.4rem 0;"></div>
        <p style="color:#FFB162; margin:0.4rem 0 0 0; font-size:0.85rem; letter-spacing:0.5px;">
            Enterprise Analytics Suite &nbsp;|&nbsp; Indian BFSI Context &nbsp;|&nbsp; TCS / HDFC / Infosys Grade
        </p>
    </div>
    """, unsafe_allow_html=True)


    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📊 ML Modules", "4",    "Credit Risk · Segmentation · NLP · Forecasting")
    k2.metric("🗂️ Training Data", "35,000+", "Rows across all modules")
    k3.metric("🤖 Models Built", "6+",  "Classifiers · Clusters · Prophet")
    k4.metric("🎯 Top Model AUC", "~0.89", "XGBoost — Credit Risk")

    st.markdown("---")
    st.subheader("🗂️ Platform Modules")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="finsight-card">
            <h3 style="color:{PALETTE['dark_base']}; margin:0;">📊 Module 1: Credit Risk Scoring</h3>
            <p style="color:{PALETTE['deep_dark']}; font-size:0.9rem; margin:0.5rem 0 0 0;">
            Predict loan default probability for new applicants using XGBoost,
            Random Forest, and Logistic Regression. Includes SHAP explainability
            and risk-tier classification aligned with RBI NBFC norms.
            </p>
            <p style="color:{PALETTE['dark_base']}; margin:0.5rem 0 0 0;"><b>Metrics:</b> ROC-AUC · F1-Score · Confusion Matrix</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="finsight-card">
            <h3 style="color:{PALETTE['dark_base']}; margin:0;">📰 Module 3: Sentiment Analysis</h3>
            <p style="color:{PALETTE['deep_dark']}; font-size:0.9rem; margin:0.5rem 0 0 0;">
            Score Indian financial news headlines (Economic Times, Mint, Business Standard)
            using VADER NLP — classify as Bullish / Neutral / Bearish for
            market intelligence and investment signals.
            </p>
            <p style="color:{PALETTE['dark_base']}; margin:0.5rem 0 0 0;"><b>Output:</b> Compound Score · Sentiment Label · Market Signal</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="finsight-card">
            <h3 style="color:{PALETTE['dark_base']}; margin:0;">👥 Module 2: Customer Segmentation</h3>
            <p style="color:{PALETTE['deep_dark']}; font-size:0.9rem; margin:0.5rem 0 0 0;">
            K-Means clustering to identify 4 distinct customer segments:
            Mass Market, Urban Aspirant, Affluent Investor, and Senior Preserver.
            Drives targeted product cross-sell and retention strategies.
            </p>
            <p style="color:{PALETTE['dark_base']}; margin:0.5rem 0 0 0;"><b>Metrics:</b> Silhouette Score · Inertia · t-SNE Visualisation</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="finsight-card">
            <h3 style="color:{PALETTE['dark_base']}; margin:0;">📈 Module 4: Financial Forecasting</h3>
            <p style="color:{PALETTE['deep_dark']}; font-size:0.9rem; margin:0.5rem 0 0 0;">
            Prophet-based time series forecasting of monthly loan disbursements
            (₹ Crores) with Indian fiscal year seasonality, COVID-era anomaly
            handling, and 12-month forward projections with 95% confidence bands.
            </p>
            <p style="color:{PALETTE['dark_base']}; margin:0.5rem 0 0 0;"><b>Metrics:</b> MAPE · RMSE · R² · Forecast Horizon</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="background:{PALETTE['dark_base']}; border-radius:10px; padding:1rem 1.5rem; color:{PALETTE['background']};">
        <b style="color:{PALETTE['accent']};">📌 How to Use This Dashboard</b><br><br>
        1. Use the <b>left sidebar</b> to navigate between modules.<br>
        2. Each module has an <b>interactive prediction panel</b> — enter inputs and click to score.<br>
        3. Charts update in real-time based on your filters.<br>
        4. Data is from the <b>synthetic Indian BFSI dataset</b> (15,000 rows per module).<br>
        5. Run <code>generate_data.py</code> and all 4 notebooks before using this dashboard.
    </div>
    """, unsafe_allow_html=True)

elif "Credit Risk" in selected_module:
    _load_page("credit_risk_page")

elif "Customer Segments" in selected_module:
    _load_page("segmentation_page")

elif "News Sentiment" in selected_module:
    _load_page("sentiment_page")

elif "Forecasting" in selected_module:
    _load_page("forecasting_page")

