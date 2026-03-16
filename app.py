"""
app.py — Be Thrifty · Streamlit UI
-------------------------------------
Run with:  streamlit run app.py

Requires:
    models/price_predictor.pkl
    models/encoders.pkl
    (run src/data_prep.py then src/train.py first)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

from src.data_prep import (
    LUXURY_BRANDS,
    MID_BRANDS,
    FAST_FASHION_BRANDS,
    BRAND_TIER_RESALE_RATIOS,
    CANONICAL_CATEGORIES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Feature order must match train.py FEATURE_COLS exactly — do not reorder.
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "brand", "category", "condition", "size",
    "brand_tier", "is_vintage", "season",
    "retail_price_ratio",
]

MODEL_PATH     = "models/price_predictor.pkl"
ENCODERS_PATH  = "models/encoders.pkl"
PROCESSED_PATH = "data/processed/fashion_training_data.csv"

CUSTOM_BRAND = "Other / Custom brand"
TOP_N_BRANDS = 200

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Be Thrifty",
    page_icon="🪡",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset & global ───────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #F8F7F4;
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #1C1C1C !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: #E8E3DC !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 400 !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    font-size: 0.8rem !important;
    color: #A09890 !important;
    line-height: 1.7 !important;
}
[data-testid="stSidebar"] code {
    background: #2C2C2C !important;
    color: #C4B8A8 !important;
    border-radius: 4px !important;
    font-size: 0.75rem !important;
}
[data-testid="stSidebar"] hr {
    border-color: #2E2E2E !important;
    margin: 1.25rem 0 !important;
}
[data-testid="stSidebarContent"] {
    padding: 2rem 1.5rem !important;
}

/* ── Main content area ────────────────────────────────────────────────────── */
.main .block-container {
    padding: 2.5rem 3rem 3rem !important;
    max-width: 820px !important;
}

/* ── Page header ──────────────────────────────────────────────────────────── */
.page-header {
    padding: 0 0 2rem 0;
    border-bottom: 1px solid #E8E3DC;
    margin-bottom: 2rem;
}
.brand-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8rem;
    font-weight: 400;
    color: #1C1C1C;
    letter-spacing: -0.01em;
    line-height: 1;
    margin: 0 0 0.4rem 0;
}
.brand-name span {
    color: #9B8B7A;
}
.tagline {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 300;
    color: #8A8480;
    letter-spacing: 0.04em;
    margin: 0;
}

/* ── Section heading ──────────────────────────────────────────────────────── */
.section-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #9B8B7A;
    margin: 0 0 1.1rem 0;
}

/* ── Form card ────────────────────────────────────────────────────────────── */
[data-testid="stForm"] {
    background: #FFFFFF !important;
    border-radius: 20px !important;
    border: 1px solid #EAE6E0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04), 0 8px 28px rgba(0,0,0,0.05) !important;
    padding: 2rem 2.25rem 2.25rem !important;
}

/* ── Input labels ─────────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #6B6560 !important;
}

/* ── Input controls ───────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input,
[data-testid="stNumberInput"] > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #E8E3DC !important;
    background: #FDFCFB !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    color: #1C1C1C !important;
    transition: border-color 0.15s ease !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stTextInput"] > div > div > input:focus,
[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: #9B8B7A !important;
    box-shadow: 0 0 0 3px rgba(155, 139, 122, 0.12) !important;
}

/* ── Field-row gap ────────────────────────────────────────────────────────── */
[data-testid="stForm"] [data-testid="column"] {
    padding-right: 0.6rem !important;
}
[data-testid="stForm"] [data-testid="column"]:last-child {
    padding-right: 0 !important;
}

/* ── Divider inside form ──────────────────────────────────────────────────── */
.form-divider {
    border: none;
    border-top: 1px solid #F0EBE5;
    margin: 1.5rem 0 1.25rem;
}

/* ── Submit button ────────────────────────────────────────────────────────── */
[data-testid="stFormSubmitButton"] > button {
    background: #1C1C1C !important;
    color: #F8F7F4 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    margin-top: 1.25rem !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
    box-shadow: 0 2px 12px rgba(28, 28, 28, 0.18) !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: #2E2E2E !important;
    transform: translateY(-1px) !important;
}
[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Result card ──────────────────────────────────────────────────────────── */
.result-card {
    background: #1C1C1C;
    border-radius: 20px;
    padding: 2.5rem 2.25rem 2rem;
    margin-top: 1.75rem;
    border: 1px solid #2A2A2A;
    box-shadow: 0 4px 32px rgba(0,0,0,0.12);
}
.result-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #6B6560;
    margin: 0 0 0.6rem 0;
}
.result-price {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4.5rem;
    font-weight: 300;
    color: #F8F7F4;
    line-height: 1;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.01em;
}
.result-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 300;
    color: #5A5550;
    margin: 0 0 1.5rem 0;
}
.result-divider {
    border: none;
    border-top: 1px solid #2A2A2A;
    margin: 0 0 1.25rem 0;
}
.tags-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.tag {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #252525;
    border: 1px solid #333333;
    border-radius: 6px;
    padding: 0.3rem 0.75rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 400;
    color: #C4B8A8;
}
.tag-key {
    color: #5A5550;
    font-weight: 500;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Warning / error ──────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-size: 0.82rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Prediction helpers (pipeline unchanged) ───────────────────────────────────

def get_brand_tier(brand: str) -> str:
    b = brand.strip().lower()
    if b in LUXURY_BRANDS:     return "luxury"
    if b in MID_BRANDS:        return "mid"
    if b in FAST_FASHION_BRANDS: return "fast_fashion"
    return "unknown"


def get_season() -> str:
    month = datetime.now().month
    if month in (12, 1, 2): return "winter"
    if month in (3, 4, 5):  return "spring"
    if month in (6, 7, 8):  return "summer"
    return "autumn"


def safe_encode(le, value: str) -> int:
    if value in le.classes_:
        return int(le.transform([value])[0])
    return 0


def build_feature_row(
    brand: str,
    category: str,
    condition: str,
    size: str,
    year: int,
    retail_price: float | None,
) -> pd.DataFrame:
    brand_tier = get_brand_tier(brand)
    is_vintage = 1 if year < 2005 else 0
    season     = get_season()

    retail_price_ratio = (
        BRAND_TIER_RESALE_RATIOS.get(brand_tier, 0.18)
        if retail_price and retail_price > 0
        else 1.0
    )

    row = {
        "brand"              : safe_encode(encoders["brand"],      brand.strip().lower()),
        "category"           : safe_encode(encoders["category"],   category.strip().lower()),
        "condition"          : safe_encode(encoders["condition"],  condition.strip().lower()),
        "size"               : safe_encode(encoders["size"],       size.strip().lower()),
        "brand_tier"         : safe_encode(encoders["brand_tier"], brand_tier),
        "is_vintage"         : is_vintage,
        "season"             : safe_encode(encoders["season"],     season),
        "retail_price_ratio" : retail_price_ratio,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model    = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        return model, encoders
    except FileNotFoundError as e:
        return None, str(e)


model, encoders = load_artifacts()


@st.cache_data
def get_top_brands(n: int = TOP_N_BRANDS) -> list[str]:
    """
    Return the top-n most frequent brands from the training data, sorted
    alphabetically, with CUSTOM_BRAND appended as the final option.

    The processed CSV stores brand as integer codes (LabelEncoder output),
    so value_counts() on that column gives exact training-set frequencies.
    The top codes are then inverse-transformed back to readable brand names.

    Falls back to all encoder classes (alphabetical) if the processed file
    is missing, so the UI still works before the pipeline has been run.
    """
    le = encoders["brand"]

    if os.path.exists(PROCESSED_PATH):
        brand_codes = pd.read_csv(PROCESSED_PATH, usecols=["brand"])["brand"]
        top_codes   = brand_codes.value_counts().head(n).index.tolist()
        top_names   = sorted(le.inverse_transform(top_codes).tolist())
    else:
        top_names = sorted(le.classes_.tolist())

    return top_names + [CUSTOM_BRAND]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <h2 style='font-family: Cormorant Garamond, serif; font-size: 1.4rem;
               font-weight: 400; color: #E8E3DC; margin: 0 0 0.25rem;'>
        Be Thrifty
    </h2>
    <p style='font-size: 0.72rem; color: #5A5550; margin: 0 0 1.5rem;
              letter-spacing: 0.04em;'>
        resale intelligence
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if model is not None:
        st.markdown("### Model")

        col_a, col_b = st.columns(2)
        with col_a:
            trees = getattr(model, "n_estimators", "—")
            st.metric("Trees", trees)
        with col_b:
            st.metric("Features", len(FEATURE_COLS))

        if hasattr(model, "feature_importances_"):
            st.markdown("---")
            st.markdown("### Feature Importance")

            imp_df = (
                pd.DataFrame({
                    "Feature"    : FEATURE_COLS,
                    "Importance" : model.feature_importances_,
                })
                .sort_values("Importance", ascending=True)
                .set_index("Feature")
            )
            st.bar_chart(imp_df, color="#9B8B7A", height=280)

    st.markdown("---")
    st.markdown("### Auto-computed")
    st.markdown("""
- **brand\\_tier** — luxury / mid / fast fashion
- **is\\_vintage** — true if year < 2005
- **season** — from today's month
- **retail\\_price\\_ratio** — tier-based estimate
    """)

    st.markdown("---")
    st.markdown("### Pipeline")
    st.code(
        "python src/data_prep.py\npython src/train.py\nstreamlit run app.py",
        language="bash",
    )


# ── Guard: model missing ──────────────────────────────────────────────────────
if model is None:
    st.error(
        f"**Model or encoders not found.**\n\n`{encoders}`\n\n"
        "Run the pipeline first:\n```bash\npython src/data_prep.py\npython src/train.py\n```"
    )
    st.stop()


# ── Build dropdown options ────────────────────────────────────────────────────
# Category: use the controlled canonical list defined in data_prep.py so the
# UI options always match exactly what the model was trained on.
category_options  = CANONICAL_CATEGORIES
condition_options = list(encoders["condition"].classes_)
size_options      = list(encoders["size"].classes_)

# Brand dropdown: all trained classes (alphabetical) + a custom fallback at the end.
# Streamlit's selectbox supports keyboard search natively — users can start typing
# to filter the list without any extra configuration.
brand_options = get_top_brands()


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1 class="brand-name">Be <span>Thrifty</span></h1>
    <p class="tagline">AI-powered resale value prediction for thrifted fashion.</p>
</div>
""", unsafe_allow_html=True)


# ── Item input form ───────────────────────────────────────────────────────────
with st.form("predict_form"):
    st.markdown('<p class="section-label">Item Details</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Category", options=category_options)
    with col2:
        brand = st.selectbox(
            "Brand",
            brand_options,
            index=None,
            placeholder="Search or select brand...",
        )

    col3, col4 = st.columns(2)
    with col3:
        condition = st.selectbox("Condition", options=condition_options)
    with col4:
        size = st.selectbox("Size", options=size_options)

    st.markdown('<hr class="form-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Additional Info</p>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        year = st.number_input(
            "Year Made",
            min_value=1900,
            max_value=datetime.now().year,
            value=2015,
            step=1,
            help="Items made before 2005 are automatically flagged as vintage.",
        )
    with col6:
        retail_price = st.number_input(
            "Original Retail Price ($)",
            min_value=0.0,
            value=0.0,
            step=5.0,
            help="Optional. Leave at 0 if unknown.",
        )

    submitted = st.form_submit_button("Get Resale Estimate")


# ── Prediction result ─────────────────────────────────────────────────────────
if submitted:
    # When the custom fallback is chosen, treat the brand as unknown so
    # safe_encode gracefully falls back and brand_tier returns "unknown".
    # None  → user left the placeholder; treat as unknown
    # CUSTOM_BRAND → explicit "other" selection
    if brand is None or brand == CUSTOM_BRAND:
        brand_value = "unknown"
        brand_label = "Custom / Other"
    else:
        brand_value = brand
        brand_label = brand.title()

    brand_tier = get_brand_tier(brand_value)
    is_vintage = 1 if year < 2005 else 0
    season     = get_season()
    rp_input   = retail_price if retail_price > 0 else None

    X     = build_feature_row(brand_value, category, condition, size, year, rp_input)
    price = max(0.50, round(float(np.expm1(model.predict(X)[0])), 2))

    tier_label    = brand_tier.replace("_", " ")
    vintage_label = "vintage" if is_vintage else "contemporary"

    st.markdown(f"""
    <div class="result-card">
        <p class="result-label">Estimated Resale Value</p>
        <p class="result-price">${price:,.2f}</p>
        <p class="result-sub">
            Based on resale market patterns for similar pre-loved pieces.
        </p>
        <hr class="result-divider">
        <div class="tags-row">
            <span class="tag">
                <span class="tag-key">brand</span>
                {brand_label}
            </span>
            <span class="tag">
                <span class="tag-key">tier</span>
                {tier_label}
            </span>
            <span class="tag">
                <span class="tag-key">season</span>
                {season}
            </span>
            <span class="tag">
                <span class="tag-key">era</span>
                {vintage_label}
            </span>
            <span class="tag">
                <span class="tag-key">year</span>
                {int(year)}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Estimates are model predictions, not appraisals. "
        "Actual resale prices vary by platform, listing quality, and timing."
    )
