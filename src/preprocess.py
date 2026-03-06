"""
FinSight AI — Data Preprocessing Module
=========================================
Centralised preprocessing pipeline with BFSI-grade transformations:
null imputation, outlier capping, feature encoding, and scaling.
All transformers are fit on training data and applied to test/live data,
following MLOps best practices for production deployment.

Author  : FinSight AI Team
Version : 1.0.0
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path

# ── Logger setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger("FinSightAI.Preprocess")

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: Credit Risk Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_credit_risk(
    df: pd.DataFrame,
    target_col: str = "default_flag",
    fit_mode: bool = True,
    transformer_path: str = None
) -> tuple:
    """
    Full preprocessing pipeline for the credit risk dataset.

    Steps:
      1. Drop irrelevant identifiers
      2. Cap outliers using IQR winsorisation
      3. Impute missing values (median for numeric, mode for categorical)
      4. Encode categorical features (ordinal encoding)
      5. Standard-scale numeric features

    Parameters
    ----------
    df            : pd.DataFrame — Raw credit risk data
    target_col    : str         — Name of the target column
    fit_mode      : bool        — True = fit+transform; False = transform only
    transformer_path : str      — Path to saved transformer (used in predict mode)

    Returns
    -------
    (X_processed, y, fitted_transformer)
    """
    logger.info("[Preprocess] Starting credit risk preprocessing...")

    df = df.copy()

    # ── Separate target variable ──────────────────────────────────────────────
    y = (df[target_col] == "Yes").astype(int)
    df = df.drop(columns=[target_col, "applicant_id"], errors="ignore")

    # ── Define feature groups ─────────────────────────────────────────────────
    numeric_features = [
        "applicant_age", "annual_income_lakh", "loan_amount_lakh",
        "loan_tenure_months", "credit_score", "existing_emis",
        "debt_to_income_ratio", "total_assets_lakh", "num_credit_accounts",
        "months_since_last_default", "employment_years"
    ]
    categorical_features = [
        "applicant_gender", "employment_type", "loan_purpose",
        "property_ownership", "credit_grade", "geographic_tier", "state_name"
    ]

    # ── Filter to only existing columns ──────────────────────────────────────
    numeric_features     = [c for c in numeric_features     if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    # ── Cap outliers at 1st–99th percentile (Winsorisation) ──────────────────
    for col in numeric_features:
        p01 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=p01, upper=p99)
        logger.debug(f"[Preprocess] Outlier cap applied: {col} → [{p01:.2f}, {p99:.2f}]")

    # ── Build sklearn ColumnTransformer pipeline ──────────────────────────────
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline,     numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ], remainder="drop")

    if fit_mode:
        X_processed = preprocessor.fit_transform(df)
        # ── Persist transformer for predict.py ───────────────────────────────
        save_path = transformer_path or str(MODELS_DIR / "credit_risk_preprocessor.pkl")
        joblib.dump(preprocessor, save_path)
        logger.info(f"[Preprocess] Transformer saved to: {save_path}")
    else:
        load_path = transformer_path or str(MODELS_DIR / "credit_risk_preprocessor.pkl")
        preprocessor = joblib.load(load_path)
        X_processed = preprocessor.transform(df)

    # ── Reconstruct feature names for interpretability ────────────────────────
    feature_names = numeric_features + categorical_features

    logger.info(f"[Preprocess] Final feature matrix: {X_processed.shape[0]:,} rows × {X_processed.shape[1]} features")
    return X_processed, y, preprocessor, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: Customer Segmentation Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_customer_segmentation(
    df: pd.DataFrame,
    fit_mode: bool = True,
    transformer_path: str = None
) -> tuple:
    """
    Preprocessing pipeline for customer segmentation (unsupervised).

    Steps:
      1. Remove identifiers
      2. Encode categoricals
      3. Scale for K-Means compatibility

    Parameters
    ----------
    df               : pd.DataFrame — Raw customer segmentation data
    fit_mode         : bool         — True = fit+transform; False = transform only
    transformer_path : str          — Path to saved transformer

    Returns
    -------
    (X_scaled, fitted_preprocessor, feature_names)
    """
    logger.info("[Preprocess] Starting customer segmentation preprocessing...")

    df = df.copy().drop(columns=["customer_id"], errors="ignore")

    numeric_features = [
        "customer_age", "monthly_income_lakh", "monthly_expenses_lakh",
        "total_savings_lakh", "total_investments_lakh", "total_debt_lakh",
        "num_products_held", "account_tenure_months", "avg_monthly_transaction_count",
        "digital_engagement_score", "credit_utilization_pct"
    ]
    categorical_features = [
        "customer_city_tier", "investment_preference", "primary_bank",
        "occupation_category", "risk_appetite", "gender"
    ]

    numeric_features     = [c for c in numeric_features     if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline,     numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ], remainder="drop")

    if fit_mode:
        X_scaled  = preprocessor.fit_transform(df)
        save_path = transformer_path or str(MODELS_DIR / "segmentation_preprocessor.pkl")
        joblib.dump(preprocessor, save_path)
        logger.info(f"[Preprocess] Transformer saved: {save_path}")
    else:
        load_path = transformer_path or str(MODELS_DIR / "segmentation_preprocessor.pkl")
        preprocessor = joblib.load(load_path)
        X_scaled = preprocessor.transform(df)

    feature_names = numeric_features + categorical_features
    logger.info(f"[Preprocess] Final matrix: {X_scaled.shape[0]:,} rows × {X_scaled.shape[1]} features")
    return X_scaled, preprocessor, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: Financial News Preprocessing (NLP)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_news_text(text_series: pd.Series) -> pd.Series:
    """
    Clean and normalise financial news headlines for NLP analysis.

    Operations:
      - Lowercase normalisation
      - Strip special characters and excess whitespace
      - Remove URLs, HTML tags, and numeric-only tokens

    Parameters
    ----------
    text_series : pd.Series — Raw news headline strings

    Returns
    -------
    pd.Series — Cleaned text ready for VADER / transformer inference
    """
    import re

    logger.info(f"[Preprocess] Cleaning {len(text_series):,} news headlines...")

    def clean_single_text(text: str) -> str:
        """Apply all cleaning operations to a single headline."""
        if not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove currency symbols while keeping numbers (for financial context)
        text = re.sub(r"[₹$€£]", " ", text)
        # Remove special punctuation except basic sentence markers
        text = re.sub(r"[^a-zA-Z0-9\s.,!?%-]", " ", text)
        # Collapse multiple whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    cleaned = text_series.apply(clean_single_text)
    logger.info(f"[Preprocess] Text cleaning complete. Avg length: {cleaned.str.len().mean():.0f} chars")
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: Time Series Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_time_series(
    df: pd.DataFrame,
    value_col: str = "total_loan_disbursement_cr",
    date_col:  str = "month"
) -> pd.DataFrame:
    """
    Prepare time series data for Prophet and LSTM modelling.

    Steps:
      1. Rename to Prophet-compatible ds/y columns
      2. Handle missing months via forward-fill
      3. Add lag features and rolling statistics

    Parameters
    ----------
    df        : pd.DataFrame — Raw time series data
    value_col : str          — Target metric column (in Crores)
    date_col  : str          — Date column name

    Returns
    -------
    pd.DataFrame — Prophet-ready dataframe with additional features
    """
    logger.info(f"[Preprocess] Preparing time series for column: {value_col}")

    df = df[[date_col, value_col]].copy()
    df = df.rename(columns={date_col: "ds", value_col: "y"})
    df["ds"] = pd.to_datetime(df["ds"])

    # ── Ensure monthly frequency with no gaps ─────────────────────────────────
    full_idx = pd.date_range(start=df["ds"].min(), end=df["ds"].max(), freq="MS")
    df = df.set_index("ds").reindex(full_idx).rename_axis("ds").reset_index()
    df["y"] = df["y"].ffill()

    # ── Add calendar features ─────────────────────────────────────────────────
    df["quarter"]         = df["ds"].dt.quarter
    df["fiscal_quarter"]  = ((df["ds"].dt.month - 4) % 12 // 3 + 1)  # India FY starts April
    df["year"]            = df["ds"].dt.year

    # ── Lag and rolling window features ──────────────────────────────────────
    df["lag_1m"]          = df["y"].shift(1)
    df["lag_3m"]          = df["y"].shift(3)
    df["lag_12m"]         = df["y"].shift(12)
    df["rolling_3m_avg"]  = df["y"].rolling(3).mean()
    df["rolling_12m_avg"] = df["y"].rolling(12).mean()
    df["yoy_growth_pct"]  = df["y"].pct_change(12) * 100  # Year-over-Year

    df = df.dropna(subset=["lag_12m"]).reset_index(drop=True)

    logger.info(f"[Preprocess] Time series ready: {len(df)} periods, {df['ds'].min().date()} → {df['ds'].max().date()}")
    return df
