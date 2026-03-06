"""
FinSight AI — Data Loader Module
=================================
Enterprise-grade data loading utilities for the FinSight AI platform.
Handles ingestion of all four dataset modules with validation,
type coercion, and logging — production-ready for BFSI environments.

Author  : FinSight AI Team
Platform: TCS iON / Infosys Nia compatible
Version : 1.0.0
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# ── Configure module-level logger ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FinSightAI.DataLoader")

# ── Base directory resolution ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: Credit Risk Dataset Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_credit_risk_data(filepath: str = None) -> pd.DataFrame:
    """
    Load and validate the credit risk / loan default dataset.

    Parameters
    ----------
    filepath : str, optional
        Absolute or relative path to the CSV file.
        Defaults to data/credit_risk_data.csv.

    Returns
    -------
    pd.DataFrame
        Validated credit risk DataFrame with correct dtypes.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    # Resolve filepath — fall back to default data directory
    target_path = Path(filepath) if filepath else DATA_DIR / "credit_risk_data.csv"

    if not target_path.exists():
        logger.error(f"Credit risk data not found at: {target_path}")
        raise FileNotFoundError(f"File not found: {target_path}")

    logger.info(f"[Module 1] Loading credit risk data from: {target_path}")
    df = pd.read_csv(target_path)

    # ── Type enforcement for BFSI compliance ─────────────────────────────────
    categorical_cols = [
        "applicant_gender", "employment_type", "loan_purpose",
        "property_ownership", "credit_grade", "geographic_tier",
        "state_name", "default_flag"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # ── Log dataset summary ───────────────────────────────────────────────────
    logger.info(f"[Module 1] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"[Module 1] Default rate: {df['default_flag'].value_counts(normalize=True).get('Yes', 0):.2%}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: Customer Segmentation Dataset Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_customer_segmentation_data(filepath: str = None) -> pd.DataFrame:
    """
    Load and validate the customer segmentation dataset.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to data/customer_segments_data.csv.

    Returns
    -------
    pd.DataFrame
        Validated customer segmentation DataFrame.
    """
    target_path = Path(filepath) if filepath else DATA_DIR / "customer_segments_data.csv"

    if not target_path.exists():
        logger.error(f"Segmentation data not found at: {target_path}")
        raise FileNotFoundError(f"File not found: {target_path}")

    logger.info(f"[Module 2] Loading customer segmentation data from: {target_path}")
    df = pd.read_csv(target_path)

    # ── Enforce categorical types ─────────────────────────────────────────────
    categorical_cols = [
        "customer_city_tier", "investment_preference", "primary_bank",
        "occupation_category", "risk_appetite", "gender"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    logger.info(f"[Module 2] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: Financial News / Sentiment Dataset Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_financial_news_data(filepath: str = None) -> pd.DataFrame:
    """
    Load and validate the financial news sentiment dataset.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to data/financial_news_data.csv.

    Returns
    -------
    pd.DataFrame
        Validated news sentiment DataFrame with parsed datetime index.
    """
    target_path = Path(filepath) if filepath else DATA_DIR / "financial_news_data.csv"

    if not target_path.exists():
        logger.error(f"News data not found at: {target_path}")
        raise FileNotFoundError(f"File not found: {target_path}")

    logger.info(f"[Module 3] Loading financial news data from: {target_path}")
    df = pd.read_csv(target_path, parse_dates=["publication_date"])

    # ── Sort chronologically for time-aware splits ────────────────────────────
    df = df.sort_values("publication_date").reset_index(drop=True)

    # ── Category enforcement ──────────────────────────────────────────────────
    categorical_cols = ["news_source", "sector", "sentiment_label", "market_impact"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    logger.info(f"[Module 3] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"[Module 3] Date range: {df['publication_date'].min().date()} → {df['publication_date'].max().date()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: Time Series / Forecasting Dataset Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_time_series_data(filepath: str = None) -> pd.DataFrame:
    """
    Load and validate the monthly financial time series dataset.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to data/time_series_data.csv.

    Returns
    -------
    pd.DataFrame
        Validated time series DataFrame indexed by month.
    """
    target_path = Path(filepath) if filepath else DATA_DIR / "time_series_data.csv"

    if not target_path.exists():
        logger.error(f"Time series data not found at: {target_path}")
        raise FileNotFoundError(f"File not found: {target_path}")

    logger.info(f"[Module 4] Loading time series data from: {target_path}")
    df = pd.read_csv(target_path, parse_dates=["month"])

    # ── Set month as index for time-series operations ─────────────────────────
    df = df.sort_values("month").reset_index(drop=True)
    df["month"] = pd.to_datetime(df["month"])

    logger.info(f"[Module 4] Dataset loaded: {df.shape[0]} monthly records")
    logger.info(f"[Module 4] Period: {df['month'].min().strftime('%b %Y')} → {df['month'].max().strftime('%b %Y')}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: Dataset Health Check
# ─────────────────────────────────────────────────────────────────────────────

def run_data_health_check(df: pd.DataFrame, module_name: str) -> dict:
    """
    Run a quick data quality audit on any loaded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to audit.
    module_name : str
        Human-readable name of the module (for logging).

    Returns
    -------
    dict
        Health report with missing values, duplicate counts, and dtypes.
    """
    logger.info(f"[HealthCheck] Running audit on {module_name}...")

    # ── Compute health metrics ────────────────────────────────────────────────
    missing_counts = df.isnull().sum()
    missing_pct    = (missing_counts / len(df) * 100).round(2)
    duplicate_rows = df.duplicated().sum()

    health_report = {
        "module"           : module_name,
        "total_rows"       : df.shape[0],
        "total_columns"    : df.shape[1],
        "duplicate_rows"   : int(duplicate_rows),
        "columns_with_nulls": int((missing_counts > 0).sum()),
        "null_summary"     : missing_pct[missing_pct > 0].to_dict(),
        "memory_usage_MB"  : round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }

    # ── Log key findings ──────────────────────────────────────────────────────
    logger.info(f"[HealthCheck] {module_name}: {health_report['total_rows']:,} rows, "
                f"{health_report['total_columns']} cols, "
                f"{health_report['duplicate_rows']} duplicates, "
                f"{health_report['columns_with_nulls']} null columns")

    return health_report


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run directly to verify all datasets load correctly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  FinSight AI — Data Loader Self-Test")
    print("="*60)

    loaders = [
        (load_credit_risk_data,          "Module 1: Credit Risk"),
        (load_customer_segmentation_data, "Module 2: Customer Segmentation"),
        (load_financial_news_data,        "Module 3: Financial News"),
        (load_time_series_data,           "Module 4: Time Series"),
    ]

    for loader_fn, name in loaders:
        try:
            df = loader_fn()
            report = run_data_health_check(df, name)
            print(f"\n✅ {name}")
            print(f"   Rows: {report['total_rows']:,} | Cols: {report['total_columns']}")
            print(f"   Memory: {report['memory_usage_MB']} MB | Duplicates: {report['duplicate_rows']}")
        except FileNotFoundError:
            print(f"\n⚠️  {name}: Data file not yet generated. Run generate_data.py first.")
