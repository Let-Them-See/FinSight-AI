"""
FinSight AI — Prediction / Inference Module
==============================================
Production inference interface for all four FinSight AI platform modules.
Loads persisted models and preprocessors to score new, unseen inputs.
Designed for API integration and Streamlit dashboard consumption.

Author  : FinSight AI Team
Version : 1.0.0
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Logger ─────────────────────────────────────────────────────────────────────
logger   = logging.getLogger("FinSightAI.Predict")
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: Credit Risk — Single Applicant Scoring
# ─────────────────────────────────────────────────────────────────────────────

def predict_credit_risk(applicant_data: dict, model_name: str = "xgboost") -> dict:
    """
    Score a single loan applicant for default risk.

    Parameters
    ----------
    applicant_data : dict
        New applicant features, e.g.:
        {
          "applicant_age"        : 35,
          "annual_income_lakh"   : 12.5,
          "loan_amount_lakh"     : 8.0,
          "loan_tenure_months"   : 60,
          "credit_score"         : 720,
          "employment_type"      : "Salaried",
          "loan_purpose"         : "Home",
          ...
        }
    model_name : str
        Which trained model to use: "xgboost", "random_forest", "logistic_regression"

    Returns
    -------
    dict — {
        "default_probability"  : float (0.0–1.0),
        "risk_label"           : str  ("Low / Medium / High"),
        "recommended_action"   : str,
        "model_used"           : str,
    }
    """
    # ── Load preprocessor and model ───────────────────────────────────────────
    preprocessor_path = MODELS_DIR / "credit_risk_preprocessor.pkl"
    model_path        = MODELS_DIR / f"credit_risk_{model_name}.pkl"

    if not preprocessor_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Models not found. Please run the Credit Risk notebook or train_model.py first."
        )

    preprocessor = joblib.load(preprocessor_path)
    model        = joblib.load(model_path)

    # ── Convert input dict → DataFrame (single row) ───────────────────────────
    applicant_df = pd.DataFrame([applicant_data])

    # ── Apply same preprocessing as training ─────────────────────────────────
    X_transformed = preprocessor.transform(applicant_df)

    # ── Predict probability of default ────────────────────────────────────────
    default_prob = model.predict_proba(X_transformed)[0][1]

    # ── Map probability to risk tier (enterprise convention) ──────────────────
    if default_prob < 0.20:
        risk_label          = "Low Risk"
        recommended_action  = "Approve — standard terms applicable."
    elif default_prob < 0.45:
        risk_label          = "Medium Risk"
        recommended_action  = "Conditional approval — enhanced KYC and co-applicant recommended."
    else:
        risk_label          = "High Risk"
        recommended_action  = "Decline or refer to Risk Management Committee (RMC)."

    result = {
        "default_probability" : round(float(default_prob), 4),
        "risk_label"          : risk_label,
        "recommended_action"  : recommended_action,
        "model_used"          : model_name.replace("_", " ").title(),
    }

    logger.info(f"[Predict] Credit Risk scored: P(default)={default_prob:.4f} → {risk_label}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: Customer Segmentation — Single Customer Cluster Assignment
# ─────────────────────────────────────────────────────────────────────────────

# Segment descriptions for dashboard display
SEGMENT_PROFILES = {
    0: {
        "label"       : "Segment A — Mass Market Saver",
        "description" : "Young, salaried customers in Tier-2/3 cities. "
                        "Low income, moderate savings, low investment activity.",
        "strategy"    : "Cross-sell: SIP mutual funds, recurring deposits.",
    },
    1: {
        "label"       : "Segment B — Urban Aspirant",
        "description" : "Mid-income professionals in metros. "
                        "Active credit card users, moderate investment awareness.",
        "strategy"    : "Upsell: Stock trading platform, term insurance.",
    },
    2: {
        "label"       : "Segment C — Affluent Investor",
        "description" : "High-net-worth individuals with diversified portfolios. "
                        "Active in equity, MF, and real estate.",
        "strategy"    : "Premium wealth management, NRI investment products.",
    },
    3: {
        "label"       : "Segment D — Senior Wealth Preserver",
        "description" : "Retired/near-retirement customers. "
                        "Low risk appetite, FD and government bond preference.",
        "strategy"    : "Pension products, senior citizen FD schemes, health insurance.",
    },
}


def predict_customer_segment(customer_data: dict) -> dict:
    """
    Assign a new customer to the nearest K-Means cluster.

    Parameters
    ----------
    customer_data : dict — New customer feature values

    Returns
    -------
    dict — {
        "cluster_id"     : int,
        "segment_label"  : str,
        "description"    : str,
        "strategy"       : str,
    }
    """
    preprocessor_path = MODELS_DIR / "segmentation_preprocessor.pkl"
    model_path        = MODELS_DIR / "customer_segmentation_kmeans.pkl"

    if not preprocessor_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Segmentation model not found. Please run the Customer Segmentation notebook first."
        )

    preprocessor = joblib.load(preprocessor_path)
    kmeans       = joblib.load(model_path)

    customer_df   = pd.DataFrame([customer_data])
    X_transformed = preprocessor.transform(customer_df)

    cluster_id    = int(kmeans.predict(X_transformed)[0])
    profile       = SEGMENT_PROFILES.get(cluster_id, {
        "label": f"Segment {cluster_id}", "description": "N/A", "strategy": "N/A"
    })

    result = {
        "cluster_id"    : cluster_id,
        "segment_label" : profile["label"],
        "description"   : profile["description"],
        "strategy"      : profile["strategy"],
    }

    logger.info(f"[Predict] Customer assigned → Cluster {cluster_id}: {profile['label']}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: News Sentiment — Single Headline Scoring
# ─────────────────────────────────────────────────────────────────────────────

def predict_headline_sentiment(headline: str) -> dict:
    """
    Score a single financial news headline using VADER.

    Parameters
    ----------
    headline : str — Financial news headline text

    Returns
    -------
    dict — {
        "headline"             : str,
        "compound_score"       : float (-1.0 to +1.0),
        "sentiment_label"      : str   (Positive / Neutral / Negative),
        "market_signal"        : str,
        "neg", "neu", "pos"    : float (component scores)
    }
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import re

    # ── Clean input headline ──────────────────────────────────────────────────
    clean_text = re.sub(r"[^a-zA-Z0-9\s.,!?%-]", " ", headline).strip()

    analyzer = SentimentIntensityAnalyzer()
    scores   = analyzer.polarity_scores(clean_text)
    compound = scores["compound"]

    # ── Map to label and market signal ────────────────────────────────────────
    if compound >= 0.05:
        sentiment_label = "Positive"
        market_signal   = "📈 Bullish — consider accumulating related sector stocks."
    elif compound <= -0.05:
        sentiment_label = "Negative"
        market_signal   = "📉 Bearish — exercise caution; risk-off positioning advised."
    else:
        sentiment_label = "Neutral"
        market_signal   = "⚖️ Neutral — no strong directional signal."

    result = {
        "headline"       : headline,
        "compound_score" : round(compound, 4),
        "sentiment_label": sentiment_label,
        "market_signal"  : market_signal,
        "neg"            : scores["neg"],
        "neu"            : scores["neu"],
        "pos"            : scores["pos"],
    }

    logger.info(f"[Predict] Sentiment: '{headline[:60]}...' → {sentiment_label} ({compound:.4f})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: Time Series — Extended Forecast
# ─────────────────────────────────────────────────────────────────────────────

def predict_financial_forecast(periods_ahead: int = 6) -> pd.DataFrame:
    """
    Generate a financial metric forecast using the trained Prophet model.

    Parameters
    ----------
    periods_ahead : int — Number of future months to forecast (default: 6)

    Returns
    -------
    pd.DataFrame — Forecast table with columns: [month, forecast, lower, upper]
    """
    model_path = MODELS_DIR / "prophet_forecaster.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "Prophet model not found. Please run the Time Series notebook first."
        )

    model  = joblib.load(model_path)
    future = model.make_future_dataframe(periods=periods_ahead, freq="MS")
    fc     = model.predict(future)

    # ── Extract only future forecast rows ─────────────────────────────────────
    forecast_rows = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods_ahead).copy()
    forecast_rows = forecast_rows.rename(columns={
        "ds"         : "Month",
        "yhat"       : "Forecast (₹ Cr)",
        "yhat_lower" : "Lower Bound (₹ Cr)",
        "yhat_upper" : "Upper Bound (₹ Cr)",
    })
    forecast_rows["Month"] = forecast_rows["Month"].dt.strftime("%b %Y")

    # ── Round to 2 decimal places for boardroom readability ──────────────────
    for col in ["Forecast (₹ Cr)", "Lower Bound (₹ Cr)", "Upper Bound (₹ Cr)"]:
        forecast_rows[col] = forecast_rows[col].round(2)

    logger.info(f"[Predict] Forecast generated for {periods_ahead} months.")
    return forecast_rows.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO — run to test inference pipeline end-to-end
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  FinSight AI — Inference Demo")
    print("="*60)

    # ── Sentiment demo (no model file needed) ────────────────────────────────
    demo_headlines = [
        "RBI hikes repo rate by 25 bps amid persistent inflation concerns",
        "HDFC Bank Q3 net profit jumps 33%, beats analyst estimates",
        "SEBI cracks down on P-note misuse, markets fall 2%",
    ]
    print("\n📰 NEWS SENTIMENT DEMO")
    for headline in demo_headlines:
        result = predict_headline_sentiment(headline)
        print(f"\n  Headline : {result['headline'][:70]}...")
        print(f"  Sentiment: {result['sentiment_label']} (compound={result['compound_score']})")
        print(f"  Signal   : {result['market_signal']}")
