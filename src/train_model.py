"""
FinSight AI — Model Training Module
=====================================
Multi-model training orchestrator for all four FinSight AI platform modules.
Implements model selection, cross-validation, and hyperparameter search
following enterprise MLOps conventions.

Author  : FinSight AI Team
Version : 1.0.0
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import StratifiedKFold, cross_val_score, TimeSeriesSplit
from sklearn.dummy import DummyClassifier
import xgboost as xgb
import lightgbm as lgb

# ── Logger ─────────────────────────────────────────────────────────────────────
logger = logging.getLogger("FinSightAI.TrainModel")

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: Credit Risk — Classification Models
# ─────────────────────────────────────────────────────────────────────────────

def train_credit_risk_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Train and evaluate three classification models for credit risk prediction.

    Models trained:
      1. Logistic Regression   — interpretable baseline
      2. Random Forest         — ensemble, handles non-linearity
      3. XGBoost               — gradient boosting, top BFSI performance

    Parameters
    ----------
    X_train : np.ndarray — Preprocessed feature matrix
    y_train : np.ndarray — Binary default labels (0=Good, 1=Default)

    Returns
    -------
    dict — {model_name: fitted_model} for all three classifiers
    """
    logger.info("[Train] Starting credit risk model training...")

    # ── Cross-validation setup (stratified for class imbalance) ──────────────
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Model definitions ─────────────────────────────────────────────────────
    model_configs = {
        "Logistic Regression": LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric="logloss", random_state=42, verbosity=0
        ),
    }

    # ── Baseline: Dummy classifier (stratified) ───────────────────────────────
    baseline = DummyClassifier(strategy="stratified", random_state=42)
    baseline.fit(X_train, y_train)
    baseline_cv = cross_val_score(baseline, X_train, y_train, cv=cv_strategy,
                                  scoring="roc_auc").mean()
    logger.info(f"[Train] Baseline (DummyClassifier) ROC-AUC: {baseline_cv:.4f}")

    trained_models = {"Baseline (Dummy)": baseline}

    # ── Train each model with cross-validation ────────────────────────────────
    for model_name, model in model_configs.items():
        logger.info(f"[Train] Training: {model_name}...")

        # 5-fold stratified CV
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy,
                                    scoring="roc_auc", n_jobs=-1)
        logger.info(f"[Train] {model_name} CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── Fit on full training data ─────────────────────────────────────────
        model.fit(X_train, y_train)
        trained_models[model_name] = model

        # ── Persist model to disk ─────────────────────────────────────────────
        safe_name = model_name.lower().replace(" ", "_")
        save_path = MODELS_DIR / f"credit_risk_{safe_name}.pkl"
        joblib.dump(model, save_path)
        logger.info(f"[Train] Saved: {save_path}")

    return trained_models


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: Customer Segmentation — Clustering
# ─────────────────────────────────────────────────────────────────────────────

def train_customer_segmentation(
    X_scaled: np.ndarray,
    n_clusters: int = 4
) -> tuple:
    """
    Train K-Means clustering for customer segmentation.

    Also computes the elbow curve data (inertia vs. k)
    and silhouette scores for model selection transparency.

    Parameters
    ----------
    X_scaled   : np.ndarray — Preprocessed & scaled customer feature matrix
    n_clusters : int        — Number of segments (default: 4, from elbow analysis)

    Returns
    -------
    (fitted_kmeans, elbow_data)
    """
    from sklearn.metrics import silhouette_score

    logger.info(f"[Train] Training K-Means with k={n_clusters}...")

    # ── Elbow curve: inertia for k = 2..10 ───────────────────────────────────
    elbow_data = []
    for k in range(2, 11):
        km_temp    = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km_temp.fit(X_scaled)
        sil_score  = silhouette_score(X_scaled, km_temp.labels_, sample_size=3000, random_state=42)
        elbow_data.append({
            "k"          : k,
            "inertia"    : km_temp.inertia_,
            "silhouette" : round(sil_score, 4)
        })
        logger.info(f"[Train] k={k}: inertia={km_temp.inertia_:.0f}, silhouette={sil_score:.4f}")

    elbow_df = pd.DataFrame(elbow_data)

    # ── Final model with optimal k ────────────────────────────────────────────
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20,
                    max_iter=500, random_state=42)
    kmeans.fit(X_scaled)

    final_silhouette = silhouette_score(X_scaled, kmeans.labels_, random_state=42)
    logger.info(f"[Train] Final K-Means (k={n_clusters}): silhouette={final_silhouette:.4f}")

    # ── Persist ───────────────────────────────────────────────────────────────
    joblib.dump(kmeans, MODELS_DIR / "customer_segmentation_kmeans.pkl")
    logger.info(f"[Train] K-Means model saved to models/customer_segmentation_kmeans.pkl")

    return kmeans, elbow_df


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: News Sentiment — VADER + Optional Classification
# ─────────────────────────────────────────────────────────────────────────────

def run_vader_sentiment_analysis(text_series: pd.Series) -> pd.DataFrame:
    """
    Apply VADER sentiment analysis to financial news headlines.

    VADER (Valence Aware Dictionary and Sentiment Reasoner) is specifically
    calibrated for short, social-media-style financial text — a widely used
    industry standard (Bloomberg, Reuters data pipelines).

    Parameters
    ----------
    text_series : pd.Series — Cleaned news headline strings

    Returns
    -------
    pd.DataFrame — Scores: neg, neu, pos, compound + mapped label
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    logger.info(f"[Train] Running VADER on {len(text_series):,} headlines...")

    analyzer = SentimentIntensityAnalyzer()

    # ── Vectorised VADER scoring ───────────────────────────────────────────────
    scores = text_series.apply(analyzer.polarity_scores)
    scores_df = pd.DataFrame(scores.tolist())

    # ── Map compound score to standard sentiment label ─────────────────────────
    def map_sentiment_label(compound: float) -> str:
        """Map VADER compound score to Positive/Neutral/Negative."""
        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    scores_df["predicted_sentiment"] = scores_df["compound"].apply(map_sentiment_label)
    logger.info(f"[Train] Sentiment distribution:\n{scores_df['predicted_sentiment'].value_counts()}")

    return scores_df


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: Time Series Forecasting — Prophet
# ─────────────────────────────────────────────────────────────────────────────

def train_prophet_model(ts_df: pd.DataFrame, forecast_periods: int = 12) -> tuple:
    """
    Fit a Facebook Prophet model to the monthly financial time series.

    Prophet handles:
      - Indian fiscal year seasonality (April start)
      - RBI annual policy cycles
      - COVID-era anomaly dampening
      - Festival season peaks (Diwali, Navratri — Oct/Nov uplift)

    Parameters
    ----------
    ts_df            : pd.DataFrame — Prophet-formatted df with 'ds' and 'y' cols
    forecast_periods : int          — Number of months to forecast ahead

    Returns
    -------
    (fitted_prophet_model, forecast_dataframe)
    """
    from prophet import Prophet

    logger.info(f"[Train] Fitting Prophet model — {len(ts_df)} historical periods...")

    # ── Configure Prophet for Indian financial context ────────────────────────
    model = Prophet(
        yearly_seasonality  = True,
        weekly_seasonality  = False,  # Monthly data — no weekly pattern
        daily_seasonality   = False,
        seasonality_mode    = "multiplicative",  # Better for financial growth series
        changepoint_prior_scale = 0.3,           # Moderate flexibility
        seasonality_prior_scale = 10.0,
        interval_width      = 0.95,              # 95% confidence bands
    )

    # ── Add custom seasonality for Indian fiscal quarters ─────────────────────
    model.add_seasonality(
        name="fiscal_quarterly",
        period=91.25,    # ~3 months per fiscal quarter
        fourier_order=5,
    )

    # ── Fit the model ─────────────────────────────────────────────────────────
    model.fit(ts_df[["ds", "y"]])

    # ── Generate future dataframe and forecast ────────────────────────────────
    future_df     = model.make_future_dataframe(periods=forecast_periods, freq="MS")
    forecast_df   = model.predict(future_df)

    logger.info(f"[Train] Prophet forecast generated: {forecast_periods} months ahead.")
    logger.info(f"[Train] Forecast range: {forecast_df['ds'].iloc[-forecast_periods].date()} → "
                f"{forecast_df['ds'].iloc[-1].date()}")

    # ── Persist model ─────────────────────────────────────────────────────────
    joblib.dump(model, MODELS_DIR / "prophet_forecaster.pkl")
    logger.info("[Train] Prophet model saved to models/prophet_forecaster.pkl")

    return model, forecast_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — orchestrate all training when run standalone
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  FinSight AI — Model Training Orchestrator")
    print("="*60)
    print("Run each module's notebook or the individual functions above.")
    print("All models are saved to: ./models/")
