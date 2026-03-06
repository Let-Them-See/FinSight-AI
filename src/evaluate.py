"""
FinSight AI — Model Evaluation Module
=======================================
Comprehensive evaluation suite for all four FinSight AI platform modules.
Generates classification reports, ROC curves, confusion matrices, silhouette
scores, and MAPE metrics — boardroom-ready output aligned with TCS iON and
HDFC Analytics standards.

Author  : FinSight AI Team
Version : 1.0.0
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, accuracy_score, mean_absolute_percentage_error
)
from sklearn.metrics import silhouette_score

# ── Brand colour palette ───────────────────────────────────────────────────────
PALETTE = {
    "background" : "#EEE9DF",
    "surface"    : "#C9C1B1",
    "dark_base"  : "#2C3B4D",
    "accent"     : "#FFB162",
    "highlight"  : "#A35139",
    "deep_dark"  : "#CD5C5C",
}

# ── Global matplotlib theme ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : PALETTE["background"],
    "axes.facecolor"   : PALETTE["background"],
    "axes.edgecolor"   : PALETTE["dark_base"],
    "axes.labelcolor"  : PALETTE["deep_dark"],
    "xtick.color"      : PALETTE["deep_dark"],
    "ytick.color"      : PALETTE["deep_dark"],
    "text.color"       : PALETTE["deep_dark"],
    "grid.color"       : PALETTE["surface"],
    "font.family"      : "DejaVu Sans",
})

logger     = logging.getLogger("FinSightAI.Evaluate")
BASE_DIR   = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: Credit Risk — Classification Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_credit_risk_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list = None,
    save_plots: bool = True
) -> pd.DataFrame:
    """
    Evaluate all credit risk classifiers and generate comparison table.

    Metrics computed:
      - Accuracy, F1-Score (macro), ROC-AUC
      - Confusion Matrix per model
      - Combined ROC curve plot

    Parameters
    ----------
    models        : dict        — {model_name: fitted_model}
    X_test        : np.ndarray  — Test feature matrix
    y_test        : np.ndarray  — True binary labels
    feature_names : list        — Column names (for feature importance)
    save_plots    : bool        — Whether to save figures to /assets

    Returns
    -------
    pd.DataFrame — Model comparison table sorted by ROC-AUC (descending)
    """
    logger.info("[Evaluate] Running credit risk model evaluation...")

    results_rows = []

    # ── Combined ROC curve figure ─────────────────────────────────────────────
    fig_roc, ax_roc = plt.subplots(figsize=(9, 6))
    ax_roc.set_facecolor(PALETTE["background"])
    ax_roc.plot([0, 1], [0, 1], "--", color=PALETTE["surface"],
                linewidth=1.5, label="Chance Level (AUC = 0.50)")

    roc_colors = [PALETTE["dark_base"], PALETTE["accent"], PALETTE["highlight"]]

    for idx, (model_name, model) in enumerate(models.items()):
        # ── Predict probabilities ─────────────────────────────────────────────
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test).astype(float)

        y_pred    = (y_prob >= 0.5).astype(int)
        accuracy  = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred, average="macro", zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_prob)

        results_rows.append({
            "Model"    : model_name,
            "Accuracy" : round(accuracy * 100, 2),
            "F1-Score" : round(f1 * 100, 2),
            "ROC-AUC"  : round(roc_auc, 4),
        })

        logger.info(f"[Evaluate] {model_name}: Acc={accuracy:.3f} F1={f1:.3f} AUC={roc_auc:.4f}")

        # ── Plot ROC curve (skip dummy) ───────────────────────────────────────
        if "Dummy" not in model_name and "Baseline" not in model_name:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            color = roc_colors[min(idx, len(roc_colors) - 1)]
            ax_roc.plot(fpr, tpr, linewidth=2.5, color=color,
                        label=f"{model_name}  (AUC = {roc_auc:.3f})")

    # ── ROC curve aesthetics ──────────────────────────────────────────────────
    ax_roc.set_xlabel("False Positive Rate", fontsize=12, color=PALETTE["deep_dark"])
    ax_roc.set_ylabel("True Positive Rate", fontsize=12, color=PALETTE["deep_dark"])
    ax_roc.set_title("ROC Curves — Credit Risk Models\nFinSight AI | Module 1",
                     fontsize=14, fontweight="bold", color=PALETTE["deep_dark"], pad=15)
    ax_roc.legend(loc="lower right", fontsize=10, framealpha=0.85,
                  facecolor=PALETTE["background"])
    ax_roc.grid(True, alpha=0.4, color=PALETTE["surface"])
    plt.tight_layout()

    if save_plots:
        fig_roc.savefig(ASSETS_DIR / "credit_risk_roc_curves.png", dpi=150, bbox_inches="tight")
        logger.info("[Evaluate] ROC curve saved: assets/credit_risk_roc_curves.png")
    plt.close(fig_roc)

    # ── Build and return results DataFrame ────────────────────────────────────
    results_df = pd.DataFrame(results_rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    logger.info(f"[Evaluate] Best model: {results_df.iloc[0]['Model']} (AUC={results_df.iloc[0]['ROC-AUC']})")
    return results_df


def plot_confusion_matrix(model, model_name: str, X_test: np.ndarray, y_test: np.ndarray):
    """
    Plot a styled confusion matrix for a given classifier.

    Parameters
    ----------
    model      : fitted sklearn classifier
    model_name : str         — Display name for the chart title
    X_test     : np.ndarray  — Test feature matrix
    y_test     : np.ndarray  — True labels
    """
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(PALETTE["background"])

    # ── Custom colormap from palette ──────────────────────────────────────────
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list(
        "finsight", [PALETTE["background"], PALETTE["dark_base"]]
    )

    sns.heatmap(cm, annot=True, fmt="d", cmap=custom_cmap, ax=ax,
                linewidths=0.5, linecolor=PALETTE["surface"],
                xticklabels=["Good Loan", "Default"],
                yticklabels=["Good Loan", "Default"],
                cbar_kws={"shrink": 0.8})

    ax.set_xlabel("Predicted Label", fontsize=11, color=PALETTE["deep_dark"])
    ax.set_ylabel("True Label",      fontsize=11, color=PALETTE["deep_dark"])
    ax.set_title(f"Confusion Matrix — {model_name}\nFinSight AI | Module 1",
                 fontsize=13, fontweight="bold", color=PALETTE["deep_dark"], pad=12)
    ax.tick_params(colors=PALETTE["deep_dark"])

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    fig.savefig(ASSETS_DIR / f"confusion_matrix_{safe_name}.png", dpi=150, bbox_inches="tight")
    logger.info(f"[Evaluate] Confusion matrix saved: assets/confusion_matrix_{safe_name}.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: Customer Segmentation — Cluster Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_customer_segmentation(
    kmeans_model,
    X_scaled: np.ndarray,
    elbow_df: pd.DataFrame,
    save_plots: bool = True
) -> dict:
    """
    Evaluate K-Means clustering quality and visualise elbow curve.

    Metrics:
      - Final silhouette score
      - Inertia (within-cluster sum of squares)
      - Elbow curve plot

    Parameters
    ----------
    kmeans_model : fitted KMeans   — Final clustering model
    X_scaled     : np.ndarray     — Preprocessed feature matrix
    elbow_df     : pd.DataFrame   — k vs inertia/silhouette data
    save_plots   : bool           — Save elbow chart to /assets

    Returns
    -------
    dict — Cluster quality metrics
    """
    labels           = kmeans_model.labels_
    final_silhouette = silhouette_score(X_scaled, labels, random_state=42)
    inertia          = kmeans_model.inertia_

    logger.info(f"[Evaluate] K-Means Silhouette Score: {final_silhouette:.4f}")
    logger.info(f"[Evaluate] K-Means Inertia: {inertia:.0f}")

    # ── Elbow curve + Silhouette dual plot ────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE["background"])

    for ax in [ax1, ax2]:
        ax.set_facecolor(PALETTE["background"])
        ax.grid(True, alpha=0.4, color=PALETTE["surface"])

    # Elbow curve
    ax1.plot(elbow_df["k"], elbow_df["inertia"], marker="o", linewidth=2.5,
             color=PALETTE["dark_base"], markersize=8, markerfacecolor=PALETTE["accent"])
    ax1.axvline(kmeans_model.n_clusters, color=PALETTE["highlight"],
                linestyle="--", linewidth=2, label=f"Optimal k={kmeans_model.n_clusters}")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=11, color=PALETTE["deep_dark"])
    ax1.set_ylabel("Inertia (WCSS)",         fontsize=11, color=PALETTE["deep_dark"])
    ax1.set_title("Elbow Curve — Optimal k Selection\nModule 2: Customer Segmentation",
                  fontsize=12, fontweight="bold", color=PALETTE["deep_dark"])
    ax1.legend(fontsize=10, facecolor=PALETTE["background"])

    # Silhouette scores
    colors_bar = [PALETTE["accent"] if k == kmeans_model.n_clusters else PALETTE["dark_base"]
                  for k in elbow_df["k"]]
    ax2.bar(elbow_df["k"].astype(str), elbow_df["silhouette"], color=colors_bar,
            edgecolor=PALETTE["deep_dark"], linewidth=0.5)
    ax2.set_xlabel("Number of Clusters (k)", fontsize=11, color=PALETTE["deep_dark"])
    ax2.set_ylabel("Silhouette Score",        fontsize=11, color=PALETTE["deep_dark"])
    ax2.set_title("Silhouette Scores by k\nModule 2: Customer Segmentation",
                  fontsize=12, fontweight="bold", color=PALETTE["deep_dark"])

    plt.tight_layout()
    if save_plots:
        fig.savefig(ASSETS_DIR / "segmentation_elbow_silhouette.png", dpi=150, bbox_inches="tight")
        logger.info("[Evaluate] Elbow/silhouette plot saved.")
    plt.close(fig)

    return {
        "n_clusters"       : int(kmeans_model.n_clusters),
        "silhouette_score" : round(final_silhouette, 4),
        "inertia"          : round(inertia, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: Time Series — Forecast Error Metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_forecast(
    actual: pd.Series,
    predicted: pd.Series,
    model_name: str = "Prophet"
) -> dict:
    """
    Compute forecast accuracy metrics for the time series module.

    Parameters
    ----------
    actual     : pd.Series — Observed values
    predicted  : pd.Series — Forecasted values (aligned index)
    model_name : str       — Model name for logging

    Returns
    -------
    dict — MAPE, RMSE, MAE, and R² metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # ── Align and drop NaN pairs ──────────────────────────────────────────────
    combined = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()

    mape = mean_absolute_percentage_error(combined["actual"], combined["predicted"]) * 100
    rmse = np.sqrt(mean_squared_error(combined["actual"], combined["predicted"]))
    mae  = mean_absolute_error(combined["actual"], combined["predicted"])
    r2   = r2_score(combined["actual"], combined["predicted"])

    metrics = {
        "model"  : model_name,
        "MAPE_%" : round(mape, 2),
        "RMSE"   : round(rmse, 2),
        "MAE"    : round(mae, 2),
        "R²"     : round(r2, 4),
    }

    logger.info(f"[Evaluate] {model_name} Forecast → MAPE={mape:.2f}% RMSE={rmse:.2f} R²={r2:.4f}")
    return metrics
