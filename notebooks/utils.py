"""Evaluation utilities for binary scoring models.

Kept minimal:
- ROC: two variants (matplotlib via sklearn, plotly), each shows just curve + AUC.
- threshold_table: per-threshold metrics.
- lift_table / plot_lift: two views of the same lift/gain analysis.
- calculate_psi: Population Stability Index between two distributions.
- plot_psi_density: overlay two score distributions with PSI badge in the centre.

Usage:

    from utils import plot_roc_mpl, plot_roc_plotly, threshold_table, lift_table, plot_lift

    plot_roc_mpl(y_val, y_proba)              # matplotlib
    fig = plot_roc_plotly(y_val, y_proba); fig.show()   # plotly

    threshold_table(y_val, y_proba)
    lift_table(y_val, y_proba, n_bins=10)
    plot_lift(y_val, y_proba).show()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder


# ----------------------------------------------------------------------
# ROC — matplotlib (sklearn native)
# ----------------------------------------------------------------------
def plot_roc_mpl(y_true, y_proba, ax=None, name: str = "Model", class_names=None):
    """ROC curve via sklearn.RocCurveDisplay. Matplotlib classic, just curve + AUC.

    Binary  : ``y_proba`` 1-D ``(n,)`` — une seule courbe.
    Multiclasse : ``y_proba`` 2-D ``(n, n_classes)`` — une courbe par classe
    (one-vs-rest). ``class_names`` est utilisé pour la légende ; à défaut on
    prend ``np.unique(y_true)``.

    Requires matplotlib (`poetry add matplotlib`).
    Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if y_proba.ndim == 1:
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name=name)
    else:
        if class_names is None:
            class_names = np.unique(y_true)
        for i, cls in enumerate(class_names):
            y_binary = (y_true == cls).astype(int)
            RocCurveDisplay.from_predictions(
                y_binary, y_proba[:, i], ax=ax, name=f"class {cls}"
            )

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return ax


# ----------------------------------------------------------------------
# ROC — plotly (minimal)
# ----------------------------------------------------------------------
def plot_roc_plotly(y_true, y_proba, title: str = "ROC Curve"):
    """ROC curve in Plotly. Just the curve, the diagonal, and the AUC."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = float(auc(fpr, tpr))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.4f}")
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Random",
        )
    )
    fig.update_layout(
        title=f"{title} — AUC = {roc_auc:.4f}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        width=600,
        height=450,
    )
    return fig


# ----------------------------------------------------------------------
# Threshold table
# ----------------------------------------------------------------------
def threshold_table(y_true, y_proba, thresholds=None, class_name=None, classes=None) -> pd.DataFrame:
    """Precision / recall / F1 / specificity / accuracy per threshold.

    Binary  : ``y_proba`` 1-D ``(n,)``.
    Multiclasse : ``y_proba`` 2-D ``(n, n_classes)`` + ``class_name`` (la classe
    à analyser en one-vs-rest). ``classes`` est la liste des labels dans
    l'ordre des colonnes (typiquement ``pipeline.named_steps["classifier"].classes_``) ;
    à défaut on prend ``np.unique(y_true)``.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_proba.ndim == 2:
        if class_name is None:
            raise ValueError("class_name est requis quand y_proba est 2-D (multiclasse).")
        if classes is None:
            classes = np.unique(y_true)
        classes = np.asarray(classes)
        col = int(np.where(classes == class_name)[0][0])

        y_true = (y_true == class_name).astype(int)
        y_proba = y_proba[:, col]

    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        rows.append(
            {
                "threshold": round(float(t), 3),
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Lift — DataFrame (decile table)
# ----------------------------------------------------------------------
def _top_k_metrics(y, p, tops):
    """Internal: compute (threshold, captured, capture_%, lift) per top K%."""
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    baseline = float(y.mean())
    total_pos = max(int(y.sum()), 1)

    order = np.argsort(-p)
    y_sorted = y[order]
    p_sorted = p[order]

    out = []
    for top in tops:
        k = max(int(round(n * top / 100)), 1)
        captured = int(y_sorted[:k].sum())
        rate_in_top = captured / k
        out.append(
            {
                "top_%": top,
                "score_threshold": round(float(p_sorted[k - 1]), 4),
                "captured": captured,
                "capture_%": round(captured / total_pos * 100, 2),
                "lift": round(rate_in_top / baseline, 2) if baseline > 0 else np.nan,
            }
        )
    return out


def lift_table(
    y_val,
    y_val_proba,
    y_train=None,
    y_train_proba=None,
    tops=(1, 5, 10),
    class_name=None,
    classes=None,
) -> pd.DataFrame:
    """Cumulative metrics at the top K% of the population (sorted by score desc).

    Same logic as `plot_lift`'s tooltip, but aggregated on the top K% you care
    about in production (default: top 1 %, 5 %, 10 %).

    Binary  : ``y_val_proba`` 1-D ``(n,)``.
    Multiclasse : ``y_val_proba`` 2-D ``(n, n_classes)`` + ``class_name`` (la
    classe à analyser en one-vs-rest). ``classes`` est la liste des labels
    dans l'ordre des colonnes de ``y_val_proba`` (typiquement
    ``pipeline.named_steps["classifier"].classes_``) ; à défaut on prend
    ``np.unique(y_val)``.

    If ``y_train`` / ``y_train_proba`` are passed, two extra columns are added
    (``lift_train``, ``score_threshold_train``) — useful to spot overfitting
    or train/val drift at the same cut-off.

    Columns
    -------
    top_%                   : cut-off as a percentage of the population
    score_threshold         : minimum val score to fall within the top K%
    captured                : number of val positives caught in the top K%
    capture_%               : % of all val positives caught cumulatively
    lift                    : cum_positive_rate_val / mean(y_val)  — × vs random
    lift_train (optional)   : same lift computed on the training set
    score_threshold_train   : same threshold computed on the training set
    """
    y_val = np.asarray(y_val)
    y_val_proba = np.asarray(y_val_proba)

    if y_val_proba.ndim == 2:
        if class_name is None:
            raise ValueError("class_name est requis quand y_val_proba est 2-D (multiclasse).")
        if classes is None:
            classes = np.unique(y_val)
        classes = np.asarray(classes)
        col = int(np.where(classes == class_name)[0][0])

        y_val = (y_val == class_name).astype(int)
        y_val_proba = y_val_proba[:, col]

        if y_train is not None and y_train_proba is not None:
            y_train = (np.asarray(y_train) == class_name).astype(int)
            y_train_proba = np.asarray(y_train_proba)[:, col]

    val = _top_k_metrics(y_val, y_val_proba, tops)
    df = pd.DataFrame(val)

    if y_train is not None and y_train_proba is not None:
        train = _top_k_metrics(y_train, y_train_proba, tops)
        df["lift_train"] = [r["lift"] for r in train]
        df["score_threshold_train"] = [r["score_threshold"] for r in train]

    return df


# ----------------------------------------------------------------------
# Lift — curve (plotly)
# ----------------------------------------------------------------------
def _lift_trace(y_binary, proba, name):
    """Construit une trace Scatter (cumulative gain) pour un vecteur binaire."""
    y_binary = np.asarray(y_binary)
    proba = np.asarray(proba)
    n = len(y_binary)
    baseline = float(y_binary.mean())

    order = np.argsort(-proba)
    y_sorted = y_binary[order]
    total_pos = max(int(y_sorted.sum()), 1)

    rank_pct = np.arange(1, n + 1) / n * 100
    cum_positives = np.cumsum(y_sorted)
    cum_capture_pct = cum_positives / total_pos * 100
    cum_positive_rate = cum_positives / np.arange(1, n + 1)
    lift = cum_positive_rate / baseline if baseline > 0 else np.full(n, np.nan)

    customdata = np.stack([lift, cum_positives], axis=-1)

    return go.Scatter(
        x=rank_pct,
        y=cum_capture_pct,
        mode="lines",
        name=name,
        customdata=customdata,
        hovertemplate=(
            f"<b>{name}</b><br>"
            "Top %{x:.1f}% of population<br>"
            "Captures %{y:.1f}% of positives<br>"
            "Lift = %{customdata[0]:.2f}×<br>"
            "Volume = %{customdata[1]:,.0f} positives<extra></extra>"
        ),
    )


def plot_lift(y_true, y_proba, class_names=None, title: str = "Cumulative Gain"):
    """Cumulative gain curve — supporte binaire et multiclasse.

    - ``y_proba`` 1-D ``(n,)``  : binaire, une seule courbe.
    - ``y_proba`` 2-D ``(n, n_classes)`` : multiclasse, une courbe par classe
      (one-vs-rest). ``class_names`` est utilisé pour la légende ; à défaut on
      prend ``np.unique(y_true)``.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    fig = go.Figure()

    if y_proba.ndim == 1:
        fig.add_trace(_lift_trace(y_true, y_proba, name="Model"))
    else:
        if class_names is None:
            class_names = np.unique(y_true)
        for i, cls in enumerate(class_names):
            y_binary = (y_true == cls).astype(int)
            fig.add_trace(_lift_trace(y_binary, y_proba[:, i], name=f"class {cls}"))

    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            line=dict(dash="dash"),
            name="Random",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="% of population (sorted by score desc)",
        yaxis_title="% of positives captured",
        template="plotly_white",
        width=600,
        height=450,
    )
    return fig


# ----------------------------------------------------------------------
# PSI — Population Stability Index
# ----------------------------------------------------------------------
def calculate_psi(expected, actual, buckettype="bins", buckets=10, axis=0):
    """Calculate the PSI (Population Stability Index) across all variables.

    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values
       buckettype: 'bins' for even splits, 'quantiles' for quantile buckets
       buckets: number of buckets to use
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal

    Returns:
       psi_values: ndarray of psi values for each variable

    Author:
       Matthew Burke (github.com/mwburke) — mwburke.github.io.com
    """

    def psi(expected_array, actual_array, buckets):
        """Calculate the PSI for a single variable."""

        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == "bins":
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == "quantiles":
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

        expected_fractions = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_fractions = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            """Avoid div-by-zero / log(0) by flooring empty buckets to a small value."""
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001
            return (e_perc - a_perc) * np.log(e_perc / a_perc)

        return sum(
            sub_psi(expected_fractions[i], actual_fractions[i])
            for i in range(0, len(expected_fractions))
        )

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[1 - axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:, i], actual[:, i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i, :], actual[i, :], buckets)

    return psi_values


# ----------------------------------------------------------------------
# PSI — density plot
# ----------------------------------------------------------------------
def plot_psi_density(
    expected,
    actual,
    buckettype: str = "quantiles",
    buckets: int = 10,
    expected_name: str = "Train",
    actual_name: str = "Validation",
    title: str = "Score distribution — PSI",
):
    """Overlay the two score densities and annotate the PSI in the centre.

    Uses Plotly histograms with ``histnorm='probability density'`` so the two
    samples are comparable regardless of their absolute size. PSI is computed
    with :func:`calculate_psi` (default: quantile buckets, like for scores).

    Colour code on the PSI badge:
    - green   : PSI < 0.10  (stable)
    - orange  : 0.10 ≤ PSI < 0.25 (moderate drift)
    - red     : PSI ≥ 0.25  (significant drift)
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    psi_value = float(calculate_psi(expected, actual, buckettype=buckettype, buckets=buckets))

    if psi_value < 0.10:
        psi_color = "#2ca02c"
        psi_label = "stable"
    elif psi_value < 0.25:
        psi_color = "#ff7f0e"
        psi_label = "moderate drift"
    else:
        psi_color = "#d62728"
        psi_label = "significant drift"

    from scipy.stats import gaussian_kde

    x_min = float(min(expected.min(), actual.min()))
    x_max = float(max(expected.max(), actual.max()))
    pad = 0.02 * (x_max - x_min) if x_max > x_min else 0.01
    grid = np.linspace(x_min - pad, x_max + pad, 400)

    kde_e = gaussian_kde(expected)(grid)
    kde_a = gaussian_kde(actual)(grid)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grid, y=kde_e, mode="lines", name=expected_name,
            line=dict(color="#1f77b4", width=2),
            fill="tozeroy", fillcolor="rgba(31,119,180,0.25)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid, y=kde_a, mode="lines", name=actual_name,
            line=dict(color="#ff7f0e", width=2),
            fill="tozeroy", fillcolor="rgba(255,127,14,0.25)",
        )
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.12,
        xanchor="right",
        yanchor="bottom",
        text=f"<b>PSI = {psi_value:.4f}</b> &nbsp;·&nbsp; <span style='font-size:11px'>{psi_label}</span>",
        showarrow=False,
        font=dict(size=14, color="white"),
        align="center",
        bgcolor=psi_color,
        bordercolor=psi_color,
        borderpad=6,
        opacity=0.95,
    )

    fig.update_layout(
        title=f"{title} ({expected_name} vs {actual_name})",
        xaxis_title="Predicted probability",
        yaxis_title="Density",
        template="plotly_white",
        width=700,
        height=470,
        margin=dict(t=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
    )
    return fig


# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def add_volume(X):
    """Ajoute la colonne ``volume = height * width * depth``."""
    X = X.copy()
    X["volume"] = X["height"] * X["width"] * X["depth"]
    return X


# ----------------------------------------------------------------------
# Model wrappers — LabelEncodedClassifier
# ----------------------------------------------------------------------
class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a classifier so it accepts arbitrary (string/object) labels.

    Encodes ``y`` with :class:`LabelEncoder` at fit time and decodes the
    predictions back to the original labels at predict time. ``predict_proba``
    returns probabilities aligned with ``self.classes_``.
    """

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y, **fit_params):
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)

        if fit_params.get("eval_set") is not None:
            fit_params["eval_set"] = [
                (Xe, self.label_encoder_.transform(ye))
                for Xe, ye in fit_params["eval_set"]
            ]

        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X, y_encoded, **fit_params)

        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        y_pred_encoded = self.classifier_.predict(X)
        return self.label_encoder_.inverse_transform(y_pred_encoded)

    def predict_proba(self, X):
        return self.classifier_.predict_proba(X)
