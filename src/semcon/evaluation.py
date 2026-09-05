"""Decision-layer evaluation: thresholds, precision/recall, confusion matrix.

Ranking metrics (PR-AUC, ROC-AUC, Brier) are computed in train_xgb.py.
This module answers the operational question: at a chosen flagging
threshold, how many fails do we catch and how many false alarms do we buy?

Rules
-----
- Thresholds are tuned on OOF (cross-validated) predictions ONLY, then
  applied once to the time-blocked holdout. Never tune on the holdout.
- Always report raw counts alongside rates: the holdout has ~15 fails.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from semcon.utils import setup_logging

logger = setup_logging()


def tune_threshold(y_true, scores, criterion: str = "mcc") -> float:
    """Choose a flagging threshold on OOF scores.

    criterion: 'f1' or 'mcc'. MCC is the stabler choice at 6.6% prevalence.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    prec, rec, thr = precision_recall_curve(y_true, scores)
    prec, rec = prec[:-1], rec[:-1]  # align with thr

    if criterion == "f1":
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        i = int(np.nanargmax(f1))
    elif criterion == "mcc":
        mccs = [matthews_corrcoef(y_true, scores >= t) for t in thr]
        i = int(np.argmax(mccs))
    else:
        raise ValueError(f"unknown criterion: {criterion}")

    logger.info(
        f"Threshold tuned on OOF ({criterion}): {thr[i]:.4f} "
        f"(precision={prec[i]:.3f}, recall={rec[i]:.3f})"
    )
    return float(thr[i])


def classification_summary(y_true, scores, threshold: float) -> pd.Series:
    """Precision/recall/F1/MCC/FPR plus raw counts at a fixed threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return pd.Series(
        {
            "threshold": threshold,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "fpr": fp / max(fp + tn, 1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    )


def operating_points(y_true, scores, recall_targets=(0.5, 0.6, 0.7)) -> pd.DataFrame:
    """The cost table: precision paid for each recall level.

    For each recall target, the threshold achieving it with the best
    precision. This is the decision table for 'a missed fail costs K
    false alarms' conversations.
    """
    prec, rec, thr = precision_recall_curve(y_true, scores)
    prec, rec = prec[:-1], rec[:-1]
    rows = []
    for rt in recall_targets:
        ok = np.flatnonzero(rec >= rt)
        if len(ok) == 0:
            continue
        i = ok[np.argmax(prec[ok])]
        rows.append(
            {
                "recall_target": rt,
                "threshold": float(thr[i]),
                "precision": float(prec[i]),
                "recall": float(rec[i]),
            }
        )
    return pd.DataFrame(rows)


def save_pr_curve(y_true, scores, out: Path, label: str = "XGBoost") -> None:
    """PR curve with the no-skill prevalence line — the README figure."""
    prec, rec, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    base = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, lw=2, label=f"{label} (AP={ap:.3f})")
    ax.axhline(base, ls="--", color="gray", label=f"no skill ({base:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall, holdout")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"PR curve saved to {out}")


def save_confusion_heatmap(
    y_true, scores, threshold: float, out: Path, title: str = "Holdout"
) -> None:
    """Confusion matrix heatmap at the tuned threshold, counts annotated."""
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["pred pass", "pred fail"],
        yticklabels=["true pass", "true fail"],
        ax=ax,
    )
    ax.set_title(f"{title} @ threshold={threshold:.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {out}")


def recall_at_flagrate(y_true, scores, q: float) -> tuple[float, int]:
    """Recall when flagging the top-q fraction by score. Transfers a RATE
    across regimes instead of a threshold — robust to calibration drift."""
    y_true = np.asarray(y_true)
    k = max(1, int(len(scores) * q))
    top = np.argsort(scores)[::-1][:k]
    return float(y_true[top].sum() / y_true.sum()), k
