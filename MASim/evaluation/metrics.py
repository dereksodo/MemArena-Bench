"""Evaluation metrics: CPS, MCE, PU-AUC, Memory Lift, Set Recall, etc."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def conflict_preservation_score(
    original_detected: bool,
    conflict_detected: bool,
    sources_correct: bool,
    temporal_correct: bool,
) -> float:
    """CPS: Harmonic mean of conflict detection sub-metrics.

    Components:
    - Detection: both facts identified (0 or 1)
    - Attribution: correct source agents (0 or 1)
    - Temporal: correct ordering (0 or 1)
    """
    detection = 1.0 if (original_detected and conflict_detected) else 0.0
    attribution = 1.0 if sources_correct else 0.0
    temporal = 1.0 if temporal_correct else 0.0

    components = [detection, attribution, temporal]
    nonzero = [c for c in components if c > 0]
    if not nonzero:
        return 0.0
    # Harmonic mean
    return len(nonzero) / sum(1.0 / c for c in nonzero) * (len(nonzero) / len(components))


def memory_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """MCE: Expected Calibration Error for memory system confidence.

    Measures how well system confidence aligns with actual correctness.
    Lower is better.
    """
    if not confidences:
        return 0.0

    conf = np.array(confidences)
    corr = np.array(correctness, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        if not mask.any():
            continue
        bin_conf = conf[mask].mean()
        bin_acc = corr[mask].mean()
        bin_weight = mask.sum() / len(conf)
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def pu_auc(
    scores_positive: Sequence[float],
    scores_unlabeled: Sequence[float],
) -> float:
    """PU-AUC: Area under ROC for Positive-Unlabeled setting.

    Used for confabulation detection where we have known fabrications
    (positive) and responses of unknown correctness (unlabeled).
    """
    if not scores_positive or not scores_unlabeled:
        return 0.5

    pos = np.array(scores_positive)
    unl = np.array(scores_unlabeled)

    # Mann-Whitney U statistic approach
    count = 0
    total = len(pos) * len(unl)
    for p in pos:
        count += np.sum(unl < p)
        count += 0.5 * np.sum(unl == p)

    return float(count / total) if total > 0 else 0.5


def memory_lift(
    score_with_memory: float,
    score_without_memory: float,
) -> float:
    """Memory Lift: relative improvement from having memory access.

    ML = (score_with - score_without) / max(score_without, epsilon)
    """
    epsilon = 1e-8
    return (score_with_memory - score_without_memory) / max(score_without_memory, epsilon)


def set_f1(predicted: set, reference: set) -> float:
    """Set-level F1 for metadata completeness (D6).

    F1 = 2 * precision * recall / (precision + recall)
    """
    if not predicted and not reference:
        return 1.0
    if not predicted or not reference:
        return 0.0

    tp = len(predicted & reference)
    precision = tp / len(predicted)
    recall = tp / len(reference)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def set_recall(predicted: set, reference: set) -> float:
    """Set-level recall."""
    if not reference:
        return 1.0
    return len(predicted & reference) / len(reference)


def aggregate_dimension_scores(
    scores: List[float],
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute aggregate statistics for a dimension's scores."""
    if not scores:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0}

    arr = np.array(scores)
    if weights is not None:
        w = np.array(weights)
        w = w / w.sum()
        mean = float(np.average(arr, weights=w))
    else:
        mean = float(arr.mean())

    return {
        "mean": mean,
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": len(scores),
    }


def composite_score(
    dimension_scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute weighted composite score across dimensions.

    Default: equal weights across all dimensions.
    """
    if not dimension_scores:
        return 0.0

    if weights is None:
        weights = {d: 1.0 for d in dimension_scores}

    total_weight = sum(weights.get(d, 1.0) for d in dimension_scores)
    score = sum(
        dimension_scores[d] * weights.get(d, 1.0)
        for d in dimension_scores
    )
    return score / total_weight if total_weight > 0 else 0.0
