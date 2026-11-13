"""Evaluation framework for KNN Recommender Systems (Hùng — Weeks 2–3).

Implements:
- Rating prediction error metrics (Week 2): RMSE, MAE
- Top-K ranking metrics (Week 3): Precision@K, Recall@K, NDCG@K
- Batch evaluation and catalog coverage helpers
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence
import numpy as np


def _to_1d_float_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def rmse(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Compute Root Mean Squared Error between true and predicted ratings."""
    yt = _to_1d_float_array(y_true, "y_true")
    yp = _to_1d_float_array(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}.")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Compute Mean Absolute Error between true and predicted ratings."""
    yt = _to_1d_float_array(y_true, "y_true")
    yp = _to_1d_float_array(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}.")
    return float(np.mean(np.abs(yt - yp)))


def precision_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Compute Precision@K for a single ranked recommendation list."""
    if k <= 0:
        raise ValueError("k must be positive.")
    top_k = list(recommended)[:k]
    if not top_k:
        return 0.0
    rel_set = set(relevant)
    hits = sum(1 for item in top_k if item in rel_set)
    return hits / float(k)


def recall_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Compute Recall@K for a single ranked recommendation list."""
    if k <= 0:
        raise ValueError("k must be positive.")
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    top_k = list(recommended)[:k]
    hits = sum(1 for item in top_k if item in rel_set)
    return hits / float(len(rel_set))


def ndcg_at_k(
    recommended: Sequence[int],
    relevant: Iterable[int],
    k: int,
    relevance_scores: Mapping[int, float] | None = None,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@K).

    Supports binary relevance (default when ``relevance_scores`` is None)
    or graded relevance mapping ``{item_id: gain}``.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    rel_set = set(relevant)
    if relevance_scores is None:
        scores_map: Dict[int, float] = {item: 1.0 for item in rel_set}
    else:
        scores_map = {item: float(val) for item, val in relevance_scores.items() if float(val) > 0}

    if not scores_map:
        return 0.0

    top_k = list(recommended)[:k]
    dcg = 0.0
    for rank_idx, item in enumerate(top_k):
        rel = scores_map.get(item, 0.0)
        if rel > 0.0:
            dcg += rel / np.log2(rank_idx + 2.0)

    ideal_rels = sorted(scores_map.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(rank_idx + 2.0) for rank_idx, rel in enumerate(ideal_rels))
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def catalog_coverage(all_recommended: Sequence[Sequence[int]], total_items: int, k: int) -> float:
    """Compute fraction of catalog items appearing in at least one user's Top-K list."""
    if total_items <= 0 or k <= 0:
        raise ValueError("total_items and k must be positive.")
    unique_items = {item for recs in all_recommended for item in list(recs)[:k]}
    return len(unique_items) / float(total_items)


def evaluate_rating_predictions(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> Dict[str, float]:
    """Return RMSE and MAE summary dictionary."""
    return {
        "RMSE": round(rmse(y_true, y_pred), 6),
        "MAE": round(mae(y_true, y_pred), 6),
    }


def evaluate_ranking(
    recommendations_by_user: Mapping[int, Sequence[int]],
    relevant_by_user: Mapping[int, Iterable[int]],
    k: int = 10,
    total_items: int | None = None,
) -> Dict[str, float]:
    """Compute macro-averaged Precision@K, Recall@K, NDCG@K (and Coverage if total_items set)."""
    precisions = []
    recalls = []
    ndcgs = []
    all_recs = []

    for uid, relevant in relevant_by_user.items():
        rel_list = list(relevant)
        if not rel_list:
            continue
        recs = list(recommendations_by_user.get(uid, []))
        all_recs.append(recs)
        precisions.append(precision_at_k(recs, rel_list, k))
        recalls.append(recall_at_k(recs, rel_list, k))
        ndcgs.append(ndcg_at_k(recs, rel_list, k))

    if not precisions:
        return {f"Precision@{k}": 0.0, f"Recall@{k}": 0.0, f"NDCG@{k}": 0.0}

    results: Dict[str, float] = {
        f"Precision@{k}": round(float(np.mean(precisions)), 6),
        f"Recall@{k}": round(float(np.mean(recalls)), 6),
        f"NDCG@{k}": round(float(np.mean(ndcgs)), 6),
    }
    if total_items is not None and total_items > 0:
        results[f"Coverage@{k}"] = round(catalog_coverage(all_recs, total_items, k), 6)
    return results

