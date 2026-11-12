"""New-Item (Cold-Start) Recommender Module — Content-based + Hybrid Switching KNN.

Author: Hùng (Week 3 deliverable, Milestone M3)
Supports:
1. Finding Top-N potential customers for a newly launched item (Item -> Users).
2. Recommending Top-N items for a user via Hybrid Switching between
   Content-based KNN (for cold-start items) and Item-based CF KNN (for warm items).
3. CLI interface returning Top-10 recommendations and holdout evaluation metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    # Optional integration with Hoàng's shared KNN core once merged to main
    from src import knn_core  # type: ignore
except ImportError:
    knn_core = None

from src.evaluate import evaluate_ranking, evaluate_rating_predictions

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "products"


@dataclass
class RecommendationItem:
    """Single recommendation output entry with score, strategy, and explanation."""

    target_id: int
    name: str
    score: float
    strategy: str
    reason: str


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute exact pairwise cosine similarity matrix for dense feature vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = vectors / safe_norms
    sim = normalized @ normalized.T
    return np.clip(sim, -1.0, 1.0)


def _top_k_indices(scores: np.ndarray, k: int, exclude_idx: Optional[int] = None) -> np.ndarray:
    """Return indices of top-k highest scores in descending order."""
    work = scores.copy()
    if exclude_idx is not None and 0 <= exclude_idx < len(work):
        work[exclude_idx] = -np.inf
    k_eff = min(k, len(work) - (1 if exclude_idx is not None else 0))
    if k_eff <= 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(work, -k_eff)[-k_eff:]
    return idx[np.argsort(work[idx])[::-1]]


class NewItemRecommender:
    """Content-based KNN + Hybrid Switching recommender for new/cold-start items."""

    def __init__(
        self,
        cold_threshold: int = 5,
        k_neighbors: int = 15,
        cat_weight: float = 2.0,
        brand_weight: float = 1.0,
        tag_weight: float = 1.2,
        price_weight: float = 0.8,
        year_weight: float = 0.4,
    ) -> None:
        self.cold_threshold = cold_threshold
        self.k_neighbors = k_neighbors
        self.cat_weight = cat_weight
        self.brand_weight = brand_weight
        self.tag_weight = tag_weight
        self.price_weight = price_weight
        self.year_weight = year_weight

        self.products_df: pd.DataFrame = pd.DataFrame()
        self.interactions_df: pd.DataFrame = pd.DataFrame()
        self.pid_to_idx: Dict[int, int] = {}
        self.idx_to_pid: Dict[int, int] = {}
        self.uid_to_idx: Dict[int, int] = {}
        self.idx_to_uid: Dict[int, int] = {}
        self.feature_matrix: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.content_sim: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.cf_sim: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.rating_matrix: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.item_interaction_counts: np.ndarray = np.empty(0, dtype=np.int64)
        self.item_mean_ratings: np.ndarray = np.empty(0, dtype=np.float64)
        self.global_mean_rating: float = 3.5

    def _build_item_features(self, products: pd.DataFrame) -> np.ndarray:
        """Encode category, brand, tags, price, and release year into L2-normalized vectors."""
        n_items = len(products)
        blocks: List[np.ndarray] = []

        # 1. Category one-hot
        cats = sorted(products["category"].astype(str).unique())
        cat_map = {c: i for i, c in enumerate(cats)}
        cat_mat = np.zeros((n_items, len(cats)), dtype=np.float64)
        for r_i, val in enumerate(products["category"].astype(str)):
            cat_mat[r_i, cat_map[val]] = self.cat_weight
        blocks.append(cat_mat)

        # 2. Brand one-hot
        brands = sorted(products["brand"].astype(str).unique())
        brand_map = {b: i for i, b in enumerate(brands)}
        brand_mat = np.zeros((n_items, len(brands)), dtype=np.float64)
        for r_i, val in enumerate(products["brand"].astype(str)):
            brand_mat[r_i, brand_map[val]] = self.brand_weight
        blocks.append(brand_mat)

        # 3. Tags multi-hot with IDF weighting
        tag_lists = [
            [t.strip().lower() for t in str(raw).split(",") if t.strip()]
            for raw in products["tags"].fillna("")
        ]
        vocab = sorted({t for tl in tag_lists for t in tl})
        if vocab:
            tag_map = {t: i for i, t in enumerate(vocab)}
            tag_mat = np.zeros((n_items, len(vocab)), dtype=np.float64)
            df_counts = np.zeros(len(vocab), dtype=np.float64)
            for r_i, tl in enumerate(tag_lists):
                for t in set(tl):
                    col = tag_map[t]
                    tag_mat[r_i, col] = 1.0
                    df_counts[col] += 1.0
            idf = np.log((1.0 + n_items) / (1.0 + df_counts)) + 1.0
            tag_mat = tag_mat * idf * self.tag_weight
            blocks.append(tag_mat)

        # 4. Numeric normalized price & release year
        if "price_minmax" in products.columns:
            price_norm = products["price_minmax"].fillna(0.5).to_numpy(dtype=np.float64)
        else:
            p = products["price_vnd"].to_numpy(dtype=np.float64)
            p_span = max(float(p.max() - p.min()), 1.0)
            price_norm = (p - p.min()) / p_span
        blocks.append((price_norm * self.price_weight).reshape(-1, 1))

        years = products["release_year"].fillna(2025).to_numpy(dtype=np.float64)
        y_span = max(float(years.max() - years.min()), 1.0)
        year_norm = (years - years.min()) / y_span
        blocks.append((year_norm * self.year_weight).reshape(-1, 1))

        raw_features = np.hstack(blocks)
        norms = np.linalg.norm(raw_features, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return raw_features / norms

    def fit(self, products_df: pd.DataFrame, interactions_df: pd.DataFrame) -> "NewItemRecommender":
        """Fit content feature matrix, content KNN similarities, and CF item similarities."""
        self.products_df = products_df.sort_values("product_id").reset_index(drop=True)
        self.interactions_df = interactions_df.copy()

        pids = self.products_df["product_id"].astype(int).tolist()
        self.pid_to_idx = {pid: i for i, pid in enumerate(pids)}
        self.idx_to_pid = {i: pid for i, pid in enumerate(pids)}

        uids = sorted(self.interactions_df["user_id"].astype(int).unique().tolist())
        self.uid_to_idx = {uid: i for i, uid in enumerate(uids)}
        self.idx_to_uid = {i: uid for i, uid in enumerate(uids)}

        self.feature_matrix = self._build_item_features(self.products_df)
        if knn_core is not None and hasattr(knn_core, "pairwise_cosine"):
            self.content_sim = knn_core.pairwise_cosine(self.feature_matrix)
        else:
            self.content_sim = _cosine_similarity_matrix(self.feature_matrix)

        n_u, n_i = len(uids), len(pids)
        self.rating_matrix = np.zeros((n_u, n_i), dtype=np.float64)
        for row in self.interactions_df.itertuples(index=False):
            u_idx = self.uid_to_idx.get(int(row.user_id))
            i_idx = self.pid_to_idx.get(int(row.product_id))
            if u_idx is not None and i_idx is not None:
                self.rating_matrix[u_idx, i_idx] = float(row.rating)

        rated_mask = self.rating_matrix > 0
        self.item_interaction_counts = rated_mask.sum(axis=0)
        all_ratings = self.rating_matrix[rated_mask]
        self.global_mean_rating = float(all_ratings.mean()) if all_ratings.size > 0 else 3.5

        item_sums = self.rating_matrix.sum(axis=0)
        self.item_mean_ratings = np.where(
            self.item_interaction_counts > 0,
            item_sums / np.maximum(self.item_interaction_counts, 1),
            self.global_mean_rating,
        )

        # Item-Item Collaborative Filtering similarity (mean-centered cosine / adjusted cosine)
        user_counts = rated_mask.sum(axis=1, keepdims=True)
        user_means = np.where(user_counts > 0, self.rating_matrix.sum(axis=1, keepdims=True) / np.maximum(user_counts, 1), 0.0)
        centered = np.where(rated_mask, self.rating_matrix - user_means, 0.0)
        self.cf_sim = _cosine_similarity_matrix(centered.T)
        np.fill_diagonal(self.cf_sim, 0.0)
        return self

    def find_similar_warm_items(self, item_id: int, k: Optional[int] = None) -> List[Tuple[int, float]]:
        """Return top-k warm items most similar in content to ``item_id``."""
        if item_id not in self.pid_to_idx:
            raise KeyError(f"Unknown product_id={item_id}")
        k_val = k or self.k_neighbors
        i_idx = self.pid_to_idx[item_id]
        sims = self.content_sim[i_idx].copy()

        # Restrict neighbors to warm items with enough interactions
        warm_mask = self.item_interaction_counts >= self.cold_threshold
        warm_mask[i_idx] = False
        sims = np.where(warm_mask, sims, -np.inf)

        top_idxs = _top_k_indices(sims, k_val)
        return [(self.idx_to_pid[int(idx)], float(sims[idx])) for idx in top_idxs if sims[idx] > 0]

    def recommend_users_for_new_item(
        self,
        item_id: int,
        top_n: int = 10,
        k: Optional[int] = None,
        min_positive_rating: float = 3.5,
    ) -> List[RecommendationItem]:
        """Recommend Top-N potential customers for a new (cold-start) item."""
        neighbors = self.find_similar_warm_items(item_id, k=k)
        if not neighbors:
            return []

        user_scores = np.zeros(len(self.uid_to_idx), dtype=np.float64)
        user_sim_sums = np.zeros(len(self.uid_to_idx), dtype=np.float64)
        best_neighbor: Dict[int, Tuple[int, float]] = {}

        for nb_pid, sim in neighbors:
            nb_idx = self.pid_to_idx[nb_pid]
            col_ratings = self.rating_matrix[:, nb_idx]
            pos_users = np.where(col_ratings >= min_positive_rating)[0]
            for u_idx in pos_users:
                contrib = sim * col_ratings[u_idx]
                user_scores[u_idx] += contrib
                user_sim_sums[u_idx] += sim
                prev = best_neighbor.get(int(u_idx))
                if prev is None or contrib > prev[1]:
                    best_neighbor[int(u_idx)] = (nb_pid, sim)

        # Exclude users who already interacted with this item
        target_idx = self.pid_to_idx[item_id]
        already_interacted = self.rating_matrix[:, target_idx] > 0
        user_scores = np.where(already_interacted, -np.inf, user_scores)

        # Normalize softly while preserving multi-neighbor consensus confidence
        valid = user_sim_sums > 0
        final_scores = np.where(valid, user_scores / np.sqrt(np.maximum(user_sim_sums, 1e-8)), -np.inf)

        top_u_idxs = _top_k_indices(final_scores, top_n)
        results: List[RecommendationItem] = []
        for u_idx in top_u_idxs:
            if not np.isfinite(final_scores[u_idx]) or final_scores[u_idx] <= 0:
                continue
            uid = self.idx_to_uid[int(u_idx)]
            nb_pid, nb_sim = best_neighbor[int(u_idx)]
            nb_name = str(self.products_df.loc[self.pid_to_idx[nb_pid], "name"])
            results.append(
                RecommendationItem(
                    target_id=uid,
                    name=f"User #{uid}",
                    score=round(float(final_scores[u_idx]), 4),
                    strategy="content_knn_item_to_user",
                    reason=f"Thich SP tuong dong '{nb_name}' (cos_sim={nb_sim:.3f})",
                )
            )
        return results

    def predict_user_item_score(
        self,
        user_id: int,
        item_id: int,
        k: Optional[int] = None,
    ) -> Tuple[float, str, str]:
        """Predict affinity score (1..5 scale) via Hybrid Switching (Content KNN vs Item-CF)."""
        if user_id not in self.uid_to_idx or item_id not in self.pid_to_idx:
            return self.global_mean_rating, "fallback_global_mean", "User hoac san pham chua co du lieu"

        u_idx = self.uid_to_idx[user_id]
        i_idx = self.pid_to_idx[item_id]
        k_val = k or self.k_neighbors

        user_ratings = self.rating_matrix[u_idx]
        rated_items = np.where(user_ratings > 0)[0]
        if rated_items.size == 0:
            return float(self.item_mean_ratings[i_idx]), "fallback_item_mean", "User chua co lich su tuong tac"

        n_inter = int(self.item_interaction_counts[i_idx])
        is_cold = n_inter < self.cold_threshold

        if is_cold:
            # Content-based KNN for cold-start / new items
            sims = self.content_sim[i_idx, rated_items]
            pos_mask = sims > 0
            if not np.any(pos_mask):
                return self.global_mean_rating, "content_knn_fallback", "Khong co dac trung trung khop"
            cand_items = rated_items[pos_mask]
            cand_sims = sims[pos_mask]
            order = np.argsort(cand_sims)[::-1][:k_val]
            top_items = cand_items[order]
            top_sims = cand_sims[order]

            pred = float(np.dot(top_sims, user_ratings[top_items]) / np.sum(top_sims))
            best_nb = int(top_items[0])
            nb_name = str(self.products_df.loc[best_nb, "name"])
            reason = f"Hang moi tuong dong '{nb_name}' (cos_sim={top_sims[0]:.3f})"
            return float(np.clip(pred, 1.0, 5.0)), "content_knn_cold_start", reason

        # Hybrid / Item-based CF KNN for warm items
        cf_sims = self.cf_sim[i_idx, rated_items]
        pos_mask = cf_sims > 0
        if not np.any(pos_mask):
            # Smooth fallback to content-based KNN when CF overlap is zero
            sims = self.content_sim[i_idx, rated_items]
            order = np.argsort(sims)[::-1][:k_val]
            top_items = rated_items[order]
            top_sims = np.maximum(sims[order], 0.0)
            denom = float(np.sum(top_sims))
            pred = float(np.dot(top_sims, user_ratings[top_items]) / denom) if denom > 0 else float(self.item_mean_ratings[i_idx])
            nb_name = str(self.products_df.loc[int(top_items[0]), "name"])
            return float(np.clip(pred, 1.0, 5.0)), "content_knn_sparse_fallback", f"Goi y noi dung theo '{nb_name}'"

        cand_items = rated_items[pos_mask]
        cand_sims = cf_sims[pos_mask]
        order = np.argsort(cand_sims)[::-1][:k_val]
        top_items = cand_items[order]
        top_sims = cand_sims[order]

        # Mean-centered Item-CF prediction
        diffs = user_ratings[top_items] - self.item_mean_ratings[top_items]
        pred = float(self.item_mean_ratings[i_idx] + np.dot(top_sims, diffs) / np.sum(top_sims))
        best_nb = int(top_items[0])
        nb_name = str(self.products_df.loc[best_nb, "name"])
        reason = f"Khach hang thich '{nb_name}' thuong chon SP nay (cf_sim={top_sims[0]:.3f})"
        return float(np.clip(pred, 1.0, 5.0)), "item_cf_hybrid", reason

    def recommend_items_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        k: Optional[int] = None,
        include_new_only: bool = False,
    ) -> List[RecommendationItem]:
        """Recommend Top-N items for a user using Hybrid Switching."""
        if user_id not in self.uid_to_idx:
            raise KeyError(f"Unknown user_id={user_id}")

        u_idx = self.uid_to_idx[user_id]
        already_seen = self.rating_matrix[u_idx] > 0

        candidates: List[RecommendationItem] = []
        for i_idx, row in self.products_df.iterrows():
            if already_seen[i_idx]:
                continue
            if include_new_only and int(row["is_new_item"]) != 1:
                continue
            pid = int(row["product_id"])
            score, strategy, reason = self.predict_user_item_score(user_id, pid, k=k)
            candidates.append(
                RecommendationItem(
                    target_id=pid,
                    name=str(row["name"]),
                    score=round(score, 4),
                    strategy=strategy,
                    reason=reason,
                )
            )

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_n]


def load_data(data_dir: Path = DEFAULT_DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned products and product interactions CSV files."""
    products = pd.read_csv(data_dir / "products_clean.csv")
    interactions = pd.read_csv(data_dir / "product_interactions_clean.csv")
    return products, interactions


def evaluate_cold_start_holdout(
    data_dir: Path = DEFAULT_DATA_DIR,
    n_holdout_items: int = 30,
    k_neighbors: int = 15,
    top_k: int = 10,
    seed: int = 42,
) -> Dict[str, object]:
    """Simulate cold-start items by hiding all interactions of ``n_holdout_items`` warm items.

    Evaluates both:
    - Item-to-User cold-start ranking (Precision@K, Recall@K, NDCG@K)
    - User-to-Item hybrid rating prediction (RMSE, MAE)
    """
    products, interactions = load_data(data_dir)
    rng = np.random.default_rng(seed)

    warm_pids = products.loc[products["is_new_item"] == 0, "product_id"].to_numpy()
    holdout_pids = set(int(x) for x in rng.choice(warm_pids, size=n_holdout_items, replace=False))

    train_inter = interactions[~interactions["product_id"].isin(holdout_pids)].reset_index(drop=True)
    test_inter = interactions[interactions["product_id"].isin(holdout_pids)].reset_index(drop=True)

    rec = NewItemRecommender(cold_threshold=5, k_neighbors=k_neighbors).fit(products, train_inter)

    # 1. Item -> User Top-K ranking evaluation
    recs_by_item: Dict[int, List[int]] = {}
    rel_by_item: Dict[int, List[int]] = {}
    for pid in sorted(holdout_pids):
        pos_users = test_inter.loc[
            (test_inter["product_id"] == pid) & (test_inter["rating"] >= 3.5), "user_id"
        ].astype(int).tolist()
        if not pos_users:
            continue
        rel_by_item[pid] = pos_users
        preds = rec.recommend_users_for_new_item(pid, top_n=top_k, k=k_neighbors)
        recs_by_item[pid] = [p.target_id for p in preds]

    ranking_metrics = evaluate_ranking(
        recs_by_item,
        rel_by_item,
        k=top_k,
        total_items=int(interactions["user_id"].nunique()),
    )

    # 2. Rating prediction RMSE & MAE on hidden cold-start interactions
    y_true, y_pred = [], []
    for row in test_inter.itertuples(index=False):
        pred, _, _ = rec.predict_user_item_score(int(row.user_id), int(row.product_id), k=k_neighbors)
        y_true.append(float(row.rating))
        y_pred.append(pred)
    rating_metrics = evaluate_rating_predictions(y_true, y_pred)

    return {
        "holdout_items": len(rel_by_item),
        "test_interactions": len(test_inter),
        "ranking_metrics": ranking_metrics,
        "rating_metrics": rating_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hùng — Module gợi ý hàng mới (Content-based + Hybrid KNN)")
    parser.add_argument(
        "--mode",
        choices=["recommend_users", "recommend_items", "evaluate"],
        default="evaluate",
        help="Chế độ chạy CLI",
    )
    parser.add_argument("--item-id", type=int, default=461, help="ID sản phẩm mới cần tìm khách hàng")
    parser.add_argument("--user-id", type=int, default=1, help="ID khách hàng cần gợi ý sản phẩm")
    parser.add_argument("--top-k", type=int, default=10, help="Số lượng gợi ý Top-K (mặc định 10)")
    parser.add_argument("--k-neighbors", type=int, default=15, help="Số láng giềng KNN")
    parser.add_argument("--new-only", action="store_true", help="Chỉ gợi ý hàng mới (is_new_item=1)")
    args = parser.parse_args()

    products, interactions = load_data()
    model = NewItemRecommender(cold_threshold=5, k_neighbors=args.k_neighbors).fit(products, interactions)

    if args.mode == "recommend_users":
        prod_row = products.loc[products["product_id"] == args.item_id]
        if prod_row.empty:
            raise SystemExit(f"Không tìm thấy product_id={args.item_id}")
        pname = prod_row.iloc[0]["name"]
        pcat = prod_row.iloc[0]["category"]
        print(f"=== TOP-{args.top_k} KHÁCH HÀNG TIỀM NĂNG CHO HÀNG MỚI #{args.item_id}: {pname} ({pcat}) ===")
        for rank, item in enumerate(model.recommend_users_for_new_item(args.item_id, top_n=args.top_k), start=1):
            print(f"{rank:2d}. {item.name:10s} | score={item.score:.4f} | {item.reason}")

    elif args.mode == "recommend_items":
        print(f"=== TOP-{args.top_k} SẢN PHẨM GỢI Ý (HYBRID) CHO USER #{args.user_id} ===")
        recs = model.recommend_items_for_user(
            args.user_id, top_n=args.top_k, include_new_only=args.new_only
        )
        for rank, item in enumerate(recs, start=1):
            print(f"{rank:2d}. [#{item.target_id:03d}] {item.name:30s} | score={item.score:.4f} | {item.strategy} | {item.reason}")

    else:
        print("=== ĐÁNH GIÁ HOLDOUT COLD-START (HÀNG MỚI) ===")
        summary = evaluate_cold_start_holdout(k_neighbors=args.k_neighbors, top_k=args.top_k)
        print(summary)


if __name__ == "__main__":
    main()

