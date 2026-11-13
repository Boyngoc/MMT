"""Unit and integration tests for Hùng's Weeks 1–3 deliverables."""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path
import numpy as np

MMT_ROOT = Path(__file__).resolve().parent.parent
if str(MMT_ROOT) not in sys.path:
    sys.path.insert(0, str(MMT_ROOT))

from src.evaluate import (
    catalog_coverage,
    evaluate_ranking,
    evaluate_rating_predictions,
    mae,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)
from src.rec_newitem import NewItemRecommender, evaluate_cold_start_holdout, load_data


class TestEvaluateMetrics(unittest.TestCase):
    def test_rmse_and_mae_hand_calculated(self) -> None:
        y_true = [3.0, 4.0, 5.0, 2.0]
        y_pred = [2.5, 4.5, 4.0, 2.0]
        # errors: -0.5, +0.5, -1.0, 0.0 -> abs sum = 2.0 -> MAE = 0.5
        # sq errors: 0.25, 0.25, 1.0, 0.0 -> mean = 0.375 -> RMSE = sqrt(0.375)
        self.assertAlmostEqual(mae(y_true, y_pred), 0.5, places=6)
        self.assertAlmostEqual(rmse(y_true, y_pred), np.sqrt(0.375), places=6)
        res = evaluate_rating_predictions(y_true, y_pred)
        self.assertIn("RMSE", res)
        self.assertIn("MAE", res)

    def test_ranking_metrics_hand_calculated(self) -> None:
        rec = [10, 20, 30, 40, 50]
        rel = [20, 40, 60]
        # Top-3: [10, 20, 30] -> 1 hit (20) -> P@3 = 1/3, R@3 = 1/3
        self.assertAlmostEqual(precision_at_k(rec, rel, 3), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(recall_at_k(rec, rel, 3), 1.0 / 3.0, places=6)
        # DCG@3 = 1/log2(3), IDCG@3 = 1/log2(2) + 1/log2(3) + 1/log2(4)
        expected_dcg = 1.0 / np.log2(3.0)
        expected_idcg = 1.0 / np.log2(2.0) + 1.0 / np.log2(3.0) + 1.0 / np.log2(4.0)
        self.assertAlmostEqual(ndcg_at_k(rec, rel, 3), expected_dcg / expected_idcg, places=6)

        cov = catalog_coverage([[1, 2, 3], [2, 3, 4]], total_items=10, k=3)
        self.assertAlmostEqual(cov, 0.4, places=6)

        batch = evaluate_ranking({1: rec}, {1: rel}, k=3, total_items=10)
        self.assertAlmostEqual(batch["Precision@3"], 1.0 / 3.0, places=5)


class TestNewItemModuleAndSchema(unittest.TestCase):
    def test_sqlite_schema_executes_cleanly(self) -> None:
        schema_sql = (MMT_ROOT / "docs" / "schema.sql").read_text(encoding="utf-8")
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_sql)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        expected = {
            "users",
            "movies",
            "movie_ratings",
            "products",
            "product_interactions",
            "documents",
            "document_interactions",
            "recommendation_logs",
        }
        self.assertTrue(expected.issubset(tables))
        conn.close()

    def test_new_item_recommender_top10_and_holdout(self) -> None:
        products, interactions = load_data()
        model = NewItemRecommender(cold_threshold=5, k_neighbors=15).fit(products, interactions)

        # Product 461 is a cold-start new item (is_new_item == 1)
        users = model.recommend_users_for_new_item(461, top_n=10)
        self.assertEqual(len(users), 10)
        self.assertTrue(all(u.strategy == "content_knn_item_to_user" for u in users))

        items = model.recommend_items_for_user(1, top_n=10)
        self.assertEqual(len(items), 10)

        new_items_only = model.recommend_items_for_user(1, top_n=10, include_new_only=True)
        self.assertEqual(len(new_items_only), 10)
        self.assertTrue(all(x.strategy == "content_knn_cold_start" for x in new_items_only))

        holdout = evaluate_cold_start_holdout(n_holdout_items=20, k_neighbors=15, top_k=10)
        self.assertGreater(holdout["ranking_metrics"]["Precision@10"], 0.04)
        self.assertLess(holdout["rating_metrics"]["RMSE"], 1.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
