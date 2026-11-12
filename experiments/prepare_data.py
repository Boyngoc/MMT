"""Data preparation, cleaning, and statistical profiling for Hùng (Weeks 1–2).

Handles:
1. Profiling the real MovieLens ml-latest-small dataset.
2. Generating realistic Zipf-distributed raw datasets for Products (new-item
   cold-start domain) and Documents (download/implicit-feedback domain),
   following the fallback strategy in Section 6 of the team plan.
3. Cleaning, deduplicating, imputing, and normalizing both datasets into
   standardized CSVs ready for downstream recommendation modules and SQLite.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
MOVIELENS_DIR = RAW_DIR / "ml-latest-small"
PRODUCTS_DIR = RAW_DIR / "products"
DOCUMENTS_DIR = RAW_DIR / "documents"

SEED = 42


def profile_movielens() -> dict:
    """Compute exact summary statistics for MovieLens ml-latest-small."""
    ratings = pd.read_csv(MOVIELENS_DIR / "ratings.csv")
    movies = pd.read_csv(MOVIELENS_DIR / "movies.csv")
    tags = pd.read_csv(MOVIELENS_DIR / "tags.csv")

    n_users = int(ratings["userId"].nunique())
    n_rated_movies = int(ratings["movieId"].nunique())
    n_total_movies = int(movies["movieId"].nunique())
    n_ratings = int(len(ratings))
    sparsity = 1.0 - (n_ratings / (n_users * n_rated_movies))
    sparsity_all = 1.0 - (n_ratings / (n_users * n_total_movies))

    rating_counts = ratings["rating"].value_counts().sort_index().to_dict()
    user_counts = ratings.groupby("userId").size()
    item_counts = ratings.groupby("movieId").size()

    return {
        "dataset": "MovieLens ml-latest-small",
        "n_users": n_users,
        "n_items_total": n_total_movies,
        "n_items_rated": n_rated_movies,
        "n_ratings": n_ratings,
        "n_tags": int(len(tags)),
        "rating_min": float(ratings["rating"].min()),
        "rating_max": float(ratings["rating"].max()),
        "rating_mean": round(float(ratings["rating"].mean()), 4),
        "rating_std": round(float(ratings["rating"].std()), 4),
        "sparsity_rated_pct": round(sparsity * 100, 4),
        "sparsity_total_pct": round(sparsity_all * 100, 4),
        "avg_ratings_per_user": round(float(user_counts.mean()), 2),
        "min_ratings_per_user": int(user_counts.min()),
        "avg_ratings_per_item": round(float(item_counts.mean()), 2),
        "rating_distribution": {str(k): int(v) for k, v in rating_counts.items()},
    }


def _zipf_probabilities(n: int, alpha: float = 1.15) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, alpha)
    return weights / weights.sum()


def generate_and_clean_products(rng: np.random.Generator) -> dict:
    """Create raw Zipf product data (with realistic noise) and clean it."""
    PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)

    categories = {
        "Laptop": {
            "brands": ["Dell", "Asus", "Lenovo", "Apple", "HP"],
            "price_range": (12_000_000, 45_000_000),
            "tags": ["ultrabook", "gaming", "van-phong", "ssd", "ram-16gb", "man-hinh-fullhd", "pin-trau"],
        },
        "Dien thoai": {
            "brands": ["Apple", "Samsung", "Xiaomi", "Oppo", "Vivo"],
            "price_range": (3_500_000, 34_000_000),
            "tags": ["5g", "amoled", "sac-nhanh", "camera-ai", "chong-nuoc", "pin-5000mah"],
        },
        "Tai nghe": {
            "brands": ["Sony", "Apple", "JBL", "Logitech", "Soundcore"],
            "price_range": (450_000, 8_500_000),
            "tags": ["bluetooth", "chong-on-anc", "khong-day", "gaming", "hi-res", "mic-loc-am"],
        },
        "Ban phim": {
            "brands": ["Keychron", "Logitech", "Akko", "Corsair", "Razer"],
            "price_range": (550_000, 4_800_000),
            "tags": ["co-hoc", "hot-swap", "rgb", "khong-day", "layout-75", "switch-linear"],
        },
        "Chuot may tinh": {
            "brands": ["Logitech", "Razer", "Pulsar", "Corsair", "Microsoft"],
            "price_range": (280_000, 3_600_000),
            "tags": ["khong-day", "sieu-nhe", "dpi-cao", "cong-thai-hoc", "gaming", "silent"],
        },
        "Man hinh": {
            "brands": ["LG", "Dell", "Samsung", "Asus", "ViewSonic"],
            "price_range": (2_800_000, 18_000_000),
            "tags": ["ips", "144hz", "2k-qhd", "4k-uhd", "usb-c", "do-hoa", "chong-choi"],
        },
        "Sach cong nghe": {
            "brands": ["NXB Tre", "O'Reilly", "NXB Giao Duc", "NXB BKHN", "Packt"],
            "price_range": (120_000, 850_000),
            "tags": ["python", "machine-learning", "knn", "mang-may-tinh", "sql", "thuat-toan"],
        },
        "Phu kien": {
            "brands": ["Anker", "Ugreen", "Baseus", "Belkin", "Logitech"],
            "price_range": (150_000, 2_500_000),
            "tags": ["usb-c", "sac-pd", "hub-chuyen-doi", "cap-boc-du", "gia-do", "nho-gon"],
        },
    }

    n_products = 500
    n_new_items = 60  # Cold-start items (last 60 items)
    n_users = 400
    cat_names = list(categories.keys())

    product_rows = []
    base_date = datetime(2026, 6, 1)

    for pid in range(1, n_products + 1):
        cat = cat_names[(pid - 1) % len(cat_names)]
        spec = categories[cat]
        brand = str(rng.choice(spec["brands"]))
        low, high = spec["price_range"]
        price = int(np.round(rng.uniform(low, high) / 10_000) * 10_000)
        chosen_tags = rng.choice(spec["tags"], size=int(rng.integers(3, 5)), replace=False)
        is_new = pid > (n_products - n_new_items)
        release_year = 2026 if is_new else int(rng.choice([2023, 2024, 2025, 2026], p=[0.15, 0.30, 0.40, 0.15]))
        created_offset = int(rng.integers(150, 165)) if is_new else int(rng.integers(0, 140))
        created_at = (base_date + timedelta(days=created_offset)).strftime("%Y-%m-%d")
        name = f"{cat} {brand} Model-{pid:03d}"
        desc = f"{name} thuoc nhom {cat}, thuong hieu {brand}, noi bat voi {', '.join(chosen_tags)}."

        # Inject controlled raw noise (~4% missing price/tags, whitespace) to demonstrate cleaning
        raw_price = float(price) if rng.random() > 0.03 else np.nan
        raw_tags = ", ".join(chosen_tags) if rng.random() > 0.03 else ""
        raw_name = f"  {name}  " if rng.random() < 0.1 else name

        product_rows.append(
            {
                "product_id": pid,
                "name": raw_name,
                "category": cat,
                "brand": brand,
                "price_vnd": raw_price,
                "release_year": release_year,
                "tags": raw_tags,
                "description": desc,
                "is_new_item": int(is_new),
                "created_at": created_at,
            }
        )

    df_prod_raw = pd.DataFrame(product_rows)
    # Inject 8 duplicate rows in raw file
    df_prod_raw = pd.concat([df_prod_raw, df_prod_raw.iloc[:8]], ignore_index=True)
    df_prod_raw.to_csv(PRODUCTS_DIR / "products_raw.csv", index=False, encoding="utf-8-sig")

    # Generate user-product interactions with category affinity + Zipf item popularity
    warm_ids = np.arange(1, n_products - n_new_items + 1)
    zipf_warm = _zipf_probabilities(len(warm_ids), alpha=1.1)

    # Each user has 2 preferred categories so collaborative & content signals align
    user_prefs = {
        uid: list(rng.choice(cat_names, size=2, replace=False))
        for uid in range(1, n_users + 1)
    }

    prod_cat_map = {r["product_id"]: r["category"] for r in product_rows}
    interaction_rows = []
    seen_pairs = set()

    for uid in range(1, n_users + 1):
        n_inter = int(rng.integers(14, 38))
        prefs = set(user_prefs[uid])

        # Bias Zipf weights toward user's preferred categories
        weights = zipf_warm.copy()
        for idx, pid in enumerate(warm_ids):
            if prod_cat_map[int(pid)] in prefs:
                weights[idx] *= 5.0
        weights /= weights.sum()

        chosen_pids = rng.choice(warm_ids, size=min(n_inter, len(warm_ids)), replace=False, p=weights)
        for pid in chosen_pids:
            pid = int(pid)
            if (uid, pid) in seen_pairs:
                continue
            seen_pairs.add((uid, pid))
            in_pref = prod_cat_map[pid] in prefs
            rating = float(rng.choice([4.0, 4.5, 5.0], p=[0.25, 0.35, 0.40])) if in_pref else float(
                rng.choice([1.0, 2.0, 3.0, 3.5], p=[0.15, 0.25, 0.35, 0.25])
            )
            itype = "purchase" if rating >= 4.0 else ("cart" if rating >= 3.0 else "view")
            ts = (base_date + timedelta(days=int(rng.integers(0, 145)))).strftime("%Y-%m-%d")
            interaction_rows.append(
                {
                    "user_id": uid,
                    "product_id": pid,
                    "interaction_type": itype,
                    "rating": rating,
                    "interacted_at": ts,
                }
            )

    # Give ~20 of the 60 new items 1–2 early interactions (near cold-start), 40 have 0 interactions
    for pid in range(n_products - n_new_items + 1, n_products - n_new_items + 21):
        n_early = int(rng.integers(1, 3))
        cat = prod_cat_map[pid]
        matching_users = [u for u, p in user_prefs.items() if cat in p]
        for uid in rng.choice(matching_users, size=n_early, replace=False):
            interaction_rows.append(
                {
                    "user_id": int(uid),
                    "product_id": pid,
                    "interaction_type": "view",
                    "rating": float(rng.choice([4.0, 4.5, 5.0])),
                    "interacted_at": "2026-11-10",
                }
            )

    df_inter_raw = pd.DataFrame(interaction_rows)
    # Inject 25 duplicate interactions and 5 invalid ratings
    dups = df_inter_raw.iloc[:25].copy()
    bad_rows = df_inter_raw.iloc[:5].copy()
    bad_rows["rating"] = 9.9  # invalid out-of-range rating
    df_inter_raw = pd.concat([df_inter_raw, dups, bad_rows], ignore_index=True)
    df_inter_raw.to_csv(PRODUCTS_DIR / "product_interactions_raw.csv", index=False, encoding="utf-8-sig")

    # --- CLEANING & NORMALIZATION ---
    df_prod = df_prod_raw.copy()
    raw_prod_rows = len(df_prod)
    df_prod["name"] = df_prod["name"].astype(str).str.strip()
    df_prod = df_prod.drop_duplicates(subset=["product_id"], keep="first").reset_index(drop=True)

    # Impute missing price with category median
    cat_medians = df_prod.groupby("category")["price_vnd"].transform("median")
    df_prod["price_vnd"] = df_prod["price_vnd"].fillna(cat_medians).astype(float)

    # Impute missing tags with category keyword fallback
    df_prod["tags"] = df_prod["tags"].fillna("").astype(str).str.strip()
    empty_tags = df_prod["tags"] == ""
    df_prod.loc[empty_tags, "tags"] = (
        df_prod.loc[empty_tags, "category"].str.lower().str.replace(" ", "-", regex=False) + ", chinh-hang"
    )

    # Min-Max and Z-score price normalization
    p_min, p_max = df_prod["price_vnd"].min(), df_prod["price_vnd"].max()
    df_prod["price_minmax"] = ((df_prod["price_vnd"] - p_min) / (p_max - p_min)).round(6)
    p_mean, p_std = df_prod["price_vnd"].mean(), df_prod["price_vnd"].std()
    df_prod["price_zscore"] = ((df_prod["price_vnd"] - p_mean) / p_std).round(6)
    df_prod.to_csv(PRODUCTS_DIR / "products_clean.csv", index=False, encoding="utf-8-sig")

    df_inter = df_inter_raw.copy()
    raw_inter_rows = len(df_inter)
    df_inter = df_inter[(df_inter["rating"] >= 1.0) & (df_inter["rating"] <= 5.0)]
    df_inter = df_inter.drop_duplicates(subset=["user_id", "product_id"], keep="last").reset_index(drop=True)
    df_inter.to_csv(PRODUCTS_DIR / "product_interactions_clean.csv", index=False, encoding="utf-8-sig")

    n_u = int(df_inter["user_id"].nunique())
    n_i_total = int(df_prod["product_id"].nunique())
    n_i_active = int(df_inter["product_id"].nunique())
    n_int = int(len(df_inter))
    sparsity = 1.0 - (n_int / (n_u * n_i_total))

    return {
        "dataset": "Products (Zipf Simulated + Cleaned)",
        "raw_product_rows": raw_prod_rows,
        "clean_product_rows": n_i_total,
        "cold_start_new_items": int(df_prod["is_new_item"].sum()),
        "zero_interaction_items": int(n_i_total - n_i_active),
        "raw_interaction_rows": raw_inter_rows,
        "clean_interaction_rows": n_int,
        "n_users": n_u,
        "sparsity_pct": round(sparsity * 100, 4),
        "rating_mean": round(float(df_inter["rating"].mean()), 4),
    }


def generate_and_clean_documents(rng: np.random.Generator) -> dict:
    """Create raw Zipf downloadable document dataset (implicit feedback) and clean it."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    topics = {
        "Tri tue nhan tao & KNN": [
            "knn", "k-nearest-neighbors", "euclidean", "cosine-similarity", "phan-lop", "hoi-quy", "kd-tree",
        ],
        "He goi y": [
            "recommender-system", "collaborative-filtering", "content-based", "cold-start", "tfidf", "svd", "ndcg",
        ],
        "Mang may tinh": [
            "tcp-ip", "http", "fastapi", "rest-api", "socket", "dns", "routing", "client-server",
        ],
        "Co so du lieu": [
            "sql", "sqlite", "erd", "normalization", "index", "query-optimization", "schema",
        ],
        "Lap trinh Python": [
            "python", "numpy", "pandas", "scipy", "vectorization", "pytest", "clean-code",
        ],
        "Toan ung dung": [
            "xac-suat", "dai-so-tuyen-tinh", "ma-tran-thua", "chuan-hoa", "z-score", "pearson",
        ],
    }
    file_types = ["PDF", "DOCX", "PPTX", "IPYNB"]
    topic_names = list(topics.keys())
    n_docs = 400
    n_users = 350
    base_date = datetime(2026, 6, 1)

    doc_rows = []
    for did in range(1, n_docs + 1):
        topic = topic_names[(did - 1) % len(topic_names)]
        kw = rng.choice(topics[topic], size=int(rng.integers(3, 5)), replace=False)
        ftype = str(rng.choice(file_types, p=[0.55, 0.20, 0.15, 0.10]))
        fsize = round(float(rng.uniform(0.3, 14.5)), 2)
        title = f"Tai lieu {topic} - Phan {did:03d}"
        abstract = (
            f"Tai lieu chuyen de {topic}, trinh bay chi tiet cac khai niem {', '.join(kw)} "
            f"kem vi du minh hoa va bai tap thuc hanh."
        )
        created_at = (base_date + timedelta(days=int(rng.integers(0, 150)))).strftime("%Y-%m-%d")

        # Raw noise
        raw_size = fsize if rng.random() > 0.03 else -1.0
        raw_tags = ", ".join(kw) if rng.random() > 0.025 else None

        doc_rows.append(
            {
                "doc_id": did,
                "title": f" {title} " if rng.random() < 0.08 else title,
                "category": topic,
                "tags": raw_tags,
                "abstract": abstract,
                "file_type": ftype.lower() if rng.random() < 0.1 else ftype,
                "file_size_mb": raw_size,
                "created_at": created_at,
            }
        )

    df_docs_raw = pd.DataFrame(doc_rows)
    df_docs_raw = pd.concat([df_docs_raw, df_docs_raw.iloc[:6]], ignore_index=True)
    df_docs_raw.to_csv(DOCUMENTS_DIR / "documents_raw.csv", index=False, encoding="utf-8-sig")

    # Implicit feedback: view (weight=1), bookmark (weight=2), download (weight=3)
    doc_ids = np.arange(1, n_docs + 1)
    zipf_docs = _zipf_probabilities(n_docs, alpha=1.1)
    doc_topic_map = {r["doc_id"]: r["category"] for r in doc_rows}
    action_weights = {"view": 1.0, "bookmark": 2.0, "download": 3.0}

    inter_rows = []
    for uid in range(1, n_users + 1):
        prefs = set(rng.choice(topic_names, size=2, replace=False))
        weights = zipf_docs.copy()
        for idx, did in enumerate(doc_ids):
            if doc_topic_map[int(did)] in prefs:
                weights[idx] *= 4.5
        weights /= weights.sum()

        n_inter = int(rng.integers(12, 32))
        chosen = rng.choice(doc_ids, size=min(n_inter, n_docs), replace=False, p=weights)
        for did in chosen:
            did = int(did)
            if doc_topic_map[did] in prefs:
                action = str(rng.choice(["view", "bookmark", "download"], p=[0.20, 0.25, 0.55]))
            else:
                action = str(rng.choice(["view", "bookmark", "download"], p=[0.65, 0.20, 0.15]))
            ts = (base_date + timedelta(days=int(rng.integers(0, 155)))).strftime("%Y-%m-%d")
            inter_rows.append(
                {
                    "user_id": uid,
                    "doc_id": did,
                    "action_type": action,
                    "weight": action_weights[action],
                    "interacted_at": ts,
                }
            )

    df_doc_inter_raw = pd.DataFrame(inter_rows)
    df_doc_inter_raw = pd.concat([df_doc_inter_raw, df_doc_inter_raw.iloc[:15]], ignore_index=True)
    df_doc_inter_raw.to_csv(DOCUMENTS_DIR / "document_interactions_raw.csv", index=False, encoding="utf-8-sig")

    # --- CLEANING ---
    df_docs = df_docs_raw.copy()
    raw_doc_rows = len(df_docs)
    df_docs["title"] = df_docs["title"].astype(str).str.strip()
    df_docs["file_type"] = df_docs["file_type"].astype(str).str.upper().str.strip()
    df_docs = df_docs.drop_duplicates(subset=["doc_id"], keep="first").reset_index(drop=True)
    df_docs.loc[df_docs["file_size_mb"] <= 0, "file_size_mb"] = np.nan
    df_docs["file_size_mb"] = df_docs["file_size_mb"].fillna(df_docs["file_size_mb"].median()).round(2)
    df_docs["tags"] = df_docs["tags"].fillna("tai-lieu, tham-khao").astype(str).str.strip()
    df_docs.to_csv(DOCUMENTS_DIR / "documents_clean.csv", index=False, encoding="utf-8-sig")

    df_doc_inter = df_doc_inter_raw.drop_duplicates(subset=["user_id", "doc_id", "action_type"]).reset_index(drop=True)
    df_doc_inter.to_csv(DOCUMENTS_DIR / "document_interactions_clean.csv", index=False, encoding="utf-8-sig")

    n_u = int(df_doc_inter["user_id"].nunique())
    n_d = int(df_docs["doc_id"].nunique())
    n_int = int(len(df_doc_inter))
    sparsity = 1.0 - (n_int / (n_u * n_d))

    return {
        "dataset": "Documents (Zipf Implicit Feedback + Cleaned)",
        "raw_doc_rows": raw_doc_rows,
        "clean_doc_rows": n_d,
        "raw_interaction_rows": int(len(df_doc_inter_raw)),
        "clean_interaction_rows": n_int,
        "n_users": n_u,
        "sparsity_pct": round(sparsity * 100, 4),
        "action_distribution": df_doc_inter["action_type"].value_counts().to_dict(),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    ml_stats = profile_movielens()
    prod_stats = generate_and_clean_products(rng)
    doc_stats = generate_and_clean_documents(rng)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "movielens": ml_stats,
        "products": prod_stats,
        "documents": doc_stats,
    }
    out_path = BASE_DIR / "experiments" / "dataset_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

