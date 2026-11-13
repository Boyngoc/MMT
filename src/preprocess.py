"""
Module tiền xử lý dữ liệu (Preprocessing) cho hệ thống gợi ý.

CHỦ SỞ HỮU: Linh (Tuần 2).
Nhiệm vụ:
  1. Đọc dữ liệu thô (ratings / interactions) và chuyển đổi thành ma trận user-item.
  2. Lọc lặp (k-core filtering) loại bỏ user và item có dưới 5 tương tác.
  3. Tính toán độ thưa (sparsity) và thống kê tương tác.
  4. Thực hiện chuẩn hóa dữ liệu: Min-Max Scaling và Z-Score Standardization (bảo toàn ô 0).
  5. Cung cấp API chuẩn để rec_movie.py, rec_document.py và rec_newitem.py tái sử dụng.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from typing import Tuple, Dict, Any, Optional, List, Union

import numpy as np

# Thử import pandas; nếu môi trường bị chặn hoặc lỗi DLL, tự động chuyển sang parser thuần
try:
    import pandas as pd
    HAS_PANDAS = True
except (ImportError, Exception):
    HAS_PANDAS = False
    pd = None

__all__ = [
    "load_raw_ratings",
    "filter_k_core",
    "build_user_item_matrix",
    "calculate_sparsity",
    "min_max_scale",
    "z_score_scale",
    "mean_center",
    "save_processed_data",
    "load_processed_data",
    "summary_statistics",
]


# ==========================================================================
# 1. Đọc và làm sạch dữ liệu tương tác thô
# ==========================================================================
def load_raw_ratings(
    file_path: str,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
    sep: str = ",",
) -> Union[List[Tuple[Any, Any, float]], Any]:
    """
    Đọc dữ liệu tương tác thô từ file CSV/TSV.
    Tương thích cả khi có pandas lẫn khi chỉ có numpy/standard library.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {file_path}")

    if HAS_PANDAS and pd is not None:
        try:
            df = pd.read_csv(file_path, sep=sep)
            df = df.dropna(subset=[user_col, item_col, rating_col])
            df = df.drop_duplicates(subset=[user_col, item_col], keep="last")
            df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
            df = df.dropna(subset=[rating_col])
            return df[[user_col, item_col, rating_col]].copy()
        except Exception:
            pass

    # Bộ đọc CSV thuần an toàn (Safe Pure Python Reader)
    records = []
    seen = set()
    with open(file_path, mode="r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=sep)
        for row in reader:
            try:
                u = row[user_col].strip()
                i = row[item_col].strip()
                r = float(row[rating_col].strip())
                key = (u, i)
                if key not in seen:
                    seen.add(key)
                    records.append((u, i, r))
            except (KeyError, ValueError, AttributeError):
                continue
    return records


# ==========================================================================
# 2. Lọc lặp K-Core (User và Item >= k tương tác)
# ==========================================================================
def filter_k_core(
    data: Union[Any, List[Tuple[Any, Any, float]]],
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
    min_user: int = 5,
    min_item: int = 5,
    max_iter: int = 10,
) -> Union[Any, List[Tuple[Any, Any, float]]]:
    """
    Thực hiện lọc lặp đệ quy (k-core pruning) để đảm bảo mọi user còn lại đều có
    ít nhất `min_user` tương tác và mọi item đều có ít nhất `min_item` tương tác.
    """
    if HAS_PANDAS and pd is not None and isinstance(data, pd.DataFrame):
        filtered = data.copy()
        prev_len = len(filtered)
        for _ in range(max_iter):
            uc = filtered[user_col].value_counts()
            filtered = filtered[filtered[user_col].isin(uc[uc >= min_user].index)]
            ic = filtered[item_col].value_counts()
            filtered = filtered[filtered[item_col].isin(ic[ic >= min_item].index)]
            curr_len = len(filtered)
            if curr_len == prev_len:
                break
            prev_len = curr_len
        return filtered.reset_index(drop=True)

    # Xử lý dạng danh sách tuples [(user, item, rating), ...]
    records = list(data)
    prev_len = len(records)
    for _ in range(max_iter):
        user_counts = Counter(r[0] for r in records)
        records = [r for r in records if user_counts[r[0]] >= min_user]
        item_counts = Counter(r[1] for r in records)
        records = [r for r in records if item_counts[r[1]] >= min_item]
        curr_len = len(records)
        if curr_len == prev_len:
            break
        prev_len = curr_len
    return records


# ==========================================================================
# 3. Dựng ma trận User - Item và thống kê độ thưa (Sparsity)
# ==========================================================================
def build_user_item_matrix(
    data: Union[Any, List[Tuple[Any, Any, float]]],
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
    min_user: int = 5,
    min_item: int = 5,
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chuyển đổi dữ liệu tương tác thành ma trận 2D User-Item.

    Returns
    -------
    R : np.ndarray (n_users, n_items)
    user_ids : np.ndarray
    item_ids : np.ndarray
    """
    if min_user > 0 or min_item > 0:
        cleaned = filter_k_core(
            data,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
            min_user=min_user,
            min_item=min_item,
        )
    else:
        cleaned = data

    if len(cleaned) == 0:
        raise ValueError("Dữ liệu rỗng sau khi áp dụng ngưỡng lọc k-core.")

    if HAS_PANDAS and pd is not None and isinstance(cleaned, pd.DataFrame):
        pivot = cleaned.pivot_table(
            index=user_col,
            columns=item_col,
            values=rating_col,
            fill_value=fill_value,
        )
        return pivot.to_numpy(dtype=np.float64), pivot.index.to_numpy(), pivot.columns.to_numpy()

    # Dựng bằng numpy thuần
    users = sorted(list(set(r[0] for r in cleaned)))
    items = sorted(list(set(r[1] for r in cleaned)))
    user_map = {u: idx for idx, u in enumerate(users)}
    item_map = {it: idx for idx, it in enumerate(items)}

    R = np.full((len(users), len(items)), fill_value, dtype=np.float64)
    for u, it, r in cleaned:
        R[user_map[u], item_map[it]] = float(r)

    return R, np.array(users, dtype=object), np.array(items, dtype=object)


def calculate_sparsity(R: np.ndarray, unrated_val: float = 0.0) -> Dict[str, Any]:
    """
    Tính toán độ thưa (Sparsity) và mật độ (Density) của ma trận.
    """
    n_users, n_items = R.shape
    total_cells = n_users * n_items
    if total_cells == 0:
        return {"sparsity_percent": 100.0, "density_percent": 0.0, "total_cells": 0}

    rated_cells = int(np.count_nonzero(R != unrated_val))
    density = (rated_cells / total_cells) * 100.0
    sparsity = 100.0 - density

    return {
        "n_users": n_users,
        "n_items": n_items,
        "total_cells": total_cells,
        "rated_cells": rated_cells,
        "density_percent": round(density, 4),
        "sparsity_percent": round(sparsity, 4),
    }


# ==========================================================================
# 4. Các phép chuẩn hóa dữ liệu (Min-Max, Z-score, Mean-Centering)
# ==========================================================================
def min_max_scale(
    R: np.ndarray,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    mask_zeros: bool = True,
) -> np.ndarray:
    """Chuẩn hóa Min-Max Scaling ma trận (bảo toàn ô unrated nếu mask_zeros=True)."""
    low, high = feature_range
    R_scaled = R.astype(np.float64).copy()

    if mask_zeros:
        mask = R_scaled != 0.0
        if not np.any(mask):
            return R_scaled
        r_min = float(np.min(R_scaled[mask]))
        r_max = float(np.max(R_scaled[mask]))

        if r_max == r_min:
            R_scaled[mask] = high
        else:
            R_scaled[mask] = low + (R_scaled[mask] - r_min) * (high - low) / (r_max - r_min)
    else:
        r_min = float(np.min(R_scaled))
        r_max = float(np.max(R_scaled))
        if r_max == r_min:
            R_scaled.fill(high)
        else:
            R_scaled = low + (R_scaled - r_min) * (high - low) / (r_max - r_min)

    return R_scaled


def z_score_scale(
    R: np.ndarray,
    axis: int = 0,
    mask_zeros: bool = True,
    eps: float = 1e-9,
) -> np.ndarray:
    """Chuẩn hóa Z-Score: (x - mean) / std."""
    R_scaled = R.astype(np.float64).copy()

    if not mask_zeros:
        mean = np.mean(R_scaled, axis=1 if axis == 0 else 0, keepdims=True)
        std = np.std(R_scaled, axis=1 if axis == 0 else 0, keepdims=True)
        std[std == 0] = 1.0
        return (R_scaled - mean) / (std + eps)

    if axis == 0:
        for i in range(R_scaled.shape[0]):
            row = R_scaled[i]
            rated_idx = np.where(row != 0.0)[0]
            if len(rated_idx) > 1:
                vals = row[rated_idx]
                u_mean = np.mean(vals)
                u_std = np.std(vals)
                row[rated_idx] = (vals - u_mean) / u_std if u_std > eps else 0.0
            elif len(rated_idx) == 1:
                row[rated_idx] = 0.0
    else:
        for j in range(R_scaled.shape[1]):
            col = R_scaled[:, j]
            rated_idx = np.where(col != 0.0)[0]
            if len(rated_idx) > 1:
                vals = col[rated_idx]
                i_mean = np.mean(vals)
                i_std = np.std(vals)
                col[rated_idx] = (vals - i_mean) / i_std if i_std > eps else 0.0
            elif len(rated_idx) == 1:
                col[rated_idx] = 0.0

    return R_scaled


def mean_center(
    R: np.ndarray,
    axis: int = 0,
    mask_zeros: bool = True,
) -> np.ndarray:
    """Thực hiện Mean-Centering: r'_ui = r_ui - mu_u."""
    R_centered = R.astype(np.float64).copy()
    if not mask_zeros:
        mean = np.mean(R_centered, axis=1 if axis == 0 else 0, keepdims=True)
        return R_centered - mean

    if axis == 0:
        for i in range(R_centered.shape[0]):
            row = R_centered[i]
            mask = row != 0.0
            if np.any(mask):
                row[mask] -= np.mean(row[mask])
    else:
        for j in range(R_centered.shape[1]):
            col = R_centered[:, j]
            mask = col != 0.0
            if np.any(mask):
                col[mask] -= np.mean(col[mask])

    return R_centered


# ==========================================================================
# 5. Lưu & Đọc dữ liệu đã tiền xử lý
# ==========================================================================
def save_processed_data(
    R: np.ndarray,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    output_dir: str = "data/processed",
    prefix: str = "user_item",
) -> str:
    """Lưu ma trận nén .npz."""
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{prefix}_matrix.npz")
    np.savez_compressed(out_file, matrix=R, user_ids=user_ids, item_ids=item_ids)
    return out_file


def load_processed_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Đọc ma trận từ .npz."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    data = np.load(file_path, allow_pickle=True)
    return data["matrix"], data["user_ids"], data["item_ids"]


def summary_statistics(data: Any, rating_col: str = "rating") -> Dict[str, Any]:
    """Thống kê mô tả điểm đánh giá."""
    if HAS_PANDAS and pd is not None and isinstance(data, pd.DataFrame):
        series = data[rating_col]
        return {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "50%": float(series.median()),
            "max": float(series.max()),
        }
    # Danh sách tuples
    vals = [r[2] for r in data]
    arr = np.array(vals, dtype=float)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "50%": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


# ==========================================================================
# 6. Kiểm thử tự động khi chạy trực tiếp
# ==========================================================================
if __name__ == "__main__":
    print("=== KIỂM THỬ MODULE PREPROCESS.PY (LINH) ===")

    # Dữ liệu tương tác mẫu 6 users x 6 items
    raw_sample = [
        (1, 101, 5.0), (1, 102, 3.0), (1, 103, 4.0), (1, 104, 2.0), (1, 105, 1.0),
        (2, 101, 4.0), (2, 102, 3.0), (2, 103, 5.0), (2, 104, 3.0), (2, 105, 2.0),
        (3, 101, 2.0), (3, 102, 4.0), (3, 103, 4.0), (3, 104, 5.0), (3, 105, 4.0),
        (4, 101, 1.0), (4, 102, 2.0), (4, 103, 3.0), (4, 104, 4.0), (4, 105, 5.0),
        (5, 101, 3.0), (5, 102, 3.0), (5, 103, 3.0), (5, 104, 3.0), (5, 105, 3.0),
        (6, 106, 5.0), # user 6 và item 106 chỉ có 1 tương tác -> sẽ bị loại
    ]

    print(f"Số tương tác ban đầu: {len(raw_sample)}")
    filtered = filter_k_core(raw_sample, min_user=5, min_item=5)
    print(f"Số tương tác sau lọc k-core (>=5): {len(filtered)}")

    R, uids, iids = build_user_item_matrix(raw_sample, min_user=5, min_item=5)
    print(f"Kích thước ma trận R: {R.shape} ({len(uids)} users x {len(iids)} items)")
    assert R.shape == (5, 5), "Kích thước ma trận sau lọc k-core không đúng!"

    stats = calculate_sparsity(R)
    print(f"Độ thưa (Sparsity): {stats['sparsity_percent']}% | Mật độ: {stats['density_percent']}%")

    R_minmax = min_max_scale(R, feature_range=(0.0, 1.0), mask_zeros=True)
    print(f"Min-Max [min: {np.min(R_minmax[R_minmax > 0])}, max: {np.max(R_minmax)}]")
    assert np.allclose(np.max(R_minmax), 1.0), "Min-max scaling lỗi!"

    R_centered = mean_center(R, axis=0, mask_zeros=True)
    print(f"Mean-centered hàng đầu (tổng lệch ~ 0): {np.round(np.sum(R_centered[0]), 4)}")
    assert abs(np.sum(R_centered[0])) < 1e-6, "Mean centering lỗi!"

    print("=== TẤT CẢ CÁC BƯỚC TEST CỦA PREPROCESS.PY ĐỀU VƯỢT QUA THÀNH CÔNG! ===")
