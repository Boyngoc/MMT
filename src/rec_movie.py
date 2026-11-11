"""
Module gợi ý xem phim — Collaborative Filtering bằng KNN.

CHỦ SỞ HỮU: Hoàng (Tuần 3).
Xây trên KNNRecommender của knn_core.py; không tự viết lại thuật toán KNN.

Dữ liệu: MovieLens ml-latest-small (data/raw/ml-latest-small/).
  - ratings.csv : userId, movieId, rating (0.5–5.0), timestamp
  - movies.csv  : movieId, title, genres

Chạy thử:  python src/rec_movie.py

--------------------------------------------------------------------------
GHI CHÚ VỀ RANH GIỚI VỚI preprocess.py (Linh)
--------------------------------------------------------------------------
Theo kế hoạch, ma trận user-item do preprocess.py của Linh dựng ra. Hiện
Linh chưa viết file đó, nên rec_movie.py tạm có hàm load_movielens() +
build_user_item() tự đọc thẳng ratings.csv. Khi Linh xong preprocess.py,
chỉ cần thay hai hàm này bằng lời gọi preprocess và giữ nguyên lớp
MovieRecommender ở dưới.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# cho phép chạy cả dạng "python src/rec_movie.py" lẫn "from src.rec_movie import ..."
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from .knn_core import KNNRecommender, KNNIndex  # noqa: F401
except ImportError:
    from knn_core import KNNRecommender, KNNIndex  # noqa: F401

__all__ = ["load_movielens", "build_user_item", "MovieRecommender", "DEFAULT_DATA_DIR"]

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "ml-latest-small",
)


# ==========================================================================
# Nạp & chuẩn bị dữ liệu  (tạm thời, sẽ chuyển sang preprocess.py của Linh)
# ==========================================================================
def load_movielens(data_dir: str = DEFAULT_DATA_DIR):
    """Đọc ratings.csv và movies.csv, trả về (ratings_df, movies_df)."""
    rpath = os.path.join(data_dir, "ratings.csv")
    mpath = os.path.join(data_dir, "movies.csv")
    if not os.path.exists(rpath):
        raise FileNotFoundError(
            f"Khong tim thay {rpath}. Hay tai MovieLens ml-latest-small va giai "
            f"nen vao {data_dir} (xem README)."
        )
    ratings = pd.read_csv(rpath)
    movies = pd.read_csv(mpath)
    return ratings, movies


def build_user_item(ratings: pd.DataFrame, min_user: int = 5, min_item: int = 5):
    """
    Dựng ma trận user-item từ bảng ratings.

    Lọc bỏ user có dưới `min_user` lượt chấm và phim có dưới `min_item` lượt,
    để ma trận tương đồng item-item không quá to và các cặp có đủ cơ sở (đây
    là bước mà knn_core cảnh báo khi chưa lọc — xem docstring adjusted_cosine).

    Trả về:
        R         : mảng numpy (n_user x n_item), 0 = chưa đánh giá
        user_ids  : mảng userId theo đúng thứ tự hàng
        item_ids  : mảng movieId theo đúng thứ tự cột
    """
    df = ratings.copy()
    # lọc lặp vài vòng cho ổn định (lọc item có thể làm user tụt xuống dưới ngưỡng)
    for _ in range(5):
        uc = df.userId.value_counts()
        df = df[df.userId.isin(uc[uc >= min_user].index)]
        ic = df.movieId.value_counts()
        df = df[df.movieId.isin(ic[ic >= min_item].index)]
    pivot = df.pivot_table(index="userId", columns="movieId", values="rating", fill_value=0.0)
    R = pivot.to_numpy(dtype=np.float64)
    return R, pivot.index.to_numpy(), pivot.columns.to_numpy()


# ==========================================================================
# Lớp gợi ý phim
# ==========================================================================
class MovieRecommender:
    """
    Bọc KNNRecommender cho bài toán phim, gắn thêm tên phim và thể loại.

    Tham số
    -------
    k           : số láng giềng
    user_based  : False = item-based (mặc định, nhanh & ổn định hơn)
    metric      : "adjusted_cosine" (item-based) hoặc "pearson" (user-based)
    """

    def __init__(self, k: int = 20, user_based: bool = False, metric: str | None = None,
                 min_support: int = 5):
        if metric is None:
            metric = "pearson" if user_based else "adjusted_cosine"
        # min_support=5: một cặp phim/người phải có ít nhất 5 lượt đồng đánh giá
        # mới được tính tương đồng, nếu không sẽ sinh ra sim=1.0 giả tạo từ các
        # cặp chỉ trùng 2-3 lượt (lỗi kinh điển của CF trên dữ liệu thưa).
        self.engine = KNNRecommender(
            k=k, metric=metric, user_based=user_based,
            min_support=min_support, rating_range=(0.5, 5.0),
        )
        self.movies_: pd.DataFrame | None = None
        self.item_index_: KNNIndex | None = None

    def fit(self, R, user_ids, item_ids, movies: pd.DataFrame):
        self.engine.fit(R, user_ids=user_ids, item_ids=item_ids)
        self.movies_ = movies.set_index("movieId")
        return self

    # ------------------------------------------------------------------
    def _title(self, movie_id) -> str:
        if self.movies_ is not None and movie_id in self.movies_.index:
            return str(self.movies_.loc[movie_id, "title"])
        return f"movieId={movie_id}"

    def _genres(self, movie_id) -> str:
        if self.movies_ is not None and movie_id in self.movies_.index:
            return str(self.movies_.loc[movie_id, "genres"])
        return ""

    def recommend(self, user_id, n: int = 10):
        """Trả về danh sách [(title, điểm dự đoán, genres), ...] cho user."""
        top = self.engine.top_n(user_id, n=n)
        return [(self._title(mid), round(score, 3), self._genres(mid)) for mid, score in top]

    def similar_movies(self, movie_id, n: int = 10):
        """Phim tương tự theo ma trận item-item (chỉ dùng được khi item-based)."""
        if self.engine.user_based:
            raise ValueError("similar_movies chỉ dùng với item-based (user_based=False)")
        sim = self.engine.sim_
        ipos = self.engine._ipos
        if movie_id not in ipos:
            raise KeyError(f"Không có movieId {movie_id} (có thể đã bị lọc vì quá ít lượt chấm)")
        i = ipos[movie_id]
        row = sim[i].copy()
        row[i] = -np.inf
        order = np.argsort(-row)[:n]
        ids = self.engine.item_ids_
        return [(self._title(ids[j]), round(float(row[j]), 3)) for j in order]

    def predict(self, user_id, movie_id) -> float:
        return self.engine.predict(user_id, movie_id)

    def liked_by(self, user_id, ratings: pd.DataFrame, top: int = 5):
        """Vài phim user đã chấm điểm cao nhất — để giải thích vì sao gợi ý."""
        sub = ratings[ratings.userId == user_id].sort_values("rating", ascending=False)
        out = []
        for _, r in sub.head(top).iterrows():
            out.append((self._title(int(r.movieId)), float(r.rating)))
        return out


# ==========================================================================
# Chia train/test theo thời gian để mô phỏng thực tế
# ==========================================================================
def time_split(ratings: pd.DataFrame, test_frac: float = 0.2):
    """
    Với mỗi user, giữ `test_frac` lượt chấm MỚI NHẤT làm test, còn lại train.
    Chia theo thời gian đúng tinh thần hệ gợi ý thật (đoán tương lai từ quá khứ).
    """
    ratings = ratings.sort_values(["userId", "timestamp"])
    train_parts, test_parts = [], []
    for _, g in ratings.groupby("userId"):
        n_test = max(1, int(len(g) * test_frac))
        test_parts.append(g.iloc[-n_test:])
        train_parts.append(g.iloc[:-n_test])
    return pd.concat(train_parts), pd.concat(test_parts)


# ==========================================================================
# DEMO / self-test:  python src/rec_movie.py
# ==========================================================================
def _demo() -> None:
    print("=" * 70)
    print("DEMO  src/rec_movie.py  —  goi y phim bang KNN Collaborative Filtering")
    print("=" * 70)

    ratings, movies = load_movielens()
    print(f"\nDu lieu: {len(ratings):,} luot cham, "
          f"{ratings.userId.nunique()} user, {ratings.movieId.nunique()} phim")

    # Loc phim >=30 luot cham: tren du lieu thua, phim qua it luot sinh ra
    # diem du doan cuc doan (toan 5.0) va tuong dong 1.0 gia tao -> goi y vo
    # nghia. Day la cach kiem soat nhieu duoi dai (long-tail); doi lai la
    # giam do phu catalog (popularity bias) — se phan tich trong muc 2.6.
    R, uids, iids = build_user_item(ratings, min_user=10, min_item=30)
    print(f"Sau khi loc (user>=10, phim>=30 luot): ma tran {R.shape[0]} user x {R.shape[1]} phim")
    sparsity = 100 * (1 - np.count_nonzero(R) / R.size)
    print(f"Do thua: {sparsity:.2f}%")

    # ---- Item-based ----
    print("\n--- Huan luyen ITEM-BASED (adjusted cosine, k=30) ---")
    rec = MovieRecommender(k=30, user_based=False).fit(R, uids, iids, movies)

    demo_user = int(uids[0])
    print(f"\nUser {demo_user} da thich nhung phim (diem cao nhat):")
    for title, rt in rec.liked_by(demo_user, ratings, top=5):
        print(f"   {rt:.1f}*  {title}")

    print(f"\n==> 10 phim goi y cho user {demo_user}:")
    for title, score, genres in rec.recommend(demo_user, n=10):
        print(f"   {score:.2f}  {title[:45]:<45}  [{genres[:30]}]")

    # phim tuong tu mot phim quen thuoc
    toy = movies[movies.title.str.startswith("Toy Story (1995)")]
    if len(toy):
        tid = int(toy.iloc[0].movieId)
        print(f"\n==> Phim tuong tu 'Toy Story (1995)':")
        try:
            for title, s in rec.similar_movies(tid, n=5):
                print(f"   {s:.3f}  {title}")
        except KeyError as e:
            print(f"   ({e})")

    # ---- Danh gia RMSE/MAE, so voi baseline ----
    print("\n--- Danh gia tren tap test (chia theo thoi gian) ---")
    train, test = time_split(ratings, test_frac=0.2)
    Rtr, utr, itr = build_user_item(train, min_user=10, min_item=30)

    def eval_cfg(name, user_based, metric):
        m = KNNRecommender(k=30, metric=metric, user_based=user_based,
                           min_support=5, rating_range=(0.5, 5.0))
        m.fit(Rtr, user_ids=utr, item_ids=itr)
        triples = list(zip(test.userId, test.movieId, test.rating))
        kq = m.evaluate(triples)
        print(f"   {name:<28} RMSE={kq['rmse']:.4f}  MAE={kq['mae']:.4f}  "
              f"(n={kq['n']:,}, bo qua {kq['bo_qua']:,} cap cold-start)")
        return kq

    kq_item = eval_cfg("Item-based (adj.cosine)", False, "adjusted_cosine")
    kq_user = eval_cfg("User-based (pearson)", True, "pearson")

    # baseline: doan bang trung binh toan cuc
    gmean = train.rating.mean()
    base_rmse = float(np.sqrt(np.mean((test.rating - gmean) ** 2)))
    print(f"   {'Baseline (trung binh)':<28} RMSE={base_rmse:.4f}")

    best = min(kq_item["rmse"], kq_user["rmse"])
    print(f"\n==> KNN tot nhat giam {100 * (1 - best / base_rmse):.1f}% RMSE so voi baseline.")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
