"""
Nhân KNN dùng chung cho cả 3 module gợi ý.

CHỦ SỞ HỮU: Hoàng — API đã chốt (Tuần 2).
Linh và Hùng xây module của mình TRÊN hai lớp ở đây, không viết lại KNN.

--------------------------------------------------------------------------
DÙNG LỚP NÀO?
--------------------------------------------------------------------------
KNNIndex        — KNN trên ma trận ĐẶC TRƯNG (content-based).
                  Linh: TF-IDF của tài liệu.
                  Hùng: vector thuộc tính sản phẩm, giải cold-start.

KNNRecommender  — KNN dự đoán RATING (collaborative filtering).
                  Hoàng: module gợi ý phim.
                  Hùng: nhánh CF trong công thức hybrid.
--------------------------------------------------------------------------

Ví dụ cho Linh (gợi ý tài liệu):

    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.knn_core import KNNIndex

    X = TfidfVectorizer(max_features=5000).fit_transform(corpus)
    idx = KNNIndex(metric="cosine", k=10).fit(X, ids=doc_ids)

    idx.similar_to("doc_42")                    # tài liệu tương tự
    idx.recommend(liked_ids=da_tai_cua_user)    # gợi ý cá nhân hóa

Ví dụ cho Hoàng (gợi ý phim):

    from src.knn_core import KNNRecommender

    rec = KNNRecommender(k=20, metric="adjusted_cosine", user_based=False)
    rec.fit(R, user_ids=uids, item_ids=iids)    # R: user x item, 0 = chưa xem

    rec.predict(user=1, item=50)                # điểm dự đoán
    rec.top_n(user=1, n=10)                     # 10 phim nên gợi ý

Ví dụ cho Hùng (cold-start hàng mới):

    idx = KNNIndex(metric="cosine", k=5).fit(dac_trung_sp, ids=ma_sp)
    tuong_tu = idx.similar_to(ma_sp_moi)        # 5 sản phẩm cũ giống nhất
    # -> lấy nhóm khách đã mua những sản phẩm đó để gợi ý hàng mới
"""

from __future__ import annotations

import warnings

import numpy as np

try:  # chạy như một package:  from src.knn_core import ...
    from . import similarity as sim_mod
except ImportError:  # chạy trực tiếp:  python src/knn_core.py
    import similarity as sim_mod

__all__ = ["KNNIndex", "KNNRecommender", "rmse", "mae"]

EPS = 1e-12


# ==========================================================================
# Tiện ích nội bộ
# ==========================================================================
def _to_dense(X, name: str = "X") -> np.ndarray:
    """
    Đưa đầu vào về mảng numpy dày 2 chiều.

    Chấp nhận cả ma trận thưa của scipy (TF-IDF của Linh) mà không cần
    import scipy — chỉ cần đối tượng có phương thức .toarray().
    """
    if hasattr(X, "toarray"):  # scipy.sparse
        n_cells = int(np.prod(X.shape))
        if n_cells > 50_000_000:
            raise MemoryError(
                f"{name} thưa có {n_cells:,} ô, chuyển sang dày sẽ tràn bộ nhớ. "
                "Hãy giảm chiều bằng TruncatedSVD (100-300 chiều) trước khi fit."
            )
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"{name} phải là ma trận 2 chiều, nhận được ndim={X.ndim}")
    return X


def _index_map(ids) -> dict:
    """Ánh xạ id gốc -> chỉ số hàng. Báo lỗi rõ ràng nếu id bị trùng."""
    out: dict = {}
    for pos, raw in enumerate(ids):
        if raw in out:
            raise ValueError(f"id bị trùng: {raw!r}")
        out[raw] = pos
    return out


def _topk_along(S: np.ndarray, k: int, axis: int) -> np.ndarray:
    """
    Mặt nạ boolean giữ lại k phần tử LỚN NHẤT theo trục đã cho.

    Ô có giá trị -inf (đã bị loại từ trước) không bao giờ được giữ.
    Dùng argpartition nên chi phí O(n) thay vì O(n log n) của argsort.
    """
    n = S.shape[axis]
    keep = np.zeros(S.shape, dtype=bool)
    if k >= n:
        np.copyto(keep, np.isfinite(S))
        return keep

    idx = np.argpartition(-S, k - 1, axis=axis)
    sel = np.take(idx, np.arange(k), axis=axis)
    np.put_along_axis(keep, sel, True, axis=axis)
    keep &= np.isfinite(S)
    return keep


def rmse(y_true, y_pred) -> float:
    """Căn bậc hai của trung bình bình phương sai số."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Sai số tuyệt đối trung bình."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


# ==========================================================================
# LỚP 1 — KNN trên ma trận đặc trưng (content-based)
# ==========================================================================
class KNNIndex:
    """
    Tìm k đối tượng gần nhất trong một không gian đặc trưng.

    Tham số
    -------
    metric : tên độ đo trong src.similarity (mặc định "cosine").
             Tên độ đo khoảng cách cũng dùng được, sẽ tự đổi sang similarity.
    k      : số láng giềng mặc định.
    min_sim: ngưỡng similarity tối thiểu; láng giềng dưới ngưỡng bị bỏ.
             Đặt 0.0 để loại các cặp không liên quan hoặc tương quan nghịch.

    Thuộc tính sau khi fit
    ----------------------
    sim_   : ma trận tương đồng n x n (đã tính sẵn, cache trong RAM)
    ids_   : mảng id gốc theo đúng thứ tự hàng
    """

    def __init__(self, metric: str = "cosine", k: int = 10, min_sim: float = 0.0):
        self.metric = metric
        self.k = int(k)
        self.min_sim = float(min_sim)
        self.sim_: np.ndarray | None = None
        self.ids_: np.ndarray | None = None
        self._pos: dict = {}

    # ---------------------------------------------------------------- fit
    def fit(self, X, ids=None) -> "KNNIndex":
        """
        X   : ma trận (n_doi_tuong x n_dac_trung). Nhận cả scipy.sparse.
        ids : id gốc của từng hàng (movieId, doc_id, ma_san_pham...).
              Bỏ trống thì dùng luôn chỉ số 0..n-1.
        """
        Xd = _to_dense(X, "X")
        n = Xd.shape[0]
        if n == 0:
            raise ValueError("X không có hàng nào")

        self.ids_ = np.arange(n) if ids is None else np.asarray(list(ids), dtype=object)
        if len(self.ids_) != n:
            raise ValueError(f"ids có {len(self.ids_)} phần tử nhưng X có {n} hàng")
        self._pos = _index_map(self.ids_)

        fn = sim_mod.get_similarity(self.metric)
        self.sim_ = np.asarray(fn(Xd), dtype=np.float64)
        np.fill_diagonal(self.sim_, 1.0)
        return self

    # ------------------------------------------------------------ truy vấn
    def _require_fit(self) -> None:
        if self.sim_ is None:
            raise RuntimeError("Phải gọi fit() trước")

    def similar_to(self, item_id, k: int | None = None) -> list[tuple]:
        """
        k đối tượng giống `item_id` nhất, trả về [(id, sim), ...] giảm dần.
        Bản thân item_id bị loại khỏi kết quả.
        """
        self._require_fit()
        if item_id not in self._pos:
            raise KeyError(f"Không có id {item_id!r} trong index")
        k = self.k if k is None else int(k)

        row = self.sim_[self._pos[item_id]].copy()
        row[self._pos[item_id]] = -np.inf  # loại chính nó
        row[row < self.min_sim] = -np.inf

        return self._take_top(row, k)

    def kneighbors_vector(self, vector, k: int | None = None) -> list[tuple]:
        """
        Giống similar_to nhưng đầu vào là một VECTOR đặc trưng mới, chưa có
        trong index. Dùng cho hàng mới hoàn toàn (cold-start của Hùng) khi
        sản phẩm chưa được fit vào index.

        Lưu ý: chỉ hỗ trợ các độ đo tính được theo từng cặp hàng, tức là
        cosine và các độ đo khoảng cách. Pearson/adjusted_cosine không dùng
        được ở đây vì chúng cần toàn bộ ma trận rating.
        """
        self._require_fit()
        if self.metric.lower() in ("pearson", "adjusted_cosine", "cosine_masked"):
            raise ValueError(
                f"metric='{self.metric}' cần toàn bộ ma trận rating, không dùng "
                "được với một vector rời. Hãy dùng metric='cosine'."
            )
        if self._X is None:
            raise RuntimeError("Index này không giữ lại X; hãy fit với keep_X=True")
        k = self.k if k is None else int(k)

        v = _to_dense(vector, "vector")
        fn = sim_mod.SIMILARITIES.get(self.metric.lower())
        if fn is None:
            row = sim_mod.distance_to_similarity(
                sim_mod.DISTANCES[self.metric.lower()](v, self._X)
            )[0]
        else:
            row = fn(v, self._X)[0]
        row = row.astype(np.float64).copy()
        row[row < self.min_sim] = -np.inf
        return self._take_top(row, k)

    def recommend(
        self,
        liked_ids,
        n: int = 10,
        weights=None,
        exclude_liked: bool = True,
    ) -> list[tuple]:
        """
        Gợi ý content-based từ những gì người dùng ĐÃ thích.

        Điểm của mỗi ứng viên = tổng similarity tới các đối tượng đã thích,
        có thể gắn trọng số. Đây chính là công thức dùng cho:
          - Linh: liked_ids = danh sách tài liệu user đã tải
          - Hùng: liked_ids = danh sách sản phẩm user đã mua

        weights: trọng số từng mục đã thích, cùng độ dài liked_ids.
                 Linh dùng để phân biệt tải (1) / mở lại (2) / bookmark (3).
        """
        self._require_fit()
        liked = [i for i in liked_ids if i in self._pos]
        if not liked:
            return []

        rows = np.array([self._pos[i] for i in liked])
        w = np.ones(len(liked)) if weights is None else np.asarray(weights, dtype=float)
        if len(w) != len(liked):
            raise ValueError("weights phải cùng độ dài với liked_ids")

        S = self.sim_[rows]  # (n_liked, n_all)
        S = np.where(S >= self.min_sim, S, 0.0)
        score = (S * w[:, None]).sum(axis=0) / max(w.sum(), EPS)

        score = score.astype(np.float64)
        if exclude_liked:
            score[rows] = -np.inf
        return self._take_top(score, n)

    def _take_top(self, row: np.ndarray, k: int) -> list[tuple]:
        valid = np.isfinite(row)
        if not valid.any():
            return []
        k = min(k, int(valid.sum()))
        cand = np.where(valid)[0]
        order = cand[np.argsort(-row[cand], kind="stable")[:k]]
        return [(self.ids_[j], float(row[j])) for j in order]

    # giữ X lại để phục vụ kneighbors_vector
    _X: np.ndarray | None = None

    def fit_keep(self, X, ids=None) -> "KNNIndex":
        """Như fit() nhưng giữ lại X để dùng được kneighbors_vector()."""
        self.fit(X, ids)
        self._X = _to_dense(X, "X")
        return self

    def __repr__(self) -> str:
        n = "chua fit" if self.sim_ is None else f"{self.sim_.shape[0]} doi tuong"
        return f"KNNIndex(metric={self.metric!r}, k={self.k}, {n})"


# ==========================================================================
# LỚP 2 — KNN dự đoán rating (collaborative filtering)
# ==========================================================================
class KNNRecommender:
    """
    Dự đoán điểm đánh giá bằng KNN, hỗ trợ cả user-based và item-based.

    Công thức item-based (user_based=False):

        r_ui = sum_{j in N(i)} sim(i,j) * r_uj  /  sum_{j in N(i)} |sim(i,j)|

        với N(i) là k item giống i nhất mà user u ĐÃ đánh giá.

    Công thức user-based (user_based=True), có khử thiên lệch cho điểm:

        r_ui = mu_u + sum_{v in N(u)} sim(u,v) * (r_vi - mu_v)
                      / sum_{v in N(u)} |sim(u,v)|

        với N(u) là k user giống u nhất mà ĐÃ đánh giá item i.

    Khi không đủ láng giềng, lần lượt rút về: trung bình của user ->
    trung bình của item -> trung bình toàn cục. Nhờ vậy hàm luôn trả về một
    số hợp lệ, không bao giờ NaN.

    Tham số
    -------
    k           : số láng giềng (quét 1,3,...,51 trong thực nghiệm Tuần 5)
    metric      : "adjusted_cosine" cho item-based, "pearson" cho user-based
    user_based  : False = item-based (NÊN DÙNG, nhanh và ổn định hơn)
    min_support : số ô đồng đánh giá tối thiểu để tin một cặp
    min_sim     : bỏ láng giềng có similarity dưới ngưỡng này
    rating_range: khoảng điểm hợp lệ để kẹp kết quả, ví dụ (0.5, 5.0)
    """

    def __init__(
        self,
        k: int = 20,
        metric: str = "adjusted_cosine",
        user_based: bool = False,
        min_support: int = 2,
        min_sim: float = 0.0,
        rating_range: tuple[float, float] | None = None,
    ):
        self.k = int(k)
        self.metric = metric
        self.user_based = bool(user_based)
        self.min_support = int(min_support)
        self.min_sim = float(min_sim)
        self.rating_range = rating_range

        self.R_: np.ndarray | None = None
        self.sim_: np.ndarray | None = None
        self.user_means_: np.ndarray | None = None
        self.item_means_: np.ndarray | None = None
        self.global_mean_: float = 0.0

    # ---------------------------------------------------------------- fit
    def fit(self, R, user_ids=None, item_ids=None) -> "KNNRecommender":
        """
        R : ma trận user x item, **0 nghĩa là chưa đánh giá**.
        Ma trận tương đồng được tính sẵn ở đây (offline) rồi cache lại, nên
        các lần predict/top_n sau đó rất nhanh — đúng như thiết kế trong
        Chương 3 của báo cáo.
        """
        self.R_ = _to_dense(R, "R")
        n_users, n_items = self.R_.shape
        self.mask_ = self.R_ != 0

        if not self.mask_.any():
            raise ValueError("R không có ô nào được đánh giá")

        self.user_ids_ = (
            np.arange(n_users) if user_ids is None
            else np.asarray(list(user_ids), dtype=object)
        )
        self.item_ids_ = (
            np.arange(n_items) if item_ids is None
            else np.asarray(list(item_ids), dtype=object)
        )
        if len(self.user_ids_) != n_users or len(self.item_ids_) != n_items:
            raise ValueError("Độ dài user_ids/item_ids không khớp kích thước R")
        self._upos = _index_map(self.user_ids_)
        self._ipos = _index_map(self.item_ids_)

        # Các mức rút về khi thiếu láng giềng
        self.user_means_ = sim_mod.masked_row_means(self.R_)
        self.item_means_ = sim_mod.masked_row_means(self.R_.T)
        self.global_mean_ = float(self.R_[self.mask_].mean())

        # Ma trận tương đồng
        fn_name = self.metric.lower()
        if self.user_based:
            if fn_name == "adjusted_cosine":
                warnings.warn(
                    "adjusted_cosine là độ đo giữa ITEM. Với user_based=True "
                    "nên dùng metric='pearson'. Đang tự chuyển sang pearson.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                fn_name = "pearson"
            self.sim_ = sim_mod.SIMILARITIES[fn_name](
                self.R_, min_support=self.min_support
            )
        else:
            if fn_name == "adjusted_cosine":
                self.sim_ = sim_mod.adjusted_cosine(
                    self.R_, min_support=self.min_support
                )
            elif fn_name in ("pearson", "cosine_masked"):
                # tương đồng giữa item -> truyền ma trận đã chuyển vị
                self.sim_ = sim_mod.SIMILARITIES[fn_name](
                    self.R_.T, min_support=self.min_support
                )
            else:
                self.sim_ = sim_mod.get_similarity(fn_name)(self.R_.T)
        self.sim_ = np.asarray(self.sim_, dtype=np.float64)
        np.fill_diagonal(self.sim_, 0.0)  # không tự làm láng giềng của mình
        return self

    def _require_fit(self) -> None:
        if self.R_ is None:
            raise RuntimeError("Phải gọi fit() trước")

    # ------------------------------------------------------------ dự đoán
    def _clip(self, x):
        if self.rating_range is None:
            return x
        lo, hi = self.rating_range
        return np.clip(x, lo, hi)

    def _predict_item_based(self, u: int, items: np.ndarray) -> np.ndarray:
        """Dự đoán điểm của user u cho một mảng chỉ số item."""
        rated = np.where(self.mask_[u])[0]
        fallback = np.where(
            self.mask_[:, items].any(axis=0), self.item_means_[items], self.global_mean_
        )
        if self.user_means_[u] > 0:
            fallback = np.where(
                self.mask_[:, items].any(axis=0), fallback, self.user_means_[u]
            )
        if rated.size == 0:
            return self._clip(fallback)

        S = self.sim_[np.ix_(items, rated)]  # (n_cand, n_rated)
        S = np.where(S >= self.min_sim, S, -np.inf)
        keep = _topk_along(S, self.k, axis=1)

        W = np.where(keep, np.where(np.isfinite(S), S, 0.0), 0.0)
        num = W @ self.R_[u, rated]
        den = np.abs(W).sum(axis=1)

        return self._clip(np.where(den > EPS, num / np.maximum(den, EPS), fallback))

    def _predict_user_based(self, u: int, items: np.ndarray) -> np.ndarray:
        """Dự đoán điểm của user u cho một mảng chỉ số item."""
        mu_u = self.user_means_[u] if self.user_means_[u] > 0 else self.global_mean_
        fallback = np.where(
            self.mask_[:, items].any(axis=0),
            mu_u + (self.item_means_[items] - self.global_mean_),
            mu_u,
        )

        sim_u = self.sim_[u].copy()
        sim_u[u] = -np.inf
        sim_u[sim_u < self.min_sim] = -np.inf

        # (n_users, n_cand): chỉ giữ user đã đánh giá item tương ứng
        valid = self.mask_[:, items]
        S = np.where(valid, sim_u[:, None], -np.inf)
        keep = _topk_along(S, self.k, axis=0)

        W = np.where(keep, np.where(np.isfinite(S), S, 0.0), 0.0)
        dev = np.where(valid, self.R_[:, items] - self.user_means_[:, None], 0.0)
        num = (W * dev).sum(axis=0)
        den = np.abs(W).sum(axis=0)

        return self._clip(np.where(den > EPS, mu_u + num / np.maximum(den, EPS), fallback))

    def predict(self, user, item) -> float:
        """Điểm dự đoán của một cặp (user, item) theo id gốc."""
        self._require_fit()
        if user not in self._upos:
            raise KeyError(f"Không có user {user!r}")
        if item not in self._ipos:
            raise KeyError(f"Không có item {item!r}")
        u = self._upos[user]
        i = np.array([self._ipos[item]])
        fn = self._predict_user_based if self.user_based else self._predict_item_based
        return float(fn(u, i)[0])

    def predict_all(self, user) -> np.ndarray:
        """Điểm dự đoán của một user cho TOÀN BỘ item, theo thứ tự item_ids_."""
        self._require_fit()
        u = self._upos[user]
        items = np.arange(self.R_.shape[1])
        fn = self._predict_user_based if self.user_based else self._predict_item_based
        return fn(u, items)

    def top_n(self, user, n: int = 10, exclude_rated: bool = True) -> list[tuple]:
        """
        n item nên gợi ý cho user, trả về [(item_id, diem_du_doan), ...].

        exclude_rated=True (mặc định) loại những item user đã đánh giá —
        gợi ý lại phim người ta vừa chấm điểm là lỗi hay gặp nhất khi demo.
        """
        self._require_fit()
        if user not in self._upos:
            raise KeyError(f"Không có user {user!r}")
        u = self._upos[user]

        scores = self.predict_all(user).astype(np.float64)
        if exclude_rated:
            scores[self.mask_[u]] = -np.inf

        valid = np.isfinite(scores)
        if not valid.any():
            return []
        cand = np.where(valid)[0]
        order = cand[np.argsort(-scores[cand], kind="stable")[: int(n)]]
        return [(self.item_ids_[j], float(scores[j])) for j in order]

    def evaluate(self, test_triples) -> dict:
        """
        RMSE và MAE trên tập kiểm tra.

        test_triples: iterable của (user_id, item_id, diem_thuc).
        Cặp có user hoặc item chưa từng thấy sẽ bị bỏ qua và được đếm riêng
        vào "bo_qua" — đây chính là phần cold-start cần nêu trong Chương 4.
        """
        self._require_fit()
        y_true, y_pred, skipped = [], [], 0
        for user, item, truth in test_triples:
            if user not in self._upos or item not in self._ipos:
                skipped += 1
                continue
            y_true.append(float(truth))
            y_pred.append(self.predict(user, item))
        if not y_true:
            return {"rmse": float("nan"), "mae": float("nan"), "n": 0, "bo_qua": skipped}
        return {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "n": len(y_true),
            "bo_qua": skipped,
        }

    def __repr__(self) -> str:
        kind = "user-based" if self.user_based else "item-based"
        state = "chua fit" if self.R_ is None else f"R={self.R_.shape}"
        return f"KNNRecommender({kind}, k={self.k}, metric={self.metric!r}, {state})"


# ==========================================================================
# SELF-TEST — chay: python src/knn_core.py
# ==========================================================================
def _ref_item_based(R, sim, u, i, k, min_sim=0.0):
    """Ban viet thang theo cong thuc, dung lam chuan doi chieu."""
    rated = [j for j in range(R.shape[1]) if R[u, j] != 0 and j != i]
    cands = [(sim[i, j], j) for j in rated if sim[i, j] >= min_sim]
    cands.sort(key=lambda t: -t[0])
    cands = cands[:k]
    num = sum(s * R[u, j] for s, j in cands)
    den = sum(abs(s) for s, _ in cands)
    if den < EPS:
        return None
    return num / den


def _ref_user_based(R, sim, mu, u, i, k, gmean, min_sim=0.0):
    """Ban viet thang theo cong thuc, dung lam chuan doi chieu."""
    others = [v for v in range(R.shape[0]) if v != u and R[v, i] != 0]
    cands = [(sim[u, v], v) for v in others if sim[u, v] >= min_sim]
    cands.sort(key=lambda t: -t[0])
    cands = cands[:k]
    num = sum(s * (R[v, i] - mu[v]) for s, v in cands)
    den = sum(abs(s) for s, _ in cands)
    mu_u = mu[u] if mu[u] > 0 else gmean
    if den < EPS:
        return None
    return mu_u + num / den


def _selftest() -> None:
    ok = fail = 0

    def check(label, got, want, tol=1e-9):
        nonlocal ok, fail
        g = np.asarray(got, dtype=float)
        w = np.asarray(want, dtype=float)
        if g.shape == w.shape and np.allclose(g, w, atol=tol):
            print(f"  [OK]   {label}")
            ok += 1
        else:
            print(f"  [FAIL] {label}\n         got  = {g}\n         want = {w}")
            fail += 1

    def check_true(label, cond, extra=""):
        nonlocal ok, fail
        if cond:
            print(f"  [OK]   {label}" + (f"  {extra}" if extra else ""))
            ok += 1
        else:
            print(f"  [FAIL] {label}" + (f"  {extra}" if extra else ""))
            fail += 1

    print("=" * 70)
    print("SELF-TEST  src/knn_core.py")
    print("=" * 70)

    # ---------------- Vi du 5x5 dung trong bao cao ----------------------
    R = np.array(
        [
            [5, 3, 0, 1, 4],
            [4, 0, 4, 1, 0],
            [1, 1, 5, 0, 2],
            [0, 2, 4, 5, 1],
            [5, 4, 0, 2, 0],
        ],
        dtype=float,
    )
    print("\n-- Ma tran rating 5 user x 5 phim (0 = chua xem) --")
    print(R.astype(int))

    print("\n[1] ITEM-BASED: vector hoa vs cong thuc viet thang")
    rec = KNNRecommender(k=3, metric="adjusted_cosine", user_based=False, min_support=1)
    rec.fit(R)
    for u, i in [(0, 2), (1, 1), (2, 3), (4, 2), (3, 0)]:
        got = rec.predict(u, i)
        want = _ref_item_based(R, rec.sim_, u, i, k=3)
        if want is None:
            check_true(f"predict({u},{i}) rut ve trung binh", np.isfinite(got),
                       f"= {got:.4f}")
        else:
            check(f"predict({u},{i})", got, want)

    print("\n[2] USER-BASED: vector hoa vs cong thuc viet thang")
    recu = KNNRecommender(k=3, metric="pearson", user_based=True, min_support=2)
    recu.fit(R)
    for u, i in [(0, 2), (1, 1), (2, 3), (4, 2), (3, 0)]:
        got = recu.predict(u, i)
        want = _ref_user_based(
            R, recu.sim_, recu.user_means_, u, i, k=3, gmean=recu.global_mean_
        )
        if want is None:
            check_true(f"predict({u},{i}) rut ve trung binh", np.isfinite(got),
                       f"= {got:.4f}")
        else:
            check(f"predict({u},{i})", got, want)

    print("\n[3] predict_all khop voi predict tung cap")
    allp = rec.predict_all(0)
    one = [rec.predict(0, i) for i in range(5)]
    check("predict_all == predict", allp, one)

    print("\n[4] top_n")
    top = rec.top_n(0, n=5)
    print(f"         top_n(user=0) = {[(int(i), round(s, 3)) for i, s in top]}")
    da_xem = set(np.where(R[0] != 0)[0].tolist())
    check_true(
        "khong goi y lai phim da xem",
        all(int(i) not in da_xem for i, _ in top),
    )
    check_true("diem giam dan", all(top[t][1] >= top[t + 1][1] for t in range(len(top) - 1)))
    check_true("so luong dung", len(top) == 5 - len(da_xem), f"= {len(top)}")

    print("\n[5] Kep diem vao khoang hop le")
    reck = KNNRecommender(k=3, metric="adjusted_cosine", min_support=1,
                          rating_range=(1.0, 5.0)).fit(R)
    preds = [reck.predict(u, i) for u in range(5) for i in range(5)]
    check_true("moi du doan nam trong [1, 5]", all(1.0 - 1e-9 <= p <= 5.0 + 1e-9 for p in preds))

    print("\n[6] k lon hon dung nhieu lang gieng hon")
    r1 = KNNRecommender(k=1, metric="adjusted_cosine", min_support=1).fit(R)
    r9 = KNNRecommender(k=9, metric="adjusted_cosine", min_support=1).fit(R)
    khac = sum(
        abs(r1.predict(u, i) - r9.predict(u, i)) > 1e-6
        for u in range(5) for i in range(5)
    )
    check_true("k=1 va k=9 cho ket qua khac nhau", khac > 0, f"({khac} cap khac)")

    print("\n[7] Khong bao gio tra ve NaN, ke ca user chi co 1 danh gia")
    R2 = R.copy()
    R2 = np.vstack([R2, [0, 0, 0, 0, 3]])  # user moi, chi danh gia 1 phim
    R2 = np.hstack([R2, np.zeros((6, 1))])  # item moi, chua ai danh gia
    rec2 = KNNRecommender(k=3, metric="adjusted_cosine", min_support=1).fit(R2)
    vals = [rec2.predict(u, i) for u in range(6) for i in range(6)]
    check_true("khong co NaN/inf", all(np.isfinite(v) for v in vals))
    print(f"         user moi (5) voi item moi (5): {rec2.predict(5, 5):.4f}")

    print("\n[8] KNNIndex tren dac trung: lang gieng phai dung bang truc giac")
    X = np.array([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]])
    idx = KNNIndex(metric="cosine", k=1).fit(X, ids=["a", "a2", "b", "b2"])
    check_true("a gan a2 nhat", idx.similar_to("a")[0][0] == "a2")
    check_true("b gan b2 nhat", idx.similar_to("b")[0][0] == "b2")
    check_true("khong tra ve chinh no", all(i != "a" for i, _ in idx.similar_to("a", k=3)))

    print("\n[9] KNNIndex.recommend tu lich su")
    docs = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],  # d0  chu de A
            [1.0, 0.9, 0.0, 0.0],  # d1  chu de A
            [0.0, 0.0, 1.0, 1.0],  # d2  chu de B
            [0.0, 0.0, 1.0, 0.9],  # d3  chu de B
            [0.9, 1.0, 0.1, 0.0],  # d4  chu de A
        ]
    )
    di = KNNIndex(metric="cosine", k=3).fit(docs, ids=[f"d{t}" for t in range(5)])
    goi_y = di.recommend(liked_ids=["d0"], n=2)
    print(f"         da tai d0 -> goi y {[(i, round(s, 3)) for i, s in goi_y]}")
    check_true("chi goi y tai lieu cung chu de A", {i for i, _ in goi_y} <= {"d1", "d4"})
    check_true("khong goi y lai d0", all(i != "d0" for i, _ in goi_y))

    print("\n[10] Trong so cua recommend co tac dung")
    g1 = dict(di.recommend(["d0", "d2"], n=5, weights=[1, 1]))
    g2 = dict(di.recommend(["d0", "d2"], n=5, weights=[1, 10]))
    check_true("tang trong so cho d2 thi d3 len diem", g2["d3"] > g1["d3"])

    print("\n[11] Co hoc duoc gi that khong: so voi baseline trung binh")
    rng = np.random.default_rng(7)
    n_u, n_i, d = 120, 60, 3
    U = rng.random((n_u, d))
    V = rng.random((n_i, d))
    full = 1.0 + 4.0 * (U @ V.T) / d  # diem "that" co cau truc an
    obs = rng.random((n_u, n_i)) < 0.30
    Rtr = np.where(obs, np.round(full * 2) / 2, 0.0)

    test = []
    for u in range(n_u):
        for i in range(n_i):
            if not obs[u, i]:
                test.append((u, i, full[u, i]))
    test = test[:3000]

    m = KNNRecommender(k=20, metric="adjusted_cosine", min_support=2,
                       rating_range=(1.0, 5.0)).fit(Rtr)
    kq = m.evaluate(test)
    base = rmse([t[2] for t in test], [m.global_mean_] * len(test))
    print(f"         KNN item-based : RMSE = {kq['rmse']:.4f}  MAE = {kq['mae']:.4f}")
    print(f"         baseline (trung binh toan cuc): RMSE = {base:.4f}")
    check_true("KNN tot hon baseline", kq["rmse"] < base,
               f"(giam {100 * (1 - kq['rmse'] / base):.1f}%)")

    mu_ = KNNRecommender(k=20, metric="pearson", user_based=True, min_support=2,
                         rating_range=(1.0, 5.0)).fit(Rtr)
    kqu = mu_.evaluate(test)
    print(f"         KNN user-based : RMSE = {kqu['rmse']:.4f}  MAE = {kqu['mae']:.4f}")
    check_true("user-based cung tot hon baseline", kqu["rmse"] < base)

    print("\n[12] Canh bao khi dung sai do do cho user-based")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        KNNRecommender(k=3, metric="adjusted_cosine", user_based=True).fit(R)
        check_true("co canh bao va tu chuyen sang pearson", len(w) == 1)

    print("\n[13] Bao loi ro rang khi dung sai")
    try:
        KNNRecommender().predict(0, 0)
        check_true("chua fit phai bao loi", False)
    except RuntimeError:
        check_true("chua fit -> RuntimeError", True)
    try:
        rec.predict(999, 0)
        check_true("user la phai bao loi", False)
    except KeyError:
        check_true("user la -> KeyError", True)

    print("\n" + "=" * 70)
    print(f"KET QUA: {ok} dat / {fail} loi")
    print("=" * 70)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    _selftest()
