"""
Các độ đo khoảng cách và độ tương đồng dùng chung cho cả 3 module gợi ý.

CHỦ SỞ HỮU: Hoàng — API đã chốt (Tuần 2).
Linh và Hùng gọi các hàm ở đây, KHÔNG tự viết lại hàm cosine/pearson riêng.

--------------------------------------------------------------------------
QUY ƯỚC QUAN TRỌNG VỀ Ý NGHĨA CỦA SỐ 0
--------------------------------------------------------------------------
Có hai loại ma trận trong đồ án này và số 0 mang ý nghĩa KHÁC NHAU:

1. Ma trận RATING (user x item) — 0 nghĩa là "chưa đánh giá" (missing).
   Dùng: pearson(), adjusted_cosine(), cosine_masked(), jaccard()

2. Ma trận ĐẶC TRƯNG (item x feature) — 0 là giá trị thật.
   Ví dụ: TF-IDF của tài liệu, vector thuộc tính sản phẩm.
   Dùng: cosine(), euclidean(), manhattan(), minkowski(), hamming()

Gọi sai nhóm hàm là lỗi logic, không phải lỗi chạy — chương trình vẫn ra
số nhưng số đó vô nghĩa. Xem bảng chọn hàm ở dưới.

--------------------------------------------------------------------------
BẢNG CHỌN HÀM
--------------------------------------------------------------------------
Bài toán                          | Hàm nên dùng
----------------------------------|-------------------------------------
Gợi ý phim, user-based CF         | pearson(R)
Gợi ý phim, item-based CF         | adjusted_cosine(R)
Gợi ý tài liệu (TF-IDF)           | cosine(X)
Gợi ý hàng mới (đặc trưng SP)     | cosine(X) sau khi đã chuẩn hóa
Dữ liệu nhị phân (đã tải/chưa)    | jaccard(X)
Minh họa lý thuyết Chương 1       | euclidean / manhattan / minkowski
--------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "cosine",
    "cosine_masked",
    "pearson",
    "adjusted_cosine",
    "jaccard",
    "euclidean",
    "manhattan",
    "minkowski",
    "hamming",
    "distance_to_similarity",
    "masked_row_means",
    "get_similarity",
    "SIMILARITIES",
    "DISTANCES",
]

# Ngưỡng chống chia cho 0
EPS = 1e-12

# Cảnh báo khi ma trận kết quả vượt ngưỡng này (số phần tử)
_WARN_CELLS = 40_000_000  # ~320 MB với float64


# ==========================================================================
# Tiện ích nội bộ
# ==========================================================================
def _as_float_2d(X, name: str = "X") -> np.ndarray:
    """Đưa đầu vào về mảng numpy 2 chiều kiểu float."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"{name} phải là ma trận 2 chiều, nhận được ndim={X.ndim}")
    return X


def _check_size(n_rows_x: int, n_rows_y: int) -> None:
    cells = n_rows_x * n_rows_y
    if cells > _WARN_CELLS:
        warnings.warn(
            f"Ma trận kết quả {n_rows_x}x{n_rows_y} = {cells:,} phần tử "
            f"(~{cells * 8 / 1e9:.1f} GB). Hãy lọc bớt user/item có ít tương tác "
            f"trước khi tính, hoặc giảm chiều bằng SVD.",
            RuntimeWarning,
            stacklevel=3,
        )


def masked_row_means(R: np.ndarray, missing: float = 0.0) -> np.ndarray:
    """
    Trung bình của từng HÀNG, chỉ tính trên các ô đã có giá trị.

    Dùng để lấy điểm trung bình mỗi user (khử thói quen cho điểm cao/thấp).
    Hàng không có ô nào được đánh giá thì trả về 0.
    """
    R = _as_float_2d(R, "R")
    mask = R != missing
    counts = mask.sum(axis=1)
    sums = np.where(mask, R, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return means


def distance_to_similarity(D: np.ndarray) -> np.ndarray:
    """
    Đổi khoảng cách thành độ tương đồng: sim = 1 / (1 + d).

    KNN luôn chọn k láng giềng có sim LỚN nhất, nên mọi độ đo khoảng cách
    phải đi qua hàm này trước khi đưa vào knn_core.
    """
    D = np.asarray(D, dtype=np.float64)
    return 1.0 / (1.0 + D)


# ==========================================================================
# Nhóm 1 — dùng cho MA TRẬN ĐẶC TRƯNG (0 là giá trị thật)
# ==========================================================================
def cosine(X, Y=None) -> np.ndarray:
    """
    Cosine similarity giữa các HÀNG.

        sim(x, y) = (x · y) / (||x|| * ||y||)

    Kết quả nằm trong [-1, 1]; với dữ liệu không âm (TF-IDF) thì trong [0, 1].
    Đây là độ đo chuẩn cho vector thưa, nhiều chiều — dùng cho module tài liệu
    của Linh và module hàng mới của Hùng.

    Hàng toàn 0 cho similarity = 0 với mọi hàng khác (không phải NaN).
    """
    X = _as_float_2d(X, "X")
    Y = X if Y is None else _as_float_2d(Y, "Y")
    _check_size(X.shape[0], Y.shape[0])

    nx = np.linalg.norm(X, axis=1, keepdims=True)
    ny = np.linalg.norm(Y, axis=1, keepdims=True)
    Xn = X / np.maximum(nx, EPS)
    Yn = Y / np.maximum(ny, EPS)
    return Xn @ Yn.T


def euclidean(X, Y=None) -> np.ndarray:
    """Khoảng cách Euclidean (L2) giữa các hàng: d = sqrt(sum (xi - yi)^2)."""
    X = _as_float_2d(X, "X")
    Y = X if Y is None else _as_float_2d(Y, "Y")
    _check_size(X.shape[0], Y.shape[0])

    # Dùng hằng đẳng thức ||x-y||^2 = ||x||^2 + ||y||^2 - 2x·y cho nhanh
    x2 = (X**2).sum(axis=1)[:, None]
    y2 = (Y**2).sum(axis=1)[None, :]
    d2 = x2 + y2 - 2.0 * (X @ Y.T)
    np.maximum(d2, 0.0, out=d2)  # khử sai số âm rất nhỏ
    return np.sqrt(d2)


def manhattan(X, Y=None) -> np.ndarray:
    """Khoảng cách Manhattan (L1): d = sum |xi - yi|."""
    X = _as_float_2d(X, "X")
    Y = X if Y is None else _as_float_2d(Y, "Y")
    _check_size(X.shape[0], Y.shape[0])
    return np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=2)


def minkowski(X, Y=None, p: float = 3.0) -> np.ndarray:
    """
    Khoảng cách Minkowski: d = (sum |xi - yi|^p)^(1/p).

    p = 1 cho Manhattan, p = 2 cho Euclidean. Hàm này để minh họa dạng tổng
    quát trong Chương 1; khi chạy thật nên gọi manhattan/euclidean cho nhanh.
    """
    if p <= 0:
        raise ValueError("p phải lớn hơn 0")
    X = _as_float_2d(X, "X")
    Y = X if Y is None else _as_float_2d(Y, "Y")
    _check_size(X.shape[0], Y.shape[0])
    diff = np.abs(X[:, None, :] - Y[None, :, :])
    return (diff**p).sum(axis=2) ** (1.0 / p)


def hamming(X, Y=None) -> np.ndarray:
    """
    Khoảng cách Hamming: SỐ vị trí có giá trị khác nhau.

    Dùng cho đặc trưng phân loại đã mã hóa (genre, thể loại tài liệu).
    """
    X = _as_float_2d(X, "X")
    Y = X if Y is None else _as_float_2d(Y, "Y")
    _check_size(X.shape[0], Y.shape[0])
    return (X[:, None, :] != Y[None, :, :]).sum(axis=2).astype(np.float64)


# ==========================================================================
# Nhóm 2 — dùng cho MA TRẬN RATING (0 = chưa đánh giá)
# ==========================================================================
def jaccard(X, Y=None) -> np.ndarray:
    """
    Jaccard similarity cho dữ liệu nhị phân: |A giao B| / |A hợp B|.

    Dùng cho phản hồi tiềm ẩn của module tài liệu: 1 = đã tải, 0 = chưa tải.
    Mọi giá trị khác 0 đều được coi là 1.
    """
    X = (_as_float_2d(X, "X") != 0).astype(np.float64)
    Y = X if Y is None else (_as_float_2d(Y, "Y") != 0).astype(np.float64)
    _check_size(X.shape[0], Y.shape[0])

    inter = X @ Y.T
    sx = X.sum(axis=1)[:, None]
    sy = Y.sum(axis=1)[None, :]
    union = sx + sy - inter
    return np.where(union > 0, inter / np.maximum(union, EPS), 0.0)


def cosine_masked(R, min_support: int = 1) -> np.ndarray:
    """
    Cosine chỉ tính trên các ô mà CẢ HAI hàng đều đã đánh giá.

    Khác với cosine() ở chỗ: ô chưa đánh giá bị loại khỏi phép tính thay vì
    bị coi là điểm 0. Coi "chưa xem" là "0 điểm" làm méo kết quả rất nặng —
    đây là lỗi kinh điển khi làm hệ gợi ý.

    min_support: số ô đồng đánh giá tối thiểu; dưới ngưỡng này trả về 0 vì
    không đủ căn cứ (2 user chỉ trùng 1 phim có thể ra sim = 1.0 giả tạo).
    """
    R = _as_float_2d(R, "R")
    _check_size(R.shape[0], R.shape[0])

    M = (R != 0).astype(np.float64)
    R0 = np.where(M > 0, R, 0.0)

    n = M @ M.T  # số ô đồng đánh giá
    num = R0 @ R0.T  # sum r_ui * r_vi  (chỉ ô đồng đánh giá góp phần)
    sq_u = (R0**2) @ M.T  # sum r_ui^2 trên ô đồng đánh giá
    sq_v = sq_u.T  # sum r_vi^2 trên ô đồng đánh giá

    den = np.sqrt(np.maximum(sq_u, 0.0) * np.maximum(sq_v, 0.0))
    sim = np.where((den > EPS) & (n >= min_support), num / np.maximum(den, EPS), 0.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def pearson(R, min_support: int = 1) -> np.ndarray:
    """
    Hệ số tương quan Pearson giữa các HÀNG, chỉ trên các ô ĐỒNG ĐÁNH GIÁ.

        sim(u,v) = sum (r_ui - mu_u)(r_vi - mu_v)
                   / sqrt(sum (r_ui - mu_u)^2 * sum (r_vi - mu_v)^2)

    trong đó mọi tổng chỉ chạy trên tập item mà cả u và v đều đã đánh giá, và
    mu_u là trung bình của u TRÊN TẬP ĐÓ (đúng theo định nghĩa Pearson, giống
    thư viện Surprise).

    Đây là độ đo nên dùng cho user-based CF: người A luôn cho 4-5 sao, người B
    luôn cho 2-3 sao — Cosine nói họ khác nhau, Pearson nói họ cùng gu.

    Cách tính: khai triển đại số để vector hóa hoàn toàn bằng 4 phép nhân
    ma trận, thay vì lặp qua từng cặp (nhanh hơn hàng trăm lần).

        num = sum(r_u r_v) - sum(r_u) * sum(r_v) / n
        den = sqrt[ (sum r_u^2 - (sum r_u)^2/n) * (sum r_v^2 - (sum r_v)^2/n) ]

    Cặp có ít hơn min_support ô đồng đánh giá, hoặc một bên không có biến
    thiên (cho cùng một điểm cho mọi item), trả về 0.
    """
    R = _as_float_2d(R, "R")
    _check_size(R.shape[0], R.shape[0])

    M = (R != 0).astype(np.float64)
    R0 = np.where(M > 0, R, 0.0)

    n = M @ M.T  # n[u,v] = số item đồng đánh giá
    sum_u = R0 @ M.T  # sum_u[u,v] = sum r_ui trên tập đồng đánh giá
    sum_v = sum_u.T  # sum_v[u,v] = sum r_vi trên tập đó
    sum_uv = R0 @ R0.T  # sum r_ui * r_vi
    sq_u = (R0**2) @ M.T  # sum r_ui^2 trên tập đó
    sq_v = sq_u.T

    n_safe = np.maximum(n, 1.0)
    num = sum_uv - (sum_u * sum_v) / n_safe
    var_u = np.maximum(sq_u - (sum_u**2) / n_safe, 0.0)
    var_v = np.maximum(sq_v - (sum_v**2) / n_safe, 0.0)
    den = np.sqrt(var_u * var_v)

    ok = (den > EPS) & (n >= max(min_support, 2))
    sim = np.where(ok, num / np.maximum(den, EPS), 0.0)
    np.clip(sim, -1.0, 1.0, out=sim)
    np.fill_diagonal(sim, 1.0)
    return sim


def adjusted_cosine(R, min_support: int = 1) -> np.ndarray:
    """
    Adjusted Cosine giữa các ITEM — độ đo chuẩn cho item-based CF.

    R là ma trận user x item (0 = chưa đánh giá). TRẢ VỀ MA TRẬN ITEM x ITEM.

    Trước khi tính cosine, mỗi điểm được trừ đi trung bình CỦA USER đó (không
    phải trung bình của item). Nhờ vậy khử được thiên lệch cho điểm của từng
    người — điều mà cosine thường không làm được khi so sánh hai item.

        sim(i,j) = sum_u (r_ui - mu_u)(r_uj - mu_u)
                   / sqrt(sum_u (r_ui - mu_u)^2 * sum_u (r_uj - mu_u)^2)

    với u chạy trên tập user đã đánh giá cả i và j.

    LƯU Ý BỘ NHỚ: kết quả là ma trận n_item x n_item. MovieLens có 9.700 phim
    thì ma trận này nặng ~750 MB. Hãy lọc bỏ item có dưới 5 lượt đánh giá
    trước khi gọi (việc của preprocess.py — Linh).
    """
    R = _as_float_2d(R, "R")
    n_items = R.shape[1]
    _check_size(n_items, n_items)

    M = (R != 0).astype(np.float64)
    mu = masked_row_means(R)[:, None]  # trung bình theo user
    Rc = np.where(M > 0, R - mu, 0.0)  # đã trừ trung bình user

    n = M.T @ M  # số user đồng đánh giá cặp item
    num = Rc.T @ Rc
    sq_i = (Rc**2).T @ M  # sum (r_ui - mu_u)^2 trên tập user đồng đánh giá
    sq_j = sq_i.T

    den = np.sqrt(np.maximum(sq_i, 0.0) * np.maximum(sq_j, 0.0))
    ok = (den > EPS) & (n >= min_support)
    sim = np.where(ok, num / np.maximum(den, EPS), 0.0)
    np.clip(sim, -1.0, 1.0, out=sim)
    np.fill_diagonal(sim, 1.0)
    return sim


# ==========================================================================
# Sổ đăng ký — để experiments/ quét qua nhiều độ đo bằng một vòng lặp
# ==========================================================================
SIMILARITIES = {
    "cosine": cosine,
    "cosine_masked": cosine_masked,
    "pearson": pearson,
    "adjusted_cosine": adjusted_cosine,
    "jaccard": jaccard,
}

DISTANCES = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "minkowski": minkowski,
    "hamming": hamming,
}


def get_similarity(name: str):
    """
    Lấy hàm độ tương đồng theo tên. Nếu tên là một độ đo KHOẢNG CÁCH thì
    tự động bọc qua distance_to_similarity().

    Dùng trong experiments/ để so sánh các độ đo:

        for ten in ["cosine_masked", "pearson", "euclidean"]:
            S = get_similarity(ten)(R)
    """
    key = name.strip().lower()
    if key in SIMILARITIES:
        return SIMILARITIES[key]
    if key in DISTANCES:
        fn = DISTANCES[key]

        def wrapped(X, Y=None, **kw):
            return distance_to_similarity(fn(X, Y, **kw))

        wrapped.__name__ = f"sim_from_{key}"
        return wrapped
    raise KeyError(
        f"Khong co do do '{name}'. Hop le: "
        f"{sorted(SIMILARITIES) + sorted(DISTANCES)}"
    )


# ==========================================================================
# SELF-TEST — chay: python src/similarity.py
# Doi chieu ban vector hoa voi ban vong lap thuan theo dung cong thuc sach.
# ==========================================================================
def _ref_pearson_pair(ru, rv, min_support=2):
    """Pearson cho 1 cap, viet thang theo dinh nghia (dung lam chuan doi chieu)."""
    both = (ru != 0) & (rv != 0)
    if both.sum() < min_support:
        return 0.0
    a, b = ru[both], rv[both]
    a = a - a.mean()
    b = b - b.mean()
    da = np.sqrt((a**2).sum())
    db = np.sqrt((b**2).sum())
    if da < EPS or db < EPS:
        return 0.0
    return float((a * b).sum() / (da * db))


def _ref_adjusted_cosine_pair(R, i, j, min_support=1):
    """Adjusted cosine cho 1 cap item, viet thang theo dinh nghia."""
    mu = masked_row_means(R)
    both = (R[:, i] != 0) & (R[:, j] != 0)
    if both.sum() < min_support:
        return 0.0
    a = R[both, i] - mu[both]
    b = R[both, j] - mu[both]
    da = np.sqrt((a**2).sum())
    db = np.sqrt((b**2).sum())
    if da < EPS or db < EPS:
        return 0.0
    return float((a * b).sum() / (da * db))


def _selftest() -> None:
    rng = np.random.default_rng(42)
    ok = 0
    fail = 0

    def check(label, got, want, tol=1e-9):
        nonlocal ok, fail
        got = np.asarray(got, dtype=float)
        want = np.asarray(want, dtype=float)
        if got.shape == want.shape and np.allclose(got, want, atol=tol):
            print(f"  [OK]   {label}")
            ok += 1
        else:
            print(f"  [FAIL] {label}")
            print(f"         got  = {got}")
            print(f"         want = {want}")
            fail += 1

    print("=" * 70)
    print("SELF-TEST  src/similarity.py")
    print("=" * 70)

    # --- vi du 5x5 dung trong bao cao (Chuong 1) ---------------------------
    # 5 user x 5 phim, 0 = chua xem
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

    print("\n-- Vi du 5x5 (dung cho Chuong 1 bao cao) --")
    print(R.astype(int))

    print("\n[1] Trung binh moi user, chi tinh tren phim da xem")
    mu = masked_row_means(R)
    # user 0: (5+3+1+4)/4 = 3.25 ; user 1: (4+4+1)/3 = 3.0
    check("masked_row_means", mu, [3.25, 3.0, 2.25, 3.0, 11 / 3])

    print("\n[2] Pearson: ban vector hoa vs ban vong lap thuan")
    S = pearson(R, min_support=2)
    ref = np.array(
        [[_ref_pearson_pair(R[i], R[j]) for j in range(5)] for i in range(5)]
    )
    np.fill_diagonal(ref, 1.0)
    check("pearson (toan ma tran 5x5)", S, ref)
    check("pearson doi xung", S, S.T)
    print(f"         sim(user0, user4) = {S[0, 4]:+.4f}")
    print(f"         sim(user0, user2) = {S[0, 2]:+.4f}")

    print("\n[3] Adjusted cosine: ban vector hoa vs ban vong lap thuan")
    A = adjusted_cosine(R, min_support=1)
    refA = np.array(
        [[_ref_adjusted_cosine_pair(R, i, j) for j in range(5)] for i in range(5)]
    )
    np.fill_diagonal(refA, 1.0)
    check("adjusted_cosine (item x item)", A, refA)

    print("\n[4] Cosine: truong hop tinh tay duoc")
    # x=(1,0), y=(0,1) vuong goc -> 0 ; x=(1,1), y=(2,2) cung huong -> 1
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    C = cosine(X)
    check("cosine vuong goc", C[0, 1], 0.0)
    check("cosine cung huong", C[2, 3], 1.0)
    check("cosine 45 do", C[0, 2], 1 / np.sqrt(2))
    check("duong cheo = 1", np.diag(C), np.ones(4))

    print("\n[5] cosine_masked KHAC cosine tren ma tran rating")
    cm = cosine_masked(R, min_support=1)
    cp = cosine(R)
    diff = np.abs(cm - cp).max()
    if diff > 1e-6:
        print(f"  [OK]   hai ham cho ket qua khac nhau (lech toi da {diff:.4f})")
        print("         -> dung cosine_masked cho rating, cosine cho dac trung")
        ok += 1
    else:
        print("  [FAIL] hai ham ra giong nhau, kiem tra lai mask")
        fail += 1

    print("\n[6] Jaccard: tinh tay")
    B = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    J = jaccard(B)
    # hang0 & hang1: giao = {item0} = 1, hop = {0,1,2} = 3 -> 1/3
    check("jaccard 1/3", J[0, 1], 1 / 3)
    check("jaccard roi nhau = 0", J[0, 2], 0.0)

    print("\n[7] Khoang cach: tinh tay")
    P = np.array([[0.0, 0.0], [3.0, 4.0]])
    check("euclidean 3-4-5", euclidean(P)[0, 1], 5.0)
    check("manhattan = 7", manhattan(P)[0, 1], 7.0)
    check("minkowski p=1 == manhattan", minkowski(P, p=1)[0, 1], 7.0)
    check("minkowski p=2 == euclidean", minkowski(P, p=2)[0, 1], 5.0)
    Q = np.array([[1, 0, 1, 1], [1, 1, 0, 1]], dtype=float)
    check("hamming = 2", hamming(Q)[0, 1], 2.0)

    print("\n[8] distance_to_similarity: don dieu giam")
    d = np.array([0.0, 1.0, 3.0])
    check("1/(1+d)", distance_to_similarity(d), [1.0, 0.5, 0.25])

    print("\n[9] Truong hop bien")
    check("hang toan 0 -> cosine 0", cosine(np.array([[0.0, 0.0], [1.0, 1.0]]))[0, 1], 0.0)
    flat = np.array([[3, 3, 3, 0], [3, 3, 3, 0]], dtype=float)
    check("khong co bien thien -> pearson 0", pearson(flat)[0, 1], 0.0)
    one = np.array([[5, 0, 0], [5, 0, 0]], dtype=float)
    check("chi 1 o trung -> pearson 0 (min_support)", pearson(one, min_support=2)[0, 1], 0.0)

    print("\n[10] get_similarity")
    check("ten similarity", get_similarity("pearson")(R), S)
    check(
        "ten distance duoc boc tu dong",
        get_similarity("euclidean")(P)[0, 1],
        1 / (1 + 5.0),
    )
    try:
        get_similarity("khong_ton_tai")
        print("  [FAIL] ten sai phai bao loi")
        fail += 1
    except KeyError:
        print("  [OK]   ten sai bao KeyError dung nhu mong doi")
        ok += 1

    print("\n[11] Ma tran lon ngau nhien: vector hoa == vong lap")
    Rb = rng.integers(0, 6, size=(30, 20)).astype(float)
    Rb[rng.random((30, 20)) < 0.5] = 0.0
    Sb = pearson(Rb, min_support=2)
    refB = np.array(
        [[_ref_pearson_pair(Rb[i], Rb[j]) for j in range(30)] for i in range(30)]
    )
    np.fill_diagonal(refB, 1.0)
    check("pearson 30x20 ngau nhien", Sb, refB)
    Ab = adjusted_cosine(Rb, min_support=1)
    refAb = np.array(
        [[_ref_adjusted_cosine_pair(Rb, i, j) for j in range(20)] for i in range(20)]
    )
    np.fill_diagonal(refAb, 1.0)
    check("adjusted_cosine 20x20 ngau nhien", Ab, refAb)

    print("\n" + "=" * 70)
    print(f"KET QUA: {ok} dat / {fail} loi")
    print("=" * 70)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    _selftest()
