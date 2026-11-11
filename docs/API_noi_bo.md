# API nội bộ — nhân KNN dùng chung

> Tài liệu chốt API cho `src/similarity.py` và `src/knn_core.py`.
> **Chủ sở hữu:** Hoàng · Chốt Tuần 2 (hạn 15/11).
> Linh và Hùng xây module của mình **dựa trên** các hàm/lớp ở đây, không tự viết lại thuật toán KNN.

Mọi ví dụ giả định chạy từ thư mục gốc dự án, đã `activate` `.venv`.

---

## 1. `src/similarity.py` — các độ đo

Nhập:

```python
from src.similarity import cosine, pearson, adjusted_cosine, jaccard, get_similarity
```

| Hàm | Đầu vào | Trả về | Dùng khi |
|---|---|---|---|
| `cosine(X, Y=None)` | ma trận **đặc trưng** (0 là giá trị thật) | sim hàng×hàng | TF-IDF tài liệu, vector sản phẩm |
| `cosine_masked(R)` | ma trận **rating** (0 = chưa đánh giá) | sim | cosine bỏ qua ô trống |
| `pearson(R)` | ma trận rating | sim user×user (hoặc item×item nếu truyền `R.T`) | user-based CF |
| `adjusted_cosine(R)` | ma trận rating user×item | sim **item×item** | item-based CF |
| `jaccard(X)` | ma trận nhị phân | sim | đã tải / chưa tải |
| `euclidean`, `manhattan`, `minkowski`, `hamming` | đặc trưng | **khoảng cách** | minh họa lý thuyết |
| `get_similarity("tên")` | tên độ đo | hàm tương ứng | quét nhiều độ đo trong `experiments/` |

**Quy ước số 0 — rất quan trọng:**
- Ma trận **rating**: `0` = *chưa đánh giá* → dùng nhóm `pearson / adjusted_cosine / cosine_masked / jaccard`.
- Ma trận **đặc trưng**: `0` = *giá trị thật* → dùng `cosine / euclidean / ...`.

Dùng sai nhóm hàm là **lỗi logic** (chương trình vẫn ra số nhưng vô nghĩa).

Độ đo khoảng cách muốn dùng như độ tương đồng thì bọc qua `distance_to_similarity(d) = 1/(1+d)`, hoặc gọi `get_similarity("euclidean")` để nó tự bọc.

---

## 2. `src/knn_core.py` — hai lớp chính

### 2.1 `KNNIndex` — cho content-based (Linh, Hùng)

Tìm k đối tượng giống nhau trong một không gian đặc trưng.

```python
from src.knn_core import KNNIndex

idx = KNNIndex(metric="cosine", k=10, min_sim=0.0).fit(X, ids=doc_ids)

idx.similar_to("doc_42")                 # [(id, sim), ...] giống nhất, bỏ chính nó
idx.recommend(liked_ids=["d1", "d7"])    # gợi ý từ lịch sử đã thích
idx.recommend(["d1","d7"], weights=[1,3])# gắn trọng số (tải=1, bookmark=3)
```

| Phương thức | Ý nghĩa |
|---|---|
| `fit(X, ids=None)` | X: ma trận (n_đối_tượng × n_đặc_trưng), nhận cả `scipy.sparse`. `ids`: id gốc mỗi hàng |
| `similar_to(id, k=None)` | k đối tượng giống nhất |
| `recommend(liked_ids, n=10, weights=None, exclude_liked=True)` | điểm = tổng similarity tới các mục đã thích |
| `fit_keep(X, ids)` + `kneighbors_vector(v)` | khi cần truy vấn bằng **vector mới** chưa có trong index (cold-start Hùng) |

**Cho Linh (tài liệu):**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
X = TfidfVectorizer(max_features=5000, ngram_range=(1,2)).fit_transform(corpus)
idx = KNNIndex(metric="cosine", k=10).fit(X, ids=doc_ids)
```

**Cho Hùng (hàng mới):**
```python
idx = KNNIndex(metric="cosine", k=5).fit(dac_trung_sp, ids=ma_sp)
tuong_tu = idx.similar_to(ma_sp_moi)   # 5 sản phẩm cũ giống nhất
```

> Lưu ý: `kneighbors_vector` **không** dùng với `pearson/adjusted_cosine/cosine_masked` (chúng cần cả ma trận rating). Với vector rời, dùng `metric="cosine"`.

### 2.2 `KNNRecommender` — cho collaborative filtering (Hoàng, nhánh CF của Hùng)

Dự đoán điểm đánh giá, hỗ trợ item-based và user-based.

```python
from src.knn_core import KNNRecommender

rec = KNNRecommender(k=20, metric="adjusted_cosine", user_based=False,
                     min_support=5, rating_range=(0.5, 5.0))
rec.fit(R, user_ids=uids, item_ids=iids)   # R: user×item, 0 = chưa đánh giá

rec.predict(user=1, item=50)               # điểm dự đoán 1 cặp
rec.top_n(user=1, n=10)                     # [(item_id, điểm), ...] nên gợi ý
rec.evaluate([(u, i, r), ...])              # {rmse, mae, n, bo_qua}
```

| Tham số | Mặc định | Ghi chú |
|---|---|---|
| `k` | 20 | số láng giềng (quét 1,3,…,51 ở thực nghiệm Tuần 5) |
| `metric` | `"adjusted_cosine"` | item-based; dùng `"pearson"` khi user-based |
| `user_based` | `False` | **False = item-based, nên dùng** |
| `min_support` | 2 | số ô đồng đánh giá tối thiểu; **nên để 5** với dữ liệu thưa |
| `rating_range` | `None` | ví dụ `(0.5, 5.0)` để kẹp kết quả |

Đặc điểm: tính sẵn ma trận tương đồng lúc `fit` rồi cache; khi thiếu láng giềng tự rút về trung bình user → item → toàn cục nên **không bao giờ trả NaN**; cặp user/item chưa từng thấy khi `evaluate` được đếm riêng vào `bo_qua` (phần cold-start).

---

## 3. `src/rec_movie.py` — tham khảo cách ghép (Hoàng, Tuần 3)

Đây là ví dụ hoàn chỉnh cách bọc `KNNRecommender` cho một bài toán thật — Linh và Hùng theo cùng khuôn này cho module của mình.

```python
from src.rec_movie import load_movielens, build_user_item, MovieRecommender

ratings, movies = load_movielens()
R, uids, iids = build_user_item(ratings, min_user=10, min_item=30)
rec = MovieRecommender(k=30, user_based=False).fit(R, uids, iids, movies)

rec.recommend(user_id=1, n=10)       # [(title, điểm, genres), ...]
rec.similar_movies(movie_id=1, n=5)  # phim tương tự (chỉ item-based)
```

Chạy thử toàn bộ: `python src/rec_movie.py`

---

## 4. Lưu ý chung

- **Bộ nhớ:** ma trận item-item là n_item × n_item. MovieLens đầy đủ (9.724 phim) ≈ 750 MB → phải lọc bớt item ít lượt trước khi `fit` (xem `build_user_item`). `knn_core` tự phát `RuntimeWarning` khi ma trận quá to.
- **Chuẩn hóa:** với đặc trưng số khác thang đo (giá, số sao…) phải chuẩn hóa Min-Max/Z-score **trước** khi đưa vào `KNNIndex`, nếu không chiều có trị lớn sẽ áp đảo khoảng cách.
- **Đổi độ đo không sửa code lõi:** chỉ đổi tham số `metric`. Muốn thêm độ đo mới thì thêm vào `similarity.py` và đăng ký trong `SIMILARITIES`/`DISTANCES` — **chỉ Hoàng sửa hai file lõi này**.
- Kiểm thử lõi: `python src/similarity.py` (25/25) · `python src/knn_core.py` (28/28).
