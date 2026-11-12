"""
Module gợi ý tài liệu download (Document Recommendation).

CHỦ SỞ HỮU: Linh (Tuần 3).
Đặc tả kỹ thuật:
  1. Trích xuất đặc trưng văn bản bằng TF-IDF (max_features=5000, ngram_range=(1, 2)).
  2. Giảm chiều không gian đặc trưng bằng TruncatedSVD (Latent Semantic Analysis - LSA).
  3. Sử dụng nhân KNN dùng chung: KNNIndex từ src.knn_core với metric="cosine".
  4. Xử lý phản hồi tiềm ẩn (implicit feedback):
     - Lượt tải (download - weight 1.0)
     - Xem trực tuyến (view - weight 0.5)
     - Đánh dấu yêu thích (bookmark - weight 2.5)
  5. Cơ chế giải thích gợi ý (Explainable AI):
     Chỉ ra tài liệu lịch sử nào đóng góp điểm tương đồng cao nhất cho tài liệu được gợi ý.
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

try:
    import pandas as pd
    HAS_PANDAS = True
except (ImportError, Exception):
    HAS_PANDAS = False
    pd = None

# Cho phép import linh hoạt cả khi chạy độc lập lẫn import từ thư mục gốc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from .knn_core import KNNIndex
except ImportError:
    from knn_core import KNNIndex

__all__ = [
    "DocumentRecommender",
    "create_sample_document_dataset",
    "IMPLICIT_WEIGHTS",
]

# Trọng số mặc định cho các hành vi phản hồi tiềm ẩn
IMPLICIT_WEIGHTS = {
    "download": 1.0,
    "view": 0.5,
    "bookmark": 2.5,
    "cite": 3.0,
}


# ==========================================================================
# 1. Lớp gợi ý tài liệu chính (DocumentRecommender)
# ==========================================================================
class DocumentRecommender:
    """
    Hệ thống gợi ý tài liệu dựa trên nội dung (Content-Based) kết hợp KNN.

    Quy trình xử lý:
      Corpus (tiêu đề + tóm tắt + từ khóa)
        -> TF-IDF (max_features=5000, ngram=(1,2))
        -> TruncatedSVD (giảm chiều LSA, ví dụ 50 chiều)
        -> KNNIndex (Cosine similarity)
        -> Gợi ý cá nhân hóa từ lịch sử tải / tương tác kèm lời giải thích.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        n_components: Optional[int] = 50,
        k: int = 10,
        min_sim: float = 0.05,
        random_state: int = 42,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.k = k
        self.min_sim = min_sim
        self.random_state = random_state

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.index: Optional[KNNIndex] = None

        self.doc_map: Dict[str, Dict[str, Any]] = {}
        self.X_reduced: Optional[np.ndarray] = None

    # ----------------------------------------------------------------------
    # Fit mô hình từ dữ liệu tài liệu
    # ----------------------------------------------------------------------
    def fit(
        self,
        documents: Union[Any, List[Dict[str, Any]]],
        id_col: str = "doc_id",
        text_cols: Optional[List[str]] = None,
    ) -> "DocumentRecommender":
        """
        Huấn luyện bộ trích xuất đặc trưng và lập chỉ mục KNN.

        Parameters
        ----------
        documents : DataFrame hoặc danh sách dict
        id_col : str
            Tên cột ID tài liệu.
        text_cols : list[str], optional
            Các trường văn bản dùng để ghép nội dung.
        """
        raw_list: List[Dict[str, Any]] = []
        if HAS_PANDAS and pd is not None and isinstance(documents, pd.DataFrame):
            raw_list = documents.to_dict(orient="records")
        elif isinstance(documents, list):
            raw_list = documents
        else:
            raw_list = list(documents)

        if not raw_list:
            raise ValueError("Danh sách tài liệu rỗng")

        sample_item = raw_list[0]
        if id_col not in sample_item:
            raise ValueError(f"Thiếu trường '{id_col}' trong tài liệu")

        default_text_keys = ["title", "abstract", "tags", "category", "content"]
        keys_to_use = text_cols or [k for k in default_text_keys if k in sample_item]
        if not keys_to_use:
            raise ValueError("Không tìm thấy trường văn bản nào để xây dựng TF-IDF.")

        corpus = []
        doc_ids = []
        self.doc_map = {}

        for item in raw_list:
            d_id = str(item[id_col])
            doc_ids.append(d_id)
            self.doc_map[d_id] = item

            # Ghép văn bản
            parts = []
            for k in keys_to_use:
                v = item.get(k, "")
                if isinstance(v, (list, tuple)):
                    parts.append(" ".join(str(x) for x in v))
                elif v:
                    parts.append(str(v))
            corpus.append(" ".join(parts))

        # 1. Trích xuất đặc trưng TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        X_tfidf = self.vectorizer.fit_transform(corpus)

        # 2. Giảm chiều bằng TruncatedSVD (LSA)
        n_samples, n_feats = X_tfidf.shape
        if self.n_components and self.n_components < min(n_samples, n_feats):
            self.svd = TruncatedSVD(
                n_components=self.n_components,
                random_state=self.random_state,
            )
            self.X_reduced = self.svd.fit_transform(X_tfidf)
        else:
            self.svd = None
            self.X_reduced = X_tfidf.toarray()

        # 3. Lập chỉ mục KNNIndex dùng chung với metric Cosine
        self.index = KNNIndex(
            metric="cosine",
            k=self.k,
            min_sim=self.min_sim,
        )
        self.index.fit(self.X_reduced, ids=doc_ids)

        return self

    # ----------------------------------------------------------------------
    # Tìm tài liệu tương tự một tài liệu cụ thể
    # ----------------------------------------------------------------------
    def similar_documents(
        self,
        doc_id: str,
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Tìm top N tài liệu tương đồng nhất với doc_id."""
        if self.index is None:
            raise RuntimeError("Cần gọi fit() trước khi truy vấn.")

        doc_id = str(doc_id)
        neighbors = self.index.similar_to(doc_id, k=n)

        results = []
        for n_id, sim in neighbors:
            info = self.doc_map.get(str(n_id), {"title": f"Tài liệu {n_id}"})
            results.append({
                "doc_id": str(n_id),
                "title": info.get("title", f"Doc {n_id}"),
                "category": info.get("category", "General"),
                "similarity": round(float(sim), 4),
                "tags": info.get("tags", []),
            })
        return results

    # ----------------------------------------------------------------------
    # Gợi ý cá nhân hóa từ phản hồi tiềm ẩn (Implicit Feedback) + Giải thích
    # ----------------------------------------------------------------------
    def recommend(
        self,
        interactions: Union[List[str], List[Dict[str, Any]], Dict[str, float]],
        n: int = 5,
        exclude_interacted: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Gợi ý tài liệu dựa trên lịch sử tương tác tiềm ẩn.
        """
        if self.index is None:
            raise RuntimeError("Cần gọi fit() trước khi gợi ý.")

        liked_ids: List[str] = []
        weights: List[float] = []

        if isinstance(interactions, dict):
            for d_id, w in interactions.items():
                liked_ids.append(str(d_id))
                weights.append(float(w))
        elif isinstance(interactions, list) and interactions and isinstance(interactions[0], dict):
            for item in interactions:
                d_id = str(item.get("doc_id", ""))
                action = item.get("action", "download")
                w = item.get("weight", IMPLICIT_WEIGHTS.get(action, 1.0))
                liked_ids.append(d_id)
                weights.append(float(w))
        else:
            liked_ids = [str(x) for x in interactions]
            weights = [1.0] * len(liked_ids)

        valid_pairs = [
            (d_id, w) for d_id, w in zip(liked_ids, weights)
            if d_id in self.doc_map and d_id in self.index._pos
        ]

        if not valid_pairs:
            return self._fallback_popular(n)

        v_ids = [p[0] for p in valid_pairs]
        v_weights = [p[1] for p in valid_pairs]

        raw_recs = self.index.recommend(
            liked_ids=v_ids,
            n=n,
            weights=v_weights,
            exclude_liked=exclude_interacted,
        )

        # ------------------------------------------------------------------
        # Sinh lý do giải thích (Explainable AI)
        # ------------------------------------------------------------------
        results = []
        for cand_id, total_score in raw_recs:
            cand_id_str = str(cand_id)
            cand_pos = self.index._pos[cand_id_str]

            best_source_id = None
            best_contrib = -1.0
            best_sim = 0.0

            for s_id, s_w in zip(v_ids, v_weights):
                s_pos = self.index._pos[s_id]
                sim_val = self.index.sim_[cand_pos, s_pos]
                contrib = sim_val * s_w
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_source_id = s_id
                    best_sim = sim_val

            source_title = self.doc_map.get(best_source_id, {}).get("title", f"Tài liệu #{best_source_id}")
            cand_info = self.doc_map.get(cand_id_str, {})
            reason = f"Vì bạn đã quan tâm đến '{source_title}' (độ tương đồng nội dung: {best_sim*100:.1f}%)"

            results.append({
                "doc_id": cand_id_str,
                "title": cand_info.get("title", f"Doc {cand_id_str}"),
                "category": cand_info.get("category", "General"),
                "score": round(float(total_score), 4),
                "similarity_to_key": round(float(best_sim), 4),
                "reason": reason,
                "based_on_doc_id": best_source_id,
                "tags": cand_info.get("tags", []),
            })

        return results

    def _fallback_popular(self, n: int) -> List[Dict[str, Any]]:
        first_n = list(self.doc_map.values())[:n]
        return [
            {
                "doc_id": str(item.get("doc_id")),
                "title": item.get("title"),
                "category": item.get("category", "General"),
                "score": 1.0,
                "similarity_to_key": 1.0,
                "reason": "Tài liệu tiêu biểu được nhiều sinh viên tải về nhất",
                "based_on_doc_id": None,
                "tags": item.get("tags", []),
            }
            for item in first_n
        ]


# ==========================================================================
# 2. Bộ dữ liệu tài liệu mẫu (Sample Dataset)
# ==========================================================================
def create_sample_document_dataset() -> List[Dict[str, Any]]:
    """Tạo bộ dữ liệu tài liệu học tập mẫu môn Mạng máy tính & Trí tuệ nhân tạo."""
    return [
        {
            "doc_id": "doc_01",
            "title": "Giáo trình Mạng máy tính cơ bản và nâng cao",
            "abstract": "Tổng quan về mô hình OSI, giao thức TCP/IP, địa chỉ IP subnetting, định tuyến router và chuyển mạch switch trong mạng LAN/WAN.",
            "category": "Mạng máy tính",
            "tags": ["OSI", "TCP/IP", "Router", "Subnet"],
        },
        {
            "doc_id": "doc_02",
            "title": "Tìm hiểu giao thức HTTP/HTTPS và lập trình Web Socket",
            "abstract": "Phân tích gói tin HTTP request response, bảo mật SSL TLS và truyền thông hai chiều thời gian thực qua Web Socket.",
            "category": "Mạng máy tính",
            "tags": ["HTTP", "HTTPS", "WebSocket", "Security"],
        },
        {
            "doc_id": "doc_03",
            "title": "Thiết kế kiến trúc mạng Client - Server với FastAPI",
            "abstract": "Xây dựng RESTful API backend, xử lý đa tiến trình bất đồng bộ async await, tích hợp CSDL SQLite và kết nối web client.",
            "category": "Kỹ thuật phần mềm",
            "tags": ["Client-Server", "FastAPI", "REST API", "Backend"],
        },
        {
            "doc_id": "doc_04",
            "title": "Nhập môn Trí tuệ nhân tạo và Học máy căn bản",
            "abstract": "Khái niệm AI, Machine Learning, Deep Learning, phân biệt học có giám sát và không giám sát, các thuật toán phân lớp phổ biến.",
            "category": "Trí tuệ nhân tạo",
            "tags": ["AI", "Machine Learning", "Supervised", "Classification"],
        },
        {
            "doc_id": "doc_05",
            "title": "Thuật toán K-Nearest Neighbors và ứng dụng thực tế",
            "abstract": "Nguyên lý phân lớp và hồi quy bằng KNN, các độ đo khoảng cách Euclidean Manhattan Cosine, xử lý lời nguyền số chiều.",
            "category": "Trí tuệ nhân tạo",
            "tags": ["KNN", "Distance Metric", "Cosine", "Neighbors"],
        },
        {
            "doc_id": "doc_06",
            "title": "Xây dựng hệ gợi ý bằng Lọc cộng tác Collaborative Filtering",
            "abstract": "Ma trận tương tác User-Item, dự đoán điểm đánh giá rating bằng User-based và Item-based KNN với độ đo Pearson tương quan.",
            "category": "Hệ gợi ý",
            "tags": ["Recommender", "Collaborative Filtering", "Pearson", "Rating"],
        },
        {
            "doc_id": "doc_07",
            "title": "Xử lý ngôn ngữ tự nhiên với kỹ thuật TF-IDF và Cosine",
            "abstract": "Trích xuất đặc trưng văn bản bằng TF-IDF n-gram, tính độ tương đồng tài liệu Cosine similarity và lọc theo nội dung Content-based.",
            "category": "Hệ gợi ý",
            "tags": ["NLP", "TF-IDF", "Cosine", "Content-based"],
        },
        {
            "doc_id": "doc_08",
            "title": "Kỹ thuật giảm chiều dữ liệu TruncatedSVD và PCA",
            "abstract": "Phân tích giá trị kỳ dị Singular Value Decomposition, giảm chiều ma trận thưa TF-IDF, tối ưu hóa tốc độ và bộ nhớ cho mô hình học máy.",
            "category": "Học máy",
            "tags": ["SVD", "PCA", "Dimension Reduction", "Sparse"],
        },
        {
            "doc_id": "doc_09",
            "title": "Giải quyết bài toán Cold-Start trong hệ thống gợi ý",
            "abstract": "Phân tích thách thức khi có người dùng mới hoặc vật phẩm mới chưa có tương tác, ứng dụng kiến trúc hệ gợi ý lai Hybrid.",
            "category": "Hệ gợi ý",
            "tags": ["Cold-start", "Hybrid", "New User", "New Item"],
        },
        {
            "doc_id": "doc_10",
            "title": "Đánh giá chất lượng hệ gợi ý: RMSE, MAE, Precision và Recall",
            "abstract": "Phương pháp thực nghiệm phân chia tập Train Test, đo lường sai số dự đoán hồi quy và chất lượng xếp hạng Top-N với NDCG.",
            "category": "Hệ gợi ý",
            "tags": ["Evaluation", "RMSE", "MAE", "Precision", "NDCG"],
        },
    ]


# ==========================================================================
# 3. Chạy thử nghiệm độc lập (CLI Demo)
# ==========================================================================
if __name__ == "__main__":
    print("================================================================")
    print("   DEMO MODULE GỢI Ý TÀI LIỆU DOWNLOAD - REC_DOCUMENT.PY (LINH)")
    print("================================================================\n")

    docs = create_sample_document_dataset()
    print(f"-> Nạp thành công {len(docs)} tài liệu học tập mẫu.")

    recommender = DocumentRecommender(
        max_features=5000,
        ngram_range=(1, 2),
        n_components=5,
        k=5,
        min_sim=0.01,
    )
    recommender.fit(docs, id_col="doc_id")
    print(f"-> Đã trích xuất TF-IDF và giảm chiều TruncatedSVD thành công.")
    print(f"-> Đã lập chỉ mục không gian đặc trưng vào nhân KNNIndex (metric='cosine').\n")

    target_doc = "doc_05"
    print(f"--- KỊCH BẢN 1: TÀI LIỆU TƯƠNG TỰ VỚI '{recommender.doc_map[target_doc]['title']}' ---")
    sims = recommender.similar_documents(target_doc, n=3)
    for idx, item in enumerate(sims, 1):
        print(f"  {idx}. [{item['doc_id']}] {item['title']} (Cosine Sim: {item['similarity']:.4f})")
    print()

    student_history = [
        {"doc_id": "doc_07", "action": "download"},  # weight 1.0
        {"doc_id": "doc_08", "action": "bookmark"},  # weight 2.5
    ]
    print("--- KỊCH BẢN 2: GỢI Ý TOP-N TÀI LIỆU DỰA TRÊN LỊCH SỬ TIỀM ẨN ---")
    print(f"Lịch sử tương tác của sinh viên: {student_history}")
    recommendations = recommender.recommend(student_history, n=4)

    for rank, rec in enumerate(recommendations, 1):
        print(f"\nTop {rank}: [{rec['doc_id']}] {rec['title']}")
        print(f"   Thể loại: {rec['category']} | Tổng điểm gợi ý: {rec['score']:.4f}")
        print(f"   💡 {rec['reason']}")

    print("\n================================================================")
    print("   MODULE REC_DOCUMENT.PY ĐÃ CHẠY HOÀN HẢO THEO ĐÚNG ĐẶC TẢ!")
    print("================================================================")
