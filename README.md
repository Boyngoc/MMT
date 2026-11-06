# Tìm hiểu thuật toán KNN và ứng dụng gợi ý

> Đồ án môn **Mạng máy tính (MMT)** · Thời gian: **05/11/2026 → 16/12/2026** (6 tuần)

Hệ gợi ý theo mô hình **client – server** dùng thuật toán **K-Nearest Neighbors** cho ba bài toán:

| Bài toán | Loại phản hồi | Phương pháp |
|---|---|---|
| Gợi ý **xem phim** | Explicit (rating 1–5 sao) | Collaborative Filtering — user-based & item-based KNN |
| Gợi ý **hàng mới** | Không có lịch sử (cold-start) | Content-based KNN + Hybrid switching |
| Gợi ý **tài liệu download** | Implicit (đã tải / chưa tải) | TF-IDF + Cosine KNN |

---

## Thành viên và phân công

| Thành viên | Module code | Chương báo cáo | Nhiệm vụ xuyên suốt |
|---|---|---|---|
| **Hoàng** (nhóm trưởng) | Gợi ý phim (CF), nhân KNN dùng chung, Server/API, CSDL | Mở đầu, Ch.3, 2.4, 4.1–4.2, Kết luận | Quản lý tiến độ, review code, ghép quyển báo cáo |
| **Linh** | Gợi ý tài liệu (TF-IDF), tiền xử lý dữ liệu, Client web | Ch.1 (toàn bộ), 2.2, 4.4–4.5 | Slide thuyết trình, chuẩn hóa hình vẽ |
| **Hùng** | Gợi ý hàng mới (cold-start), framework đánh giá, dữ liệu | Ch.2, 4.3, 4.6, Tài liệu tham khảo | Thực nghiệm, vẽ đồ thị, kiểm thử |

Chi tiết đầy đủ xem file `KeHoach_PhanCong_Nhom_KNN.docx`.

---

## Cấu trúc dự án và quyền sở hữu file

```
MMT/
├── data/
│   ├── raw/                     [Hùng]   dữ liệu thô — KHÔNG commit
│   └── processed/               [Linh]   dữ liệu đã làm sạch — KHÔNG commit
├── src/
│   ├── similarity.py            [Hoàng]  các độ đo khoảng cách
│   ├── knn_core.py              [Hoàng]  nhân KNN dùng chung — CHỐT TUẦN 2
│   ├── preprocess.py            [Linh]   tiền xử lý, ma trận user-item
│   ├── rec_movie.py             [Hoàng]  Collaborative Filtering
│   ├── rec_document.py          [Linh]   TF-IDF + Cosine
│   ├── rec_newitem.py           [Hùng]   Content-based + Hybrid
│   ├── evaluate.py              [Hùng]   RMSE, MAE, P@K, R@K, NDCG
│   └── db.py                    [Hoàng]  kết nối SQLite
├── api/main.py                  [Hoàng]  FastAPI server
├── client/                      [Linh]   giao diện web
├── experiments/                 [Hùng]   script thực nghiệm + figures/
├── notebooks/                   mỗi người một file riêng: eda_linh.ipynb ...
├── docs/                        báo cáo + slide
└── requirements.txt             [Hoàng]
```

**Quy tắc vàng:** mỗi người **chỉ sửa file mình sở hữu**. Cần đổi file của người khác thì nhắn trong nhóm, không tự sửa. Riêng `similarity.py` và `knn_core.py` là giao diện dùng chung — chỉ Hoàng được sửa sau khi đã chốt.

---

## Cài đặt môi trường

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu PowerShell chặn script khi activate, chạy một lần:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Dữ liệu

Dataset chính: **MovieLens ml-latest-small** (100k rating / 610 user / 9.700 phim).
Tải tại <https://grouplens.org/datasets/movielens/> rồi giải nén vào `data/raw/`.
Thư mục `data/` không được commit — mỗi người tự tải về máy mình.

---

## Quy ước làm việc với Git

- Nhánh `main` luôn ở trạng thái chạy được. **Không ai commit trực tiếp vào `main`.**
- Nhánh làm việc: `feat/linh-document`, `feat/hung-newitem`, `fix/hoang-api`…
- Mọi lần merge vào `main` đều qua Pull Request, **Hoàng review** rồi mới merge.
- Tiền tố commit message: `feat:` · `fix:` · `docs:` · `exp:` · `chore:`

```powershell
# bắt đầu một việc mới
git switch -c feat/linh-document

# làm xong
git add src/rec_document.py
git commit -m "feat: them module goi y tai lieu bang TF-IDF + cosine"
git push -u origin feat/linh-document
# rồi vào GitHub bấm "Compare & pull request", gán Hoàng làm reviewer

# đầu mỗi buổi làm việc, kéo code mới nhất về nhánh của mình
git switch main
git pull
git switch feat/linh-document
git merge main
```

> **Notebook là nguồn conflict số một** vì lưu cả output. Mỗi người dùng một file notebook riêng. Muốn chắc hơn thì cài: `pip install nbstripout` rồi `nbstripout --install`.

---

## Các mốc nghiệm thu

| Mốc | Hạn | Nội dung |
|---|---|---|
| M1 | 11/11 | Repo chạy được trên máy cả 3 người · Đề cương được duyệt · Đủ 3 nguồn dữ liệu |
| M2 | 18/11 | `knn_core` chạy đúng trên ví dụ tính tay 5×5 · Dữ liệu đã sạch · Ch.1 ~60% |
| M3 | 25/11 | Cả 3 module trả về được top-10 qua CLI · Ch.1 xong, Ch.2 ~80% |
| M4 | 02/12 | Hệ thống chạy end-to-end client–server · Ch.1, 2, 3 xong |
| M5 | 09/12 | **Feature freeze** · Đủ số liệu thực nghiệm · Ch.4 ~80% · Slide bản 1 |
| M6 | 16/12 | Nộp quyển + mã nguồn + slide, bảo vệ |

---

## Họp nhóm

- **Thứ 3** — mỗi người báo 3 câu: tuần trước làm gì / tuần này làm gì / đang bị chặn ở đâu.
- **Thứ 7** — nghiệm thu mốc tuần, demo phần vừa làm, điều chỉnh việc tuần sau.

**Định nghĩa XONG** — Code: đã merge vào `main`, chạy được trên máy người khác, có ít nhất 1 test hoặc ví dụ minh chứng. Báo cáo: đã nộp vào `docs/`, đủ hình và bảng, đã có người khác đọc soát.
