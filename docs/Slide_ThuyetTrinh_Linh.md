# KỊCH BẢN THUYẾT TRÌNH VÀ DÀN Ý SLIDE BẢO VỆ ĐỒ ÁN
## PHẦN TRÌNH BÀY: LÝ THUYẾT THUẬT TOÁN KNN VÀ HỆ GỢI Ý (CHƯƠNG 1)

- **Người thực hiện:** Linh  
- **Thời lượng thuyết trình:** ~10–12 phút (khoảng 40–50 giây/slide)  
- **Số lượng slide:** 18 slide chuẩn hóa  
- **Công cụ vẽ sơ đồ đồng bộ nhóm:** Draw.io (định dạng xuất PNG/SVG nền trong suốt)

---

### SLIDE 1: TRANG TIÊU ĐỀ & MỞ ĐẦU
- **Tiêu đề Slide:** NGHIÊN CỨU THUẬT TOÁN K-NEAREST NEIGHBORS (KNN) VÀ ỨNG DỤNG XÂY DỰNG HỆ THỐNG GỢI Ý ĐA NHIỆM
- **Nội dung chính trên Slide:**
  - Môn học: Mạng máy tính (MMT) — Đồ án cuối kỳ
  - Nhóm sinh viên thực hiện: Hoàng (Nhóm trưởng) · **Linh (Báo cáo phần Lý thuyết)** · Hùng
  - Giảng viên hướng dẫn: [Điền tên Giảng viên]
  - Phần 1: Cơ sở lý thuyết Trí tuệ nhân tạo & Thuật toán K-Nearest Neighbors
- **Kịch bản thuyết trình (Speaker Notes):**
  > *"Kính thưa quý thầy cô và các bạn, em là Linh, đại diện nhóm xin phép được trình bày Phần 1 của đồ án: Nghiên cứu chuyên sâu về cơ sở lý thuyết của thuật toán K-Nearest Neighbors và vai trò của thuật toán này trong việc xây dựng hệ thống gợi ý đa nhiệm theo kiến trúc client-server."*
- **Gợi ý hình vẽ / Layout Draw.io:**
  - Logo trường đại học ở góc trên bên trái.
  - Hình minh họa trừu tượng: Biểu tượng mạng nơ-ron kết nối các điểm dữ liệu phân cụm (Network Nodes) với tông màu xanh dương chủ đạo của nhóm.

---

### SLIDE 2: TỔNG QUAN VỀ TRÍ TUỆ NHÂN TẠO & HỌC MÁY
- **Tiêu đề Slide:** 1.1 Tổng quan về Trí tuệ nhân tạo (AI) và Học máy (Machine Learning)
- **Nội dung chính trên Slide:**
  - AI: Khoa học mô phỏng trí thông minh của con người trên máy tính.
  - Mối quan hệ thứ bậc: AI $\supset$ Machine Learning $\supset$ Deep Learning.
  - Phân loại bài toán Học máy:
    - Học có giám sát (Supervised): Phân lớp & Hồi quy.
    - Học không giám sát (Unsupervised): Gom cụm & Giảm chiều dữ liệu.
    - Học tăng cường (Reinforcement Learning).
  - Vị trí của bài toán Hệ gợi ý: Ứng dụng trọng tâm của AI giải quyết "quá tải thông tin".
- **Kịch bản thuyết trình:**
  > *"Để hiểu rõ nguồn gốc của KNN, trước hết ta nhìn vào bức tranh toàn cảnh của Trí tuệ nhân tạo. Học máy là hạt nhân của AI hiện đại, nơi hệ thống tự học từ dữ liệu thay vì lập trình thủ công quy tắc. Bài toán hệ thống gợi ý mà nhóm giải quyết là sự giao thoa giữa học có giám sát để dự đoán điểm số và học không giám sát để tìm kiếm độ tương đồng nội dung."*
- **Gợi ý sơ đồ Draw.io:**
  - Sơ đồ 3 vòng tròn đồng tâm lồng nhau (Venn diagram): Vòng ngoài cùng là "Artificial Intelligence", vòng giữa là "Machine Learning", vòng lõi là "Deep Learning".
  - Một nhánh sơ đồ cây (Tree diagram) phân chia 3 nhánh học máy: Supervised, Unsupervised, Reinforcement.

---

### SLIDE 3: BẢN CHẤT VÀ Ý TƯỞNG CỐT LÕI CỦA THUẬT TOÁN KNN
- **Tiêu đề Slide:** 1.2.1 Thuật toán K-Nearest Neighbors — Ý tưởng nền tảng
- **Nội dung chính trên Slide:**
  - Triết lý cốt lõi: *"Birds of a feather flock together"* — Các đối tượng có đặc điểm tương đồng sẽ nằm gần nhau trong không gian hình học.
  - Phân loại mô hình:
    - **Phi tham số (Non-parametric):** Không giả định phân phối dữ liệu gốc.
    - **Học lười biếng (Lazy Learning / Instance-based):** Không có pha huấn luyện tham số tường minh; toàn bộ tính toán dồn vào pha truy vấn.
  - Ứng dụng kép:
    - Phân lớp (Classification): Bầu chọn theo đa số (Majority Voting).
    - Hồi quy (Regression): Tính trung bình (Average) hoặc trung bình có trọng số.
- **Kịch bản thuyết trình:**
  > *"KNN là một trong những thuật toán học máy thanh lịch và trực quan nhất. Thuật toán dựa trên nguyên lý: các thực thể tương tự nhau sẽ quần tụ lại gần nhau trong không gian đặc trưng. KNN là một giải thuật Lazy Learning, nghĩa là pha training chỉ đơn giản là nạp dữ liệu vào bộ nhớ, nhưng khi có một truy vấn mới, nó sẽ so sánh khoảng cách tới từng điểm dữ liệu sẵn có để đưa ra phán đoán."*
- **Gợi ý sơ đồ Draw.io:**
  - Đồ thị 2D trực quan: Các hình tròn màu xanh (Lớp A) và hình tam giác màu cam (Lớp B). Một dấu chấm hỏi màu đỏ ở giữa. Vẽ một vòng tròn nét đứt bán kính nhỏ ($k=3$) bao quanh 2 tam giác + 1 hình tròn $\to$ Dự đoán hình tam giác.

---

### SLIDE 4: LƯU ĐỒ THỰC THI THUẬT TOÁN KNN
- **Tiêu đề Slide:** Lưu đồ và Các bước thực thi thuật toán KNN
- **Nội dung chính trên Slide:**
  - Bước 1: Tiếp nhận điểm dữ liệu truy vấn $\mathbf{x}_q$.
  - Bước 2: Tính toán khoảng cách từ $\mathbf{x}_q$ tới tất cả các mẫu trong tập dữ liệu $D$: $d_i = \text{dist}(\mathbf{x}_q, \mathbf{x}_i)$.
  - Bước 3: Sắp xếp danh sách khoảng cách tăng dần ($O(n \log n)$ hoặc $O(n)$ với phân hoạch).
  - Bước 4: Trích xuất $k$ điểm láng giềng gần nhất $N_k(\mathbf{x}_q)$.
  - Bước 5: Tổng hợp kết quả dự đoán (Voting hoặc Weighted Average).
- **Kịch bản thuyết trình:**
  > *"Quy trình hoạt động của KNN được chuẩn hóa qua 5 bước liên hoàn như trên slide. Điểm cần lưu ý nhất ở góc độ kỹ thuật là bước tính khoảng cách có độ phức tạp tuyến tính $O(n \cdot d)$, đòi hỏi các kỹ thuật tối ưu hóa chỉ mục khi hệ thống mở rộng lên hàng triệu bản ghi."*
- **Gợi ý sơ đồ Draw.io:**
  - Lưu đồ thuật toán (Flowchart): Các khối hình chữ nhật bo góc nối với nhau bằng mũi tên: [Input x_q] $\to$ [Tính ma trận khoảng cách] $\to$ [Sắp xếp QuickSelect Top-k] $\to$ [Tổng hợp láng giềng] $\to$ [Output dự đoán].

---

### SLIDE 5: CÁC ĐỘ ĐO KHOẢNG CÁCH HÌNH HỌC (EUCLIDEAN & MANHATTAN)
- **Tiêu đề Slide:** 1.2.2 Các độ đo khoảng cách hình học
- **Nội dung chính trên Slide:**
  - **Khoảng cách Euclidean ($L_2$ Norm):**
    $$d_{\text{Euclid}}(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^d (x_i - y_i)^2}$$
    *Đặc điểm:* Đo đường chim bay ngắn nhất; phù hợp dữ liệu liên tục cùng thang đo.
  - **Khoảng cách Manhattan ($L_1$ Norm / Taxicab):**
    $$d_{\text{Manhattan}}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^d |x_i - y_i|$$
    *Đặc điểm:* Bền vững hơn trước các giá trị ngoại lai (outliers); di chuyển theo dạng ô bàn cờ.
  - **Dạng tổng quát Minkowski ($L_p$ Norm):**
    Chuyển đổi linh hoạt giữa $L_1$ ($p=1$), $L_2$ ($p=2$) và Chebyshev ($p \to \infty$).
- **Kịch bản thuyết trình:**
  > *"Độ đo khoảng cách là trái tim của KNN. Nhóm đã cài đặt đầy đủ các hàm đo này trong file `src/similarity.py`. Với dữ liệu số liên tục, khoảng cách Euclidean phản ánh khoảng cách hình học thực tế, trong khi khoảng cách Manhattan đo lường độ lệch theo từng trục tọa độ và ít bị bóp méo bởi các điểm nhiễu ngoại lai."*
- **Gợi ý sơ đồ Draw.io:**
  - Hình minh họa 2 điểm $A(1,1)$ và $B(4,5)$ trên lưới tọa độ: Đường thẳng nối trực tiếp màu xanh lá biểu thị Euclidean ($d=5$); đường gấp khúc màu đỏ chạy vuông góc theo các ô lưới biểu thị Manhattan ($d=3+4=7$).

---

### SLIDE 6: ĐỘ TƯƠNG ĐỒNG ĐỊNH HƯỚNG VĂN BẢN: COSINE SIMILARITY
- **Tiêu đề Slide:** Độ tương đồng Cosine (Cosine Similarity)
- **Nội dung chính trên Slide:**
  - Công thức góc nghiêng giữa hai vector:
    $$\text{sim}_{\text{Cosine}}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2} = \cos(\theta)$$
  - Miền giá trị: $[-1, 1]$ (hoặc $[0, 1]$ với đặc trưng không âm).
  - Đặc tính ưu việt:
    - **Không phụ thuộc vào độ dài biên độ của vector.**
    - Hai tài liệu dài ngắn khác nhau nhưng cùng tỷ lệ phân bổ từ khóa sẽ có $\cos(\theta) \approx 1$.
  - Ứng dụng trong đồ án: **Module gợi ý tài liệu download (`rec_document.py`)**.
- **Kịch bản thuyết trình:**
  > *"Đối với dữ liệu văn bản, độ dài bài viết không quyết định chủ đề. Một bài tóm tắt 100 từ và một giáo trình 1,000 trang về mạng máy tính có thể có cùng tỷ lệ từ khóa. Khi đó, khoảng cách Euclidean sẽ rất lớn, nhưng độ tương đồng Cosine lại xấp xỉ bằng 1 vì góc nghiêng giữa hai vector gần bằng 0. Đây chính là lý do nhóm sử dụng Cosine làm độ đo tiêu chuẩn cho module tài liệu."*
- **Gợi ý sơ đồ Draw.io:**
  - Hai vector $\vec{u}$ và $\vec{v}$ cùng xuất phát từ gốc tọa độ $(0,0)$, vector $\vec{u}$ ngắn, vector $\vec{v}$ rất dài nhưng cùng nằm trên một phương nghiêng. Góc $\theta$ giữa hai vector rất nhỏ, chú thích: $\cos(\theta) \approx 1 \implies$ Độ tương đồng rất cao.

---

### SLIDE 7: HỆ SỐ TƯƠNG QUAN PEARSON TRONG LỌC CỘNG TÁC
- **Tiêu đề Slide:** Hệ số tương quan Pearson (Pearson Correlation)
- **Nội dung chính trên Slide:**
  - Công thức chuẩn hóa Mean-Centering:
    $$\text{sim}_{\text{Pearson}}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)(r_{v,i} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{v,i} - \bar{r}_v)^2}}$$
  - Vai trò quyết định:
    - Khử hoàn toàn **độ lệch chấm điểm cá nhân (User Rating Bias)**.
    - Người chấm "dễ tính" (trung bình 4.5 sao) và người "khó tính" (trung bình 2.5 sao) vẫn được nhận diện là có cùng sở thích nếu xu hướng đánh giá tương quan thuận với nhau.
  - Ứng dụng trong đồ án: **User-based Collaborative Filtering (`rec_movie.py`)**.
- **Kịch bản thuyết trình:**
  > *"Trong hệ gợi ý phim, mỗi người dùng có một thang đo tâm lý riêng. Có bạn cho 3 sao là phim tệ, nhưng có bạn cho 3 sao là phim đã rất hay. Nếu dùng Euclidean hay Cosine thuần túy, hai người này sẽ bị coi là xa nhau. Hệ số tương quan Pearson giải quyết triệt để vấn đề này bằng cách lấy điểm từng phim trừ đi điểm trung bình của chính người đó trước khi tính tương quan."*
- **Gợi ý sơ đồ Draw.io:**
  - Hai đường gấp khúc biểu diễn điểm rating qua 5 bộ phim của User A và User B: Đường của User A luôn cao hơn User B 2 điểm, nhưng hai đường chạy hoàn toàn song song nhau. Chú thích: Pearson = +1.0 (Đồng thuận hoàn hảo dù thang điểm lệch nhau).

---

### SLIDE 8: MINH HỌA TÍNH TAY CÔNG THỨC PEARSON (MA TRẬN 5X5)
- **Tiêu đề Slide:** Ví dụ tính toán thủ công từng bước bằng công thức Pearson
- **Nội dung chính trên Slide:**
  - Bài toán: Dự đoán điểm của người dùng $U_1$ cho phim $M_3$ với $k = 2$.
  - Bảng số liệu 5 users $\times$ 5 movies (với $\bar{r}_{U_1} = 3.75$).
  - Kết quả tính hệ số tương quan từng bước:
    - $\text{sim}(U_1, U_2) = \mathbf{+0.7980}$ (Đồng thuận cao $\to$ Láng giềng 1)
    - $\text{sim}(U_1, U_4) = \mathbf{+0.4461}$ (Đồng thuận vừa $\to$ Láng giềng 2)
    - $\text{sim}(U_1, U_3) = -0.7048$ và $\text{sim}(U_1, U_5) = -0.9748$ ($\to$ Bị loại)
  - Công thức dự đoán:
    $$\hat{r}_{U_1, M_3} = 3.75 + \frac{0.7980 \times (-2.0) + 0.4461 \times (-1.4)}{0.7980 + 0.4461} \approx \mathbf{1.97 \text{ sao}}$$
- **Kịch bản thuyết trình:**
  > *"Để kiểm chứng tính đúng đắn của code trong `knn_core.py`, nhóm đã tính toán thủ công bằng tay trên ma trận 5x5. Kết quả cho thấy $U_1$ có 2 láng giềng gần nhất là $U_2$ và $U_4$. Do cả hai người bạn này đều không thích phim $M_3$, thuật toán dự đoán chính xác $U_1$ chỉ cho $M_3$ khoảng 1.97 sao và hệ thống sẽ không đề xuất phim này. Kết quả tính tay khớp chính xác 100% với hàm kiểm thử tự động của nhóm."*
- **Gợi ý sơ đồ Draw.io:**
  - Bảng ma trận 5x5 có highlight màu: Cột $M_3$ của $U_1$ có dấu `?` màu đỏ. Hàng $U_2$ và $U_4$ được tô màu xanh lá (Neighbors). Mũi tên dẫn từ 2 hàng này sang công thức tính điểm ra con số 1.97.

---

### SLIDE 9: ĐỘ ĐO CHO DỮ LIỆU RỜI RẠC & TIÊU CHÍ CHỌN ĐỘ ĐO
- **Tiêu đề Slide:** Độ đo Jaccard, Hamming và Tiêu chuẩn lựa chọn
- **Nội dung chính trên Slide:**
  - **Độ đo Jaccard (Dữ liệu tập hợp / Nhị phân):**
    $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
    *Ứng dụng:* So sánh tập tài liệu đã tải hoặc tập kỹ năng của người dùng.
  - **Khoảng cách Hamming:** Đếm số lượng bit hoặc ký tự khác biệt.
  - **Ma trận quyết định lựa chọn độ đo (Decision Matrix):**
    - Văn bản TF-IDF $\to$ **Cosine Similarity**
    - Rating có độ lệch cá nhân $\to$ **Pearson Correlation**
    - Rating Item-Item $\to$ **Adjusted Cosine**
    - Dữ liệu tương tác nhị phân (Đã tải / Chưa tải) $\to$ **Jaccard**
    - Thuộc tính số thông số kỹ thuật $\to$ **Euclidean / Manhattan (sau chuẩn hóa)**
- **Kịch bản thuyết trình:**
  > *"Tùy thuộc vào bản chất dữ liệu, việc chọn sai độ đo sẽ dẫn đến sai lầm logic nghiêm trọng. Trong đồ án, nhóm đã phân loại rõ: nhóm ma trận rating dùng Pearson và Adjusted Cosine để xử lý ô trống; nhóm ma trận đặc trưng văn bản dùng Cosine và Jaccard."*
- **Gợi ý sơ đồ Draw.io:**
  - Sơ đồ cây quyết định (Decision Tree): [Dữ liệu vào] $\to$ Rẽ 2 nhánh: [Ma trận Rating thưa] $\to$ Pearson / Adjusted Cosine; [Ma trận Đặc trưng văn bản] $\to$ TF-IDF + Cosine.

---

### SLIDE 10: VẤN ĐỀ LỰA CHỌN THAM SỐ K
- **Tiêu đề Slide:** 1.2.3 Chiến lược lựa chọn tham số $k$ tối ưu
- **Nội dung chính trên Slide:**
  - Bản chất cân bằng **Độ chệch (Bias)** và **Phương sai (Variance)**:
    - $k$ quá nhỏ ($k=1$): Phương sai cao $\to$ **Overfitting** (Nhạy cảm với nhiễu).
    - $k$ quá lớn ($k \to n$): Độ chệch cao $\to$ **Underfitting** (Dự đoán cào bằng về giá trị trung bình toàn cục).
  - Quy tắc kinh nghiệm:
    - Chọn $k$ lẻ để tránh hòa phiếu khi phân lớp.
    - Xuất phát điểm: $k \approx \sqrt{n}$.
  - Thực nghiệm quét tham số (Grid Search Cross-Validation):
    - Đồ án quét $k \in [1, 3, 5, 7, ..., 51]$ để tìm điểm cực tiểu của RMSE trên tập MovieLens.
- **Kịch bản thuyết trình:**
  > *"Tham số $k$ quyết định độ bao quát của mô hình. Nếu chọn $k$ quá nhỏ, một đánh giá tiêu cực ngẫu nhiên sẽ làm hỏng kết quả gợi ý. Nhưng nếu chọn $k$ quá lớn, hệ thống sẽ chỉ đề xuất các bộ phim bom tấn đại trà. Nhóm đã thực hiện quét thực nghiệm qua nhiều giá trị $k$ để tìm ra điểm cân bằng tối ưu nhất."*
- **Gợi ý sơ đồ Draw.io:**
  - Đồ thị đường cong chữ U (U-shaped curve): Trục hoành là $k$, trục tung là Sai số RMSE trên tập kiểm thử. Điểm đáy chữ U tại khoảng $k=20-30$ được đóng khung vuông có chú thích: $k^*$ tối ưu.

---

### SLIDE 11: KNN CÓ TRỌNG SỐ (WEIGHTED KNN)
- **Tiêu đề Slide:** Thuật toán KNN có trọng số (Weighted KNN)
- **Nội dung chính trên Slide:**
  - Hạn chế của KNN cơ bản: Mọi láng giềng đều có tiếng nói ngang nhau dù ở rất gần hay rất xa.
  - Giải pháp trọng số nghịch đảo khoảng cách:
    $$w_i = \frac{1}{d(\mathbf{x}_q, \mathbf{x}_i)^2 + \epsilon} \quad \text{hoặc} \quad w_i = \text{sim}(\mathbf{x}_q, \mathbf{x}_i)$$
  - Công thức tổng hợp dự đoán:
    $$\hat{y}_q = \frac{\sum_{i=1}^k w_i y_i}{\sum_{i=1}^k w_i}$$
  - Lợi ích: Láng giềng càng giống thì đóng góp càng lớn vào quyết định; giảm thiểu tác động của các láng giềng ở rìa ranh giới.
- **Kịch bản thuyết trình:**
  > *"Để nâng cao độ chính xác, nhóm áp dụng biến thể Weighted KNN. Một người dùng có gu xem phim giống ta 95% rõ ràng phải có tiếng nói uy tín hơn một người chỉ giống 60%. Công thức trọng số nghịch đảo khoảng cách cho phép ta phản ánh đúng mức độ tin cậy này."*
- **Gợi ý sơ đồ Draw.io:**
  - Hình minh họa điểm truy vấn ở giữa và 3 láng giềng xung quanh. Láng giềng 1 ở rất gần có mũi tên dày với trọng số $w_1 = 0.8$; hai láng giềng ở xa có mũi tên mỏng với trọng số $w_2 = 0.15, w_3 = 0.05$.

---

### SLIDE 12: CHUẨN HÓA DỮ LIỆU (MIN-MAX & Z-SCORE)
- **Tiêu đề Slide:** Tầm quan trọng của Tiền xử lý & Chuẩn hóa dữ liệu
- **Nội dung chính trên Slide:**
  - Vấn đề thang đo: Các thuộc tính có biên độ đo lớn sẽ áp đảo và vô hiệu hóa các thuộc tính có biên độ đo nhỏ.
  - Hai kỹ thuật cài đặt trong `src/preprocess.py`:
    - **Min-Max Scaling:** Đưa miền giá trị về $[0, 1]$:
      $$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
    - **Z-Score Standardization:** Chuẩn hóa theo phân phối chuẩn $\mathcal{N}(0, 1)$:
      $$x' = \frac{x - \mu}{\sigma}$$
  - **Quy tắc bảo toàn ô 0 (Masking unrated cells):**
    - Chỉ chuẩn hóa trên các ô ĐÃ đánh giá; ô 0 (chưa đánh giá) giữ nguyên là 0.
- **Kịch bản thuyết trình:**
  > *"Chuẩn hóa dữ liệu là khâu sống còn mà em trực tiếp đảm nhiệm trong file `preprocess.py`. Đối với ma trận rating, một lỗi phổ biến là trừ mean toàn bộ khiến các ô 0 trở thành số âm, làm phá vỡ tính thưa của ma trận. Nhóm đã xử lý chuẩn xác bằng cơ chế mask-zeros, chỉ chuẩn hóa trên các ô người dùng thực sự đã tương tác."*
- **Gợi ý sơ đồ Draw.io:**
  - Hai biểu đồ Histogram đặt cạnh nhau: Biểu đồ trước chuẩn hóa (Trục hoành từ 0 đến 100,000) $\to$ Mũi tên biến đổi $\to$ Biểu đồ sau chuẩn hóa Z-Score (Trục hoành chuẩn hóa từ -3 đến +3 đối xứng quanh 0).

---

### SLIDE 13: ĐỘ PHỨC TẠP TÍNH TOÁN & THÁCH THỨC SCALABILITY
- **Tiêu đề Slide:** 1.2.4 Độ phức tạp tính toán và Thách thức mở rộng
- **Nội dung chính trên Slide:**
  - Độ phức tạp của thuật toán Brute-Force:
    - Huấn luyện: $O(1)$ (Không tốn chi phí).
    - Truy vấn 1 mẫu: $O(n \cdot d + n \log k)$.
    - Bộ nhớ lưu trữ ma trận tương đồng: $O(n^2)$ hoặc $O(m^2)$.
  - Thách thức thực tế:
    - Khi hệ thống có $n = 100,000$ người dùng $\to$ Ma trận tương đồng chứa $10^{10}$ phần tử $\approx 80 \text{ GB RAM}$.
  - Giải pháp đã áp dụng trong đồ án:
    - Lọc bỏ các user/item có dưới 5 tương tác (`filter_k_core`).
    - Cache ma trận tương đồng item-item trong bộ nhớ RAM (`rec_movie.py`).
- **Kịch bản thuyết trình:**
  > *"Nhược điểm lớn nhất của KNN chính là chi phí tính toán trong pha suy luận. Với một ma trận tương đồng n nhân n, dung lượng RAM sẽ bùng nổ theo hàm bậc hai. Để giải quyết, nhóm đã áp dụng chiến lược lọc K-Core để loại bỏ các điểm dữ liệu rác và thực hiện cache ma trận tính sẵn để đạt thời gian phản hồi API dưới 200 miligiây."*
- **Gợi ý sơ đồ Draw.io:**
  - Biểu đồ tăng trưởng độ phức tạp $O(n^2)$: Trục hoành là số lượng đối tượng $N$, trục tung là dung lượng RAM cần thiết (GB), đường cong vút lên màu đỏ cảnh báo nguy cơ tràn bộ nhớ.

---

### SLIDE 14: CÁC KỸ THUẬT TĂNG TỐC TRUY VẤN LÁNG GIỀNG
- **Tiêu đề Slide:** Cấu trúc dữ liệu và Kỹ thuật tăng tốc KNN
- **Nội dung chính trên Slide:**
  - **KD-Tree (K-Dimensional Tree):**
    - Phân chia không gian bằng các siêu phẳng trực giao theo trục có phương sai lớn nhất.
    - Giảm chi phí tìm kiếm trung bình xuống $O(d \log n)$ (Hiệu quả khi $d < 20$).
  - **Ball-Tree:**
    - Phân hoạch không gian thành các siêu cầu lồng nhau; hoạt động tốt hơn KD-Tree ở số chiều trung bình.
  - **Locality-Sensitive Hashing (LSH):**
    - Sử dụng hàm băm xác suất để gom các vector gần nhau vào cùng bucket $\to O(1)$.
  - **HNSW (Hierarchical Navigable Small World):**
    - Đồ thị đa tầng phân cấp, chuẩn mực công nghiệp cho cơ sở dữ liệu vector hiện đại.
- **Kịch bản thuyết trình:**
  > *"Để tăng tốc độ tìm kiếm láng giềng, các cấu trúc dữ liệu không gian phân cấp như KD-Tree và Ball-Tree giúp ta loại bỏ hàng loạt các vùng không gian ở xa mà không cần tính khoảng cách chi tiết. Với các hệ thống hiện đại, giải pháp đồ thị HNSW và kỹ thuật băm LSH đang là tiêu chuẩn hàng đầu để đưa tốc độ truy vấn về thời gian thực."*
- **Gợi ý sơ đồ Draw.io:**
  - Hình minh họa cấu trúc KD-Tree: Bên trái là không gian 2D bị chia cắt bởi các đường thẳng $x_1, y_1, x_2$. Bên phải là cây nhị phân tương ứng với các nút lá chứa các điểm dữ liệu.

---

### SLIDE 15: LỜI NGUYỀN SỐ CHIỀU (CURSE OF DIMENSIONALITY)
- **Tiêu đề Slide:** 1.2.5 Lời nguyền số chiều và Giải pháp xử lý
- **Nội dung chính trên Slide:**
  - Bản chất hiện tượng:
    - Khi số chiều $d$ tăng cao (ví dụ $d = 5,000$ từ vựng TF-IDF), không gian trở nên cực kỳ loãng.
    - **Hiện tượng khoảng cách hội tụ (Distance Concentration):**
      $$\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} = 0$$
    - Mọi điểm dữ liệu đều có khoảng cách gần như bằng nhau tới điểm truy vấn $\to$ Mất khả năng phân biệt láng giềng!
  - Giải pháp áp dụng trong đồ án:
    - **Giảm chiều dữ liệu bằng TruncatedSVD (Latent Semantic Analysis):** Nén từ 5,000 chiều từ khóa xuống 50 chiều tiềm ẩn giàu ngữ nghĩa.
- **Kịch bản thuyết trình:**
  > *"Khi làm việc với văn bản ở module tài liệu, một thách thức lớn là 'Lời nguyền số chiều'. Trong không gian 5,000 từ khóa, mọi văn bản đều có xu hướng cách đều nhau, khiến KNN mất đi độ nhạy. Em đã giải quyết bài toán này bằng cách đưa vào thuật toán TruncatedSVD để trích xuất 50 chiều ngữ nghĩa tiềm ẩn cốt lõi nhất, vừa khử nhiễu vừa giúp phép tính Cosine đạt độ chính xác tối ưu."*
- **Gợi ý sơ đồ Draw.io:**
  - Hình minh họa khối nén: Khối chữ nhật lớn (5,000 chiều TF-IDF) $\to$ Đi qua phễu nén TruncatedSVD $\to$ Khối nhỏ gọn (50 chiều LSA Latent Space) với các cụm dữ liệu phân tách rõ ràng.

---

### SLIDE 16: LỌC THEO NỘI DUNG (CONTENT-BASED) KẾT HỢP KNN
- **Tiêu đề Slide:** 1.3.2 Lọc theo nội dung (Content-Based) kết hợp KNN
- **Nội dung chính trên Slide:**
  - Mô hình không gian vector (Vector Space Model):
    - Trích xuất đặc trưng văn bản bằng **TF-IDF n-gram (1, 2)** với `max_features=5000`.
    - Xây dựng Hồ sơ Vật phẩm (Item Profile) và Hồ sơ Người dùng (User Profile).
  - Vai trò của KNN:
    - Tìm $k$ tài liệu tương đồng nhất bằng `KNNIndex` với độ đo Cosine.
    - Gợi ý cá nhân hóa từ tổng độ tương đồng với lịch sử tương tác tiềm ẩn.
  - Ưu thế vượt trội:
    - **Giải quyết hoàn toàn Cold-Start cho vật phẩm mới (New Item).**
    - Không phụ thuộc vào cộng đồng người dùng khác.
- **Kịch bản thuyết trình:**
  > *"Mục 1.3.2 là cầu nối trực tiếp giữa lý thuyết KNN và module gợi ý tài liệu download mà em đảm nhiệm. Bằng cách kết hợp TF-IDF với KNN, hệ thống có thể lập tức đề xuất một tài liệu giáo trình vừa tải lên cho đúng những sinh viên đang nghiên cứu chuyên đề đó, mà không cần phải chờ đợi có người khác vào đánh giá trước."*
- **Gợi ý sơ đồ Draw.io:**
  - Sơ đồ quy trình: [Kho tài liệu PDF] $\to$ [TF-IDF Vectorizer] $\to$ [TruncatedSVD] $\to$ [KNN Index] $\leftrightarrow$ [Lịch sử tải của Sinh viên] $\to$ [Top-N Tài liệu gợi ý].

---

### SLIDE 17: TỔNG KẾT ƯU VÀ NHƯỢC ĐIỂM CỦA KNN
- **Tiêu đề Slide:** Tổng kết: Ưu điểm, Nhược điểm và Ứng dụng thực tế
- **Nội dung chính trên Slide:**
  - **Ưu điểm lớn nhất:**
    - Tính minh bạch và giải thích được (**Explainable AI** — *"Gợi ý vì bạn đã thích X"*).
    - Không mất thời gian đào tạo lại (Thêm dữ liệu mới là dùng được ngay).
    - Không cần giả định phân phối dữ liệu (Non-parametric).
  - **Nhược điểm cần khắc phục:**
    - Chi phí bộ nhớ RAM và thời gian truy vấn pha suy luận.
    - Nhạy cảm với độ thưa dữ liệu và số chiều lớn.
  - **Bài học kinh nghiệm:**
    - Cần kết hợp kỹ thuật tiền xử lý chặt chẽ (Lọc K-Core, chuẩn hóa Min-Max, giảm chiều SVD).
- **Kịch bản thuyết trình:**
  > *"Tóm lại, KNN là một thuật toán có tính thực tiễn rất cao. Điểm giá trị nhất của KNN so với các mô hình hộp đen Deep Learning chính là khả năng giải thích minh bạch: người dùng luôn biết được tại sao họ nhận được gợi ý đó. Những nhược điểm về tốc độ và bộ nhớ hoàn toàn có thể được khắc phục bằng các kỹ thuật tiền xử lý tối ưu mà nhóm đã cài đặt thành công."*
- **Gợi ý sơ đồ Draw.io:**
  - Bảng 2 cột cân xứng: Cột xanh lá mang dấu tích xanh (Ưu điểm), Cột cam mang dấu chấm than (Nhược điểm & Giải pháp tương ứng của nhóm).

---

### SLIDE 18: KẾT NỐI SANG CHƯƠNG 2 & LỜI CẢM ƠN
- **Tiêu đề Slide:** Chuyển giao sang Chương 2 — Hiện thực hóa Ứng dụng thực tế
- **Nội dung chính trên Slide:**
  - Cơ sở lý thuyết vững chắc từ Chương 1 là nền tảng để nhóm xây dựng:
    - Module tiền xử lý `src/preprocess.py` (Linh)
    - Module gợi ý tài liệu `src/rec_document.py` (Linh)
    - Giao diện Client Web 3 tab tương tác (Linh)
    - Nhân dùng chung `knn_core.py` và module phim (Hoàng)
    - Module hàng mới và framework đánh giá thực nghiệm (Hùng)
  - Xin trân trọng cảm ơn Thầy Cô và các bạn đã lắng nghe!
  - *Chuyển phần trình bày cho bạn Hoàng / Hùng tiếp tục Chương 2.*
- **Kịch bản thuyết trình:**
  > *"Phần trình bày lý thuyết của em đến đây là hoàn thành. Toàn bộ nền tảng toán học về các độ đo khoảng cách, chuẩn hóa và KNN đã được nhóm hiện thực hóa thành mã nguồn chạy được trong hệ thống client-server. Tiếp theo đây, bạn nhóm trưởng sẽ trình bày chi tiết về phần thiết kế kiến trúc hệ thống và các module ứng dụng ở Chương 2. Em xin chân thành cảm ơn!"*
- **Gợi ý sơ đồ Draw.io:**
  - Sơ đồ tổng thể hệ sinh thái đồ án: Khối Lý thuyết Chương 1 (Bệ phóng) $\to$ Mũi tên nâng đỡ 3 module ứng dụng ở trên (Phim, Tài liệu, Hàng mới) cùng giao diện Client Web.

---
*(Hết kịch bản 18 slide của Linh)*

