# BÁO CÁO HỌC THUẬT CHI TIẾT — PHẦN THỰC HIỆN CỦA LINH

**Đề tài:** Tìm hiểu thuật toán KNN và ứng dụng gợi ý xem phim / hàng mới / tài liệu download  
**Môn học:** Mạng máy tính (MMT)  
**Người thực hiện:** Linh  
**Phân công đảm nhiệm:**  
- Chương 1: Mục 1.1, Mục 1.2, Mục 1.3.2  
- Chương 2: Mục 2.4, Mục 2.5  
- Quy chuẩn định dạng: Times New Roman 13pt, căn đều hai bên (Justified), thụt đầu dòng 1 cm, in đậm các tiêu đề.

---

# CHƯƠNG 1. TỔNG QUAN VỀ TRÍ TUỆ NHÂN TẠO VÀ CÁC THUẬT TOÁN

## 1.1. Tổng quan về Trí tuệ nhân tạo

### 1.1.1. Khái niệm và các nhánh của Trí tuệ nhân tạo
Trí tuệ nhân tạo (Artificial Intelligence - AI) là một ngành khoa học máy tính liên ngành rộng lớn, tập trung vào việc nghiên cứu, thiết kế và phát triển các hệ thống hoặc máy móc có khả năng mô phỏng các quá trình nhận thức, suy luận, học hỏi, thích nghi và giải quyết vấn đề tương tự như trí thông minh của con người. Thuật ngữ này được đề xuất chính thức lần đầu tiên vào năm 1956 tại Hội nghị Dartmouth bởi John McCarthy cùng các nhà khoa học tiên phong như Marvin Minsky, Nathaniel Rochester và Claude Shannon. Kể từ đó, AI đã phát triển từ các hệ thống chuyên gia dựa trên luật suy diễn logic hình thức (Symbolic AI) sang các hệ thống định hướng dữ liệu hiện đại có khả năng tự động học hỏi từ thông tin thực nghiệm quy mô lớn.

Để hiểu đúng vị trí và bản chất của các công nghệ hiện đại, cần làm rõ mối quan hệ thứ bậc giữa ba khái niệm thường bị nhầm lẫn: Trí tuệ nhân tạo (AI), Học máy (Machine Learning - ML) và Học sâu (Deep Learning - DL). Trí tuệ nhân tạo là vòng tròn bao quát lớn nhất, đại diện cho mọi kỹ thuật cho phép máy tính bắt chước hành vi thông minh của con người, bao gồm cả các thuật toán tìm kiếm trên đồ thị, logic mờ và hệ cơ sở tri thức tĩnh. Học máy là một tập con trọng yếu của AI, ra đời dựa trên quan điểm rằng thay vì lập trình cứng từng quy tắc xử lý cho máy tính, chúng ta cung cấp dữ liệu và các thuật toán học để máy tính tự động tìm ra các quy luật tiềm ẩn. Học sâu lại là một tập con chuyên sâu của học máy, dựa trên kiến trúc mạng nơ-ron nhân tạo đa tầng (Deep Artificial Neural Networks) với khả năng tự động trích xuất các tầng đặc trưng trừu tượng từ dữ liệu thô như hình ảnh, âm thanh hay văn bản không cấu trúc mà không cần đến sự can thiệp thủ công của con người.

Trong cấu trúc phân ngành học thuật, Trí tuệ nhân tạo hiện đại được chia thành nhiều nhánh nghiên cứu và ứng dụng chủ đạo:
1. **Xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP):** Nghiên cứu các phương pháp cho phép máy tính đọc hiểu, phân tích, dịch thuật và sinh ngôn ngữ tự nhiên của con người. Các kỹ thuật nền tảng như tách từ (tokenization), trích xuất trọng số TF-IDF, mô hình hóa chủ đề (Topic Modeling) và các kiến trúc Transformer hiện đại (như BERT, GPT) là những đại diện tiêu biểu của nhánh này.
2. **Thị giác máy tính (Computer Vision - CV):** Cho phép máy tính trích xuất, phân tích và hiểu các thông tin có ý nghĩa từ các hình ảnh kỹ thuật số, video hoặc dữ liệu không gian nhiều chiều.
3. **Hệ thống chuyên gia và lập luận tri thức (Expert Systems & Knowledge Representation):** Xây dựng các đồ thị tri thức (Knowledge Graphs), hệ suy diễn logic phục vụ việc đưa ra quyết định dựa trên các kho cơ sở tri thức miền chuyên gia.
4. **Học máy và Khai phá dữ liệu (Machine Learning & Data Mining):** Trọng tâm nghiên cứu việc tìm kiếm các cấu trúc, mẫu hình và quy luật thống kê từ các tập dữ liệu lớn phục vụ dự báo và tự động hóa.
5. **Hệ thống gợi ý (Recommender Systems):** Một nhánh ứng dụng đặc thù và phổ biến hàng đầu trong kỷ nguyên Internet hiện đại, kết hợp chặt chẽ giữa học máy, khai phá dữ liệu và tâm lý học hành vi người dùng nhằm giải quyết tình trạng "quá tải thông tin" (Information Overload), mang lại trải nghiệm cá nhân hóa tối ưu.

### 1.1.2. Học máy và phân loại bài toán học máy
Học máy (Machine Learning) được định nghĩa một cách kinh điển bởi Tom Mitchell (1997): *"Một chương trình máy tính được gọi là học từ kinh nghiệm E đối với một lớp tác vụ T và độ đo hiệu năng P, nếu hiệu năng của nó đối với các tác vụ trong T, được đo bởi P, cải thiện theo kinh nghiệm E."* Thay vì tiếp cận theo hướng tiếp cận truyền thống (truyền thống: Dữ liệu vào + Thuật toán/Quy tắc = Kết quả ra), học máy đảo ngược quy trình (Học máy: Dữ liệu vào + Kết quả mong muốn = Quy tắc/Mô hình toán học).

Dựa trên phương thức học và sự hiện diện của tín hiệu phản hồi (nhãn dữ liệu), các bài toán học máy được chia thành ba nhóm kinh điển:
1. **Học có giám sát (Supervised Learning):** Dữ liệu huấn luyện bao gồm cả tập véc-tơ đầu vào $\mathbf{x} \in \mathbb{R}^d$ và nhãn đầu ra tương ứng $y$. Mục tiêu là tìm hàm ánh xạ $f: \mathbf{x} \to y$ sao cho sai số dự báo trên dữ liệu chưa từng thấy là nhỏ nhất. Học có giám sát lại chia thành hai bài toán cơ bản:
   - *Phân lớp (Classification):* Khi nhãn mục tiêu mang tính rời rạc (ví dụ: phân loại email rác/không rác, nhận diện thể loại tài liệu).
   - *Hồi quy (Regression):* Khi nhãn mục tiêu mang giá trị liên tục trong tập số thực $\mathbb{R}$ (ví dụ: dự đoán điểm đánh giá rating từ 1.0 đến 5.0, dự báo giá cả thị trường).
2. **Học không giám sát (Unsupervised Learning):** Dữ liệu huấn luyện chỉ có đầu vào $\mathbf{x}$ mà không có nhãn đầu ra đi kèm. Mục tiêu là tự khám phá cấu trúc tiềm ẩn, quy luật phân bố hoặc các cụm dữ liệu tự nhiên. Các bài toán tiêu biểu gồm: Gom cụm (Clustering - K-Means, DBSCAN), Giảm chiều dữ liệu (Dimensionality Reduction - PCA, TruncatedSVD), Khai phá luật kết hợp (Association Rules).
3. **Học tăng cường (Reinforcement Learning):** Một tác tử (Agent) tương tác với môi trường động (Environment), thực hiện các hành động (Action) để chuyển đổi trạng thái (State) và nhận về các phần thưởng hoặc hình phạt (Reward/Penalty), nhằm tối đa hóa tổng phần thưởng tích lũy dài hạn.

**Vị trí của thuật toán K-Nearest Neighbors (KNN) trong bản đồ Học máy:**
Thuật toán K-Nearest Neighbors là một giải thuật học máy có giám sát thuộc nhóm phi tham số (non-parametric) và học lười biếng (instance-based / lazy learning). Điểm độc đáo của KNN nằm ở chỗ nó vừa có thể giải quyết bài toán phân lớp (thông qua cơ chế bỏ phiếu theo đa số - Majority Voting) vừa giải quyết bài toán hồi quy (thông qua phép tính trung bình hoặc trung bình có trọng số các láng giềng). Trong lĩnh vực hệ gợi ý (Recommender Systems), bài toán dự đoán điểm đánh giá rating chính là một bài toán hồi quy kinh điển, trong khi bài toán tìm kiếm các tài liệu hoặc sản phẩm tương tự lại là bài toán tìm láng giềng gần nhất trong không gian vector đặc trưng. Sự trực quan, tính minh bạch và khả năng giải thích rõ ràng kết quả suy luận là những ưu điểm vượt trội đưa KNN trở thành thuật toán nền tảng trong cả lý thuyết lẫn ứng dụng công nghiệp.

---

## 1.2. Thuật toán K-Nearest Neighbors (KNN)

### 1.2.1. Ý tưởng, mã giả và bản chất Lazy Learning
Ý tưởng nền tảng của thuật toán K-Nearest Neighbors bắt nguồn từ triết lý dân gian sâu sắc: *"Gần mực thì đen, gần đèn thì rạng"* hay ngạn ngữ phương Tây *"Birds of a feather flock together"*. Trong không gian biểu diễn hình học nhiều chiều, các đối tượng có thuộc tính, đặc điểm hoặc hành vi tương đồng sẽ nằm gần nhau về mặt khoảng cách không gian. Do đó, để xác định tính chất (nhãn lớp hoặc giá trị dự báo) của một điểm dữ liệu mới truy vấn $\mathbf{x}_q$, ta chỉ cần tìm ra $k$ điểm dữ liệu đã biết trong tập huấn luyện nằm gần $\mathbf{x}_q$ nhất, sau đó tổng hợp thông tin từ $k$ người láng giềng này để đưa ra quyết định.

Thuật toán KNN mang bản chất là một thuật toán **Học lười biếng (Lazy Learning / Instance-Based Learning)**:
- Trong giai đoạn "huấn luyện" (training phase), thuật toán không xây dựng một hàm mục tiêu toán học tường minh hay tối ưu hóa các tham số nội tại (như trọng số $W, b$ trong hồi quy logistic hay mạng nơ-ron). Giai đoạn huấn luyện của KNN thực chất chỉ đơn giản là nạp toàn bộ tập dữ liệu mẫu vào bộ nhớ RAM. Do đó, chi phí tính toán khi huấn luyện gần như bằng 0 ($O(1)$).
- Trái lại, trong giai đoạn "dự báo / truy vấn" (inference phase), khi có một điểm dữ liệu mới cần dự đoán, thuật toán mới bắt đầu thực hiện toàn bộ khối lượng tính toán nặng nhọc: quét qua toàn bộ các mẫu huấn luyện, tính toán khoảng cách từ điểm mới tới từng mẫu, sắp xếp để chọn ra $k$ mẫu gần nhất. Vì vậy, KNN chuyển toàn bộ gánh nặng tính toán về pha suy luận ($O(n \cdot d)$ với mỗi điểm truy vấn).

**Mã giả thuật toán K-Nearest Neighbors tổng quát:**
```text
Thuật toán: K-Nearest Neighbors (KNN)
Đầu vào:
  - Tập dữ liệu huấn luyện D = {(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)}, trong đó x_i in R^d
  - Điểm dữ liệu truy vấn mới x_q in R^d
  - Số lượng láng giềng k (số nguyên dương)
  - Hàm đo khoảng cách dist(x, x')
Đầu ra:
  - Giá trị dự đoán y_q cho x_q

Các bước thực hiện:
1. Khởi tạo danh sách khoảng cách: L = []
2. Với mỗi mẫu (x_i, y_i) trong D:
     d_i = dist(x_q, x_i)
     Thêm cặp (d_i, y_i) vào danh sách L
3. Sắp xếp danh sách L theo thứ tự khoảng cách d_i tăng dần.
4. Chọn ra k phần tử đầu tiên của L, gọi tập này là N_k(x_q).
5. Tổng hợp kết quả:
   - Nếu là bài toán Phân lớp:
       y_q = argmax_c sum_{(d_i, y_i) in N_k} I(y_i == c)  [Majority Voting]
   - Nếu là bài toán Hồi quy / Dự đoán điểm:
       y_q = (1 / k) * sum_{(d_i, y_i) in N_k} y_i        [Trung bình cộng]
       hoặc có trọng số:
       w_i = 1 / (d_i + epsilon)
       y_q = sum(w_i * y_i) / sum(w_i)
6. Trả về kết quả y_q.
```

### 1.2.2. Các độ đo khoảng cách và độ tương đồng
Hiệu năng và độ chính xác của thuật toán KNN phụ thuộc trực tiếp vào việc lựa chọn hàm đo khoảng cách (Distance Metric) hoặc độ tương đồng (Similarity Measure). Khoảng cách $d(\mathbf{x}, \mathbf{y})$ đo lường mức độ xa cách hình học ($d \ge 0$, giá trị càng nhỏ càng giống nhau), trong khi độ tương đồng $s(\mathbf{x}, \mathbf{y})$ đo lường mức độ trùng khớp ($s$ càng lớn càng giống nhau).

Dưới đây là các độ đo cốt lõi được ứng dụng trong đồ án:

1. **Khoảng cách Euclidean (Chuẩn $L_2$):**
   Là độ đo khoảng cách hình học ngắn nhất nối liền hai điểm trong không gian Euclid đa chiều:
   $$d_{\text{Euclid}}(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^d (x_i - y_i)^2}$$
   *Đặc điểm:* Rất trực quan, phù hợp với dữ liệu không gian liên tục cùng thang đo. Tuy nhiên rất nhạy cảm với các thuộc tính có thang đo lớn và ngoại lai (outliers).

2. **Khoảng cách Manhattan (Chuẩn $L_1$ / Taxicab Distance):**
   Đo khoảng cách giữa hai điểm dọc theo các trục tọa độ trực giao (tương tự việc di chuyển theo mạng lưới đường ô bàn cờ tại thành phố Manhattan):
   $$d_{\text{Manhattan}}(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_1 = \sum_{i=1}^d |x_i - y_i|$$
   *Đặc điểm:* Bền vững hơn với ngoại lai so với Euclidean, thường cho kết quả ổn định hơn khi số chiều $d$ tăng lên.

3. **Khoảng cách Minkowski (Dạng tổng quát chuẩn $L_p$):**
   $$d_{\text{Minkowski}}(\mathbf{x}, \mathbf{y}) = \left(\sum_{i=1}^d |x_i - y_i|^p\right)^{\frac{1}{p}}$$
   Khi $p = 1$, Minkowski trở thành khoảng cách Manhattan; khi $p = 2$, trở thành khoảng cách Euclidean; khi $p \to \infty$, trở thành khoảng cách Chebyshev.

4. **Độ tương đồng Cosine (Cosine Similarity):**
   Đo góc nghiêng hình học giữa hai vector trong không gian đa chiều, không phụ thuộc vào độ dài biên độ của vector:
   $$\text{sim}_{\text{Cosine}}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2} = \frac{\sum_{i=1}^d x_i y_i}{\sqrt{\sum_{i=1}^d x_i^2} \sqrt{\sum_{i=1}^d y_i^2}}$$
   *Đặc điểm:* Có miền giá trị $[-1, 1]$ (hoặc $[0, 1]$ với đặc trưng không âm như TF-IDF). Đây là độ đo tiêu chuẩn cho xử lý văn bản, tài liệu download và phân tích giỏ hàng vì nó phản ánh xu hướng phân bố tỷ lệ từ khóa thay vì độ dài tuyệt đối của bài viết.

5. **Hệ số tương quan Pearson (Pearson Correlation Coefficient):**
   Đo mối quan hệ tương quan tuyến tính giữa hai biến ngẫu nhiên, tự động khử đi xu hướng thiên lệch trung bình (Mean-Centering) của từng đối tượng:
   $$\text{sim}_{\text{Pearson}}(\mathbf{x}, \mathbf{y}) = \frac{\sum_{i \in I_{xy}} (x_i - \bar{x}) (y_i - \bar{y})}{\sqrt{\sum_{i \in I_{xy}} (x_i - \bar{x})^2} \sqrt{\sum_{i \in I_{xy}} (y_i - \bar{y})^2}}$$
   Trong đó $I_{xy}$ là tập các thuộc tính (hoặc các item) mà cả hai đối tượng cùng tương tác, $\bar{x}$ và $\bar{y}$ là giá trị trung bình tương ứng. Đây là độ đo xương sống của thuật toán Collaborative Filtering (Lọc cộng tác) vì nó triệt tiêu hoàn toàn sự khác biệt giữa người dùng "chấm điểm hào phóng" (thường cho 4–5 sao) và người dùng "chấm điểm khắt khe" (thường chỉ cho 1–3 sao).

6. **Độ đo Adjusted Cosine (Cosine hiệu chỉnh):**
   Áp dụng riêng cho Item-based Collaborative Filtering, tính độ tương đồng giữa hai vật phẩm $i$ và $j$ bằng cách trừ đi điểm trung bình của người dùng $\bar{u}$:
   $$\text{sim}_{\text{AdjCos}}(i, j) = \frac{\sum_{u \in U_{ij}} (r_{u,i} - \bar{r}_u)(r_{u,j} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{ij}} (r_{u,i} - \bar{r}_u)^2} \sqrt{\sum_{u \in U_{ij}} (r_{u,j} - \bar{r}_u)^2}}$$

7. **Độ tương đồng Jaccard (Jaccard Similarity) và Khoảng cách Hamming:**
   - *Jaccard:* Đo mức độ giao nhau giữa hai tập hợp rời rạc: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$. Rất phù hợp với bài toán phản hồi tiềm ẩn nhị phân (người dùng đã tải tài liệu hay chưa).
   - *Hamming:* Đếm số lượng vị trí ký tự hoặc bit khác nhau giữa hai chuỗi có cùng độ dài.

**Bảng so sánh và tiêu chí lựa chọn độ đo:**
| Độ đo | Miền giá trị | Loại dữ liệu phù hợp | Trường hợp sử dụng tiêu biểu trong dự án |
|---|---|---|---|
| **Euclidean** | $[0, +\infty)$ | Dữ liệu số liên tục, ít chiều, cùng thang đo | Đo khoảng cách vector thuộc tính sản phẩm |
| **Manhattan** | $[0, +\infty)$ | Dữ liệu có nhiều nhiễu, số chiều trung bình | So sánh đặc trưng thông số kỹ thuật |
| **Cosine** | $[-1, 1]$ (hoặc $[0, 1]$) | Dữ liệu thưa, nhiều chiều, không phụ thuộc độ dài | **Gợi ý tài liệu download bằng TF-IDF** |
| **Pearson** | $[-1, 1]$ | Dữ liệu rating đánh giá có độ lệch bias người dùng | **User-based Collaborative Filtering** |
| **Adjusted Cosine**| $[-1, 1]$ | Dữ liệu rating người dùng - vật phẩm | **Item-based Collaborative Filtering (Module phim)** |
| **Jaccard** | $[0, 1]$ | Dữ liệu nhị phân (tải/chưa tải, mua/chưa mua) | Phân tích tập tài liệu sinh viên đã tải |

---

### 1.2.3. Chọn tham số k, KNN có trọng số và Chuẩn hóa dữ liệu

#### A. Tác động của tham số $k$ và chiến lược lựa chọn tối ưu
Tham số $k$ quy định số lượng người láng giềng gần nhất tham gia vào quá trình quyết định. Giá trị $k$ kiểm soát sự cân bằng cơ bản giữa độ chệch (Bias) và phương sai (Variance) của mô hình:
- **Khi $k$ quá nhỏ (ví dụ $k = 1$):** Mô hình có độ chệch thấp nhưng phương sai cực kỳ cao. Ranh giới quyết định rất phức tạp và nhạy cảm quá mức với nhiễu hoặc các điểm dữ liệu dị biệt (Outliers), dẫn đến hiện tượng **Overfitting (Quá khớp)**.
- **Khi $k$ quá lớn (ví dụ $k \approx n$):** Ranh giới quyết định trở nên quá phẳng và đơn giản hóa. Kết quả dự đoán bị chi phối hoàn toàn bởi lớp đa số chiếm ưu thế trong tập dữ liệu, mô hình có phương sai thấp nhưng độ chệch rất cao, dẫn đến hiện tượng **Underfitting (Chưa khớp)**.

Chiến lược thực nghiệm chuẩn hóa để chọn $k$:
1. Luôn ưu tiên chọn $k$ là **số lẻ** đối với bài toán phân lớp 2 nhãn để tránh tình trạng hòa phiếu (tie-breaking).
2. Sử dụng quy tắc kinh nghiệm xuất phát điểm: $k \approx \sqrt{n}$ (với $n$ là kích thước tập dữ liệu huấn luyện).
3. Sử dụng kỹ thuật kiểm chuẩn chéo (K-Fold Cross-Validation) quét qua một lưới các giá trị $k \in \{1, 3, 5, 7, 9, 15, 21, 31, 51\}$ và chọn $k^*$ tối thiểu hóa sai số RMSE trên tập kiểm thử (như được thực hiện trong Chương 2 của đồ án).

#### B. Thuật toán KNN có trọng số (Weighted KNN)
Trong thuật toán KNN cơ bản, mọi láng giềng trong top-$k$ đều có quyền biểu quyết ngang nhau, bất kể láng giềng thứ nhất cách điểm truy vấn chỉ 0.01 đơn vị còn láng giềng thứ $k$ cách tới 10 đơn vị. Điều này gây bất hợp lý.
Thuật toán KNN có trọng số khắc phục điều này bằng cách gán trọng số nghịch đảo khoảng cách (Inverse Distance Weighting):
$$w_i = \frac{1}{d(\mathbf{x}_q, \mathbf{x}_i)^2 + \epsilon} \quad \text{hoặc} \quad w_i = \text{sim}(\mathbf{x}_q, \mathbf{x}_i)$$
Khi đó giá trị dự đoán hồi quy trở thành trung bình có trọng số:
$$\hat{y}_q = \frac{\sum_{i=1}^k w_i y_i}{\sum_{i=1}^k w_i}$$

#### C. Tầm quan trọng của Chuẩn hóa dữ liệu (Feature Scaling)
Bởi vì khoảng cách Euclidean và Manhattan phụ thuộc trực tiếp vào độ lớn tuyệt đối của từng chiều thuộc tính, một thuộc tính có biên độ đo lớn (ví dụ: Thu nhập hàng chục triệu đồng) sẽ hoàn toàn áp đảo một thuộc tính có biên độ nhỏ (ví dụ: Số sao đánh giá từ 1 đến 5), biến thuộc tính biên độ nhỏ thành vô nghĩa. Vì vậy, chuẩn hóa dữ liệu là tiền điều kiện bắt buộc trước khi chạy KNN:
1. **Chuẩn hóa Min-Max (Min-Max Normalization):**
   Ánh xạ miền giá trị về đoạn $[0, 1]$:
   $$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
2. **Chuẩn hóa Z-Score (Standardization):**
   Biến đổi dữ liệu có kỳ vọng bằng 0 và độ lệch chuẩn bằng 1:
   $$x' = \frac{x - \mu}{\sigma}$$
   *Lưu ý đặc biệt cho Hệ gợi ý:* Đối với ma trận rating thưa, phép chuẩn hóa chỉ được thực hiện trên các ô ĐÃ đánh giá, các ô 0 (chưa đánh giá) bắt buộc phải bảo toàn để không làm biến dạng cấu trúc thưa của ma trận.

---

### 1.2.4. Độ phức tạp tính toán và các kỹ thuật tăng tốc
Độ phức tạp tính toán của thuật toán KNN ngây thơ (Brute-Force Search):
- **Thời gian huấn luyện:** $O(1)$ (chỉ lưu trữ dữ liệu).
- **Thời gian truy vấn (cho 1 điểm mới):** $O(n \cdot d + n \log k)$ với $n$ là số mẫu huấn luyện, $d$ là số chiều thuộc tính.
- **Không gian bộ nhớ:** $O(n \cdot d)$ để lưu trữ dữ liệu và ma trận tương đồng $O(n^2)$ hoặc $O(m^2)$.

Khi $n$ lên tới hàng triệu mẫu (như hệ thống của YouTube hay Shopee), thuật toán Brute-Force hoàn toàn không đáp ứng được yêu cầu thời gian thực (Real-time latency < 50ms). Do đó, các kỹ thuật tăng tốc chỉ mục được nghiên cứu:
1. **KD-Tree (K-Dimensional Tree):**
   Cây nhị phân phân chia không gian không gian lặp đi lặp lại bằng các siêu phẳng trực giao theo từng chiều thuộc tính có phương sai lớn nhất. Cho phép cắt tỉa các nhánh cây ở xa điểm truy vấn, giảm thời gian tìm kiếm trung bình xuống $O(d \cdot \log n)$. Tuy nhiên khi số chiều $d > 20$, KD-Tree thoái hóa về tìm kiếm vét cạn do hiện tượng "thăm lại hầu hết các lá".
2. **Ball-Tree:**
   Phân hoạch không gian thành các siêu cầu lồng nhau. Hoạt động hiệu quả hơn KD-Tree trong không gian có số chiều trung bình cao nhờ tính bất biến quay của hình cầu.
3. **Locality-Sensitive Hashing (LSH):**
   Sử dụng các hàm băm xác suất sao cho các điểm dữ liệu gần nhau trong không gian gốc sẽ có xác suất rơi vào cùng một bucket băm rất cao. Chuyển bài toán tìm kiếm chính xác thành bài toán xấp xỉ láng giềng gần nhất (Approximate Nearest Neighbors - ANN) với độ phức tạp truy vấn trung bình chỉ $O(1)$.
4. **HNSW (Hierarchical Navigable Small World):**
   Cấu trúc đồ thị nhiều tầng phân cấp lấy cảm hứng từ hiện tượng "thế giới nhỏ" (Small-world network), hiện đang là giải thuật tìm kiếm láng giềng xấp xỉ đạt tốc độ và độ hồi quy (Recall) cao nhất trong các cơ sở dữ liệu vector hiện đại (như Milvus, Faiss, Pinecone).

---

### 1.2.5. Ưu, nhược điểm và Lời nguyền số chiều (Curse of Dimensionality)

#### A. Ưu điểm của KNN
1. **Đơn giản, trực quan:** Dễ hiểu, dễ cài đặt và không đòi hỏi các giả định phân phối xác suất phức tạp của dữ liệu (Non-parametric).
2. **Không có giai đoạn huấn luyện phức tạp:** Thêm mới dữ liệu không cần phải đào tạo lại toàn bộ mô hình (Incremental learning tự nhiên).
3. **Tính minh bạch và giải thích cao (Explainable AI):** Có thể lý giải ngay lập tức quyết định của hệ thống bằng cách hiển thị $k$ đối tượng tương tự láng giềng (*"Chúng tôi đề xuất phim này vì bạn đã thích phim A, B, C"*).

#### B. Nhược điểm của KNN
1. **Chi phí truy vấn cao:** Tốc độ suy luận chậm trên các tập dữ liệu lớn.
2. **Đòi hỏi bộ nhớ RAM lớn:** Phải lưu trữ toàn bộ tập dữ liệu mẫu và ma trận tương đồng.
3. **Nhạy cảm với nhiễu và thuộc tính thừa:** Nếu đưa vào các thuộc tính không liên quan, khoảng cách tính toán sẽ bị sai lệch hoàn toàn.

#### C. Lời nguyền số chiều (Curse of Dimensionality)
Hiện tượng lời nguyền số chiều xảy ra khi số chiều thuộc tính $d$ tăng lên rất lớn (ví dụ trong biểu diễn văn bản TF-IDF với $d = 5,000$ từ vựng). Trong không gian số chiều cao:
- Thể tích của không gian tăng theo hàm mũ so với số chiều ($V \propto r^d$), khiến dữ liệu trở nên cực kỳ loãng.
- **Hiện tượng khoảng cách hội tụ (Distance Concentration):** Khoảng cách giữa điểm xa nhất $d_{\max}$ và điểm gần nhất $d_{\min}$ từ một điểm truy vấn tiến dần về 0 theo tỷ lệ tương đối:
  $$\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} = 0$$
Điều này khiến cho khái niệm "láng giềng gần nhất" mất đi ý nghĩa phân biệt, vì mọi điểm dữ liệu đều có khoảng cách xấp xỉ như nhau tới điểm truy vấn.

**Giải pháp khắc phục được áp dụng trong đồ án:**
1. Lọc bỏ các từ dừng (Stopwords) và giới hạn số lượng từ khóa đặc trưng tối đa `max_features=5000`.
2. Áp dụng kỹ thuật giảm chiều không gian tiềm ẩn **TruncatedSVD (Phân tích giá trị kỳ dị rút gọn / Latent Semantic Analysis)** để nén không gian đặc trưng từ 5,000 chiều xuống 50 chiều tiềm ẩn giàu ngữ nghĩa nhất, loại bỏ nhiễu và triệt tiêu lời nguyền số chiều.

---

### 1.2.6. VÍ DỤ TÍNH TOÁN THỦ CÔNG CHI TIẾT BẰNG CÔNG THỨC PEARSON
*(Phần bắt buộc theo quy định đồ án: Minh họa từng bước tính toán bằng tay trên ma trận 5x5)*

Để hiểu rõ bản chất toán học của thuật toán Collaborative Filtering sử dụng hệ số tương quan Pearson, xét tập dữ liệu thu nhỏ gồm 5 người dùng ($U_1, U_2, U_3, U_4, U_5$) và 5 bộ phim ($M_1, M_2, M_3, M_4, M_5$) với thang điểm rating từ 1 đến 5 sao. Ký hiệu `?` là ô chưa đánh giá.

**Bảng 1.1. Ma trận đánh giá User - Item mẫu (5x5):**
| Người dùng | $M_1$ (Hành động) | $M_2$ (Hành động) | $M_3$ (Hài hước) | $M_4$ (Khoa học) | $M_5$ (Tình cảm) | Điểm trung bình $\bar{r}_u$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$U_1$ (Truy vấn)** | 5 | 4 | ? | 4 | 2 | ? |
| **$U_2$** | 4 | 5 | 1 | 4 | 1 | ? |
| **$U_3$** | 2 | 1 | 5 | 2 | 4 | ? |
| **$U_4$** | 4 | 3 | 2 | 5 | 3 | ? |
| **$U_5$** | 1 | 2 | 4 | 2 | 5 | ? |

**Mục tiêu bài toán:**
Dự đoán điểm đánh giá của người dùng **$U_1$** cho bộ phim **$M_3$** (tức tính $\hat{r}_{U_1, M_3}$) với số láng giềng $k = 2$.

---

#### BƯỚC 1: Tính điểm trung bình đánh giá của từng người dùng
*Lưu ý:* Điểm trung bình chỉ tính trên các ô NGƯỜI ĐÓ ĐÃ ĐÁNH GIÁ (bỏ qua ô `?`).
- Người dùng $U_1$: Đã chấm $\{M_1: 5, M_2: 4, M_4: 4, M_5: 2\}$. Số lượng: 4 phim.
  $$\bar{r}_{U_1} = \frac{5 + 4 + 4 + 2}{4} = \frac{15}{4} = 3.75$$
- Người dùng $U_2$: Đã chấm $\{M_1: 4, M_2: 5, M_3: 1, M_4: 4, M_5: 1\}$. Số lượng: 5 phim.
  $$\bar{r}_{U_2} = \frac{4 + 5 + 1 + 4 + 1}{5} = \frac{15}{5} = 3.00$$
- Người dùng $U_3$: Đã chấm $\{M_1: 2, M_2: 1, M_3: 5, M_4: 2, M_5: 4\}$. Số lượng: 5 phim.
  $$\bar{r}_{U_3} = \frac{2 + 1 + 5 + 2 + 4}{5} = \frac{14}{5} = 2.80$$
- Người dùng $U_4$: Đã chấm $\{M_1: 4, M_2: 3, M_3: 2, M_4: 5, M_5: 3\}$. Số lượng: 5 phim.
  $$\bar{r}_{U_4} = \frac{4 + 3 + 2 + 5 + 3}{5} = \frac{17}{5} = 3.40$$
- Người dùng $U_5$: Đã chấm $\{M_1: 1, M_2: 2, M_3: 4, M_4: 2, M_5: 5\}$. Số lượng: 5 phim.
  $$\bar{r}_{U_5} = \frac{1 + 2 + 4 + 2 + 5}{5} = \frac{14}{5} = 2.80$$

---

#### BƯỚC 2: Tính độ tương quan Pearson giữa $U_1$ và từng người dùng còn lại
Công thức hệ số tương quan Pearson giữa hai người dùng $u$ và $v$:
$$\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u) (r_{v,i} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)^2} \cdot \sqrt{\sum_{i \in I_{uv}} (r_{v,i} - \bar{r}_v)^2}}$$
Tập các phim mà $U_1$ đã đánh giá là $\{M_1, M_2, M_4, M_5\}$. Vì các user $U_2, U_3, U_4, U_5$ đều đã đánh giá đủ các phim này nên tập giao $I_{U_1, v} = \{M_1, M_2, M_4, M_5\}$.

Độ lệch chuẩn hóa của $U_1$ trên các phim giao:
- Phim $M_1$: $r_{1,1} - \bar{r}_1 = 5 - 3.75 = +1.25$
- Phim $M_2$: $r_{1,2} - \bar{r}_1 = 4 - 3.75 = +0.25$
- Phim $M_4$: $r_{1,4} - \bar{r}_1 = 4 - 3.75 = +0.25$
- Phim $M_5$: $r_{1,5} - \bar{r}_1 = 2 - 3.75 = -1.75$

Tổng bình phương độ lệch của $U_1$:
$$\sum_{i \in I} (r_{1,i} - \bar{r}_1)^2 = (1.25)^2 + (0.25)^2 + (0.25)^2 + (-1.75)^2 = 1.5625 + 0.0625 + 0.0625 + 3.0625 = 4.7500$$
Suy ra: $\sqrt{\sum (r_{1,i} - \bar{r}_1)^2} = \sqrt{4.75} \approx 2.1794$.

##### 1. Tính tương quan giữa $U_1$ và $U_2$:
Độ lệch của $U_2$ (với $\bar{r}_2 = 3.00$):
- $M_1$: $4 - 3.0 = +1.0$
- $M_2$: $5 - 3.0 = +2.0$
- $M_4$: $4 - 3.0 = +1.0$
- $M_5$: $1 - 3.0 = -2.0$

Tích vô hướng:
$$\sum = (1.25 \times 1.0) + (0.25 \times 2.0) + (0.25 \times 1.0) + (-1.75 \times -2.0)$$
$$\sum = 1.25 + 0.50 + 0.25 + 3.50 = 5.5000$$

Tổng bình phương độ lệch của $U_2$:
$$\sum = 1.0^2 + 2.0^2 + 1.0^2 + (-2.0)^2 = 1 + 4 + 1 + 4 = 10.0000 \implies \sqrt{10} \approx 3.1623$$

Hệ số tương quan:
$$\text{sim}(U_1, U_2) = \frac{5.5000}{2.1794 \times 3.1623} = \frac{5.5000}{6.8920} \approx \mathbf{+0.7980}$$
*(Kết luận: $U_1$ và $U_2$ có tương quan đồng thuận rất cao, cùng thích phim hành động $M_1, M_2$ và không thích phim tình cảm $M_5$)*.

##### 2. Tính tương quan giữa $U_1$ và $U_3$:
Độ lệch của $U_3$ (với $\bar{r}_3 = 2.80$):
- $M_1$: $2 - 2.8 = -0.8$
- $M_2$: $1 - 2.8 = -1.8$
- $M_4$: $2 - 2.8 = -0.8$
- $M_5$: $4 - 2.8 = +1.2$

Tích vô hướng:
$$\sum = (1.25 \times -0.8) + (0.25 \times -1.8) + (0.25 \times -0.8) + (-1.75 \times 1.2)$$
$$\sum = -1.00 - 0.45 - 0.20 - 2.10 = -3.7500$$

Tổng bình phương độ lệch của $U_3$:
$$\sum = (-0.8)^2 + (-1.8)^2 + (-0.8)^2 + 1.2^2 = 0.64 + 3.24 + 0.64 + 1.44 = 5.9600 \implies \sqrt{5.96} \approx 2.4413$$

Hệ số tương quan:
$$\text{sim}(U_1, U_3) = \frac{-3.7500}{2.1794 \times 2.4413} = \frac{-3.7500}{5.3206} \approx \mathbf{-0.7048}$$
*(Kết luận: $U_1$ và $U_3$ tương quan nghịch chiều, sở thích hoàn toàn trái ngược nhau)*.

##### 3. Tính tương quan giữa $U_1$ và $U_4$:
Độ lệch của $U_4$ (với $\bar{r}_4 = 3.40$):
- $M_1$: $4 - 3.4 = +0.6$
- $M_2$: $3 - 3.4 = -0.4$
- $M_4$: $5 - 3.4 = +1.6$
- $M_5$: $3 - 3.4 = -0.4$

Tích vô hướng:
$$\sum = (1.25 \times 0.6) + (0.25 \times -0.4) + (0.25 \times 1.6) + (-1.75 \times -0.4)$$
$$\sum = 0.75 - 0.10 + 0.40 + 0.70 = 1.7500$$

Tổng bình phương độ lệch của $U_4$:
$$\sum = 0.6^2 + (-0.4)^2 + 1.6^2 + (-0.4)^2 = 0.36 + 0.16 + 2.56 + 0.16 = 3.2400 \implies \sqrt{3.24} = 1.8000$$

Hệ số tương quan:
$$\text{sim}(U_1, U_4) = \frac{1.7500}{2.1794 \times 1.8000} = \frac{1.7500}{3.9230} \approx \mathbf{+0.4461}$$
*(Kết luận: $U_1$ và $U_4$ có tương quan thuận ở mức độ trung bình)*.

##### 4. Tính tương quan giữa $U_1$ và $U_5$:
Độ lệch của $U_5$ (với $\bar{r}_5 = 2.80$):
- $M_1$: $1 - 2.8 = -1.8$
- $M_2$: $2 - 2.8 = -0.8$
- $M_4$: $2 - 2.8 = -0.8$
- $M_5$: $5 - 2.8 = +2.2$

Tích vô hướng:
$$\sum = (1.25 \times -1.8) + (0.25 \times -0.8) + (0.25 \times -0.8) + (-1.75 \times 2.2)$$
$$\sum = -2.25 - 0.20 - 0.20 - 3.85 = -6.5000$$

Tổng bình phương độ lệch của $U_5$:
$$\sum = (-1.8)^2 + (-0.8)^2 + (-0.8)^2 + 2.2^2 = 3.24 + 0.64 + 0.64 + 4.84 = 9.3600 \implies \sqrt{9.36} \approx 3.0594$$

Hệ số tương quan:
$$\text{sim}(U_1, U_5) = \frac{-6.5000}{2.1794 \times 3.0594} = \frac{-6.5000}{6.6677} \approx \mathbf{-0.9748}$$
*(Kết luận: $U_1$ và $U_5$ có sở thích hoàn toàn đối lập)*.

---

#### BƯỚC 3: Chọn ra $k = 2$ láng giềng gần nhất có tương quan dương
Bảng xếp hạng độ tương đồng với $U_1$:
1. $U_2$: $\text{sim} = +0.7980$ (Láng giềng 1)
2. $U_4$: $\text{sim} = +0.4461$ (Láng giềng 2)
3. $U_3$: $\text{sim} = -0.7048$ (Bị loại vì tương quan âm)
4. $U_5$: $\text{sim} = -0.9748$ (Bị loại vì tương quan âm)

Như vậy, tập $k = 2$ láng giềng được chọn là: $\mathbf{N = \{U_2, U_4\}}$.
Cả hai láng giềng này đều đã đánh giá bộ phim $M_3$:
- $U_2$ đánh giá phim $M_3$: $r_{2,3} = 1.0$
- $U_4$ đánh giá phim $M_3$: $r_{4,3} = 2.0$

---

#### BƯỚC 4: Dự đoán điểm đánh giá $\hat{r}_{U_1, M_3}$ theo công thức chuẩn hóa
Công thức dự đoán điểm có khử bias trung bình của người dùng (Mean-centered Prediction):
$$\hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in N} \text{sim}(u, v) \cdot (r_{v,i} - \bar{r}_v)}{\sum_{v \in N} |\text{sim}(u, v)|}$$

Thay các giá trị số cụ thể vào công thức:
- $\bar{r}_{U_1} = 3.75$
- Láng giềng $U_2$: $\text{sim} = 0.7980$, điểm đánh giá $r_{2,3} = 1.0$, độ lệch: $r_{2,3} - \bar{r}_2 = 1.0 - 3.0 = -2.0$
- Láng giềng $U_4$: $\text{sim} = 0.4461$, điểm đánh giá $r_{4,3} = 2.0$, độ lệch: $r_{4,3} - \bar{r}_4 = 2.0 - 3.4 = -1.4$

Tính tử số (Tổng độ lệch có trọng số):
$$\text{Tử số} = [0.7980 \times (-2.0)] + [0.4461 \times (-1.4)] = -1.5960 - 0.6245 = -2.2205$$

Tính mẫu số (Tổng độ lớn trọng số):
$$\text{Mẫu số} = 0.7980 + 0.4461 = 1.2441$$

Tỷ số điều chỉnh:
$$\Delta r = \frac{-2.2205}{1.2441} \approx -1.7848$$

Điểm dự đoán cuối cùng:
$$\hat{r}_{U_1, M_3} = 3.75 + (-1.7848) = 1.9652 \approx \mathbf{1.97 \text{ sao}}$$

**Nhận xét phân tích kết quả:**
Kết quả dự đoán $\hat{r}_{U_1, M_3} \approx 1.97$ sao phản ánh chính xác thực tế hành vi: $U_1$ là người rất thích phim hành động/viễn tưởng (chấm $M_1, M_2, M_4$ từ 4 đến 5 sao) và không thích phim nhẹ nhàng. Hai láng giềng có gu tương đồng nhất với $U_1$ là $U_2$ và $U_4$ đều chấm phim hài $M_3$ điểm rất thấp (1 sao và 2 sao, thấp hơn mức trung bình của chính họ). Do đó hệ thống đưa ra dự đoán $U_1$ cũng sẽ không thích phim $M_3$ và sẽ **không** gợi ý phim này trong danh sách Top-N.

---

## 1.3. Hệ gợi ý và vai trò của KNN
*(Các mục 1.3.1, 1.3.3, 1.3.4, 1.3.5, 1.3.6 do Hùng và Hoàng viết. Dưới đây là mục 1.3.2 do Linh phụ trách)*

### 1.3.2. Lọc theo nội dung (Content-Based) kết hợp KNN
Phương pháp lọc theo nội dung (Content-Based Filtering) xây dựng trên nguyên tắc: *"Nếu một người dùng từng yêu thích một vật phẩm trong quá khứ, họ sẽ có xu hướng yêu thích các vật phẩm khác có nội dung, thuộc tính hoặc đặc trưng tương tự trong tương lai."*

#### 1. Nguyên lý biểu diễn dữ liệu và Vector Space Model
Để so sánh nội dung giữa các vật phẩm, mọi thông tin mô tả văn bản (tiêu đề, tóm tắt, từ khóa, tác giả, chuyên mục) cần được chuyển đổi thành các vector số học trong không gian nhiều chiều:
- **Tập từ vựng (Bag of Words):** Đếm tần suất xuất hiện thuần túy của từng từ. Phương pháp này có nhược điểm là các từ nối phổ biến (như "và", "của", "is", "the") xuất hiện nhiều nhưng không mang giá trị phân loại.
- **Trọng số TF-IDF (Term Frequency - Inverse Document Frequency):** Kỹ thuật đo lường tầm quan trọng thống kê của một từ $t$ trong tài liệu $d$ thuộc tập tài liệu $D$:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
  Trong đó:
  - $\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$ (Tần suất xuất hiện của từ $t$ trong tài liệu $d$).
  - $\text{IDF}(t, D) = \log \left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$ (Nghịch đảo tần suất văn bản, phạt nặng các từ xuất hiện đại trà ở mọi tài liệu).

#### 2. Xây dựng Hồ sơ Vật phẩm (Item Profile) và Hồ sơ Người dùng (User Profile)
- **Hồ sơ vật phẩm (Item Profile $\mathbf{v}_j$):** Mỗi tài liệu hoặc sản phẩm được đại diện bởi vector thuộc tính đã được chuẩn hóa độ dài đơn vị:
  $$\mathbf{v}_j = \frac{\mathbf{x}_j}{\|\mathbf{x}_j\|_2} \in \mathbb{R}^d$$
- **Hồ sơ sở thích người dùng (User Profile $\mathbf{u}_i$):** Được tổng hợp từ các vector vật phẩm mà người dùng $i$ đã tương tác trong lịch sử, có thể gắn kèm trọng số hành vi $w_{ij}$:
  $$\mathbf{u}_i = \sum_{j \in \text{Liked}(i)} w_{ij} \mathbf{v}_j$$
  Với dữ liệu phản hồi tiềm ẩn (Implicit Feedback), $w_{ij}$ phản ánh mức độ quan tâm: Tải về ($w = 1.0$), Đánh dấu yêu thích / Bookmark ($w = 2.5$), Trích dẫn ($w = 3.0$).

#### 3. Ứng dụng KNN trong lọc theo nội dung
Trong hệ thống Content-based, thuật toán KNN hoạt động theo hai kịch bản trực tiếp:
1. **Tìm $k$ vật phẩm giống nhất với vật phẩm hiện tại (Item-to-Item Similarity):**
   Khi người dùng đang đọc tài liệu $A$, hệ thống sử dụng `KNNIndex` với độ đo Cosine quét qua toàn bộ kho dữ liệu để tìm ra $k$ tài liệu có vector đặc trưng gần $A$ nhất.
2. **Sinh danh sách gợi ý cá nhân hóa Top-N:**
   Điểm phù hợp giữa người dùng $u$ và một tài liệu ứng viên $c$ được tính bằng tổng độ tương đồng giữa $c$ và các tài liệu trong lịch sử của $u$:
   $$\text{Score}(u, c) = \sum_{j \in \text{Liked}(u)} w_{uj} \cdot \text{sim}_{\text{Cosine}}(\mathbf{v}_c, \mathbf{v}_j)$$
   Hệ thống chọn ra $N$ tài liệu có điểm $\text{Score}$ cao nhất, loại trừ các tài liệu người dùng đã từng đọc.

#### 4. Ưu điểm và Hạn chế của Content-Based + KNN
- **Ưu điểm nổi bật:**
  - *Giải quyết hoàn toàn bài toán Cold-Start cho vật phẩm mới (New Item):* Khi một tài liệu mới được tải lên hệ thống, ngay lập tức ta có thể trích xuất vector TF-IDF từ nội dung của nó và đưa vào chỉ mục KNN để gợi ý, không cần chờ có người khác đọc trước.
  - *Độc lập giữa các người dùng:* Không phụ thuộc vào số lượng người dùng khác trên hệ thống (không bị ảnh hưởng bởi độ thưa ma trận rating cộng tác).
  - *Minh bạch và dễ giải thích:* Dễ dàng giải thích tại sao tài liệu được gợi ý dựa trên sự trùng khớp từ khóa hoặc chủ đề.
- **Hạn chế:**
  - *Hiện tượng đóng khung sở thích (Over-specialization / Serendipity Problem):* Hệ thống chỉ gợi ý những gì tương tự thứ người dùng đã biết, không thể gợi ý những chủ đề mới lạ mà người dùng tiềm năng quan tâm nhưng chưa từng tìm kiếm.
  - *Vẫn gặp vấn đề Cold-Start cho người dùng mới hoàn toàn (New User):* Khi người dùng vừa lập tài khoản và chưa có bất kỳ tương tác nào, không thể xây dựng được hồ sơ người dùng.

---

# CHƯƠNG 2. XÂY DỰNG ỨNG DỤNG
*(Các mục 2.1, 2.2, 2.3 và 2.6 do Hoàng và Hùng đảm nhiệm)*

## 2.4. Module gợi ý tài liệu download (TF-IDF, Implicit Feedback)

### 2.4.1. Đặc tả bài toán và nguồn dữ liệu tài liệu số
Trong môi trường giáo dục đại học hoặc các cổng thông tin số, sinh viên và giảng viên thường xuyên tìm kiếm, xem trước và tải về các tài liệu học tập, slide bài giảng, đề thi và giáo trình. Khác với bài toán xem phim (nơi người dùng chủ động chấm 1 đến 5 sao - Phản hồi tường minh / Explicit Feedback), hệ thống tài liệu hầu như chỉ ghi nhận các hành vi không lời: người dùng click vào tài liệu, xem trực tuyến, tải file PDF hoặc đánh dấu bookmark. Đây là dữ liệu **Phản hồi tiềm ẩn (Implicit Feedback)**.

Tập dữ liệu tài liệu gồm hai thành phần:
1. **Metadata tài liệu:** Mỗi tài liệu gồm mã định danh `doc_id`, tiêu đề `title`, bản tóm tắt `abstract`, thể loại `category` và danh sách từ khóa `tags`.
2. **Lịch sử tương tác tiềm ẩn:** Ghi nhận các hành vi của sinh viên kèm nhãn hành vi: `download`, `view`, `bookmark`.

### 2.4.2. Quy trình trích xuất đặc trưng TF-IDF và N-Gram
Để mô hình hóa toàn diện ngữ nghĩa của tài liệu, chuỗi văn bản đầu vào được ghép từ toàn bộ các trường thông tin: `text = title + " " + abstract + " " + tags`.
Quy trình biến đổi văn bản thành vector số học gồm các công đoạn:
1. **Tiền xử lý văn bản:** Chuyển đổi toàn bộ sang chữ thường (lowercase), loại bỏ các ký tự đặc biệt, dấu câu và lọc bỏ danh sách từ dừng tiếng Anh/tiếng Việt phổ biến.
2. **Khai thác N-Gram:** Thiết lập `ngram_range=(1, 2)` để không chỉ trích xuất các từ đơn lẻ (Unigram như "mạng", "máy tính") mà còn bảo toàn được các cụm từ ghép có ý nghĩa kỹ thuật quan trọng (Bigram như "mạng máy tính", "trí tuệ nhân tạo", "hệ gợi ý").
3. **Giới hạn từ vựng `max_features=5000`:** Chọn lọc 5,000 từ và cụm từ có điểm TF-IDF trung bình cao nhất trên toàn bộ kho tài liệu, loại bỏ các từ vựng quá hiếm xuất hiện (nhiễu ngẫu nhiên) hoặc quá phổ biến.

### 2.4.3. Giảm chiều dữ liệu với TruncatedSVD (Latent Semantic Analysis - LSA)
Sau bước TF-IDF, ma trận đặc trưng $X_{\text{TF-IDF}}$ có số chiều rất lớn (5,000 chiều) nhưng độ thưa rất cao (hơn 95% giá trị bằng 0). Để tối ưu hóa hiệu năng tính toán của KNN và loại bỏ hiện tượng đồng nghĩa/đa nghĩa trong ngôn ngữ, đồ án tích hợp thuật toán **TruncatedSVD (Singular Value Decomposition rút gọn)**:
$$X \approx U_k \Sigma_k V_k^T$$
Kỹ thuật này ánh xạ ma trận tài liệu từ không gian 5,000 chiều từ khóa ban đầu về không gian ngữ nghĩa tiềm ẩn gồm $c$ chiều (ví dụ $c = 50$). Ma trận thu gọn $X_{\text{reduced}}$ có các ưu điểm:
- Giảm 100 lần dung lượng bộ nhớ RAM cần thiết để lưu trữ và tính toán khoảng cách.
- Triệt tiêu hoàn toàn tác động tiêu cực của "Lời nguyền số chiều" lên hàm đo Cosine.
- Nhận diện được mối liên hệ ngữ nghĩa giữa hai tài liệu dùng các từ ngữ khác nhau nhưng cùng một chủ đề (ví dụ tài liệu chứa từ "Machine Learning" và tài liệu chứa từ "Học máy").

### 2.4.4. Xử lý phản hồi tiềm ẩn và Thiết kế lớp `DocumentRecommender`
Module `rec_document.py` được thiết kế theo mô hình hướng đối tượng, tích hợp trực tiếp với nhân `KNNIndex` dùng chung (`src/knn_core.py`):
1. **Gán trọng số tương tác:**
   $$\text{download} \to 1.0, \quad \text{view} \to 0.5, \quad \text{bookmark} \to 2.5$$
2. **Tổng hợp điểm gợi ý và Sinh lời giải thích (Explainable AI):**
   Khi người dùng $u$ yêu cầu gợi ý, hệ thống tính điểm cho từng tài liệu ứng viên $c$:
   $$\text{Score}(u, c) = \sum_{d \in \text{History}(u)} w_d \cdot \text{sim}_{\text{Cosine}}(\mathbf{x}_c, \mathbf{x}_d)$$
   Đồng thời, hệ thống tìm ra tài liệu lịch sử $d^*$ có đóng góp lớn nhất vào số điểm này:
   $$d^* = \arg\max_{d \in \text{History}(u)} \left(w_d \cdot \text{sim}_{\text{Cosine}}(\mathbf{x}_c, \mathbf{x}_d)\right)$$
   Từ đó sinh ra câu giải thích trực quan: *"Gợi ý vì bạn đã tải/đánh dấu tài liệu '{Title của $d^*$}' (độ tương đồng nội dung: {sim}%)"*.

---

## 2.5. Giao diện demo (Client Web)

### 2.5.1. Mục tiêu thiết kế và Yêu cầu kỹ thuật
Giao diện người dùng đóng vai trò cầu nối trực tiếp giữa các thuật toán học máy phức tạp ở máy chủ backend và người sử dụng cuối cùng. Mục tiêu thiết kế của giao diện Client Web bao gồm:
1. **Trực quan hóa đa bài toán:** Cung cấp trải nghiệm liền mạch cho cả ba bài toán gợi ý của đề tài: Gợi ý xem phim, Gợi ý tài liệu và Gợi ý hàng mới ra mắt.
2. **Khả năng giải thích minh bạch (Explainable AI):** Mọi kết quả đề xuất Top-N bắt buộc phải đi kèm lý do giải thích cụ thể để người dùng hiểu được căn nguyên của gợi ý, nâng cao mức độ tin cậy vào hệ thống.
3. **Độc lập và linh hoạt:** Thiết kế theo kiến trúc Single Page Application (SPA) bằng HTML5, CSS3 hiện đại và JavaScript thuần, hỗ trợ chế độ tự động nhận diện máy chủ FastAPI (Backend Mode) hoặc chuyển sang dữ liệu mô phỏng (Offline Demo Mode) khi máy chủ chưa khởi động.

### 2.5.2. Kiến trúc giao diện 3 Tab chức năng
Giao diện được phân bổ thành ba tab điều hướng chính:
1. **Tab 1 — Gợi ý phim (MovieLens Collaborative Filtering):** Cho phép chọn các người dùng mẫu có gu phim khác biệt (Action/Sci-Fi, Hoạt hình, Tâm lý tội phạm); hỗ trợ chuyển đổi giữa hai biến thể `Item-based KNN` (Adjusted Cosine) và `User-based KNN` (Pearson).
2. **Tab 2 — Gợi ý tài liệu download (TF-IDF + Cosine KNN):** Mô phỏng trải nghiệm bạn đọc tại thư viện trường đại học. Người dùng chọn hồ sơ sinh viên để nhận về các tài liệu nghiên cứu phù hợp nhất với các tài liệu đã tải hoặc đánh dấu trong quá khứ.
3. **Tab 3 — Gợi ý hàng mới (Cold-Start Hybrid KNN):** Minh họa giải pháp cho bài toán sản phẩm vừa nhập kho chưa có lịch sử mua hàng, hệ thống dựa vào vector đặc trưng kỹ thuật để tìm $k$ sản phẩm cũ tương đồng nhất và đề xuất cho nhóm khách hàng phù hợp.

### 2.5.3. Bố cục các thành phần và Cơ chế giải thích gợi ý
Mỗi màn hình được chia thành ba khu vực công năng chuẩn:
- **Thanh điều khiển (Control Panel):** Gồm bộ chọn người dùng (User Selector), thanh trượt chọn số lượng Top-N (từ 2 đến 12 kết quả), danh sách lựa chọn thuật toán/độ đo và nút kích hoạt truy vấn "Lấy gợi ý ngay".
- **Hộp bối cảnh người dùng (Context Banner):** Tóm tắt trực quan lịch sử tương tác của người dùng hiện tại (danh sách các phim đã xem, tài liệu đã tải).
- **Lưới hiển thị kết quả (Cards Grid):** Mỗi vật phẩm gợi ý được thể hiện dưới dạng thẻ Card hiện đại gồm: Số thứ tự xếp hạng (#Rank), Thể loại, Tiêu đề, Tóm tắt nội dung, Nhãn tag, Thanh chỉ số điểm tin cậy và **Hộp thoại giải thích (Explain Box)** màu xanh nổi bật với biểu tượng bóng đèn 💡.

---
### CÁC KHUNG VỊ TRÍ CHÈN ẢNH CHỤP MÀN HÌNH (PLACEHOLDERS CHO BÁO CÁO):

> **[Hình 2.5.1: Giao diện tổng thể hệ thống gợi ý đa nhiệm KNN - Tab Gợi ý xem phim]**  
> *(Hình chụp màn hình toàn bộ trang web khi chọn Tab Gợi ý xem phim cho User 1, hiển thị danh sách Top-6 bộ phim đề xuất kèm hộp thoại lý do gợi ý "Vì bạn đã chấm 5★ cho phim The Matrix")*

> **[Hình 2.5.2: Giao diện Tab Gợi ý tài liệu download bằng TF-IDF và Cosine KNN]**  
> *(Hình chụp màn hình kết quả gợi ý tài liệu học tập cho Sinh viên 01, hiển thị độ tương đồng nội dung 89.2% và lý do đề xuất từ tài liệu đã bookmark)*

> **[Hình 2.5.3: Giao diện Tab Gợi ý hàng mới giải quyết bài toán Cold-Start]**  
> *(Hình chụp màn hình sản phẩm công nghệ mới ra mắt được gợi ý cho khách hàng dựa trên thuộc tính tương đồng với các sản phẩm đã mua trước đây)*

> **[Hình 2.5.4: Chi tiết cấu trúc Thẻ kết quả gợi ý minh họa tính năng Explainable AI]**  
> *(Hình chụp phóng to một Card kết quả thể hiện rõ tiêu đề, điểm số và khối giải thích lý do gợi ý trực quan)*

---
*(Hết phần nội dung báo cáo của Linh)*

