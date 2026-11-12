"""
Script tạo file Word BaoCao_Phan_Linh.docx từ nội dung học thuật chi tiết.
Tuân thủ tuyệt đối quy định giảng viên:
  - Font: Times New Roman
  - Size: 13pt
  - Căn đều: Justified
  - Thụt đầu dòng: 1cm (first_line_indent = Cm(1.0))
  - Tiêu đề in đậm, phân cấp rõ ràng
"""

import os
import docx
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_report_docx(output_path="docs/BaoCao_Phan_Linh.docx"):
    doc = docx.Document()

    # Cấu hình lề trang chuẩn A4: Trái 3cm, Phải 2cm, Trên 2cm, Dưới 2cm
    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.0)

    # Cấu hình Normal Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(13)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.3
    normal_style.paragraph_format.space_after = Pt(4)
    normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal_style.paragraph_format.first_line_indent = Cm(1.0)

    def add_p(text, bold_prefix="", italic=False):
        p = doc.add_paragraph()
        if bold_prefix:
            run_b = p.add_run(bold_prefix)
            run_b.bold = True
            run_b.font.name = 'Times New Roman'
        run_t = p.add_run(text)
        run_t.italic = italic
        run_t.font.name = 'Times New Roman'
        return p

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(15)
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(14)
        return p

    def add_h3(text):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(13)
        return p

    def add_box(title, text):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        cell.width = Cm(15.5)
        set_cell_background(cell, "F8FAFC")
        set_cell_margins(cell, 150, 150, 200, 200)
        p = cell.paragraphs[0]
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r_title = p.add_run(title + "\n")
        r_title.bold = True
        r_title.font.name = 'Times New Roman'
        r_title.font.size = Pt(12)
        r_text = p.add_run(text)
        r_text.font.name = 'Times New Roman'
        r_text.font.size = Pt(12)
        # Cách đoạn sau bảng
        sp_p = doc.add_paragraph()
        sp_p.paragraph_format.first_line_indent = Cm(0)
        sp_p.paragraph_format.space_after = Pt(4)

    # ==================== TRANG BÌA PHỤ / ĐẦU ĐỀ ====================
    title_p = doc.add_paragraph()
    title_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.first_line_indent = Cm(0)
    title_p.paragraph_format.space_before = Pt(20)
    title_p.paragraph_format.space_after = Pt(12)
    r1 = title_p.add_run("BÁO CÁO HỌC THUẬT CHI TIẾT ĐỒ ÁN MÔN MẠNG MÁY TÍNH\n")
    r1.bold = True
    r1.font.size = Pt(16)
    r2 = title_p.add_run("ĐỀ TÀI: TÌM HIỂU THUẬT TOÁN KNN VÀ ỨNG DỤNG HỆ GỢI Ý\n")
    r2.bold = True
    r2.font.size = Pt(14)
    r3 = title_p.add_run("PHẦN ĐẢM NHIỆM: SINH VIÊN LINH (MỤC 1.1, 1.2, 1.3.2, 2.4, 2.5)")
    r3.italic = True
    r3.font.size = Pt(12)

    # ==================== CHƯƠNG 1 ====================
    add_h1("CHƯƠNG 1. TỔNG QUAN VỀ TRÍ TUỆ NHÂN TẠO VÀ CÁC THUẬT TOÁN")

    add_h2("1.1. Tổng quan về Trí tuệ nhân tạo")
    add_h3("1.1.1. Khái niệm và các nhánh của Trí tuệ nhân tạo")
    add_p(
        "Trí tuệ nhân tạo (Artificial Intelligence - AI) là một ngành khoa học máy tính liên ngành rộng lớn, "
        "tập trung vào việc nghiên cứu, thiết kế và phát triển các hệ thống hoặc máy móc có khả năng mô phỏng các "
        "quá trình nhận thức, suy luận, học hỏi, thích nghi và giải quyết vấn đề tương tự như trí tuệ con người. "
        "Khái niệm này được đề xuất chính thức lần đầu tiên vào năm 1956 tại Hội nghị Dartmouth bởi John McCarthy "
        "cùng các nhà khoa học tiên phong như Marvin Minsky, Nathaniel Rochester và Claude Shannon. Kể từ đó, "
        "AI đã trải qua nhiều giai đoạn chuyển biến mạnh mẽ, từ các hệ thống chuyên gia dựa trên luật suy diễn logic "
        "hình thức (Symbolic AI) sang các hệ thống định hướng dữ liệu hiện đại có khả năng tự động học hỏi từ thông "
        "tin thực nghiệm quy mô lớn."
    )
    add_p(
        "Để nắm bắt đúng bản chất kỹ thuật, cần phân biệt rõ ba khái niệm có mối quan hệ thứ bậc chặt chẽ: "
        "Trí tuệ nhân tạo (AI), Học máy (Machine Learning - ML) và Học sâu (Deep Learning - DL). Trí tuệ nhân tạo là "
        "tập hợp bao quát nhất, đại diện cho mọi phương pháp cho phép máy tính bắt chước hành vi thông minh. Học máy là "
        "tập con cốt lõi của AI, dựa trên nguyên lý cung cấp dữ liệu mẫu để thuật toán tự động tìm ra các quy luật tiềm "
        "ẩn mà không cần lập trình thủ công từng luật xử lý. Học sâu lại là tập con chuyên sâu của học máy, ứng dụng "
        "kiến trúc mạng nơ-ron nhân tạo nhiều tầng (Deep Neural Networks) để tự động trích xuất các đặc trưng trừu "
        "tượng từ dữ liệu phi cấu trúc như hình ảnh, âm thanh hay văn bản."
    )
    add_p(
        "Các nhánh nghiên cứu chủ đạo của AI hiện đại bao gồm: (1) Xử lý ngôn ngữ tự nhiên (NLP) phục vụ việc đọc "
        "hiểu và phân tích văn bản; (2) Thị giác máy tính (Computer Vision) giúp máy trích xuất thông tin thị giác; "
        "(3) Hệ chuyên gia và biểu diễn tri thức; (4) Khai phá dữ liệu và Học máy; và (5) Hệ thống gợi ý (Recommender "
        "Systems) — nhánh ứng dụng thực tiễn giải quyết trực tiếp thách thức 'quá tải thông tin' trên không gian mạng."
    )

    add_h3("1.1.2. Học máy và phân loại bài toán học máy")
    add_p(
        "Học máy (Machine Learning) được định nghĩa kinh điển bởi Tom Mitchell: 'Một chương trình máy tính được xem là "
        "học từ kinh nghiệm E đối với một lớp tác vụ T và độ đo hiệu năng P, nếu hiệu năng của nó đối với các tác vụ "
        "trong T, được đo bởi P, cải thiện dần theo kinh nghiệm E'. Dựa trên cách thức tiếp nhận dữ liệu và phản hồi, "
        "học máy được chia thành ba nhánh kinh điển: Học có giám sát (Supervised Learning với hai bài toán Phân lớp "
        "và Hồi quy); Học không giám sát (Unsupervised Learning với bài toán Gom cụm và Giảm chiều dữ liệu); và Học "
        "tăng cường (Reinforcement Learning dựa trên tương tác tác tử - môi trường)."
    )
    add_p(
        "Vị trí của thuật toán K-Nearest Neighbors (KNN): KNN là một giải thuật học máy có giám sát thuộc nhóm phi tham số "
        "(non-parametric) và học lười biếng (lazy learning). KNN giải quyết cả hai bài toán: Phân lớp (Majority Voting) "
        "và Hồi quy (dự đoán rating trong hệ gợi ý). Tính trực quan, minh bạch và khả năng giải thích rõ ràng giúp KNN "
        "trở thành giải thuật kinh điển và nền tảng trong cả nghiên cứu lẫn ứng dụng công nghiệp."
    )

    # 1.2 THUẬT TOÁN KNN
    add_h2("1.2. Thuật toán K-Nearest Neighbors (KNN)")
    add_h3("1.2.1. Ý tưởng, mã giả và bản chất Lazy Learning")
    add_p(
        "Ý tưởng cốt lõi của KNN dựa trên tiên đề hình học: các đối tượng có thuộc tính, đặc điểm hoặc hành vi tương "
        "đồng sẽ nằm gần nhau trong không gian đặc trưng. Thuật toán là một giải thuật 'học lười biếng' (Lazy Learning): "
        "pha huấn luyện chỉ đơn giản là nạp dữ liệu vào bộ nhớ RAM (chi phí tính toán O(1)). Ngược lại, toàn bộ chi phí "
        "tính toán nặng nhọc bị dồn sang pha suy luận (Inference phase) khi hệ thống phải duyệt qua toàn bộ tập dữ liệu "
        "để tính khoảng cách và chọn ra k láng giềng gần nhất (chi phí O(n·d))."
    )
    add_box(
        "MÃ GIẢ THUẬT TOÁN K-NEAREST NEIGHBORS TỔNG QUÁT",
        "Đầu vào: Tập dữ liệu D = {(x_1, y_1), ..., (x_n, y_n)}, điểm truy vấn x_q, tham số k, hàm khoảng cách dist().\n"
        "Đầu ra: Giá trị dự đoán y_q.\n"
        "1. Khởi tạo danh sách khoảng cách L = []\n"
        "2. Với mỗi mẫu (x_i, y_i) trong D:\n"
        "     d_i = dist(x_q, x_i)\n"
        "     Thêm (d_i, y_i) vào danh sách L\n"
        "3. Sắp xếp danh sách L theo d_i tăng dần.\n"
        "4. Chọn ra k phần tử đầu tiên, ký hiệu là tập N_k(x_q).\n"
        "5. Nếu là phân lớp: y_q = argmax_c sum(I(y_i == c)).\n"
        "   Nếu là hồi quy: y_q = (1/k) * sum(y_i) hoặc trung bình có trọng số w_i = 1 / (d_i + eps).\n"
        "6. Trả về y_q."
    )

    add_h3("1.2.2. Các độ đo khoảng cách và độ tương đồng")
    add_p(
        "Hiệu năng của KNN phụ thuộc căn bản vào việc lựa chọn hàm đo. Các độ đo cốt lõi được nghiên cứu và cài đặt gồm: "
        "(1) Khoảng cách Euclidean (chuẩn L2) đo đường chim bay ngắn nhất; (2) Khoảng cách Manhattan (chuẩn L1) đo theo "
        "các trục tọa độ trực giao, bền vững hơn với nhiễu ngoại lai; (3) Khoảng cách Minkowski dạng tổng quát chuẩn Lp; "
        "(4) Độ tương đồng Cosine đo góc nghiêng giữa hai vector, chuẩn hóa theo độ dài, lý tưởng cho dữ liệu văn bản TF-IDF; "
        "(5) Hệ số tương quan Pearson tự động khử đi xu hướng thiên lệch trung bình của từng người dùng, là xương sống của "
        "lọc cộng tác User-based; (6) Adjusted Cosine cho Item-based CF; (7) Độ đo Jaccard cho dữ liệu tập hợp nhị phân."
    )

    add_h3("1.2.3. Chọn tham số k, KNN có trọng số và Chuẩn hóa dữ liệu")
    add_p(
        "Tham số k kiểm soát sự cân bằng giữa Độ chệch (Bias) và Phương sai (Variance). Chọn k quá nhỏ (k=1) dẫn đến "
        "hiện tượng Quá khớp (Overfitting), mô hình nhạy cảm quá mức với nhiễu. Chọn k quá lớn dẫn đến hiện tượng Chưa khớp "
        "(Underfitting) do dự đoán bị cào bằng về trung bình toàn cục. Trong đồ án, k được xác định thông qua kiểm chuẩn "
        "chéo thực nghiệm quét qua dải k = 1, 3, 5, ..., 51."
    )
    add_p(
        "Thuật toán KNN có trọng số (Weighted KNN) khắc phục nhược điểm cào bằng quyền biểu quyết bằng cách gán trọng số "
        "nghịch đảo khoảng cách w_i = 1 / (d_i^2 + eps) hoặc w_i = sim(x_q, x_i). Đồng thời, chuẩn hóa dữ liệu bằng Min-Max "
        "hoặc Z-Score là bước bắt buộc trong file preprocess.py để ngăn chặn các thuộc tính có thang đo lớn áp đảo khoảng cách."
    )

    add_h3("1.2.4. Độ phức tạp tính toán và Kỹ thuật tăng tốc")
    add_p(
        "Thuật toán Brute-Force có độ phức tạp truy vấn O(n·d + n log k) và bộ nhớ ma trận tương đồng O(n^2), dễ gây quá tải "
        "khi dữ liệu lên tới hàng triệu mẫu. Để khắc phục, các cấu trúc dữ liệu không gian phân cấp được ứng dụng: KD-Tree "
        "(phân chia không gian bằng siêu phẳng trực giao), Ball-Tree (phân hoạch siêu cầu lồng nhau), Locality-Sensitive "
        "Hashing (LSH - băm xác suất đưa độ phức tạp về O(1)), và đồ thị HNSW (Hierarchical Navigable Small World) — chuẩn "
        "mực của các cơ sở dữ liệu vector hiện đại."
    )

    add_h3("1.2.5. Ưu, nhược điểm và Lời nguyền số chiều (Curse of Dimensionality)")
    add_p(
        "Lời nguyền số chiều xảy ra khi số thuộc tính d tăng cao (như ma trận TF-IDF với 5,000 từ khóa). Khi đó không gian "
        "trở nên cực kỳ thưa thớt và khoảng cách giữa điểm xa nhất và gần nhất tiến dần về bằng nhau (Distance Concentration). "
        "Để xử lý triệt để, đồ án tích hợp giải thuật TruncatedSVD nén không gian 5,000 chiều từ khóa xuống 50 chiều ngữ nghĩa "
        "tiềm ẩn LSA, vừa khử nhiễu vừa tối ưu hóa bộ nhớ và tốc độ."
    )

    add_h3("1.2.6. Ví dụ tính toán thủ công chi tiết bằng công thức Pearson (Ma trận 5x5)")
    add_p(
        "Xét tập dữ liệu thu nhỏ gồm 5 người dùng (U1 đến U5) và 5 bộ phim (M1 đến M5). Cần dự đoán điểm rating của người "
        "dùng U1 cho bộ phim M3 (ô dấu ?) với k = 2 láng giềng."
    )

    # Bảng 5x5
    tbl = doc.add_table(rows=6, cols=7)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Người dùng", "M1 (Hành động)", "M2 (Hành động)", "M3 (Hài)", "M4 (Khoa học)", "M5 (Tình cảm)", "Mean (r_u)"]
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        set_cell_background(cell, "E2E8F0")

    data_rows = [
        ["U1 (Truy vấn)", "5", "4", "?", "4", "2", "3.75"],
        ["U2", "4", "5", "1", "4", "1", "3.00"],
        ["U3", "2", "1", "5", "2", "4", "2.80"],
        ["U4", "4", "3", "2", "5", "3", "3.40"],
        ["U5", "1", "2", "4", "2", "5", "2.80"],
    ]
    for i, row in enumerate(data_rows, 1):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = val
            if j == 0 or j == 6:
                cell.paragraphs[0].runs[0].bold = True
            if val == "?":
                set_cell_background(cell, "FECACA")

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_p(
        "Bước 1: Tính điểm trung bình của từng người dùng trên các phim đã xem: "
        "r_U1 = (5+4+4+2)/4 = 3.75; r_U2 = (4+5+1+4+1)/5 = 3.00; r_U3 = (2+1+5+2+4)/5 = 2.80; "
        "r_U4 = (4+3+2+5+3)/5 = 3.40; r_U5 = (1+2+4+2+5)/5 = 2.80."
    )
    add_p(
        "Bước 2: Tính hệ số tương quan Pearson giữa U1 và các người dùng còn lại trên tập phim giao I = {M1, M2, M4, M5}: \n"
        "• sim(U1, U2): Tích vô hướng độ lệch = 5.5000, Mẫu số = sqrt(4.75) * sqrt(10) = 6.8920 => sim = +0.7980 (Tương quan thuận rất cao).\n"
        "• sim(U1, U3): Tích vô hướng = -3.7500, Mẫu số = 5.3206 => sim = -0.7048 (Tương quan nghịch chiều, loại).\n"
        "• sim(U1, U4): Tích vô hướng = 1.7500, Mẫu số = 3.9230 => sim = +0.4461 (Tương quan thuận trung bình).\n"
        "• sim(U1, U5): Tích vô hướng = -6.5000, Mẫu số = 6.6677 => sim = -0.9748 (Sở thích đối lập hoàn toàn, loại)."
    )
    add_p(
        "Bước 3: Chọn k = 2 láng giềng gần nhất có tương quan dương: Tập láng giềng được chọn là N = {U2, U4}.\n"
        "Điểm đánh giá phim M3 của hai người này: r_U2,M3 = 1.0 (độ lệch: 1.0 - 3.0 = -2.0); r_U4,M3 = 2.0 (độ lệch: 2.0 - 3.4 = -1.4)."
    )
    add_p(
        "Bước 4: Áp dụng công thức dự đoán rating có khử bias trung bình: \n"
        "r_hat(U1, M3) = 3.75 + [0.7980 * (-2.0) + 0.4461 * (-1.4)] / [0.7980 + 0.4461] \n"
        "= 3.75 + (-2.2205 / 1.2441) = 3.75 - 1.7848 = 1.9652 sao (~1.97 sao)."
    )
    add_p(
        "Kết luận phân tích: Dự đoán U1 chỉ đánh giá phim M3 1.97 sao phản ánh chính xác thực tế vì U1 thích phim hành động, "
        "trong khi các láng giềng cùng gu đều không thích bộ phim hài M3 này. Kết quả tính tay khớp hoàn toàn với mã kiểm thử."
    )

    # 1.3.2 LỌC THEO NỘI DUNG
    add_h2("1.3.2. Lọc theo nội dung (Content-Based) kết hợp KNN")
    add_p(
        "Lọc theo nội dung hoạt động trên nguyên tắc: đề xuất các vật phẩm có thuộc tính nội dung tương tự các vật phẩm "
        "người dùng từng thích trong quá khứ. Thông tin văn bản được chuyển thành vector đặc trưng trong không gian đa chiều "
        "nhờ trọng số TF-IDF kết hợp kỹ thuật n-gram. Hồ sơ người dùng (User Profile) được tổng hợp từ các vector tài liệu "
        "đã tương tác có gắn kèm trọng số hành vi."
    )
    add_p(
        "Thuật toán KNN phát huy vai trò thông qua hai kịch bản: (1) Tìm kiếm các tài liệu tương tự một tài liệu cho trước "
        "bằng hàm đo Cosine; (2) Sinh danh sách gợi ý cá nhân hóa Top-N dựa trên tổng độ tương đồng tích lũy. Ưu điểm vượt trội "
        "của phương pháp này là giải quyết triệt để bài toán Cold-Start cho tài liệu mới đưa lên hệ thống, đồng thời mang lại "
        "khả năng giải thích minh bạch cho kết quả gợi ý."
    )

    # ==================== CHƯƠNG 2 ====================
    add_h1("CHƯƠNG 2. XÂY DỰNG ỨNG DỤNG")

    add_h2("2.4. Module gợi ý tài liệu download (TF-IDF, Implicit Feedback)")
    add_p(
        "Bài toán gợi ý tài liệu download phục vụ nhu cầu tra cứu và nghiên cứu của sinh viên. Hệ thống ghi nhận phản hồi tiềm ẩn "
        "(Implicit Feedback) qua các hành vi: lượt tải file (download - trọng số 1.0), xem trực tuyến (view - trọng số 0.5), "
        "và đánh dấu yêu thích (bookmark - trọng số 2.5)."
    )
    add_p(
        "Chuỗi văn bản kết hợp từ tiêu đề, tóm tắt và từ khóa được đưa qua bộ trích xuất TfidfVectorizer với max_features=5000 "
        "và ngram_range=(1, 2). Để giải quyết bài toán lời nguyền số chiều và khai thác quan hệ ngữ nghĩa tiềm ẩn, thuật toán "
        "TruncatedSVD nén không gian về 50 chiều tiềm ẩn trước khi đưa vào chỉ mục KNNIndex (độ đo Cosine)."
    )
    add_p(
        "Module rec_document.py tích hợp cơ chế Giải thích gợi ý (Explainable AI): với mỗi tài liệu đề xuất, hệ thống xác định "
        "tài liệu lịch sử nào có tích vô hướng đóng góp lớn nhất vào điểm số, từ đó sinh ra thông điệp minh bạch: 'Vì bạn đã quan tâm "
        "đến tài liệu X (độ tương đồng nội dung: Y%)'."
    )

    add_h2("2.5. Giao diện demo (Client Web)")
    add_p(
        "Giao diện người dùng được xây dựng theo kiến trúc Single Page Application (SPA) hiện đại, đáp ứng ba yêu cầu trọng tâm: "
        "trực quan hóa đa bài toán, giải thích minh bạch gợi ý và khả năng vận hành độc lập. Hệ thống hỗ trợ tự động kết nối "
        "tới máy chủ backend FastAPI hoặc chạy trên tập dữ liệu mô phỏng khi máy chủ chưa khởi động."
    )
    add_p(
        "Cấu trúc giao diện được tổ chức thành ba tab chuyên biệt: Tab 1 — Gợi ý phim (MovieLens CF); Tab 2 — Gợi ý tài liệu "
        "(TF-IDF + Cosine); và Tab 3 — Gợi ý hàng mới ra mắt (Cold-Start Hybrid). Mỗi màn hình trang bị thanh điều khiển lựa chọn "
        "hồ sơ người dùng, thanh trượt điều chỉnh số lượng Top-N và danh sách các thẻ Card hiện đại."
    )

    add_box(
        "[HÌNH 2.5.1: GIAO DIỆN TỔNG THỂ HỆ THỐNG GỢI Ý ĐA NHIỆM KNN - TAB GỢI Ý XEM PHIM]",
        "Hình chụp màn hình toàn bộ giao diện web khi chọn Tab Gợi ý xem phim cho User 1, hiển thị danh sách Top-6 bộ phim đề xuất kèm hộp thoại lý do gợi ý 'Vì bạn đã chấm 5 sao cho phim The Matrix'."
    )
    add_box(
        "[HÌNH 2.5.2: GIAO DIỆN TAB GỢI Ý TÀI LIỆU DOWNLOAD BẰNG TF-IDF VÀ COSINE KNN]",
        "Hình chụp màn hình kết quả gợi ý tài liệu học tập cho Sinh viên 01, hiển thị độ tương đồng nội dung 89.2% và lý do đề xuất từ tài liệu đã bookmark."
    )
    add_box(
        "[HÌNH 2.5.3: GIAO DIỆN TAB GỢI Ý HÀNG MỚI GIẢI QUYẾT BÀI TOÁN COLD-START]",
        "Hình chụp màn hình sản phẩm công nghệ mới ra mắt được gợi ý cho khách hàng dựa trên thuộc tính tương đồng với các sản phẩm đã mua trước đây."
    )
    add_box(
        "[HÌNH 2.5.4: CHI TIẾT CẤU TRÚC THẺ KẾT QUẢ GỢI Ý MINH HỌA TÍNH NĂNG EXPLAINABLE AI]",
        "Hình chụp phóng to một Card kết quả thể hiện rõ tiêu đề, điểm số và khối giải thích lý do gợi ý trực quan."
    )

    doc.save(output_path)
    print(f"Đã tạo thành công file Word báo cáo tại: {output_path}")

if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    create_report_docx()

