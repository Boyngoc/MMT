/**
 * ==========================================================================
 * JavaScript Logic cho Client Web - Hệ thống gợi ý KNN
 * Người thực hiện: Linh
 *
 * Chức năng:
 *   1. Quản lý trạng thái 3 Tab: Gợi ý Phim, Gợi ý Tài liệu, Gợi ý Hàng mới.
 *   2. Thao tác điều khiển: Chọn User, Chọn Top-N, Chọn thuật toán.
 *   3. Kết nối API Backend (FastAPI tại http://localhost:8000) với cơ chế
 *      tự động chuyển sang Mock Dataset nếu server chưa chạy.
 *   4. BẮT BUỘC: Hiển thị kèm lý do gợi ý rõ ràng ("Vì bạn đã thích/tải X").
 * ==========================================================================
 */

const API_BASE_URL = "http://localhost:8000";

// Trạng thái ứng dụng
const state = {
    currentTab: "movie", // 'movie' | 'document' | 'newitem'
    currentUser: "user_1",
    topN: 6,
    isBackendOnline: false,
};

// ==========================================================================
// DỮ LIỆU DEMO MOCK (Đầy đủ và phong phú cho cả 3 bài toán)
// ==========================================================================
const MOCK_DATA = {
    // 1. DỮ LIỆU PHIM (MovieLens CF)
    movie: {
        users: [
            { id: "1", name: "User 1 (Fan phim Hành động & Sci-Fi)", history: "Đã thích: The Matrix (5★), Inception (4.5★), Terminator 2 (5★)" },
            { id: "2", name: "User 2 (Fan Hoạt hình & Gia đình)", history: "Đã thích: Toy Story (5★), Finding Nemo (4★), Shrek (4.5★)" },
            { id: "3", name: "User 3 (Fan Tâm lý - Tội phạm)", history: "Đã thích: The Godfather (5★), Pulp Fiction (5★), Fight Club (4★)" },
            { id: "42", name: "User 42 (Người dùng mới - Khảo sát)", history: "Đã xem: Interstellar (4.5★)" },
        ],
        recs: {
            "1": [
                {
                    id: "260",
                    title: "Star Wars: Episode IV - A New Hope (1977)",
                    category: "Sci-Fi | Action",
                    description: "Luke Skywalker dấn thân vào hành trình giải cứu Công chúa Leia khỏi Đế chế Ngân hà.",
                    tags: ["Sci-Fi", "Classic", "Space", "Action"],
                    scoreLabel: "Điểm dự đoán: 4.85 / 5.0 ★",
                    reason: "Vì bạn đã chấm 5★ cho phim 'The Matrix (1999)' (độ tương quan Adjusted Cosine: 0.89)"
                },
                {
                    id: "1196",
                    title: "Star Wars: Episode V - The Empire Strikes Back (1980)",
                    category: "Sci-Fi | Adventure",
                    description: "Phần phim kinh điển tiếp nối cuộc chiến chống lại Chúa tể Darth Vader.",
                    tags: ["Sci-Fi", "Masterpiece", "Adventure"],
                    scoreLabel: "Điểm dự đoán: 4.78 / 5.0 ★",
                    reason: "Vì bạn đã đánh giá cao 'Terminator 2: Judgment Day' (độ tương đồng láng giềng: 0.84)"
                },
                {
                    id: "589",
                    title: "Terminator 2: Judgment Day (1991)",
                    category: "Action | Sci-Fi",
                    description: "Kẻ hủy diệt T-800 được gửi về quá khứ để bảo vệ cậu bé John Connor.",
                    tags: ["Cyborg", "Action", "Arnold"],
                    scoreLabel: "Điểm dự đoán: 4.70 / 5.0 ★",
                    reason: "Cùng nằm trong cụm k-láng giềng yêu thích với 'The Matrix' của 15 người dùng tương tự"
                },
                {
                    id: "2571",
                    title: "The Matrix Reloaded (2003)",
                    category: "Action | Sci-Fi",
                    description: "Neo và các chiến binh Zion tiếp tục chiến đấu chống lại đội quân máy móc.",
                    tags: ["Cyberpunk", "Kung-fu", "Sci-Fi"],
                    scoreLabel: "Điểm dự đoán: 4.62 / 5.0 ★",
                    reason: "Vì bạn đã xem phần 1 'The Matrix' (độ tương quan item-item: 0.93)"
                },
                {
                    id: "1240",
                    title: "The Terminator (1984)",
                    category: "Action | Sci-Fi",
                    description: "Khởi đầu huyền thoại người máy sát thủ được điều khiển bởi Skynet.",
                    tags: ["Thriller", "Action", "Classic"],
                    scoreLabel: "Điểm dự đoán: 4.55 / 5.0 ★",
                    reason: "Được 89% người xem có gu giống bạn đánh giá trên 4 sao"
                },
                {
                    id: "1097",
                    title: "E.T. the Extra-Terrestrial (1982)",
                    category: "Children | Drama | Sci-Fi",
                    description: "Tình bạn xúc động giữa cậu bé Elliot và một sinh vật ngoài hành tinh.",
                    tags: ["Spielberg", "Aliens", "Family"],
                    scoreLabel: "Điểm dự đoán: 4.41 / 5.0 ★",
                    reason: "Gợi ý mở rộng thể loại Sci-Fi từ sở thích phim viễn tưởng của bạn"
                }
            ],
            "2": [
                {
                    id: "3114",
                    title: "Toy Story 2 (1999)",
                    category: "Animation | Children",
                    description: "Woody bị một kẻ buôn đồ chơi đánh cắp, Buzz Lightyear dẫn đầu nhóm giải cứu.",
                    tags: ["Pixar", "Animation", "Comedy"],
                    scoreLabel: "Điểm dự đoán: 4.90 / 5.0 ★",
                    reason: "Vì bạn đã chấm 5★ cho phim 'Toy Story (1995)' (độ tương quan: 0.94)"
                },
                {
                    id: "6377",
                    title: "Finding Nemo (2003)",
                    category: "Animation | Adventure",
                    description: "Hành trình vượt đại dương tìm con trai của chú cá hề Marlin.",
                    tags: ["Pixar", "Ocean", "Family"],
                    scoreLabel: "Điểm dự đoán: 4.82 / 5.0 ★",
                    reason: "Vì bạn yêu thích thể loại hoạt hình gia đình của Pixar"
                },
                {
                    id: "4306",
                    title: "Shrek (2001)",
                    category: "Animation | Comedy | Fantasy",
                    description: "Chàng chằn tinh tốt bụng Shrek cùng chú lừa Donkey đi giải cứu công chúa Fiona.",
                    tags: ["DreamWorks", "Humor", "Fairy Tale"],
                    scoreLabel: "Điểm dự đoán: 4.65 / 5.0 ★",
                    reason: "Thuộc nhóm phim hoạt hình có phân phối người xem trùng khớp 91% với bạn"
                }
            ]
        }
    },

    // 2. DỮ LIỆU TÀI LIỆU DOWNLOAD (TF-IDF + Cosine KNN, Implicit Feedback)
    document: {
        users: [
            { id: "student_01", name: "Sinh viên 01 (Nghiên cứu AI & Học máy)", history: "Đã tải: 'Nhập môn Trí tuệ nhân tạo' · Bookmark: 'Thuật toán KNN'" },
            { id: "student_02", name: "Sinh viên 02 (Chuyên ngành Mạng máy tính)", history: "Đã tải: 'Giáo trình Mạng máy tính' · Bookmark: 'Giao thức HTTP/HTTPS'" },
            { id: "student_03", name: "Sinh viên 03 (Làm đồ án Hệ gợi ý)", history: "Đã tải: 'Xây dựng hệ gợi ý Collaborative Filtering'" },
        ],
        recs: {
            "student_01": [
                {
                    id: "doc_07",
                    title: "Xử lý ngôn ngữ tự nhiên với kỹ thuật TF-IDF và Cosine",
                    category: "Trí tuệ nhân tạo",
                    description: "Hướng dẫn trích xuất vector đặc trưng văn bản, n-gram và tính ma trận tương đồng.",
                    tags: ["NLP", "TF-IDF", "Cosine", "Vector Space"],
                    scoreLabel: "Độ tương đồng nội dung: 89.2%",
                    reason: "Vì bạn đã tải 'Nhập môn Trí tuệ nhân tạo' và quan tâm đến thuật toán KNN"
                },
                {
                    id: "doc_08",
                    title: "Kỹ thuật giảm chiều dữ liệu TruncatedSVD và PCA",
                    category: "Học máy",
                    description: "Phân tích giá trị kỳ dị (SVD) trên ma trận thưa TF-IDF, tối ưu hóa bộ nhớ RAM.",
                    tags: ["SVD", "PCA", "LSA", "Dimension Reduction"],
                    scoreLabel: "Độ tương đồng nội dung: 84.7%",
                    reason: "Vì tài liệu này có chung tập từ khóa giải thuật với 'Thuật toán KNN' bạn đã bookmark"
                },
                {
                    id: "doc_06",
                    title: "Xây dựng hệ gợi ý bằng Lọc cộng tác Collaborative Filtering",
                    category: "Hệ gợi ý",
                    description: "Ứng dụng ma trận tương tác User-Item và công thức độ đo Pearson chuẩn hóa.",
                    tags: ["Recommender", "KNN", "Pearson", "Evaluation"],
                    scoreLabel: "Độ tương đồng nội dung: 81.5%",
                    reason: "Tài liệu mở rộng ứng dụng KNN trong hệ thống đề xuất thông minh"
                },
                {
                    id: "doc_10",
                    title: "Đánh giá chất lượng hệ gợi ý: RMSE, MAE, Precision và Recall",
                    category: "Học máy",
                    description: "Phương pháp thực nghiệm phân chia Train/Test và đo lường sai số dự đoán.",
                    tags: ["RMSE", "MAE", "Metrics", "Validation"],
                    scoreLabel: "Độ tương đồng nội dung: 76.0%",
                    reason: "Cùng chuyên mục Học máy với các tài liệu bạn đã tương tác"
                }
            ],
            "student_02": [
                {
                    id: "doc_03",
                    title: "Thiết kế kiến trúc mạng Client - Server với FastAPI",
                    category: "Mạng máy tính",
                    description: "Xây dựng RESTful API backend, xử lý bất đồng bộ async/await và kết nối web client.",
                    tags: ["Client-Server", "FastAPI", "REST", "Socket"],
                    scoreLabel: "Độ tương đồng nội dung: 92.4%",
                    reason: "Vì bạn đã tải 'Giáo trình Mạng máy tính' và bookmark 'Giao thức HTTP/HTTPS'"
                },
                {
                    id: "doc_02",
                    title: "Tìm hiểu giao thức HTTP/HTTPS và lập trình Web Socket",
                    category: "Mạng máy tính",
                    description: "Phân tích gói tin HTTP request/response, bắt tay TLS/SSL và bảo mật truyền thông.",
                    tags: ["HTTP", "HTTPS", "SSL", "Network Security"],
                    scoreLabel: "Độ tương đồng nội dung: 88.1%",
                    reason: "Tương đồng 88% về nội dung giao thức tầng ứng dụng bạn đang tìm hiểu"
                }
            ]
        }
    },

    // 3. DỮ LIỆU HÀNG MỚI (Cold-Start & Hybrid KNN)
    newitem: {
        users: [
            { id: "cust_01", name: "Khách hàng 01 (Thích đồ công nghệ cao cấp)", history: "Đã mua: MacBook Pro M2, Chuột Logitech MX Master 3S" },
            { id: "cust_02", name: "Khách hàng 02 (Dân văn phòng & Học sinh)", history: "Đã mua: Bàn phím cơ không dây, Đèn bàn thông minh" },
            { id: "cust_03", name: "Khách hàng 03 (Gamer)", history: "Đã mua: Màn hình 144Hz Gaming, Tai nghe HyperX" },
        ],
        recs: {
            "cust_01": [
                {
                    id: "prod_new_01",
                    title: "Bàn phím cơ Apple Magic Keyboard Touch ID 2026",
                    category: "Hàng mới ra mắt",
                    description: "Bàn phím không dây siêu mỏng tích hợp cảm biến vân tay và kết nối Bluetooth đa thiết bị.",
                    tags: ["Mới về", "Apple", "Wireless", "TouchID"],
                    scoreLabel: "Độ phù hợp Hybrid: 94.6%",
                    reason: "Sản phẩm mới ra mắt có thuộc tính kỹ thuật tương đồng 94% với 'MacBook Pro M2' bạn từng sở hữu"
                },
                {
                    id: "prod_new_02",
                    title: "Chuột công thái học ErgoMaster Pro Wireless",
                    category: "Hàng mới ra mắt",
                    description: "Thiết kế góc nghiêng 57 độ bảo vệ cổ tay, độ phân giải 8000 DPI cảm biến Darkfield.",
                    tags: ["Công thái học", "Logitech", "Bluetooth"],
                    scoreLabel: "Độ phù hợp Hybrid: 91.0%",
                    reason: "Cùng phân khúc văn phòng cao cấp với 'Chuột Logitech MX Master 3S' bạn đã mua"
                },
                {
                    id: "prod_new_03",
                    title: "Hub chuyển đổi Type-C Thunderbolt 4 Đa Năng 10-in-1",
                    category: "Phụ kiện mới",
                    description: "Xuất màn hình kép 4K 60Hz, sạc nhanh PD 100W, cổng mạng LAN Gigabit.",
                    tags: ["Thunderbolt 4", "Type-C", "Hub"],
                    scoreLabel: "Độ phù hợp Hybrid: 86.8%",
                    reason: "92% khách hàng mua MacBook Pro đều lựa chọn kèm phụ kiện mở rộng cổng kết nối này"
                }
            ],
            "cust_02": [
                {
                    id: "prod_new_04",
                    title: "Đế nâng laptop nhôm nguyên khối xoay 360 độ",
                    category: "Bàn làm việc",
                    description: "Tản nhiệt tối ưu, điều chỉnh độ cao linh hoạt cho góc nhìn chuẩn công thái học.",
                    tags: ["Setup", "Aluminum", "Ergonomic"],
                    scoreLabel: "Độ phù hợp Hybrid: 88.5%",
                    reason: "Phù hợp góc làm việc thông minh dựa trên lịch sử mua đèn bàn và bàn phím của bạn"
                }
            ]
        }
    }
};

// ==========================================================================
// KHỞI TẠO VÀ SỰ KIỆN GIAO DIỆN
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
    initElements();
    checkBackendConnection();
    bindEvents();
    renderUserSelector();
    loadRecommendations();
});

let el = {};

function initElements() {
    el = {
        tabs: document.querySelectorAll(".tab-btn"),
        userSelect: document.getElementById("userSelect"),
        userSelectLabel: document.getElementById("userSelectLabel"),
        topNSlider: document.getElementById("topNSlider"),
        topNValue: document.getElementById("topNValue"),
        metricSelect: document.getElementById("metricSelect"),
        btnFetchRecs: document.getElementById("btnFetchRecs"),
        userContextBanner: document.getElementById("userContextBanner"),
        sectionTitle: document.getElementById("sectionTitle"),
        sectionDesc: document.getElementById("sectionDesc"),
        resultsCount: document.getElementById("resultsCount"),
        recommendationsGrid: document.getElementById("recommendationsGrid"),
        loadingSpinner: document.getElementById("loadingSpinner"),
        emptyState: document.getElementById("emptyState"),
        noticeBanner: document.getElementById("noticeBanner"),
        backendStatusBadge: document.getElementById("backendStatusBadge"),
        statusDot: document.getElementById("statusDot"),
        statusText: document.getElementById("statusText"),
    };
}

// Kiểm tra kết nối tới FastAPI Backend
async function checkBackendConnection() {
    try {
        const res = await fetch(`${API_BASE_URL}/`, { method: "GET", signal: AbortSignal.timeout(1500) });
        if (res.ok) {
            state.isBackendOnline = true;
            el.backendStatusBadge.className = "status-badge online";
            el.statusText.textContent = "Backend FastAPI: Sẵn sàng (Port 8000)";
            showNotice("Đã kết nối trực tiếp đến máy chủ FastAPI. Dữ liệu tính toán từ nhân KNN thực tế!", "info");
        } else {
            throw new Error("Server not OK");
        }
    } catch (e) {
        state.isBackendOnline = false;
        el.backendStatusBadge.className = "status-badge offline";
        el.statusText.textContent = "Chế độ Demo (Offline Mockup)";
    }
}

function showNotice(msg, type = "warn") {
    el.noticeBanner.textContent = msg;
    el.noticeBanner.className = `notice-banner ${type}`;
    el.noticeBanner.classList.remove("hidden");
}

function bindEvents() {
    // Chuyển Tab
    el.tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            el.tabs.forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            state.currentTab = tab.dataset.tab;
            onTabChanged();
        });
    });

    // Thay đổi số lượng Top-N
    el.topNSlider.addEventListener("input", (e) => {
        state.topN = parseInt(e.target.value, 10);
        el.topNValue.textContent = state.topN;
    });

    // Thay đổi User
    el.userSelect.addEventListener("change", (e) => {
        state.currentUser = e.target.value;
        updateUserContextInfo();
        loadRecommendations();
    });

    // Nút Lấy Gợi Ý
    el.btnFetchRecs.addEventListener("click", () => {
        loadRecommendations();
    });
}

// Cập nhật giao diện khi chuyển Tab
function onTabChanged() {
    if (state.currentTab === "movie") {
        el.sectionTitle.textContent = "🎬 Danh sách Phim gợi ý (MovieLens CF)";
        el.sectionDesc.textContent = "Dự đoán rating bằng thuật toán KNN Lọc cộng tác (Collaborative Filtering).";
        el.userSelectLabel.textContent = "Chọn Người xem phim:";
        el.metricSelect.innerHTML = `
            <option value="item_based">Item-based KNN (Adjusted Cosine - Mặc định)</option>
            <option value="user_based">User-based KNN (Pearson Correlation)</option>
        `;
    } else if (state.currentTab === "document") {
        el.sectionTitle.textContent = "📄 Danh sách Tài liệu gợi ý tải về (TF-IDF + Cosine)";
        el.sectionDesc.textContent = "Gợi ý tài liệu học tập từ phản hồi tiềm ẩn (Implicit Feedback) và phân tích ngữ nghĩa LSA.";
        el.userSelectLabel.textContent = "Chọn Sinh viên / Bạn đọc:";
        el.metricSelect.innerHTML = `
            <option value="tfidf_cosine">TF-IDF + Cosine Similarity (max_features=5000)</option>
            <option value="svd_cosine">TruncatedSVD + Cosine (Giảm chiều LSA)</option>
        `;
    } else if (state.currentTab === "newitem") {
        el.sectionTitle.textContent = "🛍️ Gợi ý Hàng mới ra mắt (Cold-Start Hybrid)";
        el.sectionDesc.textContent = "Khắc phục hiện tượng mặt hàng mới chưa có tương tác thông qua vector đặc trưng sản phẩm.";
        el.userSelectLabel.textContent = "Chọn Khách hàng:";
        el.metricSelect.innerHTML = `
            <option value="hybrid">Content-Based KNN + Hybrid Switching</option>
            <option value="item_feature">Cosine Similarity trên Vector đặc trưng</option>
        `;
    }

    renderUserSelector();
    loadRecommendations();
}

// Đổ dữ liệu người dùng vào Select
function renderUserSelector() {
    const tabData = MOCK_DATA[state.currentTab];
    if (!tabData) return;

    el.userSelect.innerHTML = "";
    tabData.users.forEach((u) => {
        const opt = document.createElement("option");
        opt.value = u.id;
        opt.textContent = u.name;
        el.userSelect.appendChild(opt);
    });

    state.currentUser = tabData.users[0].id;
    updateUserContextInfo();
}

function updateUserContextInfo() {
    const tabData = MOCK_DATA[state.currentTab];
    const u = tabData.users.find((x) => x.id === state.currentUser) || tabData.users[0];
    el.userContextBanner.innerHTML = `
        <strong>Lịch sử tương tác của ${u.name}:</strong> ${u.history}
    `;
}

// ==========================================================================
// TẢI VÀ HIỂN THỊ DANH SÁCH GỢI Ý (BẮT BUỘC KÈM LÝ DO GIẢI THÍCH)
// ==========================================================================
async function loadRecommendations() {
    el.loadingSpinner.classList.remove("hidden");
    el.recommendationsGrid.innerHTML = "";
    el.emptyState.classList.add("hidden");

    let recs = [];

    // 1. Thử gọi API Backend nếu online
    if (state.isBackendOnline) {
        try {
            let endpoint = `${API_BASE_URL}/recommend/${state.currentTab}?user_id=${state.currentUser}&n=${state.topN}`;
            const res = await fetch(endpoint);
            if (res.ok) {
                const data = await res.json();
                recs = data.recommendations || [];
            }
        } catch (err) {
            console.warn("Không gọi được API, tự động chuyển về dữ liệu mẫu:", err);
        }
    }

    // 2. Nếu không có dữ liệu API, dùng Mock Dataset
    if (!recs || recs.length === 0) {
        await new Promise((r) => setTimeout(r, 200)); // Hiệu ứng mượt mà
        const tabData = MOCK_DATA[state.currentTab];
        const userRecs = tabData.recs[state.currentUser] || Object.values(tabData.recs)[0] || [];
        recs = userRecs.slice(0, state.topN);
    }

    el.loadingSpinner.classList.add("hidden");

    if (!recs || recs.length === 0) {
        el.emptyState.classList.remove("hidden");
        el.resultsCount.textContent = "0 kết quả";
        return;
    }

    el.resultsCount.textContent = `${recs.length} kết quả Top-${state.topN}`;

    // Render Cards
    recs.forEach((item, index) => {
        const card = createRecommendationCard(item, index + 1);
        el.recommendationsGrid.appendChild(card);
    });
}

/**
 * Tạo phần tử DOM Card cho một kết quả gợi ý.
 * BẮT BUỘC CÓ: Khối .explain-box giải thích "Vì bạn đã thích X"
 */
function createRecommendationCard(item, rank) {
    const card = document.createElement("article");
    card.className = "rec-card";

    const tagsHtml = (item.tags || [])
        .map((t) => `<span class="tag-chip">#${t}</span>`)
        .join("");

    card.innerHTML = `
        <div>
            <div class="card-top">
                <span class="card-rank">#${rank}</span>
                <span class="card-category">${item.category || "Chung"}</span>
            </div>

            <h3 class="card-title">${item.title}</h3>
            <p class="card-description">${item.description || "Nội dung đề xuất phù hợp với sở thích của bạn."}</p>

            <div class="card-tags">
                ${tagsHtml}
            </div>
        </div>

        <div>
            <div class="card-metric-row">
                <span class="score-badge">
                    <span>⚡</span> ${item.scoreLabel || `Điểm phù hợp: ${(item.score || 0.95).toFixed(2)}`}
                </span>
            </div>

            <!-- KHỐI GIẢI THÍCH LÝ DO GỢI Ý (EXPLAINABLE AI) -->
            <div class="explain-box">
                <span class="explain-icon">💡</span>
                <div class="explain-text">
                    <strong>Lý do gợi ý:</strong> ${item.reason || "Dựa trên các hành vi tương tự trong quá khứ của bạn."}
                </div>
            </div>
        </div>
    `;

    return card;
}

