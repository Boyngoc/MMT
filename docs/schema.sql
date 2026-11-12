-- Database Schema for KNN Recommender System (3 Domains: Movies, New Products, Documents)
-- Designed by Hung (Week 2 deliverable) for SQLite integration in src/db.py (Week 4)

PRAGMA foreign_keys = ON;

-- 1. Users table (shared / domain-scoped identifiers)
CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT NOT NULL,
    domain      TEXT NOT NULL CHECK (domain IN ('movie', 'product', 'document', 'global')),
    username    TEXT NOT NULL,
    segment     TEXT DEFAULT 'standard',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (domain, external_id)
);

-- 2. Movies domain (MovieLens ml-latest-small)
CREATE TABLE IF NOT EXISTS movies (
    movie_id     INTEGER PRIMARY KEY,
    title        TEXT NOT NULL,
    genres       TEXT NOT NULL,
    release_year INTEGER
);

CREATE TABLE IF NOT EXISTS movie_ratings (
    rating_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    movie_id   INTEGER NOT NULL,
    rating     REAL NOT NULL CHECK (rating >= 0.5 AND rating <= 5.0),
    rated_at   INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    UNIQUE (user_id, movie_id)
);

-- 3. New Products domain (Content-based + Hybrid cold-start)
CREATE TABLE IF NOT EXISTS products (
    product_id   INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    category     TEXT NOT NULL,
    brand        TEXT NOT NULL,
    price_vnd    REAL NOT NULL CHECK (price_vnd >= 0),
    price_minmax REAL,
    price_zscore REAL,
    release_year INTEGER,
    tags         TEXT NOT NULL,
    description  TEXT,
    is_new_item  INTEGER NOT NULL DEFAULT 0 CHECK (is_new_item IN (0, 1)),
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS product_interactions (
    interaction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL,
    product_id       INTEGER NOT NULL,
    interaction_type TEXT NOT NULL CHECK (interaction_type IN ('view', 'cart', 'purchase')),
    rating           REAL NOT NULL CHECK (rating >= 1.0 AND rating <= 5.0),
    interacted_at    TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    UNIQUE (user_id, product_id)
);

-- 4. Downloadable Documents domain (TF-IDF + Cosine KNN, implicit feedback)
CREATE TABLE IF NOT EXISTS documents (
    doc_id       INTEGER PRIMARY KEY,
    title        TEXT NOT NULL,
    category     TEXT NOT NULL,
    tags         TEXT NOT NULL,
    abstract     TEXT,
    file_type    TEXT NOT NULL CHECK (file_type IN ('PDF', 'DOCX', 'PPTX', 'IPYNB')),
    file_size_mb REAL NOT NULL CHECK (file_size_mb > 0),
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL,
    doc_id         INTEGER NOT NULL,
    action_type    TEXT NOT NULL CHECK (action_type IN ('view', 'bookmark', 'download')),
    weight         REAL NOT NULL CHECK (weight > 0),
    interacted_at  TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

-- 5. API & Recommendation Request Logs (for Week 4 FastAPI server & Week 5 latency benchmarks)
CREATE TABLE IF NOT EXISTS recommendation_logs (
    log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    domain       TEXT NOT NULL CHECK (domain IN ('movie', 'product', 'document')),
    query_mode   TEXT NOT NULL,
    target_id    INTEGER NOT NULL,
    algorithm    TEXT NOT NULL,
    top_k        INTEGER NOT NULL,
    result_ids   TEXT NOT NULL, -- JSON array of recommended IDs
    latency_ms   REAL NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes for fast retrieval
CREATE INDEX IF NOT EXISTS idx_movie_ratings_user ON movie_ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_movie_ratings_movie ON movie_ratings(movie_id);
CREATE INDEX IF NOT EXISTS idx_products_new ON products(is_new_item, category);
CREATE INDEX IF NOT EXISTS idx_prod_inter_user ON product_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_prod_inter_prod ON product_interactions(product_id);
CREATE INDEX IF NOT EXISTS idx_doc_inter_user ON document_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_doc_inter_doc ON document_interactions(doc_id);
CREATE INDEX IF NOT EXISTS idx_rec_logs_domain ON recommendation_logs(domain, created_at);

