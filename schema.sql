-- WikiVisage Database Schema
-- MariaDB (ToolsDB on Wikimedia Toolforge)
-- Database name format: {credential_user}__wikiface

-- Users table: stores authenticated Wikimedia users and their OAuth tokens.
CREATE TABLE IF NOT EXISTS users (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    wiki_user_id    BIGINT UNSIGNED NOT NULL UNIQUE,
    wiki_username   VARCHAR(255)    NOT NULL,
    access_token    VARBINARY(2048) NOT NULL,
    refresh_token   VARBINARY(2048) NOT NULL,
    token_expires_at DATETIME       NOT NULL,
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_users_username (wiki_username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Sessions table: server-side session storage for Flask-Session.
CREATE TABLE IF NOT EXISTS sessions (
    id              VARCHAR(255)    PRIMARY KEY,
    user_id         BIGINT UNSIGNED NULL,
    data            MEDIUMBLOB      NOT NULL,
    expires_at      DATETIME        NOT NULL,
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_sessions_user_id (user_id),
    INDEX idx_sessions_expires_at (expires_at),

    CONSTRAINT fk_sessions_user
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Projects table: each project targets one Wikidata entity in one Commons category.
CREATE TABLE IF NOT EXISTS projects (
    id                  BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id             BIGINT UNSIGNED NOT NULL,
    wikidata_qid        VARCHAR(20)     NOT NULL COMMENT 'e.g. Q42',
    commons_category    VARCHAR(255)    NOT NULL COMMENT 'Category name without "Category:" prefix',
    label               VARCHAR(255)    NOT NULL DEFAULT '' COMMENT 'Human-readable label for the target entity',
    distance_threshold  FLOAT           NOT NULL DEFAULT 0.6 COMMENT 'Face distance threshold for autonomous inference',
    min_confirmed       INT UNSIGNED    NOT NULL DEFAULT 5 COMMENT 'Minimum confirmed faces before autonomous mode',
    status              ENUM('active', 'paused', 'completed') NOT NULL DEFAULT 'active',
    images_total        INT UNSIGNED    NOT NULL DEFAULT 0,
    images_processed    INT UNSIGNED    NOT NULL DEFAULT 0,
    faces_confirmed     INT UNSIGNED    NOT NULL DEFAULT 0,
    created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_projects_user_id (user_id),
    INDEX idx_projects_status (status),
    UNIQUE INDEX idx_projects_user_qid_cat (user_id, wikidata_qid, commons_category),

    CONSTRAINT fk_projects_user
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Images table: tracks every file discovered in a project's Commons category.
CREATE TABLE IF NOT EXISTS images (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    project_id      BIGINT UNSIGNED NOT NULL,
    commons_page_id BIGINT UNSIGNED NOT NULL COMMENT 'MediaWiki page ID on Commons',
    file_title      VARCHAR(512)    NOT NULL COMMENT 'Full file title including "File:" prefix',
    status          ENUM('pending', 'processed', 'enriched', 'error') NOT NULL DEFAULT 'pending',
    face_count      SMALLINT UNSIGNED NOT NULL DEFAULT 0 COMMENT 'Number of faces detected',
    error_message   VARCHAR(1024)   NULL COMMENT 'Error details if status is error',
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_images_project_id (project_id),
    INDEX idx_images_status (status),
    INDEX idx_images_project_status (project_id, status),
    UNIQUE INDEX idx_images_project_page (project_id, commons_page_id),

    CONSTRAINT fk_images_project
        FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Faces table: stores each detected face encoding and its classification state.
-- encoding: 128-dimensional float64 numpy array stored as raw bytes (1024 bytes).
-- is_target: NULL = unclassified, TRUE = confirmed match, FALSE = confirmed non-match.
CREATE TABLE IF NOT EXISTS faces (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    image_id        BIGINT UNSIGNED NOT NULL,
    encoding        BLOB            NOT NULL COMMENT '128D float64 numpy array as raw bytes (1024 bytes)',
    bbox_top        SMALLINT UNSIGNED NOT NULL,
    bbox_right      SMALLINT UNSIGNED NOT NULL,
    bbox_bottom     SMALLINT UNSIGNED NOT NULL,
    bbox_left       SMALLINT UNSIGNED NOT NULL,
    is_target       TINYINT(1)      NULL DEFAULT NULL COMMENT 'NULL=unclassified, 1=match, 0=non-match',
    confidence      FLOAT           NULL COMMENT 'Face distance from known target centroid',
    classified_by   ENUM('human', 'model', 'bootstrap') NULL COMMENT 'How this face was classified',
    sdc_written     TINYINT(1)      NOT NULL DEFAULT 0 COMMENT 'Whether P180 claim was written to SDC',
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_faces_image_id (image_id),
    INDEX idx_faces_is_target (is_target),
    INDEX idx_faces_classification (image_id, is_target, classified_by),
    INDEX idx_faces_sdc (is_target, sdc_written),

    CONSTRAINT fk_faces_image
        FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
