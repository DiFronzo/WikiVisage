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
    leaderboard_opt_out TINYINT(1)   NOT NULL DEFAULT 0 COMMENT '1=user opted out of leaderboard',
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_users_username (wiki_username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Projects table: each project targets one Wikidata entity in one Commons category.
CREATE TABLE IF NOT EXISTS projects (
    id                  BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id             BIGINT UNSIGNED NOT NULL,
    wikidata_qid        VARCHAR(20)     NOT NULL COMMENT 'e.g. Q42',
    commons_category    VARCHAR(255)    NOT NULL COMMENT 'Category name without "Category:" prefix',
    label               VARCHAR(255)    NOT NULL DEFAULT '' COMMENT 'Human-readable label for the target entity',
    distance_threshold  FLOAT           NOT NULL DEFAULT 0.5 COMMENT 'Cosine distance threshold for autonomous inference (provisional default)',
    min_confirmed       INT UNSIGNED    NOT NULL DEFAULT 5 COMMENT 'Minimum confirmed faces before autonomous mode',
    status              ENUM('active', 'paused', 'completed', 'deleted') NOT NULL DEFAULT 'active',
    completion_reason   ENUM('no_faces', 'insufficient_faces') NULL COMMENT 'Why the project was auto-completed by the worker',
    p18_thumb_url       VARCHAR(1024)   NULL COMMENT 'Cached Wikidata P18 image thumbnail URL',
    images_total        INT UNSIGNED    NOT NULL DEFAULT 0,
    images_processed    INT UNSIGNED    NOT NULL DEFAULT 0,
    faces_confirmed     INT UNSIGNED    NOT NULL DEFAULT 0,
    last_inference_threshold FLOAT       NULL COMMENT 'distance_threshold used in last inference run',
    last_inference_min_confirmed INT UNSIGNED NULL COMMENT 'min_confirmed used in last inference run',
    sdc_write_requested TINYINT(1)      NOT NULL DEFAULT 0 COMMENT '1=user requested SDC writes, worker picks up',
    sdc_write_user_id   BIGINT UNSIGNED NULL COMMENT 'User who triggered SDC writes (their token is used)',
    sdc_write_error     VARCHAR(1024)   NULL COMMENT 'Error message from last SDC write attempt',
    invite_code         VARCHAR(8)       NULL DEFAULT NULL COMMENT 'Unique code for others to join this project',
    invite_code_created_at DATETIME     NULL DEFAULT NULL COMMENT 'When invite_code was generated (for TTL expiry)',
    prefilter_mode      ENUM('off', 'shadow', 'enforce') NOT NULL DEFAULT 'off' COMMENT 'LiftWing metadata pre-filter (experimental): off, shadow=record verdicts only',
    worker_claimed_by   VARCHAR(255)    NULL COMMENT 'Worker instance ID that claimed this project',
    worker_claimed_at   DATETIME        NULL COMMENT 'When the worker claimed this project',
    created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_projects_user_id (user_id),
    INDEX idx_projects_status (status),
    INDEX idx_projects_sdc_write_requested (sdc_write_requested),
    INDEX idx_projects_worker_claim (worker_claimed_by, worker_claimed_at),
    INDEX idx_projects_status_claim (status, worker_claimed_by, worker_claimed_at),
    INDEX idx_projects_sdc_claim (sdc_write_requested, worker_claimed_by, worker_claimed_at),
    UNIQUE INDEX idx_projects_user_qid_cat (user_id, wikidata_qid, commons_category),
    UNIQUE INDEX idx_projects_invite_code (invite_code),

    CONSTRAINT fk_projects_user
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Images table: tracks every file discovered in a project's Commons category.
CREATE TABLE IF NOT EXISTS images (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    project_id      BIGINT UNSIGNED NOT NULL,
    commons_page_id BIGINT UNSIGNED NOT NULL COMMENT 'MediaWiki page ID on Commons',
    file_title      VARCHAR(512)    NOT NULL COMMENT 'Full file title including "File:" prefix',
    status          ENUM('pending', 'processed', 'enriched', 'error', 'skipped') NOT NULL DEFAULT 'pending',
    face_count      SMALLINT UNSIGNED NOT NULL DEFAULT 0 COMMENT 'Number of faces detected',
    detection_width MEDIUMINT UNSIGNED NULL COMMENT 'Image width in pixels at which face detection was run',
    detection_height MEDIUMINT UNSIGNED NULL COMMENT 'Image height in pixels at which face detection was run',
    bootstrapped    TINYINT(1)      NOT NULL DEFAULT 0 COMMENT '1=image found via P180 bootstrap',
    prefilter_keep  TINYINT(1)      NULL COMMENT 'LiftWing metadata pre-filter verdict (experimental): 1=keep, 0=drop, NULL=not checked',
    prefilter_reason VARCHAR(160)   NULL COMMENT 'LiftWing reason for the pre-filter verdict',
    prefilter_checked_at DATETIME   NULL COMMENT 'When the LiftWing pre-filter checked this image',
    error_message   VARCHAR(1024)   NULL COMMENT 'Error details if status is error',
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_images_project_id (project_id),
    INDEX idx_images_status (status),
    INDEX idx_images_project_status (project_id, status),
    INDEX idx_images_project_bootstrapped (project_id, bootstrapped),
    UNIQUE INDEX idx_images_project_page (project_id, commons_page_id),

    CONSTRAINT fk_images_project
        FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Faces table: stores each detected face encoding and its classification state.
-- encoding: 512-dimensional L2-normalised float32 embedding stored as raw
--   little-endian bytes (2048 bytes). Only comparable with embeddings of the
--   same embedding_model.
-- embedding_model: which model produced the encoding (e.g. 'scrfd500m+arcface_mbf').
-- det_score: detector confidence (NULL for faces drawn by hand).
-- aligned: 0 = embedding computed without landmark alignment (a hand-drawn box
--   the detector did not confirm) — such embeddings are noticeably weaker.
-- is_target: NULL = unclassified, TRUE = confirmed match, FALSE = confirmed non-match.
-- superseded_by: FK to replacement face row (after bbox edit). NULL = active face.
CREATE TABLE IF NOT EXISTS faces (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    image_id        BIGINT UNSIGNED NOT NULL,
    encoding        BLOB            NOT NULL COMMENT 'float32 embedding, 2048 bytes for 512D',
    embedding_model VARCHAR(32)     NOT NULL COMMENT 'Model that produced the encoding, e.g. scrfd500m+arcface_mbf',
    bbox_top        SMALLINT UNSIGNED NOT NULL,
    bbox_right      SMALLINT UNSIGNED NOT NULL,
    bbox_bottom     SMALLINT UNSIGNED NOT NULL,
    bbox_left       SMALLINT UNSIGNED NOT NULL,
    is_target       TINYINT(1)      NULL DEFAULT NULL COMMENT 'NULL=unclassified, 1=match, 0=non-match',
    confidence      FLOAT           NULL COMMENT 'Cosine distance (1 - cosine similarity) from the target centroid',
    det_score       FLOAT           NULL COMMENT 'Detector confidence (NULL for manually drawn faces)',
    aligned         TINYINT(1)      NOT NULL DEFAULT 1 COMMENT '0=embedded without landmark alignment (drawn box, detector missed it)',
    classified_by   ENUM('human', 'model', 'bootstrap') NULL COMMENT 'How this face was classified',
    classified_by_user_id BIGINT UNSIGNED NULL COMMENT 'User who classified this face (human classifications)',
    classified_at   DATETIME        NULL DEFAULT NULL COMMENT 'When face was classified (set once per classification, not auto-updated)',
    sdc_written     TINYINT(1)      NOT NULL DEFAULT 0 COMMENT 'Whether P180 claim was written to SDC',
    sdc_removal_pending TINYINT(1) NOT NULL DEFAULT 0 COMMENT '1=P180 claim removal queued (rejected bootstrap face)',
    superseded_by   BIGINT UNSIGNED NULL COMMENT 'FK to replacement face after bbox edit. NULL=active face',
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_faces_image_id (image_id),
    INDEX idx_faces_is_target (is_target),
    INDEX idx_faces_classification (image_id, is_target, classified_by),
    INDEX idx_faces_sdc (is_target, sdc_written),
    INDEX idx_faces_sdc_removal (sdc_removal_pending),
    INDEX idx_faces_classified_by_user (classified_by_user_id),
    INDEX idx_faces_user_classified_at (classified_by_user_id, classified_at),
    INDEX idx_faces_superseded (superseded_by),
    INDEX idx_faces_image_target_superseded (image_id, is_target, superseded_by),

    CONSTRAINT fk_faces_image
        FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE,
    CONSTRAINT fk_faces_classified_by_user
        FOREIGN KEY (classified_by_user_id) REFERENCES users (id) ON DELETE SET NULL,
    CONSTRAINT fk_faces_superseded_by
        FOREIGN KEY (superseded_by) REFERENCES faces (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- User stats table: archived leaderboard stats that persist after project deletion.
-- When a project is hard-deleted, the worker aggregates face classification counts
-- per user into this table before CASCADE deletes the faces. The leaderboard query
-- combines live face data with these archived totals.
CREATE TABLE IF NOT EXISTS user_stats (
    user_id         BIGINT UNSIGNED PRIMARY KEY,
    classifications BIGINT UNSIGNED NOT NULL DEFAULT 0 COMMENT 'Archived classification count from deleted projects',
    sdc_tags        BIGINT UNSIGNED NOT NULL DEFAULT 0 COMMENT 'Archived SDC written count from deleted projects',
    updated_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT fk_user_stats_user
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- SDC claims table: cross-project deduplication of P180 depicts claims.
-- Ensures that only one project writes a given P180 claim per Commons page,
-- even when multiple projects target the same Wikidata entity.
CREATE TABLE IF NOT EXISTS sdc_claims (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    commons_page_id BIGINT UNSIGNED NOT NULL COMMENT 'MediaWiki page ID on Commons',
    wikidata_qid    VARCHAR(20)     NOT NULL COMMENT 'e.g. Q42',
    project_id      BIGINT UNSIGNED NOT NULL COMMENT 'Project that claimed this write',
    face_id         BIGINT UNSIGNED NOT NULL COMMENT 'Face that triggered this write',
    claimed_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    written_at      DATETIME        NULL COMMENT 'When the API write succeeded',

    UNIQUE INDEX idx_sdc_claims_page_qid (commons_page_id, wikidata_qid),
    INDEX idx_sdc_claims_project (project_id),
    INDEX idx_sdc_claims_face (face_id, written_at),

    CONSTRAINT fk_sdc_claims_project
        FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE,
    CONSTRAINT fk_sdc_claims_face
        FOREIGN KEY (face_id) REFERENCES faces (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Project members table: allows multiple users to collaborate on a project.
-- The project owner is always in users.id via projects.user_id. This table
-- tracks additional members who joined the project.
CREATE TABLE IF NOT EXISTS project_members (
    project_id  BIGINT UNSIGNED NOT NULL,
    user_id     BIGINT UNSIGNED NOT NULL,
    role        ENUM('owner', 'member') NOT NULL DEFAULT 'member',
    status      ENUM('active', 'banned') NOT NULL DEFAULT 'active',
    joined_at   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (project_id, user_id),

    CONSTRAINT fk_pm_project
        FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE,
    CONSTRAINT fk_pm_user
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Worker heartbeat table: single-row table tracking when the background worker last ran.
-- Used by the web app to detect worker downtime and display a banner.
CREATE TABLE IF NOT EXISTS worker_heartbeat (
    id          INT             NOT NULL DEFAULT 1 PRIMARY KEY,
    last_seen   DATETIME        NOT NULL,

    CONSTRAINT single_row CHECK (id = 1)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Service status table: last-known reachability of external services that are
-- optional to the pipeline. The worker's probe writes it, and the web app only
-- reads it (never a live call from a request handler). The one row is seeded
-- below and is NOT touched by the embedding-model wipe.
CREATE TABLE IF NOT EXISTS service_status (
    name                VARCHAR(32)     NOT NULL PRIMARY KEY,
    status              ENUM('up', 'down', 'unknown') NOT NULL DEFAULT 'unknown',
    consecutive_failures INT UNSIGNED   NOT NULL DEFAULT 0,
    last_checked        DATETIME        NULL COMMENT 'When the worker last probed the service',
    last_ok             DATETIME        NULL COMMENT 'When the probe last succeeded',
    last_error          VARCHAR(255)    NULL COMMENT 'Why the last probe failed',
    updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT IGNORE INTO service_status (name) VALUES ('liftwing');
