-- Per-finding rows for the PR review audit workflow.
-- Each row is one distinct Centaur code-review finding, classified as
-- tp / fp / und, with an optional structural FP bucket (A-K).
CREATE TABLE IF NOT EXISTS pr_review_audit_findings (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    repo          text      NOT NULL,
    pr_number     int       NOT NULL,
    run_at        timestamptz NOT NULL,
    classification text     NOT NULL CHECK (classification IN ('tp', 'fp', 'und')),
    bucket        text      CHECK (bucket IS NULL OR bucket IN ('A','B','C','D','E','F','G','H','I','J','K')),
    file_path     text,
    claim         text,
    why           text,
    merged        boolean,
    audit_version text      NOT NULL DEFAULT '1',
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_findings_repo_pr
    ON pr_review_audit_findings (repo, pr_number);

CREATE INDEX IF NOT EXISTS idx_audit_findings_run_at
    ON pr_review_audit_findings (run_at);

CREATE INDEX IF NOT EXISTS idx_audit_findings_classification
    ON pr_review_audit_findings (classification);

CREATE INDEX IF NOT EXISTS idx_audit_findings_bucket
    ON pr_review_audit_findings (bucket) WHERE bucket IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_audit_findings_audit_version
    ON pr_review_audit_findings (audit_version);
