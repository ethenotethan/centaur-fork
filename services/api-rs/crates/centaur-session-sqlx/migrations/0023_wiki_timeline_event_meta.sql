-- Event-time + label columns on the wiki source ledger, powering the
-- cross-stream Timeline view (GET /wiki/timeline). `occurred_at` is the
-- source's REAL-WORLD event time (PR merge / Linear update / Slack ts / Drive
-- modifiedTime), normalized at ingest by wiki_maintainer._mark_ingested, so the
-- timeline has a single event-time axis independent of ingest order/time. The
-- workflow also adds these idempotently at runtime (_ensure_ingested); this
-- migration keeps the canonical schema correct on fresh DBs.

alter table wiki_ingested_sources add column if not exists occurred_at timestamptz;
alter table wiki_ingested_sources add column if not exists title text not null default '';
alter table wiki_ingested_sources add column if not exists url text not null default '';

create index if not exists idx_wiki_ingested_sources_occurred
    on wiki_ingested_sources (occurred_at);
