create table if not exists wiki_page_revisions (
    id bigserial primary key,
    document_id text not null,
    title text not null,
    body text not null,
    content_hash text not null,
    revised_at timestamptz not null default now()
);

create index if not exists idx_wiki_page_revisions_doc_time
    on wiki_page_revisions (document_id, revised_at);

create table if not exists wiki_ingested_sources (
    source_key text primary key,
    kind text not null,
    ingested_at timestamptz not null default now()
);
