//! Read-only public wiki endpoints — serve the LLM Wiki graph + pages to the
//! docs-site SPA (live, from Postgres), so the frontend never relies on a
//! committed snapshot. Public by design (a browser can't safely hold an API
//! key); read-only, no mutations. Data: `company_context_documents` WHERE
//! `source = 'wiki'`.
//!
//! Ported from the original FastAPI router (`api/routers/wiki.py`). JSON field
//! names are kept byte-for-byte identical because the docs-site SPA depends on
//! them.

use std::collections::{BTreeMap, BTreeSet};

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::{HeaderValue, Method},
    routing::get,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use sqlx::{PgPool, Row};
use time::{Duration, OffsetDateTime, format_description::well_known::Rfc3339};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

use crate::{ApiError, routes::AppState};

const PAGE_TYPES: [&str; 4] = ["wiki_goal", "wiki_project", "wiki_entity", "wiki_topic"];

/// Build the `/wiki/*` sub-router with a permissive (or `CORS_ORIGINS`-scoped)
/// CORS layer. These routes are public, read-only GETs.
pub(crate) fn router() -> Router<AppState> {
    Router::new()
        .route("/wiki/graph", get(wiki_graph))
        .route("/wiki/search", get(wiki_search))
        .route("/wiki/changes", get(wiki_changes))
        .route("/wiki/timeline", get(wiki_timeline))
        .route("/wiki/diff", get(wiki_diff))
        // A wiki `document_id` is a path that may contain slashes/colons
        // (e.g. `wiki:topic:foo`), so capture the remainder as a wildcard.
        // The `/revisions` sub-resource can't be a static suffix after an
        // axum wildcard, so the same handler peels a trailing `/revisions`.
        .route("/wiki/page/{*document_id}", get(wiki_page_or_revisions))
        .layer(cors_layer())
}

/// CORS: if `CORS_ORIGINS` is set (a JSON array of origins, matching the Python
/// service), allow exactly those; otherwise allow any origin (safe here — every
/// route is a public read-only GET).
fn cors_layer() -> CorsLayer {
    let base = CorsLayer::new()
        .allow_methods([Method::GET])
        .allow_headers(Any);
    match parse_cors_origins() {
        Some(origins) if !origins.is_empty() => base.allow_origin(AllowOrigin::list(origins)),
        _ => base.allow_origin(Any),
    }
}

fn parse_cors_origins() -> Option<Vec<HeaderValue>> {
    let raw = std::env::var("CORS_ORIGINS").ok()?;
    let parsed: Vec<String> = serde_json::from_str(&raw).ok()?;
    Some(
        parsed
            .into_iter()
            .filter_map(|origin| HeaderValue::from_str(&origin).ok())
            .collect(),
    )
}

fn pool(state: &AppState) -> Result<PgPool, ApiError> {
    state.pool()
}

/// Format an optional timestamp as ISO-8601 (RFC 3339), or `""` if absent —
/// matching the Python `dt.isoformat()` / `""` behaviour.
fn iso(ts: Option<OffsetDateTime>) -> String {
    ts.and_then(|ts| ts.format(&Rfc3339).ok())
        .unwrap_or_default()
}

fn strip_wiki_prefix(source_type: &str) -> String {
    source_type.replacen("wiki_", "", 1)
}

// ---------------------------------------------------------------------------
// Time window resolution (mirrors `_window`)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ChangesQuery {
    days: Option<f64>,
    since: Option<String>,
    until: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SearchQuery {
    q: String,
    limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct DiffQuery {
    id: String,
    days: Option<f64>,
    since: Option<String>,
    until: Option<String>,
}

/// Resolve a time window. `days=N` (relative) takes precedence; else
/// `since`/`until` ISO timestamps; default = last 1 day.
fn window(
    days: Option<f64>,
    since: Option<&str>,
    until: Option<&str>,
) -> (OffsetDateTime, OffsetDateTime) {
    let now = OffsetDateTime::now_utc();
    if let Some(days) = days {
        let days = days.max(0.0);
        let span = Duration::seconds_f64(days * 86_400.0);
        return (now - span, now);
    }
    let parse = |s: Option<&str>, default: OffsetDateTime| -> OffsetDateTime {
        let Some(s) = s.filter(|s| !s.is_empty()) else {
            return default;
        };
        // Accept a trailing `Z` (Python replaces it with `+00:00`).
        let normalized = s.replace('Z', "+00:00");
        OffsetDateTime::parse(&normalized, &Rfc3339).unwrap_or(default)
    };
    (
        parse(since, now - Duration::days(1)),
        parse(until, now),
    )
}

/// Derive a title/url from a `wiki_ingested_sources` key like
/// `pr:Layr-Labs/d-inference#338`, `linear:DAR-160`, `meeting:<id>`.
fn source_display(source_key: &str) -> Map<String, Value> {
    let mut m = Map::new();
    if let Some(rest) = source_key.strip_prefix("pr:") {
        let (repo, num) = match rest.split_once('#') {
            Some((repo, num)) => (repo, num),
            None => (rest, ""),
        };
        m.insert("kind".into(), json!("pr"));
        m.insert("label".into(), json!(format!("{repo}#{num}")));
        m.insert(
            "url".into(),
            json!(if num.is_empty() {
                String::new()
            } else {
                format!("https://github.com/{repo}/pull/{num}")
            }),
        );
        return m;
    }
    if let Some(rest) = source_key.strip_prefix("linear_project:") {
        m.insert("kind".into(), json!("linear_project"));
        m.insert("label".into(), json!(rest));
        m.insert("url".into(), json!(""));
        return m;
    }
    if let Some(rest) = source_key.strip_prefix("linear:") {
        m.insert("kind".into(), json!("linear"));
        m.insert("label".into(), json!(rest));
        m.insert("url".into(), json!(""));
        return m;
    }
    if source_key.starts_with("meeting:") {
        m.insert("kind".into(), json!("meeting"));
        m.insert("label".into(), json!("meeting notes"));
        m.insert("url".into(), json!(""));
        return m;
    }
    if let Some(rest) = source_key.strip_prefix("slack:") {
        // Slack source_key is `slack:<document_id>` where document_id encodes
        // the thread/digest. Surface a readable label and keep `kind` accurate
        // so callers (and the agent narrative) don't see "other".
        m.insert("kind".into(), json!("slack"));
        m.insert("label".into(), json!(rest));
        m.insert("url".into(), json!(""));
        return m;
    }
    if let Some(rest) = source_key.strip_prefix("doc:") {
        m.insert("kind".into(), json!("doc"));
        m.insert("label".into(), json!(rest));
        m.insert("url".into(), json!(""));
        return m;
    }
    if let Some(rest) = source_key.strip_prefix("release:") {
        // release:<repo>@<tag>  → label is "<repo> <tag>" + GitHub release URL.
        let (repo, tag) = match rest.split_once('@') {
            Some((repo, tag)) => (repo, tag),
            None => (rest, ""),
        };
        m.insert("kind".into(), json!("release"));
        m.insert("label".into(), json!(format!("{repo} {tag}")));
        m.insert(
            "url".into(),
            json!(if tag.is_empty() {
                String::new()
            } else {
                format!("https://github.com/{repo}/releases/tag/{tag}")
            }),
        );
        return m;
    }
    if let Some(rest) = source_key.strip_prefix("tweet:") {
        // tweet:<tweet_id> — wiki_ingested_sources.title carries "@handle PREFIX: text",
        // but for the timeline endpoint we just give a numeric label fallback +
        // a clickable URL. The timeline route reads .title from the DB row for
        // the human label, so this is only the fallback if the title is empty.
        m.insert("kind".into(), json!("tweet"));
        m.insert("label".into(), json!(format!("tweet {rest}")));
        m.insert(
            "url".into(),
            json!(if rest.is_empty() {
                String::new()
            } else {
                // We don't know the handle from the source_key alone, so we
                // use the i/web URL form which X resolves regardless of handle.
                format!("https://x.com/i/status/{rest}")
            }),
        );
        return m;
    }
    if let Some(rest) = source_key.strip_prefix("directive:") {
        // directive:<slack_user_id>:<message_ts> — label is just the actor for
        // a minimal fallback; the timeline endpoint LEFT JOINs wiki_directives
        // and overrides label/actor/quote with the richer fields below.
        let actor = rest.split(':').next().unwrap_or("").to_owned();
        m.insert("kind".into(), json!("directive"));
        m.insert("label".into(), json!(format!("directive from {actor}")));
        m.insert("url".into(), json!(""));
        return m;
    }
    m.insert("kind".into(), json!("other"));
    m.insert("label".into(), json!(source_key));
    m.insert("url".into(), json!(""));
    m
}

// ---------------------------------------------------------------------------
// GET /wiki/graph
// ---------------------------------------------------------------------------

struct PageRow {
    document_id: String,
    source_type: String,
    title: String,
    body: String,
    url: String,
    updated_at: Option<OffsetDateTime>,
}

async fn wiki_graph(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
let pool = pool(&state)?;
    let rows = sqlx::query(
        "SELECT document_id, source_type, title, body, url, updated_at \
         FROM company_context_documents \
         WHERE source = 'wiki' AND source_type = ANY($1::text[]) \
         ORDER BY title",
    )
    .bind(&PAGE_TYPES[..])
    .fetch_all(&pool)
    .await?;

    let pages: Vec<PageRow> = rows
        .iter()
        .map(|r| PageRow {
            document_id: r.try_get("document_id").unwrap_or_default(),
            source_type: r.try_get("source_type").unwrap_or_default(),
            title: r.try_get("title").unwrap_or_default(),
            body: r.try_get("body").unwrap_or_default(),
            url: r.try_get("url").unwrap_or_default(),
            updated_at: r.try_get("updated_at").ok(),
        })
        .collect();

    // node id by lower-cased title (mirrors `title_to_id`).
    let mut title_to_id: BTreeMap<String, String> = BTreeMap::new();
    let mut nodes: Vec<Map<String, Value>> = Vec::with_capacity(pages.len());
    for p in &pages {
        title_to_id.insert(p.title.trim().to_lowercase(), p.document_id.clone());
        let mut node = Map::new();
        node.insert("id".into(), json!(p.document_id));
        node.insert("title".into(), json!(p.title));
        node.insert("type".into(), json!(strip_wiki_prefix(&p.source_type)));
        node.insert("url".into(), json!(p.url));
        node.insert("updated_at".into(), json!(iso(p.updated_at)));
        nodes.push(node);
    }

    let mut edges: Vec<Value> = Vec::new();
    let mut deg: BTreeMap<String, i64> = pages.iter().map(|p| (p.document_id.clone(), 0)).collect();
    let mut backl: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();

    for p in &pages {
        for m in wikilinks(&p.body) {
            let key = m.split('|').next().unwrap_or("").trim().to_lowercase();
            let Some(tgt) = title_to_id.get(&key) else {
                continue;
            };
            if tgt == &p.document_id || seen.contains(&(p.document_id.clone(), tgt.clone())) {
                continue;
            }
            seen.insert((p.document_id.clone(), tgt.clone()));
            edges.push(json!({ "source": p.document_id, "target": tgt }));
            *deg.entry(p.document_id.clone()).or_insert(0) += 1;
            *deg.entry(tgt.clone()).or_insert(0) += 1;
            backl
                .entry(tgt.clone())
                .or_default()
                .insert(p.document_id.clone());
        }
    }

    for (node, p) in nodes.iter_mut().zip(pages.iter()) {
        node.insert("degree".into(), json!(deg.get(&p.document_id).copied().unwrap_or(0)));
        let backlinks: Vec<String> = backl
            .get(&p.document_id)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();
        node.insert("backlinks".into(), json!(backlinks));
    }

    Ok(Json(json!({
        "node_count": nodes.len(),
        "edge_count": edges.len(),
        "nodes": nodes,
        "edges": edges,
    })))
}

/// Extract `[[wikilink]]` targets from a body (mirrors `_WIKILINK_RE`).
fn wikilinks(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'[' && bytes[i + 1] == b'[' {
            // Find the closing `]` (the regex captures `[^\]]+`).
            if let Some(end_rel) = body[i + 2..].find(']') {
                let inner = &body[i + 2..i + 2 + end_rel];
                if !inner.is_empty() {
                    out.push(inner.to_string());
                }
                i = i + 2 + end_rel + 1;
                continue;
            } else {
                break;
            }
        }
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// GET /wiki/search?q=…&limit=N
// ---------------------------------------------------------------------------

async fn wiki_search(
    State(state): State<AppState>,
    Query(params): Query<SearchQuery>,
) -> Result<Json<Value>, ApiError> {
    let pool = pool(&state)?;
    let limit = params.limit.unwrap_or(20).min(50);
    let q = params.q.trim().to_string();
    if q.is_empty() {
        return Ok(Json(json!({"results": []})));
    }
    // ParadeDB BM25 full-text search over wiki page titles + bodies.
    let rows = sqlx::query(
        "SELECT document_id, source_type, title, body, url, updated_at \
         FROM company_context_documents \
         WHERE source = 'wiki' AND source_type = ANY($1::text[]) \
         AND (title ||| $2::text OR body ||| $2::text) \
         ORDER BY paradedb.score(document_id) DESC, updated_at DESC \
         LIMIT $3",
    )
    .bind(&PAGE_TYPES[..])
    .bind(&q)
    .bind(limit)
    .fetch_all(&pool)
    .await?;

    let results: Vec<Value> = rows
        .iter()
        .map(|r| {
            let body: String = r.try_get("body").unwrap_or_default();
            let snippet = truncate_body(&body, &q);
            json!({
                "id": r.try_get::<String, _>("document_id").unwrap_or_default(),
                "title": r.try_get::<String, _>("title").unwrap_or_default(),
                "type": strip_wiki_prefix(&r.try_get::<String, _>("source_type").unwrap_or_default()),
                "snippet": snippet,
                "url": r.try_get::<String, _>("url").unwrap_or_default(),
                "updated_at": iso(r.try_get("updated_at").ok()),
            })
        })
        .collect();

    Ok(Json(json!({"results": results})))
}

/// Extract a ~200-char snippet from `body` around the first occurrence of any
/// search term. Falls back to the leading 200 chars if no term is found.
fn truncate_body(body: &str, query: &str) -> String {
    let terms: Vec<&str> = query.split_whitespace().collect();
    let lower = body.to_lowercase();
    let pos = terms
        .iter()
        .filter_map(|t| lower.find(t))
        .min()
        .unwrap_or(0);
    let start = pos.saturating_sub(60);
    let end = (pos + 160).min(body.len());
    let mut s = String::with_capacity(end - start + 8);
    if start > 0 {
        s.push_str("…");
    }
    s.push_str(&body[start..end]);
    if end < body.len() {
        s.push_str("…");
    }
    s
}

// ---------------------------------------------------------------------------
// GET /wiki/page/{*document_id}  (+ trailing /revisions)
// ---------------------------------------------------------------------------

async fn wiki_page_or_revisions(
    State(state): State<AppState>,
    Path(captured): Path<String>,
) -> Result<Json<Value>, ApiError> {
    if let Some(document_id) = captured.strip_suffix("/revisions") {
        wiki_revisions(&state, document_id).await
    } else {
        wiki_page(&state, &captured).await
    }
}

async fn wiki_page(state: &AppState, document_id: &str) -> Result<Json<Value>, ApiError> {
let pool = pool(&state)?;
    let row = sqlx::query(
        "SELECT document_id, source_type, title, body, url, updated_at \
         FROM company_context_documents WHERE document_id = $1 AND source = 'wiki'",
    )
    .bind(document_id)
    .fetch_optional(&pool)
    .await?;

    let Some(row) = row else {
        return Err(ApiError::NotFound("wiki page not found".to_owned()));
    };

    let source_type: String = row.try_get("source_type").unwrap_or_default();
    let updated_at: Option<OffsetDateTime> = row.try_get("updated_at").ok();
    Ok(Json(json!({
        "id": row.try_get::<String, _>("document_id").unwrap_or_default(),
        "title": row.try_get::<String, _>("title").unwrap_or_default(),
        "type": strip_wiki_prefix(&source_type),
        "body": row.try_get::<String, _>("body").unwrap_or_default(),
        "url": row.try_get::<String, _>("url").unwrap_or_default(),
        "updated_at": iso(updated_at),
    })))
}

async fn wiki_revisions(state: &AppState, document_id: &str) -> Result<Json<Value>, ApiError> {
let pool = pool(&state)?;
    // The table may not exist before first ingest — Python swallows the error
    // and returns an empty list.
    let rows = sqlx::query(
        "SELECT revised_at, content_hash, length(body) AS len FROM wiki_page_revisions \
         WHERE document_id = $1 ORDER BY revised_at DESC LIMIT 200",
    )
    .bind(document_id)
    .fetch_all(&pool)
    .await;

    let rows = match rows {
        Ok(rows) => rows,
        Err(_) => {
            return Ok(Json(json!({ "id": document_id, "revisions": [] })));
        }
    };

    let revisions: Vec<Value> = rows
        .iter()
        .map(|r| {
            let revised_at: Option<OffsetDateTime> = r.try_get("revised_at").ok();
            let content_hash: String = r.try_get("content_hash").unwrap_or_default();
            let len: i32 = r.try_get("len").unwrap_or_default();
            json!({
                "revised_at": iso(revised_at),
                "content_hash": content_hash,
                "length": len,
            })
        })
        .collect();

    Ok(Json(json!({ "id": document_id, "revisions": revisions })))
}

// ---------------------------------------------------------------------------
// GET /wiki/changes
// ---------------------------------------------------------------------------

async fn wiki_changes(
    State(state): State<AppState>,
    Query(q): Query<ChangesQuery>,
) -> Result<Json<Value>, ApiError> {
    let pool = pool(&state)?;
    let (start, end) = window(q.days, q.since.as_deref(), q.until.as_deref());

    let page_rows = sqlx::query(
        "SELECT document_id, source_type, title, url, updated_at \
         FROM company_context_documents \
         WHERE source = 'wiki' AND source_type = ANY($1::text[]) \
         AND updated_at >= $2 AND updated_at < $3 \
         ORDER BY updated_at DESC",
    )
    .bind(&PAGE_TYPES[..])
    .bind(start)
    .bind(end)
    .fetch_all(&pool)
    .await?;

    let mut pages: Vec<Value> = Vec::with_capacity(page_rows.len());
    let mut by_type: BTreeMap<String, i64> = BTreeMap::new();
    for r in &page_rows {
        let source_type: String = r.try_get("source_type").unwrap_or_default();
        let typ = strip_wiki_prefix(&source_type);
        let updated_at: Option<OffsetDateTime> = r.try_get("updated_at").ok();
        *by_type.entry(typ.clone()).or_insert(0) += 1;
        pages.push(json!({
            "id": r.try_get::<String, _>("document_id").unwrap_or_default(),
            "title": r.try_get::<String, _>("title").unwrap_or_default(),
            "type": typ,
            "url": r.try_get::<String, _>("url").unwrap_or_default(),
            "updated_at": iso(updated_at),
        }));
    }

    // The sources table may not exist before first ingest — swallow errors.
    // Query by EVENT time (occurred_at) — when the source actually happened
    // in the world — falling back to ingested_at for any older row missing
    // event time. "What changed in the window June 10-12" should return PRs
    // merged / Linear updated / Slack threads from that window even if they
    // were backfilled into the wiki today. Same shape as /wiki/timeline.
    let mut sources: Vec<Value> = Vec::new();
    let mut src_by_kind: BTreeMap<String, i64> = BTreeMap::new();
    let src_rows = sqlx::query(
        "SELECT source_key, kind, ingested_at, occurred_at FROM wiki_ingested_sources \
         WHERE COALESCE(occurred_at, ingested_at) >= $1 \
           AND COALESCE(occurred_at, ingested_at) < $2 \
         ORDER BY COALESCE(occurred_at, ingested_at) DESC",
    )
    .bind(start)
    .bind(end)
    .fetch_all(&pool)
    .await;
    if let Ok(src_rows) = src_rows {
        for r in &src_rows {
            let source_key: String = r.try_get("source_key").unwrap_or_default();
            let ingested_at: Option<OffsetDateTime> = r.try_get("ingested_at").ok();
            let occurred_at: Option<OffsetDateTime> = r.try_get("occurred_at").ok().flatten();
            let mut disp = source_display(&source_key);
            let kind = disp
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("other")
                .to_owned();
            *src_by_kind.entry(kind).or_insert(0) += 1;
            // Expose both: occurred_at (the meaningful date for "what happened
            // when") and ingested_at (pipeline-debug overlay).
            disp.insert("occurred_at".into(), json!(iso(occurred_at)));
            disp.insert("ingested_at".into(), json!(iso(ingested_at)));
            sources.push(Value::Object(disp));
        }
    }

    Ok(Json(json!({
        "since": iso(Some(start)),
        "until": iso(Some(end)),
        "page_count": pages.len(),
        "source_count": sources.len(),
        "pages_by_type": by_type,
        "sources_by_kind": src_by_kind,
        "pages": pages,
        "sources": sources,
    })))
}

// ---------------------------------------------------------------------------
// GET /wiki/timeline
// ---------------------------------------------------------------------------

/// Cross-stream event timeline: every ingested source as an event on a single
/// EVENT-time axis (real-world `occurred_at`, falling back to `ingested_at` for
/// rows ingested before the event-time columns existed). This is distinct from
/// `/wiki/changes` (page-mutation centric) — it answers "what happened, when,
/// across GitHub / Linear / Slack / Drive / meetings".
async fn wiki_timeline(
    State(state): State<AppState>,
    Query(q): Query<ChangesQuery>,
) -> Result<Json<Value>, ApiError> {
    let pool = pool(&state)?;
    let (start, end) = window(q.days, q.since.as_deref(), q.until.as_deref());

    let mut events: Vec<Value> = Vec::new();
    let mut by_kind: BTreeMap<String, i64> = BTreeMap::new();
    // The table (and the occurred_at/title/url columns) may not exist before the
    // first ingest on a fresh DB — swallow errors and return an empty timeline.
    //
    // LEFT JOIN wiki_directives so directive rows carry the rich attribution
    // payload (actor name + verbatim quote + target pages + status) inline.
    // Non-directive rows get NULLs for these columns and the frontend just
    // omits them. The COALESCEs above on (occurred_at, ingested_at) still
    // drive the time axis.
    let rows = sqlx::query(
        "SELECT s.source_key, s.kind, s.ingested_at, s.occurred_at, s.title, s.url, \
                d.actor_slack_id, d.actor_name, d.body AS directive_body, \
                d.target_pages, d.status AS directive_status, \
                d.resulting_revision_ids \
         FROM wiki_ingested_sources s \
         LEFT JOIN wiki_directives d ON d.source_key = s.source_key \
         WHERE COALESCE(s.occurred_at, s.ingested_at) >= $1 \
           AND COALESCE(s.occurred_at, s.ingested_at) < $2 \
         ORDER BY COALESCE(s.occurred_at, s.ingested_at) DESC",
    )
    .bind(start)
    .bind(end)
    .fetch_all(&pool)
    .await;

    if let Ok(rows) = rows {
        for r in &rows {
            let source_key: String = r.try_get("source_key").unwrap_or_default();
            let ingested_at: Option<OffsetDateTime> = r.try_get("ingested_at").ok();
            let occurred_at: Option<OffsetDateTime> = r.try_get("occurred_at").ok().flatten();
            let stored_title: String = r.try_get("title").unwrap_or_default();
            let stored_url: String = r.try_get("url").unwrap_or_default();

            // Prefer the kind column; fall back to parsing the source_key.
            let disp = source_display(&source_key);
            let kind: String = r
                .try_get::<String, _>("kind")
                .ok()
                .filter(|k| !k.is_empty())
                .or_else(|| disp.get("kind").and_then(Value::as_str).map(str::to_owned))
                .unwrap_or_else(|| "other".to_owned());
            let label = if stored_title.is_empty() {
                disp.get("label")
                    .and_then(Value::as_str)
                    .unwrap_or(&source_key)
                    .to_owned()
            } else {
                stored_title
            };
            let url = if stored_url.is_empty() {
                disp.get("url").and_then(Value::as_str).unwrap_or("").to_owned()
            } else {
                stored_url
            };

            *by_kind.entry(kind.clone()).or_insert(0) += 1;

            // Directive-specific enrichment from the LEFT JOIN. These are
            // None on non-directive rows; serialized as null in JSON so the
            // SPA can simply check `event.actor_name` etc.
            let actor_slack_id: Option<String> = r.try_get("actor_slack_id").ok();
            let actor_name: Option<String> = r.try_get("actor_name").ok();
            let directive_body: Option<String> = r.try_get("directive_body").ok();
            let target_pages: Option<Vec<String>> = r.try_get("target_pages").ok();
            let directive_status: Option<String> = r.try_get("directive_status").ok();
            let resulting_revision_ids: Option<Vec<i64>> =
                r.try_get("resulting_revision_ids").ok();

            // For directives, prefer the directive body as the label (verbatim
            // quote is what you want to see on the dot), capped to a tooltip-
            // friendly length. The full body is also exposed under
            // `directive_body` for the popover.
            let directive_excerpt = directive_body.as_ref().map(|b| {
                let cleaned = b.trim();
                if cleaned.chars().count() <= 200 {
                    cleaned.to_owned()
                } else {
                    let truncated: String = cleaned.chars().take(200).collect();
                    format!("{truncated}…")
                }
            });
            let final_label = if kind == "directive" {
                directive_excerpt.clone().unwrap_or(label)
            } else {
                label
            };

            events.push(json!({
                "source_key": source_key,
                "kind": kind,
                "label": final_label,
                "url": url,
                // The event-time axis: when it happened in the world.
                "occurred_at": iso(occurred_at),
                // When the wiki ingested it (pipeline-debug overlay).
                "ingested_at": iso(ingested_at),
                // True when we only have ingest time (pre-column rows).
                "event_time_estimated": occurred_at.is_none(),
                // Directive-only enrichment (null for other kinds). The SPA
                // tooltip uses these to render "directive from <actor>:
                // \"<quote>\"" + chips of the target_pages.
                "actor_slack_id": actor_slack_id,
                "actor_name": actor_name,
                "directive_body": directive_body,
                "directive_excerpt": directive_excerpt,
                "target_pages": target_pages,
                "directive_status": directive_status,
                "resulting_revision_ids": resulting_revision_ids,
            }));
        }
    }

    Ok(Json(json!({
        "since": iso(Some(start)),
        "until": iso(Some(end)),
        "event_count": events.len(),
        "events_by_kind": by_kind,
        "events": events,
    })))
}

// ---------------------------------------------------------------------------
// GET /wiki/diff
// ---------------------------------------------------------------------------
/// Latest revision body at/just-before `when` (None if no revision that early).
/// Returns `(body, revised_at_iso)`.
async fn body_at(
    pool: &PgPool,
    document_id: &str,
    when: OffsetDateTime,
) -> Result<Option<(String, String)>, ApiError> {
    let row = sqlx::query(
        "SELECT body, revised_at FROM wiki_page_revisions \
         WHERE document_id = $1 AND revised_at <= $2 \
         ORDER BY revised_at DESC LIMIT 1",
    )
    .bind(document_id)
    .bind(when)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|r| {
        let body: String = r.try_get("body").unwrap_or_default();
        let revised_at: Option<OffsetDateTime> = r.try_get("revised_at").ok();
        (body, iso(revised_at))
    }))
}

async fn wiki_diff(
    State(state): State<AppState>,
    Query(q): Query<DiffQuery>,
) -> Result<Json<Value>, ApiError> {
    let id = q.id;
    let (start, end) = window(q.days, q.since.as_deref(), q.until.as_deref());
    let pool = pool(&state)?;

    let page = sqlx::query(
        "SELECT title FROM company_context_documents WHERE document_id = $1 AND source = 'wiki'",
    )
    .bind(&id)
    .fetch_optional(&pool)
    .await?;
    let Some(page) = page else {
        return Err(ApiError::NotFound("wiki page not found".to_owned()));
    };
    let title: String = page.try_get("title").unwrap_or_default();

    let before = body_at(&pool, &id, start).await?;
    let after = body_at(&pool, &id, end).await?;

    let (before_body, before_at) = before.clone().unwrap_or_else(|| (String::new(), String::new()));

    // If no revision at/after the window end, fall back to the latest body.
    let (after_body, after_at) = match after {
        Some(after) => after,
        None => {
            let latest = sqlx::query(
                "SELECT body, revised_at FROM wiki_page_revisions WHERE document_id = $1 \
                 ORDER BY revised_at DESC LIMIT 1",
            )
            .bind(&id)
    .fetch_optional(&pool)
            .await?;
            match latest {
                Some(r) => {
                    let body: String = r.try_get("body").unwrap_or_default();
                    let revised_at: Option<OffsetDateTime> = r.try_get("revised_at").ok();
                    (body, iso(revised_at))
                }
                None => (String::new(), String::new()),
            }
        }
    };

    let changed = before_body != after_body;
    let from_label = format!(
        "{title} @ {}",
        if before_at.is_empty() { "start" } else { &before_at }
    );
    let to_label = format!(
        "{title} @ {}",
        if after_at.is_empty() { "now" } else { &after_at }
    );

    let diff_lines: Vec<String> = if changed {
        unified_diff(&before_body, &after_body, &from_label, &to_label, 2)
    } else {
        Vec::new()
    };

    let added: Vec<String> = diff_lines
        .iter()
        .filter(|ln| ln.starts_with('+') && !ln.starts_with("+++"))
        .map(|ln| ln[1..].to_string())
        .collect();
    let removed: Vec<String> = diff_lines
        .iter()
        .filter(|ln| ln.starts_with('-') && !ln.starts_with("---"))
        .map(|ln| ln[1..].to_string())
        .collect();

    Ok(Json(json!({
        "id": id,
        "title": title,
        "since": iso(Some(start)),
        "until": iso(Some(end)),
        "before_at": before_at,
        "after_at": after_at,
        "had_prior": before.is_some(),
        "changed": changed,
        "diff": diff_lines,
        "added": added,
        "removed": removed,
        "added_count": added.len(),
        "removed_count": removed.len(),
    })))
}

/// Produce a unified diff as a list of lines (no trailing newlines), matching
/// Python's `difflib.unified_diff(..., lineterm="", n=2)` output shape:
/// `--- <from>`, `+++ <to>`, `@@ -a,b +c,d @@` hunk headers, and ` `/`-`/`+`
/// prefixed body lines.
fn unified_diff(
    before: &str,
    after: &str,
    from_label: &str,
    to_label: &str,
    context: usize,
) -> Vec<String> {
    use similar::TextDiff;

    let diff = TextDiff::from_lines(before, after);
    let mut out: Vec<String> = Vec::new();
    let mut wrote_header = false;

    let mut udiff = diff.unified_diff();
    udiff.context_radius(context);
    for hunk in udiff.iter_hunks() {
        if !wrote_header {
            out.push(format!("--- {from_label}"));
            out.push(format!("+++ {to_label}"));
            wrote_header = true;
        }
        out.push(hunk.header().to_string());
        for change in hunk.iter_changes() {
            let sign = match change.tag() {
                similar::ChangeTag::Delete => '-',
                similar::ChangeTag::Insert => '+',
                similar::ChangeTag::Equal => ' ',
            };
            // `change.value()` keeps the line's trailing newline; strip it so
            // each entry is one logical line (matching `lineterm=""`).
            let value = change.value();
            let value = value.strip_suffix('\n').unwrap_or(value);
            let value = value.strip_suffix('\r').unwrap_or(value);
            out.push(format!("{sign}{value}"));
        }
    }
    out
}
