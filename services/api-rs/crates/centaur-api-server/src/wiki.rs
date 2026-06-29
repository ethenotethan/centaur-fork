//! Read-only public wiki endpoints — serve the LLM Wiki graph + pages to the
//! docs-site SPA (live, from Postgres), so the frontend never relies on a
//! committed snapshot. Public by design (a browser can't safely hold an API
//! key); read-only, no mutations. Data: `company_context_documents` WHERE
//! `source = $WIKI_SOURCE` (defaults to `'wiki'`; set `WIKI_SOURCE=wiki_v2`
//! to flip reads to the v2 KB — same env honored by the agent's `wiki` tool
//! and the standup_digest workflow so a single env flip cuts every reader
//! over at once).
//!
//! Ported from the original FastAPI router (`api/routers/wiki.py`). JSON field
//! names are kept byte-for-byte identical because the docs-site SPA depends on
//! them.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderValue, Method, StatusCode},
    response::IntoResponse,
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use sqlx::{PgPool, Row};
use time::{Duration, OffsetDateTime, format_description::well_known::Rfc3339};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

use crate::{ApiError, routes::AppState};

const PAGE_TYPES: [&str; 4] = ["wiki_goal", "wiki_project", "wiki_entity", "wiki_topic"];

/// Which wiki slice this api reads from. Resolved once at process start from
/// `WIKI_SOURCE` (default `"wiki"`). Set `WIKI_SOURCE=wiki_v2` and roll
/// api-rs to flip the docs-site SPA + agent + standup over to the v2 KB.
/// Kept as a `&'static str` (leaked once) so query strings can interpolate
/// the table names with no per-request allocation, and so the values used to
/// `format!` table names are always drawn from a closed, vetted set (no user
/// input ever reaches the SQL string).
fn resolve_wiki_source() -> &'static str {
    match std::env::var("WIKI_SOURCE")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .as_deref()
    {
        Some("wiki_v2") => "wiki_v2",
        // Anything else (unset, "wiki", or an unrecognized value) falls back
        // to the v1 slice. We deliberately don't accept arbitrary source
        // names — only the two known slices — to keep the table-name
        // derivation below safe.
        _ => "wiki",
    }
}

/// `company_context_documents.source` filter for every read (bound as `$N`).
static WIKI_SOURCE: LazyLock<&'static str> = LazyLock::new(resolve_wiki_source);

/// `wiki_ingested_sources[_v2]` table name (interpolated via `format!` into
/// the `/changes` + `/timeline` queries).
static INGESTED_TABLE: LazyLock<&'static str> = LazyLock::new(|| match *WIKI_SOURCE {
    "wiki_v2" => "wiki_ingested_sources_v2",
    _ => "wiki_ingested_sources",
});

/// `wiki_page_revisions[_v2]` table name (interpolated into the `/diff` +
/// `/revisions` queries).
static REVISIONS_TABLE: LazyLock<&'static str> = LazyLock::new(|| match *WIKI_SOURCE {
    "wiki_v2" => "wiki_page_revisions_v2",
    _ => "wiki_page_revisions",
});

/// `wiki_directives[_v2]` table name (LEFT JOINed in the `/timeline` query
/// for directive event attribution).
static DIRECTIVES_TABLE: LazyLock<&'static str> = LazyLock::new(|| match *WIKI_SOURCE {
    "wiki_v2" => "wiki_directives_v2",
    _ => "wiki_directives",
});

/// Build the `/wiki/*` sub-router with a permissive (or `CORS_ORIGINS`-scoped)
/// CORS layer. These routes are public, read-only GETs.
pub(crate) fn router() -> Router<AppState> {
    Router::new()
        .route("/wiki/graph", get(wiki_graph))
        .route("/wiki/search", get(wiki_search))
        .route("/wiki/changes", get(wiki_changes))
        .route("/wiki/timeline", get(wiki_timeline))
        .route("/wiki/revisions-timeline", get(wiki_revisions_timeline))
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
    (parse(since, now - Duration::days(1)), parse(until, now))
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
    if let Some(rest) = source_key.strip_prefix("coordinator:") {
        // coordinator:version:<commit>  — a coordinator build detected via /health
        // coordinator:stats:<YYYY-MM-DD> — a daily network-stats snapshot from /v1/stats
        let (sub_kind, label) = if let Some(commit) = rest.strip_prefix("version:") {
            ("coordinator_version", format!("coordinator {commit}"))
        } else if let Some(date) = rest.strip_prefix("stats:") {
            ("coordinator_stats", format!("coordinator stats {date}"))
        } else {
            ("coordinator", rest.to_owned())
        };
        m.insert("kind".into(), json!(sub_kind));
        m.insert("label".into(), json!(label));
        m.insert("url".into(), json!("https://api.darkbloom.dev"));
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
    /// `metadata->>'goal_kind'` — only populated on `wiki_goal` rows. Used by
    /// the docs SPA to render type badges (strategic / capability /
    /// operational) and to group goals on the Goals view. Empty string when
    /// missing or non-goal.
    goal_kind: String,
    /// `metadata->>'glossary'` ("true" on curated glossary entries — a
    /// `wiki_topic` row tagged `metadata.glossary=true`), plus the term +
    /// definition. Mirrors the `goal_kind` precedent: surfaced conditionally so
    /// the SPA can render a dedicated Glossary view. Empty when not a glossary row.
    glossary: String,
    term: String,
    definition: String,
    /// `metadata->>'person'` ("true" on Contributor entries — a `wiki_entity`
    /// row tagged `metadata.person=true`), plus display_name / github_login /
    /// roles. Surfaced conditionally like glossary so the SPA can render a
    /// Contributors view. Empty when not a person row.
    person: String,
    display_name: String,
    github_login: String,
    roles: String,
    bio: String,
    last_active: String,
    active: String,
}

async fn wiki_graph(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let pool = pool(&state)?;
    let rows = sqlx::query(
        "SELECT document_id, source_type, title, body, url, updated_at, \
                COALESCE(metadata->>'goal_kind', '') AS goal_kind, \
                COALESCE(metadata->>'glossary', '') AS glossary, \
                COALESCE(metadata->>'term', '') AS term, \
                COALESCE(metadata->>'definition', '') AS definition, \
                COALESCE(metadata->>'person', '') AS person, \
                COALESCE(metadata->>'display_name', '') AS display_name, \
                COALESCE(metadata->>'github_login', '') AS github_login, \
                COALESCE(metadata->>'roles', '') AS roles, \
                COALESCE(metadata->>'bio', '') AS bio, \
                COALESCE(metadata->>'last_active', '') AS last_active, \
                COALESCE(metadata->>'active', '') AS active \
         FROM company_context_documents \
         WHERE source = $1 AND source_type = ANY($2::text[]) \
         ORDER BY title",
    )
    .bind(*WIKI_SOURCE)
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
            goal_kind: r.try_get("goal_kind").unwrap_or_default(),
            glossary: r.try_get("glossary").unwrap_or_default(),
            term: r.try_get("term").unwrap_or_default(),
            definition: r.try_get("definition").unwrap_or_default(),
            person: r.try_get("person").unwrap_or_default(),
            display_name: r.try_get("display_name").unwrap_or_default(),
            github_login: r.try_get("github_login").unwrap_or_default(),
            roles: r.try_get("roles").unwrap_or_default(),
            bio: r.try_get("bio").unwrap_or_default(),
            last_active: r.try_get("last_active").unwrap_or_default(),
            active: r.try_get("active").unwrap_or_default(),
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
        // Only emit goal_kind on goal nodes (empty string elsewhere is noise).
        if p.source_type == "wiki_goal" && !p.goal_kind.is_empty() {
            node.insert("goal_kind".into(), json!(p.goal_kind));
        }
        // Only emit glossary fields on glossary-tagged rows (curated terms).
        if p.glossary == "true" {
            node.insert("glossary".into(), json!(true));
            node.insert("term".into(), json!(p.term));
            node.insert("definition".into(), json!(p.definition));
        }
        // Only emit person fields on contributor-tagged rows.
        if p.person == "true" {
            node.insert("person".into(), json!(true));
            node.insert("display_name".into(), json!(p.display_name));
            node.insert("github_login".into(), json!(p.github_login));
            node.insert("roles".into(), json!(p.roles));
            node.insert("bio".into(), json!(p.bio));
            node.insert("last_active".into(), json!(p.last_active));
            node.insert("active".into(), json!(p.active == "true"));
        }
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
        node.insert(
            "degree".into(),
            json!(deg.get(&p.document_id).copied().unwrap_or(0)),
        );
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
         WHERE source = $1 AND source_type = ANY($2::text[]) \
         AND (title ||| $3::text OR body ||| $3::text) \
         ORDER BY paradedb.score(document_id) DESC, updated_at DESC \
         LIMIT $4",
    )
    .bind(*WIKI_SOURCE)
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
        "SELECT document_id, source_type, title, body, url, updated_at, \
                COALESCE(metadata->>'goal_kind', '') AS goal_kind, \
                COALESCE(metadata->>'glossary', '') AS glossary, \
                COALESCE(metadata->>'term', '') AS term, \
                COALESCE(metadata->>'definition', '') AS definition, \
                COALESCE(metadata->>'person', '') AS person, \
                COALESCE(metadata->>'display_name', '') AS display_name, \
                COALESCE(metadata->>'github_login', '') AS github_login, \
                COALESCE(metadata->>'roles', '') AS roles, \
                COALESCE(metadata->>'bio', '') AS bio, \
                COALESCE(metadata->>'last_active', '') AS last_active, \
                COALESCE(metadata->>'active', '') AS active \
         FROM company_context_documents WHERE document_id = $1 AND source = $2",
    )
    .bind(document_id)
    .bind(*WIKI_SOURCE)
    .fetch_optional(&pool)
    .await?;

    let Some(row) = row else {
        return Err(ApiError::NotFound("wiki page not found".to_owned()));
    };

    let source_type: String = row.try_get("source_type").unwrap_or_default();
    let updated_at: Option<OffsetDateTime> = row.try_get("updated_at").ok();
    let goal_kind: String = row.try_get("goal_kind").unwrap_or_default();
    let glossary: String = row.try_get("glossary").unwrap_or_default();
    let term: String = row.try_get("term").unwrap_or_default();
    let definition: String = row.try_get("definition").unwrap_or_default();
    let person: String = row.try_get("person").unwrap_or_default();
    let display_name: String = row.try_get("display_name").unwrap_or_default();
    let github_login: String = row.try_get("github_login").unwrap_or_default();
    let roles: String = row.try_get("roles").unwrap_or_default();
    let bio: String = row.try_get("bio").unwrap_or_default();
    let last_active: String = row.try_get("last_active").unwrap_or_default();
    let active: String = row.try_get("active").unwrap_or_default();
    let mut out = Map::new();
    out.insert(
        "id".into(),
        json!(row.try_get::<String, _>("document_id").unwrap_or_default()),
    );
    out.insert(
        "title".into(),
        json!(row.try_get::<String, _>("title").unwrap_or_default()),
    );
    out.insert("type".into(), json!(strip_wiki_prefix(&source_type)));
    out.insert(
        "body".into(),
        json!(row.try_get::<String, _>("body").unwrap_or_default()),
    );
    out.insert(
        "url".into(),
        json!(row.try_get::<String, _>("url").unwrap_or_default()),
    );
    out.insert("updated_at".into(), json!(iso(updated_at)));
    // Only emit goal_kind on goal pages (irrelevant on entities/projects/topics).
    if source_type == "wiki_goal" && !goal_kind.is_empty() {
        out.insert("goal_kind".into(), json!(goal_kind));
    }
    // Only emit glossary fields on glossary-tagged rows (curated terms).
    if glossary == "true" {
        out.insert("glossary".into(), json!(true));
        out.insert("term".into(), json!(term));
        out.insert("definition".into(), json!(definition));
    }
    // Only emit person fields on contributor-tagged rows.
    if person == "true" {
        out.insert("person".into(), json!(true));
        out.insert("display_name".into(), json!(display_name));
        out.insert("github_login".into(), json!(github_login));
        out.insert("roles".into(), json!(roles));
        out.insert("bio".into(), json!(bio));
        out.insert("last_active".into(), json!(last_active));
        out.insert("active".into(), json!(active == "true"));
    }
    Ok(Json(Value::Object(out)))
}

async fn wiki_revisions(state: &AppState, document_id: &str) -> Result<Json<Value>, ApiError> {
    let pool = pool(&state)?;
    // The table may not exist before first ingest — Python swallows the error
    // and returns an empty list. Reads from `wiki_page_revisions` (v1) or
    // `wiki_page_revisions_v2` depending on `WIKI_SOURCE`.
    let rows = sqlx::query(&format!(
        "SELECT revised_at, content_hash, length(body) AS len FROM {} \
         WHERE document_id = $1 ORDER BY revised_at DESC LIMIT 200",
        *REVISIONS_TABLE
    ))
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

    // Page list anchored on EVENT time (source_updated_at), not row mtime
    // (updated_at), so the v2 backfill stays positionally aligned with the
    // Event Feed. During backfill, every page row's `updated_at` is
    // wall-clock `now()` (when the synthesis ran), which would lump every
    // backfilled page into "this week" and make the /diff endpoint return
    // empty (because the revision history points back to the real event
    // times). `source_updated_at` is set to the source's event time by
    // `_upsert_page`, so this matches the timeline source dots + the
    // revision timestamps that `body_at(start)` / `body_at(end)` resolve to.
    let page_rows = sqlx::query(
        "SELECT document_id, source_type, title, url, source_updated_at AS updated_at \
         FROM company_context_documents \
         WHERE source = $1 AND source_type = ANY($2::text[]) \
         AND source_updated_at >= $3 AND source_updated_at < $4 \
         ORDER BY source_updated_at DESC",
    )
    .bind(*WIKI_SOURCE)
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
    let src_rows = sqlx::query(&format!(
        "SELECT source_key, kind, ingested_at, occurred_at FROM {} \
         WHERE COALESCE(occurred_at, ingested_at) >= $1 \
           AND COALESCE(occurred_at, ingested_at) < $2 \
         ORDER BY COALESCE(occurred_at, ingested_at) DESC",
        *INGESTED_TABLE
    ))
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
    let rows = sqlx::query(&format!(
        "SELECT s.source_key, s.kind, s.ingested_at, s.occurred_at, s.title, s.url, \
                d.actor_slack_id, d.actor_name, d.body AS directive_body, \
                d.target_pages, d.status AS directive_status, \
                d.resulting_revision_ids \
         FROM {ingested} s \
         LEFT JOIN {directives} d ON d.source_key = s.source_key \
         WHERE COALESCE(s.occurred_at, s.ingested_at) >= $1 \
           AND COALESCE(s.occurred_at, s.ingested_at) < $2 \
         ORDER BY COALESCE(s.occurred_at, s.ingested_at) DESC",
        ingested = *INGESTED_TABLE,
        directives = *DIRECTIVES_TABLE,
    ))
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
                disp.get("url")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned()
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
            let resulting_revision_ids: Option<Vec<i64>> = r.try_get("resulting_revision_ids").ok();

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
// GET /wiki/revisions-timeline
// ---------------------------------------------------------------------------
/// Wiki page-edit volume over time — the *stateful output* counterpart to
/// `/wiki/timeline` (which is the raw *input* events). Buckets
/// `wiki_page_revisions[_v2].revised_at` into `date_trunc` buckets whose unit
/// adapts to the window (hour / day / week / month) and returns per-bucket
/// revision counts + the cumulative total BEFORE the window start, so the docs
/// SPA can draw a true cumulative "knowledge accrued" curve overlaid on the
/// event-volume bars. Read-only; v1/v2-aware via `*REVISIONS_TABLE`.
async fn wiki_revisions_timeline(
    State(state): State<AppState>,
    Query(q): Query<ChangesQuery>,
) -> Result<Json<Value>, ApiError> {
    let pool = pool(&state)?;
    let (start, end) = window(q.days, q.since.as_deref(), q.until.as_deref());

    // Choose a Postgres date_trunc unit matching the SPA's adaptive bucketing.
    let span_days = (end - start).as_seconds_f64() / 86_400.0;
    let unit = if span_days <= 2.0 {
        "hour"
    } else if span_days <= 45.0 {
        "day"
    } else if span_days <= 240.0 {
        "week"
    } else {
        "month"
    };

    // Per-bucket revision counts within the window.
    let mut buckets: Vec<Value> = Vec::new();
    let mut total_in_window: i64 = 0;
    let rows = sqlx::query(&format!(
        "SELECT date_trunc('{unit}', revised_at) AS bucket, count(*) AS n \
         FROM {rev} \
         WHERE revised_at >= $1 AND revised_at < $2 \
         GROUP BY 1 ORDER BY 1",
        unit = unit,
        rev = *REVISIONS_TABLE,
    ))
    .bind(start)
    .bind(end)
    .fetch_all(&pool)
    .await;

    if let Ok(rows) = rows {
        for r in &rows {
            let bucket: Option<OffsetDateTime> = r.try_get("bucket").ok();
            let n: i64 = r.try_get("n").unwrap_or(0);
            total_in_window += n;
            buckets.push(json!({ "bucket": iso(bucket), "count": n }));
        }
    }

    // Cumulative revisions that already existed BEFORE the window opened, so the
    // SPA can seed the cumulative curve at the correct height (not from zero).
    let baseline: i64 = sqlx::query_scalar(&format!(
        "SELECT count(*) FROM {rev} WHERE revised_at < $1",
        rev = *REVISIONS_TABLE,
    ))
    .bind(start)
    .fetch_one(&pool)
    .await
    .unwrap_or(0);

    Ok(Json(json!({
        "since": iso(Some(start)),
        "until": iso(Some(end)),
        "unit": unit,
        "baseline": baseline,
        "total_in_window": total_in_window,
        "buckets": buckets,
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
    let row = sqlx::query(&format!(
        "SELECT body, revised_at FROM {} \
         WHERE document_id = $1 AND revised_at <= $2 \
         ORDER BY revised_at DESC LIMIT 1",
        *REVISIONS_TABLE
    ))
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
        "SELECT title FROM company_context_documents WHERE document_id = $1 AND source = $2",
    )
    .bind(&id)
    .bind(*WIKI_SOURCE)
    .fetch_optional(&pool)
    .await?;
    let Some(page) = page else {
        return Err(ApiError::NotFound("wiki page not found".to_owned()));
    };
    let title: String = page.try_get("title").unwrap_or_default();

    let before = body_at(&pool, &id, start).await?;
    let after = body_at(&pool, &id, end).await?;

    let (before_body, before_at) = before
        .clone()
        .unwrap_or_else(|| (String::new(), String::new()));

    // If no revision at/after the window end, fall back to the latest body.
    let (after_body, after_at) = match after {
        Some(after) => after,
        None => {
            let latest = sqlx::query(&format!(
                "SELECT body, revised_at FROM {} WHERE document_id = $1 \
                 ORDER BY revised_at DESC LIMIT 1",
                *REVISIONS_TABLE
            ))
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
        if before_at.is_empty() {
            "start"
        } else {
            &before_at
        }
    );
    let to_label = format!(
        "{title} @ {}",
        if after_at.is_empty() {
            "now"
        } else {
            &after_at
        }
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

// ===========================================================================
// Live product-lifecycle sources (DAR-369 / DAR-368 follow-up)
// ===========================================================================
//
// The wiki tools above serve the *synthesized* knowledge base — accurate but
// lagging (wiki-sync/ETL cadence) and lossy (only sources the maintainer chose
// to ingest). To answer "what actually happened in the last 24h" with the same
// fidelity as asking the Centaur agent directly, we read the LIVE sources the
// agent reads: GitHub (merged PRs + releases), Linear (issues + state), and
// Slack (channel history). The api-rs container already holds the same plaintext
// credentials the agent's tools use (envFrom centaur-infra-env), so the relay
// calls these APIs directly — no sandbox/tool-server hop. Recipes mirror the
// wiki_maintainer `_fetch_*` functions verbatim.

/// Repos to scan for GitHub activity. `WIKI_REPOS` (comma-separated owner/repo)
/// or a sensible default. Shared with the wiki maintainer's repo set.
fn lifecycle_repos() -> Vec<String> {
    std::env::var("WIKI_REPOS")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|r| r.trim().to_owned())
                .filter(|r| !r.is_empty())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec!["Layr-Labs/d-inference".to_owned()])
}

/// Fine-grained GitHub PAT that works against the Layr-Labs org. NEVER the
/// classic `GITHUB_TOKEN`/`GITHUB_PAT` (those 403 on the org).
fn github_token() -> Option<String> {
    std::env::var("CODE_REVIEW_GITHUB_TOKEN")
        .ok()
        .filter(|t| !t.is_empty())
        .or_else(|| {
            std::env::var("GITHUB_GATEWAY_TOKEN")
                .ok()
                .filter(|t| !t.is_empty())
        })
}

/// Internal + provider Slack channels the lifecycle feed scans (channel history
/// via the bot token). `WIKI_SLACK_CHANNEL_IDS` + `WIKI_INTERNAL_SLACK_CHANNEL_IDS`
/// (comma-separated); default = providers/support/eng/product.
fn lifecycle_slack_channels() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for var in ["WIKI_SLACK_CHANNEL_IDS", "WIKI_INTERNAL_SLACK_CHANNEL_IDS"] {
        if let Ok(v) = std::env::var(var) {
            out.extend(
                v.split(',')
                    .map(|c| c.trim().to_owned())
                    .filter(|c| !c.is_empty()),
            );
        }
    }
    if out.is_empty() {
        // providers, support, eng, product
        out = ["C0B0CAQC8P5", "C0B0JMULP3L", "C0B6S8MUDRR", "C0B6YGM83N1"]
            .iter()
            .map(|s| s.to_string())
            .collect();
    }
    out.sort();
    out.dedup();
    out
}

fn lifecycle_http() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("centaur-mcp-lifecycle")
        .timeout(std::time::Duration::from_secs(25))
        .build()
        .unwrap_or_default()
}

/// Minimal percent-encoding for query-string values (the reqwest `query`
/// feature isn't enabled in this build, so we build URLs by hand). Encodes
/// everything that isn't an unreserved char.
fn qenc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// GitHub merged PRs across `repos` since `start`. Returns lifecycle items.
async fn fetch_live_merged_prs(
    client: &reqwest::Client,
    token: &str,
    repos: &[String],
    start: OffsetDateTime,
    per_repo: usize,
) -> Vec<Value> {
    let since = start.format(&Rfc3339).unwrap_or_default();
    let mut out = Vec::new();
    for repo in repos {
        let q = format!("repo:{repo} is:pr is:merged merged:>={since}");
        let url = format!(
            "https://api.github.com/search/issues?q={}&per_page=50&sort=created&order=desc",
            qenc(&q)
        );
        let resp = client
            .get(&url)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .bearer_auth(token)
            .send()
            .await;
        let Ok(resp) = resp else { continue };
        if !resp.status().is_success() {
            continue;
        }
        let Ok(body) = resp.json::<Value>().await else {
            continue;
        };
        let items = body
            .get("items")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for it in items.into_iter().take(per_repo) {
            let num = it.get("number").and_then(Value::as_i64).unwrap_or_default();
            let title = it
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            let author = it
                .get("user")
                .and_then(|u| u.get("login"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            // PR-search uses `closed_at` as the merge proxy on is:merged results.
            let when = it
                .get("closed_at")
                .and_then(Value::as_str)
                .or_else(|| it.get("updated_at").and_then(Value::as_str))
                .unwrap_or("")
                .to_owned();
            let body_text = it
                .get("body")
                .and_then(Value::as_str)
                .map(|b| b.chars().take(800).collect::<String>())
                .unwrap_or_default();
            out.push(json!({
                "kind": "pr",
                "repo": repo,
                "id": format!("{repo}#{num}"),
                "title": title,
                "author": author,
                "occurred_at": when,
                "url": format!("https://github.com/{repo}/pull/{num}"),
                "body": body_text,
            }));
        }
    }
    out
}

/// GitHub releases across `repos` published since `start`.
async fn fetch_live_releases(
    client: &reqwest::Client,
    token: &str,
    repos: &[String],
    start: OffsetDateTime,
) -> Vec<Value> {
    let mut out = Vec::new();
    for repo in repos {
        let url = format!("https://api.github.com/repos/{repo}/releases?per_page=20");
        let resp = client
            .get(&url)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .bearer_auth(token)
            .send()
            .await;
        let Ok(resp) = resp else { continue };
        if !resp.status().is_success() {
            continue;
        }
        let Ok(arr) = resp.json::<Value>().await else {
            continue;
        };
        for rel in arr.as_array().cloned().unwrap_or_default() {
            if rel.get("draft").and_then(Value::as_bool).unwrap_or(false) {
                continue;
            }
            let published = rel
                .get("published_at")
                .and_then(Value::as_str)
                .unwrap_or("");
            if published.is_empty() {
                continue;
            }
            // Lexical RFC3339 compare is valid for the Zulu timestamps GitHub returns.
            if published < start.format(&Rfc3339).unwrap_or_default().as_str() {
                continue;
            }
            let tag = rel
                .get("tag_name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            let name = rel
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(&tag)
                .to_owned();
            out.push(json!({
                "kind": "release",
                "repo": repo,
                "id": format!("{repo}@{tag}"),
                "title": name,
                "occurred_at": published,
                "url": rel.get("html_url").and_then(Value::as_str).unwrap_or("").to_owned(),
                "body": rel.get("body").and_then(Value::as_str)
                    .map(|b| b.chars().take(800).collect::<String>()).unwrap_or_default(),
            }));
        }
    }
    out
}

/// Linear issues updated since `start` for the configured team.
async fn fetch_live_linear(
    client: &reqwest::Client,
    key: &str,
    start: OffsetDateTime,
) -> Vec<Value> {
    let team = std::env::var("WIKI_LINEAR_TEAM_ID")
        .ok()
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| "120662cb-3a74-4b46-8105-a80adee59391".to_owned());
    let since = start.format(&Rfc3339).unwrap_or_default();
    let query = "query($teamId:String!,$since:DateTimeOrDuration!){ team(id:$teamId){ \
        issues(filter:{updatedAt:{gt:$since}}, first:50, orderBy:updatedAt){ nodes{ \
        identifier title url updatedAt state{name type} assignee{name} } } } }";
    let payload = json!({"query": query, "variables": {"teamId": team, "since": since}});
    let resp = client
        .post("https://api.linear.app/graphql")
        // Linear personal API keys go in Authorization BARE (no "Bearer ").
        .header("Authorization", key)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await;
    let Ok(resp) = resp else { return Vec::new() };
    if !resp.status().is_success() {
        return Vec::new();
    }
    let Ok(body) = resp.json::<Value>().await else {
        return Vec::new();
    };
    let nodes = body
        .pointer("/data/team/issues/nodes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    nodes
        .into_iter()
        .map(|n| {
            let ident = n.get("identifier").and_then(Value::as_str).unwrap_or("").to_owned();
            json!({
                "kind": "linear",
                "id": ident,
                "title": n.get("title").and_then(Value::as_str).unwrap_or("").to_owned(),
                "state": n.get("state").and_then(|s| s.get("name")).and_then(Value::as_str).unwrap_or("").to_owned(),
                "assignee": n.get("assignee").and_then(|a| a.get("name")).and_then(Value::as_str).unwrap_or("").to_owned(),
                "occurred_at": n.get("updatedAt").and_then(Value::as_str).unwrap_or("").to_owned(),
                "url": n.get("url").and_then(Value::as_str).unwrap_or("").to_owned(),
            })
        })
        .collect()
}

/// Slack channel-history activity across the configured channels since `start`.
/// Uses the bot token + `conversations.history` (the `search.messages` user
/// token isn't a search-capable token in this deployment).
async fn fetch_live_slack(
    client: &reqwest::Client,
    bot_token: &str,
    channels: &[String],
    start: OffsetDateTime,
    per_channel: usize,
) -> Vec<Value> {
    let oldest = (start.unix_timestamp()).to_string();
    let mut out = Vec::new();
    for chan in channels {
        let url = format!(
            "https://slack.com/api/conversations.history?channel={}&oldest={}&limit=60",
            qenc(chan),
            qenc(&oldest)
        );
        let resp = client.get(&url).bearer_auth(bot_token).send().await;
        let Ok(resp) = resp else { continue };
        let Ok(body) = resp.json::<Value>().await else {
            continue;
        };
        if !body.get("ok").and_then(Value::as_bool).unwrap_or(false) {
            continue;
        }
        let msgs = body
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for m in msgs.into_iter().take(per_channel) {
            // Skip channel-join/system subtypes; keep real messages.
            if m.get("subtype").and_then(Value::as_str).is_some() {
                continue;
            }
            let text = m.get("text").and_then(Value::as_str).unwrap_or("");
            if text.trim().is_empty() {
                continue;
            }
            let ts = m.get("ts").and_then(Value::as_str).unwrap_or("");
            // Slack ts is "<epoch>.<seq>"; derive an ISO occurred_at.
            let occurred = ts
                .split('.')
                .next()
                .and_then(|s| s.parse::<i64>().ok())
                .and_then(|e| OffsetDateTime::from_unix_timestamp(e).ok())
                .and_then(|dt| dt.format(&Rfc3339).ok())
                .unwrap_or_default();
            out.push(json!({
                "kind": "slack",
                "channel_id": chan,
                "id": format!("slack:{chan}:{ts}"),
                "title": text.chars().take(160).collect::<String>(),
                "occurred_at": occurred,
                "url": "",
                "body": text.chars().take(800).collect::<String>(),
            }));
        }
    }
    out
}

/// Aggregate the live product lifecycle across all sources in a window.
/// `sources` filters which streams to include (default: all).
async fn live_recent_activity(q: &ChangesQuery, sources: &[String]) -> Value {
    let (start, _end) = window(q.days, q.since.as_deref(), q.until.as_deref());
    let client = lifecycle_http();
    let want = |name: &str| sources.is_empty() || sources.iter().any(|s| s == name);

    let mut items: Vec<Value> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    if want("github") || want("pr") || want("release") {
        match github_token() {
            Some(tok) => {
                let repos = lifecycle_repos();
                if want("github") || want("pr") {
                    items.extend(fetch_live_merged_prs(&client, &tok, &repos, start, 50).await);
                }
                if want("github") || want("release") {
                    items.extend(fetch_live_releases(&client, &tok, &repos, start).await);
                }
            }
            None => errors.push("github: no CODE_REVIEW_GITHUB_TOKEN/GITHUB_GATEWAY_TOKEN".into()),
        }
    }
    if want("linear") {
        match std::env::var("LINEAR_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
        {
            Some(key) => items.extend(fetch_live_linear(&client, &key, start).await),
            None => errors.push("linear: no LINEAR_API_KEY".into()),
        }
    }
    if want("slack") {
        match std::env::var("SLACK_BOT_TOKEN")
            .ok()
            .filter(|k| !k.is_empty())
        {
            Some(tok) => {
                let chans = lifecycle_slack_channels();
                items.extend(fetch_live_slack(&client, &tok, &chans, start, 60).await);
            }
            None => errors.push("slack: no SLACK_BOT_TOKEN".into()),
        }
    }

    // Oldest → newest by occurred_at.
    items.sort_by(|a, b| {
        a.get("occurred_at")
            .and_then(Value::as_str)
            .unwrap_or("")
            .cmp(b.get("occurred_at").and_then(Value::as_str).unwrap_or(""))
    });
    let mut by_kind: BTreeMap<String, i64> = BTreeMap::new();
    for it in &items {
        if let Some(k) = it.get("kind").and_then(Value::as_str) {
            *by_kind.entry(k.to_owned()).or_insert(0) += 1;
        }
    }

    json!({
        "since": iso(Some(start)),
        "live": true,
        "note": "Live read from GitHub/Linear/Slack APIs (not the wiki's synthesized state) \
                 — same sources the Centaur agent reads.",
        "count": items.len(),
        "by_kind": by_kind,
        "errors": errors,
        "items": items,
    })
}

// ===========================================================================
// MCP relay — read-only Model Context Protocol server over the wiki (DAR-368)
// ===========================================================================
//
// Exposes the read-only wiki surface as a remote, Streamable-HTTP MCP server so
// external MCP clients (Claude Desktop, Cursor, etc. — e.g. EigenLabs) can query
// the knowledge base directly. It is a THIN PROTOCOL SHIM: every tool delegates
// to the existing `wiki_*` handlers above (same PgPool, same query logic, same
// JSON), so there is zero duplication of the read logic.
//
// Transport: a single `POST /mcp` JSON-RPC 2.0 endpoint handling `initialize`,
// `tools/list`, and `tools/call` (synchronous JSON responses — these are short
// reads, no SSE streaming needed for v1). `notifications/initialized` is acked.
//
// Auth: bearer token. `MCP_RELAY_TOKEN` (env) must match the request's
// `Authorization: Bearer <token>`. Unset → the relay is disabled (404), so it
// can't accidentally serve unauthenticated in an unconfigured environment.

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// `/mcp` sub-router. No CORS layer (MCP clients are servers/desktop apps, not
/// browsers); auth is enforced per-request inside the handler.
pub(crate) fn mcp_router() -> Router<AppState> {
    Router::new().route("/mcp", post(mcp_handler))
}

fn rpc_result(id: Value, result: Value) -> Json<Value> {
    Json(json!({ "jsonrpc": "2.0", "id": id, "result": result }))
}

fn rpc_error(id: Value, code: i64, message: &str) -> Json<Value> {
    Json(json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    }))
}

/// The read-only tool catalog. Each entry is (name, description, inputSchema).
fn mcp_tools() -> Value {
    json!([
        {
            "name": "search_wiki",
            "description": "Full-text search the Darkbloom/Centaur knowledge base. Returns matching wiki pages (title, type, snippet, url).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "q": { "type": "string", "description": "Search query." },
                    "limit": { "type": "integer", "description": "Max results (default 20)." }
                },
                "required": ["q"]
            }
        },
        {
            "name": "read_page",
            "description": "Read a single wiki page's full markdown body by document_id (e.g. 'wiki:entity:coordinator-deployment').",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "document_id": { "type": "string", "description": "The wiki page id, e.g. wiki:topic:release-process." }
                },
                "required": ["document_id"]
            }
        },
        {
            "name": "wiki_graph",
            "description": "The full wiki knowledge graph: every page (goal/project/entity/topic) as a node, with [[wikilink]] edges, degree, and backlinks. Use to list all pages or understand structure.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "recent_changes",
            "description": "Pages changed and sources ingested in a recent window. Args: days (number) OR since/until (ISO timestamps).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "days": { "type": "number", "description": "Look back this many days (e.g. 7)." },
                    "since": { "type": "string", "description": "ISO start time (alternative to days)." },
                    "until": { "type": "string", "description": "ISO end time (optional)." }
                }
            }
        },
        {
            "name": "event_timeline",
            "description": "Cross-stream event feed (PRs, Linear issues, provider Slack, releases, directives, coordinator telemetry, news) in a window, oldest->newest. Args: days OR since/until.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "days": { "type": "number" },
                    "since": { "type": "string" },
                    "until": { "type": "string" }
                }
            }
        },
        {
            "name": "page_diff",
            "description": "Unified diff of a wiki page's body over a window (what changed). Args: id (document_id), plus days OR since/until.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The wiki page document_id." },
                    "days": { "type": "number" },
                    "since": { "type": "string" },
                    "until": { "type": "string" }
                },
                "required": ["id"]
            }
        },
        {
            "name": "recent_activity",
            "description": "LIVE product-lifecycle feed: what actually happened across GitHub (merged PRs + releases), Linear (issues + state), and provider/internal Slack in a recent window, oldest->newest. Reads the SAME live sources the Centaur agent reads — NOT the wiki's synthesized/lagging state. Use this for 'what happened in the last 24h / this week'. Args: days (number, default 1) OR since/until (ISO); optional sources (array subset of [\"github\",\"pr\",\"release\",\"linear\",\"slack\"]).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "days": { "type": "number", "description": "Look back this many days (default 1)." },
                    "since": { "type": "string", "description": "ISO start (alternative to days)." },
                    "until": { "type": "string", "description": "ISO end (optional)." },
                    "sources": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional subset: github, pr, release, linear, slack. Default: all."
                    }
                }
            }
        },
        {
            "name": "list_merged_prs",
            "description": "LIVE merged GitHub pull requests (with bodies) across the tracked repos in a window. Args: days OR since/until.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "days": { "type": "number" },
                    "since": { "type": "string" },
                    "until": { "type": "string" }
                }
            }
        },
        {
            "name": "list_linear_issues",
            "description": "LIVE Linear issues updated in a window (identifier, title, state, assignee, url). Args: days OR since/until.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "days": { "type": "number" },
                    "since": { "type": "string" },
                    "until": { "type": "string" }
                }
            }
        },
        {
            "name": "list_slack_activity",
            "description": "LIVE Slack messages from the tracked provider + internal channels in a window (channel history). Args: days OR since/until.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "days": { "type": "number" },
                    "since": { "type": "string" },
                    "until": { "type": "string" }
                }
            }
        }
    ])
}

/// Wrap a tool's JSON output in MCP `tools/call` result shape (text content with
/// the pretty-printed JSON — the standard way to return structured data).
fn tool_content(value: &Value) -> Value {
    let text = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
    json!({ "content": [ { "type": "text", "text": text } ], "isError": false })
}

/// Dispatch a single `tools/call` to the underlying read-only wiki handler.
async fn mcp_call_tool(state: &AppState, name: &str, args: &Value) -> Result<Value, ApiError> {
    let s = |k: &str| args.get(k).and_then(Value::as_str).map(str::to_owned);
    let f = |k: &str| args.get(k).and_then(Value::as_f64);
    match name {
        "search_wiki" => {
            let q =
                s("q").ok_or_else(|| ApiError::BadRequest("search_wiki requires 'q'".into()))?;
            let limit = args.get("limit").and_then(Value::as_i64);
            let Json(v) =
                wiki_search(State(state.clone()), Query(SearchQuery { q, limit })).await?;
            Ok(v)
        }
        "read_page" => {
            let id = s("document_id")
                .ok_or_else(|| ApiError::BadRequest("read_page requires 'document_id'".into()))?;
            let Json(v) = wiki_page(state, &id).await?;
            Ok(v)
        }
        "wiki_graph" => {
            let Json(v) = wiki_graph(State(state.clone())).await?;
            Ok(v)
        }
        "recent_changes" => {
            let Json(v) = wiki_changes(
                State(state.clone()),
                Query(ChangesQuery {
                    days: f("days"),
                    since: s("since"),
                    until: s("until"),
                }),
            )
            .await?;
            Ok(v)
        }
        "event_timeline" => {
            let Json(v) = wiki_timeline(
                State(state.clone()),
                Query(ChangesQuery {
                    days: f("days"),
                    since: s("since"),
                    until: s("until"),
                }),
            )
            .await?;
            Ok(v)
        }
        "page_diff" => {
            let id =
                s("id").ok_or_else(|| ApiError::BadRequest("page_diff requires 'id'".into()))?;
            let Json(v) = wiki_diff(
                State(state.clone()),
                Query(DiffQuery {
                    id,
                    days: f("days"),
                    since: s("since"),
                    until: s("until"),
                }),
            )
            .await?;
            Ok(v)
        }
        // --- Live product-lifecycle tools (read external APIs directly) ---
        "recent_activity" => {
            let sources: Vec<String> = args
                .get("sources")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            let q = ChangesQuery {
                days: f("days"),
                since: s("since"),
                until: s("until"),
            };
            Ok(live_recent_activity(&q, &sources).await)
        }
        "list_merged_prs" => {
            let q = ChangesQuery {
                days: f("days"),
                since: s("since"),
                until: s("until"),
            };
            Ok(live_recent_activity(&q, &["pr".to_owned()]).await)
        }
        "list_linear_issues" => {
            let q = ChangesQuery {
                days: f("days"),
                since: s("since"),
                until: s("until"),
            };
            Ok(live_recent_activity(&q, &["linear".to_owned()]).await)
        }
        "list_slack_activity" => {
            let q = ChangesQuery {
                days: f("days"),
                since: s("since"),
                until: s("until"),
            };
            Ok(live_recent_activity(&q, &["slack".to_owned()]).await)
        }
        other => Err(ApiError::BadRequest(format!("unknown tool: {other}"))),
    }
}

/// Bearer-token check. `MCP_RELAY_TOKEN` unset → relay disabled (404, so the
/// endpoint's existence isn't revealed). Otherwise require an exact match.
fn mcp_authorized(headers: &HeaderMap) -> Result<(), StatusCode> {
    let Some(expected) = std::env::var("MCP_RELAY_TOKEN")
        .ok()
        .filter(|t| !t.is_empty())
    else {
        return Err(StatusCode::NOT_FOUND);
    };
    let presented = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .unwrap_or("");
    // Length-checked, constant-time-ish byte compare.
    if presented.len() == expected.len()
        && presented
            .bytes()
            .zip(expected.bytes())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
    {
        Ok(())
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

async fn mcp_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Json<Value>,
) -> axum::response::Response {
    if let Err(code) = mcp_authorized(&headers) {
        return code.into_response();
    }
    let req = body.0;
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");

    match method {
        "initialize" => rpc_result(
            id,
            json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "darkbloom-wiki-mcp", "version": "1.0.0" }
            }),
        )
        .into_response(),
        // Client lifecycle notification — no id; just ack.
        "notifications/initialized" => StatusCode::ACCEPTED.into_response(),
        "tools/list" => rpc_result(id, json!({ "tools": mcp_tools() })).into_response(),
        "tools/call" => {
            let params = req.get("params").cloned().unwrap_or(Value::Null);
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            match mcp_call_tool(&state, name, &args).await {
                Ok(v) => rpc_result(id, tool_content(&v)).into_response(),
                // Tool-level errors → successful JSON-RPC result with isError=true
                // (MCP convention), so clients surface it without a transport error.
                Err(e) => rpc_result(
                    id,
                    json!({
                        "content": [ { "type": "text", "text": format!("error: {e}") } ],
                        "isError": true
                    }),
                )
                .into_response(),
            }
        }
        "ping" => rpc_result(id, json!({})).into_response(),
        other => rpc_error(id, -32601, &format!("method not found: {other}")).into_response(),
    }
}
