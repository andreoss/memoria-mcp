#![forbid(unsafe_code)]
#![allow(clippy::multiple_crate_versions)]

use memoria_core::embedding::{EmbeddingProvider, LocalHashEmbeddingProvider};
use memoria_core::llm::{LlmProvider, LocalSentenceLlmProvider, Message, Role};
use memoria_core::memory::Memory;
use memoria_core::vector_store::{InMemoryVectorStore, VectorStore};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

type BoxedLlm = Box<dyn LlmProvider + Send + Sync>;
type BoxedEmbedding = Box<dyn EmbeddingProvider + Send + Sync>;
type BoxedVectorStore = Box<dyn VectorStore + Send + Sync>;
type SharedMemory = Arc<Memory<BoxedLlm, BoxedEmbedding, BoxedVectorStore>>;

#[derive(Clone)]
enum Backend {
    Local(SharedMemory),
    Remote(RemoteClient),
}

#[derive(Clone)]
struct RemoteClient {
    base_url: String,
    api_key: Option<String>,
    http: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct RemoteErrorBody {
    error: String,
}

#[derive(Deserialize)]
struct IdsResponse {
    ids: Vec<String>,
}

#[derive(Deserialize)]
struct SearchResponse {
    results: Vec<memoria_core::vector_store::SearchResult>,
}

#[derive(Deserialize)]
struct HistoryResponseBody {
    entries: Vec<memoria_core::memory::HistoryEntry>,
}

#[derive(Deserialize)]
struct DeleteAllResponseBody {
    deleted: usize,
}

#[derive(Deserialize)]
struct EntitiesResponseBody {
    entities: Vec<memoria_core::memory::EntitySummary>,
}

async fn error_message(response: reqwest::Response) -> Result<String, String> {
    let status = response.status();
    let bytes = response.bytes().await.map_err(|err| format!("failed to read server response: {err}"))?;
    let message = serde_json::from_slice::<RemoteErrorBody>(&bytes).map_or_else(|_| String::from_utf8_lossy(&bytes).into_owned(), |body| body.error);
    Ok(format!("server returned {status}: {message}"))
}

impl RemoteClient {
    fn new(base_url: &str, api_key: Option<String>) -> Self {
        Self { base_url: base_url.trim_end_matches('/').to_string(), api_key, http: reqwest::Client::new() }
    }

    fn authed(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => builder.bearer_auth(key),
            None => builder,
        }
    }

    async fn ensure_success(&self, builder: reqwest::RequestBuilder) -> Result<(), String> {
        let response = self.authed(builder).send().await.map_err(|err| format!("request to server failed: {err}"))?;
        if response.status().is_success() { Ok(()) } else { Err(error_message(response).await?) }
    }

    async fn send<T: serde::de::DeserializeOwned>(&self, builder: reqwest::RequestBuilder) -> Result<T, String> {
        let response = self.authed(builder).send().await.map_err(|err| format!("request to server failed: {err}"))?;
        if response.status().is_success() {
            let bytes = response.bytes().await.map_err(|err| format!("failed to read server response: {err}"))?;
            serde_json::from_slice(&bytes).map_err(|err| format!("malformed response from server: {err}"))
        } else {
            Err(error_message(response).await?)
        }
    }

    async fn get_record(&self, id: &str) -> Result<Option<memoria_core::vector_store::VectorRecord>, String> {
        let response = self.authed(self.http.get(format!("{}/memories/{id}", self.base_url))).send().await.map_err(|err| format!("request to server failed: {err}"))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if response.status().is_success() {
            let bytes = response.bytes().await.map_err(|err| format!("failed to read server response: {err}"))?;
            serde_json::from_slice(&bytes).map(Some).map_err(|err| format!("malformed response from server: {err}"))
        } else {
            Err(error_message(response).await?)
        }
    }

    async fn search(&self, request: &SearchMemoriesRequest) -> Result<Vec<memoria_core::vector_store::SearchResult>, String> {
        let body = serde_json::json!({
            "query": request.query,
            "user_id": request.user_id,
            "agent_id": request.agent_id,
            "run_id": request.run_id,
            "top_k": request.top_k,
            "threshold": request.threshold,
            "show_expired": request.show_expired,
            "explain": request.explain,
        });
        let parsed: SearchResponse = self.send(self.http.post(format!("{}/search", self.base_url)).json(&body)).await?;
        Ok(parsed.results)
    }

    async fn list_ids(&self, user_id: Option<&str>, agent_id: Option<&str>, run_id: Option<&str>, offset: usize, limit: usize, show_expired: bool) -> Result<Vec<String>, String> {
        let mut url = reqwest::Url::parse(&format!("{}/memories", self.base_url)).map_err(|err| format!("invalid server URL: {err}"))?;
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("offset", &offset.to_string());
            query.append_pair("limit", &limit.to_string());
            if let Some(v) = user_id {
                query.append_pair("user_id", v);
            }
            if let Some(v) = agent_id {
                query.append_pair("agent_id", v);
            }
            if let Some(v) = run_id {
                query.append_pair("run_id", v);
            }
            if show_expired {
                query.append_pair("show_expired", "true");
            }
        }
        let parsed: IdsResponse = self.send(self.http.get(url)).await?;
        Ok(parsed.ids)
    }

    async fn history(&self, id: &str, offset: usize, limit: usize) -> Result<Vec<memoria_core::memory::HistoryEntry>, String> {
        let url = format!("{}/memories/{id}/history?offset={offset}&limit={limit}", self.base_url);
        let parsed: HistoryResponseBody = self.send(self.http.get(url)).await?;
        Ok(parsed.entries)
    }

    async fn add(&self, request: &AddMemoryRequest) -> Result<Vec<String>, String> {
        let body = serde_json::json!({
            "content": request.content,
            "user_id": request.user_id,
            "agent_id": request.agent_id,
            "run_id": request.run_id,
            "infer": request.infer,
            "images": request.images,
        });
        let parsed: IdsResponse = self.send(self.http.post(format!("{}/memories", self.base_url)).json(&body)).await?;
        Ok(parsed.ids)
    }

    async fn update(&self, id: &str, content: Option<&str>, metadata: Option<&HashMap<String, String>>) -> Result<(), String> {
        let body = serde_json::json!({ "content": content, "metadata": metadata });
        self.ensure_success(self.http.put(format!("{}/memories/{id}", self.base_url)).json(&body)).await
    }

    async fn delete(&self, id: &str) -> Result<(), String> {
        self.ensure_success(self.http.delete(format!("{}/memories/{id}", self.base_url))).await
    }

    async fn delete_all(&self, user_id: Option<&str>, agent_id: Option<&str>, run_id: Option<&str>) -> Result<usize, String> {
        let body = serde_json::json!({ "user_id": user_id, "agent_id": agent_id, "run_id": run_id });
        let parsed: DeleteAllResponseBody = self.send(self.http.delete(format!("{}/memories", self.base_url)).json(&body)).await?;
        Ok(parsed.deleted)
    }

    async fn list_entities(&self) -> Result<Vec<memoria_core::memory::EntitySummary>, String> {
        let parsed: EntitiesResponseBody = self.send(self.http.get(format!("{}/entities", self.base_url))).await?;
        Ok(parsed.entities)
    }
}

fn scope_from_optional(user_id: Option<String>, agent_id: Option<String>, run_id: Option<String>) -> HashMap<String, String> {
    let mut scope = HashMap::new();
    if let Some(v) = user_id {
        scope.insert("user_id".to_string(), v);
    }
    if let Some(v) = agent_id {
        scope.insert("agent_id".to_string(), v);
    }
    if let Some(v) = run_id {
        scope.insert("run_id".to_string(), v);
    }
    scope
}

pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn check_secret(configured: Option<&str>, provided: Option<&str>) -> Result<(), String> {
    configured.map_or(Ok(()), |expected| {
        if provided.is_some_and(|value| constant_time_eq(value.as_bytes(), expected.as_bytes())) {
            Ok(())
        } else {
            Err("missing or incorrect secret (MEMORIA_MCP_SECRET is configured on this server; see ADR-38)".to_string())
        }
    })
}

const fn default_top_k() -> usize {
    10
}

const fn default_list_limit() -> usize {
    50
}

const fn default_history_limit() -> usize {
    100
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
struct MemoryResult {
    id: String,
    score: f32,
    payload: HashMap<String, String>,
    score_details: Option<ScoreDetailsResult>,
}

impl From<memoria_core::vector_store::SearchResult> for MemoryResult {
    fn from(result: memoria_core::vector_store::SearchResult) -> Self {
        Self { id: result.id, score: result.score, payload: result.payload, score_details: result.score_details.map(ScoreDetailsResult::from) }
    }
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
struct ScoreDetailsResult {
    semantic_score: f32,
    bm25_score: Option<f32>,
    entity_boost: Option<f32>,
    raw_score: f32,
    final_score: f32,
}

impl From<memoria_core::vector_store::ScoreDetails> for ScoreDetailsResult {
    fn from(details: memoria_core::vector_store::ScoreDetails) -> Self {
        Self {
            semantic_score: details.semantic_score,
            bm25_score: details.bm25_score,
            entity_boost: details.entity_boost,
            raw_score: details.raw_score,
            final_score: details.final_score,
        }
    }
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
struct MemoryRecord {
    id: String,
    payload: HashMap<String, String>,
}

impl From<memoria_core::vector_store::VectorRecord> for MemoryRecord {
    fn from(record: memoria_core::vector_store::VectorRecord) -> Self {
        Self { id: record.id, payload: record.payload }
    }
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
struct HistoryEntry {
    event: String,
    content: String,
}

impl From<memoria_core::memory::HistoryEntry> for HistoryEntry {
    fn from(entry: memoria_core::memory::HistoryEntry) -> Self {
        let event = match entry.event {
            memoria_core::memory::HistoryEvent::Added => "added",
            memoria_core::memory::HistoryEvent::Deleted => "deleted",
        };
        Self { event: event.to_string(), content: entry.content }
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SearchMemoriesRequest {
    #[schemars(description = "The natural-language query to search for")]
    query: String,
    #[serde(default)]
    #[schemars(description = "Restrict results to this user")]
    user_id: Option<String>,
    #[serde(default)]
    #[schemars(description = "Restrict results to this agent")]
    agent_id: Option<String>,
    #[serde(default)]
    #[schemars(description = "Restrict results to this run")]
    run_id: Option<String>,
    #[serde(default = "default_top_k")]
    #[schemars(description = "Maximum number of results to return (default 10)")]
    top_k: usize,
    #[serde(default)]
    #[schemars(description = "Minimum similarity score required to keep a result")]
    threshold: Option<f32>,
    #[serde(default)]
    #[schemars(description = "Include expired memories (default false)")]
    show_expired: bool,
    #[serde(default)]
    #[schemars(description = "Include a score_details breakdown (semantic/keyword/entity-boost components) in each result (default false)")]
    explain: bool,
    #[serde(default)]
    #[schemars(description = "Required if MEMORIA_MCP_SECRET is configured on the server; omit otherwise")]
    secret: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GetMemoryRequest {
    #[schemars(description = "The id of the memory to fetch")]
    id: String,
    #[serde(default)]
    #[schemars(description = "Required if MEMORIA_MCP_SECRET is configured on the server; omit otherwise")]
    secret: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct GetMemoriesRequest {
    #[serde(default)]
    #[schemars(description = "Restrict results to this user")]
    user_id: Option<String>,
    #[serde(default)]
    #[schemars(description = "Restrict results to this agent")]
    agent_id: Option<String>,
    #[serde(default)]
    #[schemars(description = "Restrict results to this run")]
    run_id: Option<String>,
    #[serde(default)]
    #[schemars(description = "Number of results to skip (default 0)")]
    offset: usize,
    #[serde(default = "default_list_limit")]
    #[schemars(description = "Maximum number of results to return (default 50)")]
    limit: usize,
    #[serde(default)]
    #[schemars(description = "Include expired memories (default false)")]
    show_expired: bool,
    #[serde(default)]
    #[schemars(description = "Required if MEMORIA_MCP_SECRET is configured on the server; omit otherwise")]
    secret: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct MemoryHistoryRequest {
    #[schemars(description = "The id of the memory to fetch history for")]
    id: String,
    #[serde(default)]
    #[schemars(description = "Number of entries to skip (default 0)")]
    offset: usize,
    #[serde(default = "default_history_limit")]
    #[schemars(description = "Maximum number of entries to return (default 100)")]
    limit: usize,
    #[serde(default)]
    #[schemars(description = "Required if MEMORIA_MCP_SECRET is configured on the server; omit otherwise")]
    secret: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct AddMemoryRequest {
    #[schemars(description = "The message content to extract facts from (or store verbatim if infer=false)")]
    content: String,
    #[serde(default)]
    #[schemars(description = "Scope this memory to a user")]
    user_id: Option<String>,
    #[serde(default)]
    #[schemars(description = "Scope this memory to an agent")]
    agent_id: Option<String>,
    #[serde(default)]
    #[schemars(description = "Scope this memory to a run")]
    run_id: Option<String>,
    #[serde(default = "default_infer")]
    #[schemars(description = "Extract facts via the LLM provider (default true); false stores content verbatim")]
    infer: bool,
    #[serde(default)]
    #[schemars(description = "Base64-encoded images; a vision-capable LLM provider describes them before storage")]
    images: Vec<String>,
    #[serde(default)]
    #[schemars(description = "Required if MEMORIA_MCP_SECRET is configured on the server; omit otherwise")]
    secret: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct UpdateMemoryRequest {
    #[schemars(description = "The id of the memory to update")]
    id: String,
    #[serde(default)]
    #[schemars(description = "New content; leave unset to keep the existing content")]
    content: Option<String>,
    #[serde(default)]
    #[schemars(description = "Metadata keys to add or overwrite; must not include user_id/agent_id/run_id")]
    metadata: Option<HashMap<String, String>>,
    #[serde(default)]
    #[schemars(description = "Required if MEMORIA_MCP_SECRET is configured on the server; omit otherwise")]
    secret: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct DeleteMemoryRequest {
    #[schemars(description = "The id of the memory to delete")]
    id: String,
    #[serde(default)]
    #[schemars(description = "Required if MEMORIA_MCP_SECRET is configured on the server; omit otherwise")]
    secret: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct DeleteAllMemoriesRequest {
    #[serde(default)]
    #[schemars(description = "Restrict deletion to this user")]
    user_id: Option<String>,
    #[serde(default)]
    #[schemars(description = "Restrict deletion to this agent")]
    agent_id: Option<String>,
    #[serde(default)]
    #[schemars(description = "Restrict deletion to this run")]
    run_id: Option<String>,
    #[serde(default)]
    #[schemars(description = "Required if MEMORIA_MCP_SECRET is configured on the server; omit otherwise")]
    secret: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ListEntitiesRequest {
    #[serde(default)]
    #[schemars(description = "Required if MEMORIA_MCP_SECRET is configured on the server; omit otherwise")]
    secret: Option<String>,
}

const fn default_infer() -> bool {
    true
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
struct DeleteAllResult {
    deleted: usize,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
struct SearchMemoriesResult {
    results: Vec<MemoryResult>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
struct GetMemoriesResult {
    memories: Vec<MemoryRecord>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
struct MemoryHistoryResult {
    entries: Vec<HistoryEntry>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
struct AddMemoryResult {
    ids: Vec<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
struct DeleteMemoryResult {
    deleted: bool,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
struct ListEntitiesResult {
    entities: Vec<EntitySummary>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
struct EntitySummary {
    entity_type: String,
    entity_id: String,
    memory_count: usize,
}

impl From<memoria_core::memory::EntitySummary> for EntitySummary {
    fn from(summary: memoria_core::memory::EntitySummary) -> Self {
        Self { entity_type: summary.entity_type, entity_id: summary.entity_id, memory_count: summary.memory_count }
    }
}

fn write_atomically(path: &Path, data: &[u8]) -> std::io::Result<()> {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temp_path = path.with_extension(format!("json.tmp.{}.{unique}", std::process::id()));
    std::fs::write(&temp_path, data)?;
    std::fs::rename(&temp_path, path)
}

fn save_store<L, E, V>(memory: &Memory<L, E, V>, path: &Path) -> std::io::Result<()>
where
    L: LlmProvider,
    E: EmbeddingProvider,
    V: VectorStore,
{
    let ids = memory.list_all(0, usize::MAX, true, None).unwrap_or_default();
    let records: Vec<memoria_core::vector_store::VectorRecord> = ids.iter().filter_map(|id| memory.get(id).ok().flatten()).collect();
    let data = serde_json::to_vec(&records).unwrap_or_default();
    write_atomically(path, &data)
}

fn resolve_history_path(store_path: &Path) -> PathBuf {
    store_path.with_file_name("history.json")
}

fn load_history(path: &Path) -> HashMap<String, Vec<memoria_core::memory::HistoryEntry>> {
    std::fs::read_to_string(path).ok().and_then(|data| serde_json::from_str(&data).ok()).unwrap_or_default()
}

fn save_history<L, E, V>(memory: &Memory<L, E, V>, path: &Path) -> std::io::Result<()>
where
    L: LlmProvider,
    E: EmbeddingProvider,
    V: VectorStore,
{
    let data = serde_json::to_vec(&memory.history_snapshot()).unwrap_or_default();
    write_atomically(path, &data)
}

#[derive(Clone)]
struct MemoriaMcpServer {
    backend: Backend,
    mcp_secret: Option<String>,
    store_path: PathBuf,
    history_path: PathBuf,
    persist_json_snapshot: bool,
    persist_history: bool,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MemoriaMcpServer {
    fn new(backend: Backend, mcp_secret: Option<String>, store_path: PathBuf, history_path: PathBuf, persist_json_snapshot: bool) -> Self {
        Self { backend, mcp_secret, store_path, history_path, persist_json_snapshot, persist_history: true, tool_router: Self::tool_router() }
    }

    fn persist_after_mutation(&self) {
        let Backend::Local(memory) = &self.backend else {
            return;
        };
        if self.persist_json_snapshot {
            let _ = save_store(memory, &self.store_path);
        }
        if self.persist_history {
            let _ = save_history(memory, &self.history_path);
        }
    }

    #[tool(description = "Search memories with a semantic query, optionally scoped to a user, agent, or run")]
    async fn search_memories(&self, Parameters(request): Parameters<SearchMemoriesRequest>) -> Result<Json<SearchMemoriesResult>, String> {
        check_secret(self.mcp_secret.as_deref(), request.secret.as_deref())?;
        let results = match &self.backend {
            Backend::Local(memory) => {
                let scope = scope_from_optional(request.user_id.clone(), request.agent_id.clone(), request.run_id.clone());
                memory
                    .search(&request.query, request.top_k, &scope, request.threshold, request.show_expired, None, false, request.explain)
                    .map_err(|err| err.to_string())?
            }
            Backend::Remote(client) => client.search(&request).await?,
        };
        Ok(Json(SearchMemoriesResult { results: results.into_iter().map(MemoryResult::from).collect() }))
    }

    #[tool(description = "Fetch a single memory by id")]
    async fn get_memory(&self, Parameters(request): Parameters<GetMemoryRequest>) -> Result<Json<MemoryRecord>, String> {
        check_secret(self.mcp_secret.as_deref(), request.secret.as_deref())?;
        let found = match &self.backend {
            Backend::Local(memory) => memory.get(&request.id).map_err(|err| err.to_string())?,
            Backend::Remote(client) => client.get_record(&request.id).await?,
        };
        match found {
            Some(record) => Ok(Json(MemoryRecord::from(record))),
            None => Err(format!("no memory found with id {}", request.id)),
        }
    }

    #[tool(description = "List memories, optionally scoped to a user, agent, or run")]
    async fn get_memories(&self, Parameters(request): Parameters<GetMemoriesRequest>) -> Result<Json<GetMemoriesResult>, String> {
        check_secret(self.mcp_secret.as_deref(), request.secret.as_deref())?;
        let memories = match &self.backend {
            Backend::Local(memory) => {
                let scope = scope_from_optional(request.user_id, request.agent_id, request.run_id);
                let ids = memory.list(&scope, request.offset, request.limit, request.show_expired, None).map_err(|err| err.to_string())?;
                ids.into_iter().filter_map(|id| memory.get(&id).ok().flatten()).map(MemoryRecord::from).collect()
            }
            Backend::Remote(client) => {
                let ids = client
                    .list_ids(request.user_id.as_deref(), request.agent_id.as_deref(), request.run_id.as_deref(), request.offset, request.limit, request.show_expired)
                    .await?;
                let mut tasks = tokio::task::JoinSet::new();
                for (index, id) in ids.into_iter().enumerate() {
                    let client = client.clone();
                    tasks.spawn(async move { (index, client.get_record(&id).await) });
                }
                let mut fetched = Vec::with_capacity(tasks.len());
                while let Some(joined) = tasks.join_next().await {
                    fetched.push(joined.map_err(|err| format!("record fetch task failed: {err}"))?);
                }
                fetched.sort_by_key(|(index, _)| *index);
                let mut memories = Vec::with_capacity(fetched.len());
                for (_, result) in fetched {
                    if let Some(record) = result? {
                        memories.push(MemoryRecord::from(record));
                    }
                }
                memories
            }
        };
        Ok(Json(GetMemoriesResult { memories }))
    }

    #[tool(description = "Fetch the change history (add/delete events) for a single memory")]
    async fn memory_history(&self, Parameters(request): Parameters<MemoryHistoryRequest>) -> Result<Json<MemoryHistoryResult>, String> {
        check_secret(self.mcp_secret.as_deref(), request.secret.as_deref())?;
        let entries = match &self.backend {
            Backend::Local(memory) => memory.history(&request.id, request.offset, request.limit).map_err(|err| err.to_string())?,
            Backend::Remote(client) => client.history(&request.id, request.offset, request.limit).await?,
        };
        Ok(Json(MemoryHistoryResult { entries: entries.into_iter().map(HistoryEntry::from).collect() }))
    }

    #[tool(description = "Extract facts from a message and store them under a scope (or store content verbatim if infer=false)")]
    async fn add_memory(&self, Parameters(request): Parameters<AddMemoryRequest>) -> Result<Json<AddMemoryResult>, String> {
        check_secret(self.mcp_secret.as_deref(), request.secret.as_deref())?;
        let ids = match &self.backend {
            Backend::Local(memory) => {
                let scope = scope_from_optional(request.user_id.clone(), request.agent_id.clone(), request.run_id.clone());
                memory
                    .add(&[Message::with_images(Role::User, request.content.clone(), request.images.clone())], scope, request.infer)
                    .map_err(|err| err.to_string())?
            }
            Backend::Remote(client) => client.add(&request).await?,
        };
        self.persist_after_mutation();
        Ok(Json(AddMemoryResult { ids }))
    }

    #[tool(description = "Update a memory's content and/or metadata")]
    async fn update_memory(&self, Parameters(request): Parameters<UpdateMemoryRequest>) -> Result<Json<MemoryRecord>, String> {
        check_secret(self.mcp_secret.as_deref(), request.secret.as_deref())?;
        let found = match &self.backend {
            Backend::Local(memory) => {
                memory.update(&request.id, request.content.as_deref(), request.metadata.clone()).map_err(|err| err.to_string())?;
                memory.get(&request.id).map_err(|err| err.to_string())?
            }
            Backend::Remote(client) => {
                client.update(&request.id, request.content.as_deref(), request.metadata.as_ref()).await?;
                client.get_record(&request.id).await?
            }
        };
        self.persist_after_mutation();
        match found {
            Some(record) => Ok(Json(MemoryRecord::from(record))),
            None => Err(format!("no memory found with id {}", request.id)),
        }
    }

    #[tool(description = "Delete a single memory by id")]
    async fn delete_memory(&self, Parameters(request): Parameters<DeleteMemoryRequest>) -> Result<Json<DeleteMemoryResult>, String> {
        check_secret(self.mcp_secret.as_deref(), request.secret.as_deref())?;
        match &self.backend {
            Backend::Local(memory) => memory.delete(&request.id).map_err(|err| err.to_string())?,
            Backend::Remote(client) => client.delete(&request.id).await?,
        }
        self.persist_after_mutation();
        Ok(Json(DeleteMemoryResult { deleted: true }))
    }

    #[tool(description = "Delete every memory matching a scope (user, agent, and/or run) -- at least one is required")]
    async fn delete_all_memories(&self, Parameters(request): Parameters<DeleteAllMemoriesRequest>) -> Result<Json<DeleteAllResult>, String> {
        check_secret(self.mcp_secret.as_deref(), request.secret.as_deref())?;
        if request.user_id.is_none() && request.agent_id.is_none() && request.run_id.is_none() {
            return Err(
                "delete_all_memories requires at least one of user_id, agent_id, or run_id -- server has no separate admin tier to gate an unscoped wipe the way the REST API does".to_string(),
            );
        }
        let deleted = match &self.backend {
            Backend::Local(memory) => {
                let scope = scope_from_optional(request.user_id, request.agent_id, request.run_id);
                memory.reset(&scope, None).map_err(|err| err.to_string())?
            }
            Backend::Remote(client) => client.delete_all(request.user_id.as_deref(), request.agent_id.as_deref(), request.run_id.as_deref()).await?,
        };
        self.persist_after_mutation();
        Ok(Json(DeleteAllResult { deleted }))
    }

    #[tool(description = "List distinct users, agents, and runs with a memory count for each")]
    async fn list_entities(&self, Parameters(request): Parameters<ListEntitiesRequest>) -> Result<Json<ListEntitiesResult>, String> {
        check_secret(self.mcp_secret.as_deref(), request.secret.as_deref())?;
        let entities = match &self.backend {
            Backend::Local(memory) => memory.list_entities().map_err(|err| err.to_string())?,
            Backend::Remote(client) => client.list_entities().await?,
        };
        Ok(Json(ListEntitiesResult { entities: entities.into_iter().map(EntitySummary::from).collect() }))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemoriaMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "memoria: a local-first memory layer for AI agents. Tools: search_memories, get_memory, \
             get_memories, memory_history, add_memory, update_memory, delete_memory, \
             delete_all_memories, list_entities. See ADR-38.",
        )
    }
}

fn load_store(path: &Path) -> InMemoryVectorStore {
    let store = InMemoryVectorStore::new();
    if let Ok(data) = std::fs::read_to_string(path) {
        if let Ok(records) = serde_json::from_str::<Vec<memoria_core::vector_store::VectorRecord>>(&data) {
            for record in records {
                let _ = store.insert(record);
            }
        }
    }
    store
}

fn resolve_store_path(env_override: Option<&str>, home: &str) -> PathBuf {
    env_override.map_or_else(|| Path::new(home).join(".memoria").join("mcp-store.json"), PathBuf::from)
}

fn resolve_sqlite_path(env_override: Option<&str>, home: &str) -> PathBuf {
    env_override.map_or_else(|| Path::new(home).join(".memoria").join("mcp-store.db"), PathBuf::from)
}

fn resolve_llm_provider(
    provider_choice: Option<&str>,
    model: Option<String>,
    base_url: Option<String>,
    candle_cache_dir: Option<String>,
) -> Result<(BoxedLlm, String), String> {
    match provider_choice.unwrap_or("local") {
        "local" => Ok((Box::new(LocalSentenceLlmProvider::new()), "LocalSentenceLlmProvider (local, non-AI; see ADR-12)".to_string())),
        "ollama" => {
            #[cfg(feature = "ollama")]
            {
                let model = model.unwrap_or_else(|| "qwen2.5:0.5b".to_string());
                let resolved_base_url = base_url.clone().unwrap_or_else(|| "http://localhost:11434".to_string());
                let config = memoria_core::llm::LlmConfig { model: model.clone(), base_url, api_key: None, temperature: None };
                let provider = memoria_core::llm::OllamaLlmProvider::from_config(config).map_err(|err| err.to_string())?;
                let label = format!("OllamaLlmProvider (model={model}, base_url={resolved_base_url}; see ADR-25)");
                Ok((Box::new(provider), label))
            }
            #[cfg(not(feature = "ollama"))]
            {
                let _ = (model, base_url, candle_cache_dir);
                Err("MEMORIA_LLM_PROVIDER=ollama requires the mcp binary to be built with --features ollama".to_string())
            }
        }
        "candle" => {
            #[cfg(feature = "candle")]
            {
                let model = model.unwrap_or_else(|| "qwen2.5-0.5b-instruct-q4_0".to_string());
                let config = memoria_core::llm::LlmConfig { model: model.clone(), base_url, api_key: None, temperature: None };
                let cache_dir = candle_cache_dir.map(PathBuf::from);
                let provider = memoria_core::llm::CandleLlmProvider::from_config(&config, cache_dir).map_err(|err| err.to_string())?;
                let label = format!("CandleLlmProvider (model={model}; see ADR-40)");
                Ok((Box::new(provider), label))
            }
            #[cfg(not(feature = "candle"))]
            {
                let _ = (model, base_url, candle_cache_dir);
                Err("MEMORIA_LLM_PROVIDER=candle requires the mcp binary to be built with --features candle".to_string())
            }
        }
        other => Err(format!("unknown MEMORIA_LLM_PROVIDER value {other:?} (expected \"local\", \"ollama\", or \"candle\")")),
    }
}

fn resolve_embedding_provider(
    provider_choice: Option<&str>,
    model: Option<String>,
    base_url: Option<String>,
    fastembed_cache_dir: Option<String>,
) -> Result<(BoxedEmbedding, String), String> {
    match provider_choice.unwrap_or("local") {
        "local" => Ok((Box::new(LocalHashEmbeddingProvider::new()), "LocalHashEmbeddingProvider (local, non-AI; see ADR-12)".to_string())),
        "ollama" => {
            #[cfg(feature = "ollama")]
            {
                let model = model.unwrap_or_else(|| "nomic-embed-text".to_string());
                let resolved_base_url = base_url.clone().unwrap_or_else(|| "http://localhost:11434".to_string());
                let config = memoria_core::embedding::EmbeddingConfig { model: model.clone(), base_url, api_key: None, dimensions: None };
                let provider = memoria_core::embedding::OllamaEmbeddingProvider::from_config(config).map_err(|err| err.to_string())?;
                let label = format!("OllamaEmbeddingProvider (model={model}, base_url={resolved_base_url}; see ADR-24)");
                Ok((Box::new(provider), label))
            }
            #[cfg(not(feature = "ollama"))]
            {
                let _ = (model, base_url, fastembed_cache_dir);
                Err("MEMORIA_EMBEDDING_PROVIDER=ollama requires the mcp binary to be built with --features ollama".to_string())
            }
        }
        "fastembed" => {
            #[cfg(feature = "fastembed")]
            {
                let model = model.unwrap_or_else(|| "all-MiniLM-L6-v2".to_string());
                let config = memoria_core::embedding::EmbeddingConfig { model: model.clone(), base_url: None, api_key: None, dimensions: None };
                let cache_dir = fastembed_cache_dir.clone().map(PathBuf::from);
                let provider = memoria_core::embedding::FastEmbedEmbeddingProvider::from_config(&config, cache_dir).map_err(|err| err.to_string())?;
                let cache_label = fastembed_cache_dir.unwrap_or_else(|| ".fastembed_cache (default)".to_string());
                let label = format!("FastEmbedEmbeddingProvider (model={model}, cache_dir={cache_label}; see ADR-32)");
                Ok((Box::new(provider), label))
            }
            #[cfg(not(feature = "fastembed"))]
            {
                let _ = (model, base_url, fastembed_cache_dir);
                Err("MEMORIA_EMBEDDING_PROVIDER=fastembed requires the mcp binary to be built with --features fastembed".to_string())
            }
        }
        other => Err(format!("unknown MEMORIA_EMBEDDING_PROVIDER value {other:?} (expected \"local\", \"ollama\", or \"fastembed\")")),
    }
}

#[derive(Clone, Copy, Default)]
#[cfg_attr(
    not(any(feature = "postgres", feature = "qdrant", feature = "chroma", feature = "milvus")),
    allow(dead_code)
)]
struct NetworkedStoreEnv<'a> {
    url: Option<&'a str>,
    dimension: Option<&'a str>,
    name: Option<&'a str>,
}

#[allow(clippy::too_many_lines)]
fn resolve_vector_store(
    choice: Option<&str>,
    json_snapshot_path: &Path,
    sqlite_path: &Path,
    postgres: NetworkedStoreEnv<'_>,
    qdrant: NetworkedStoreEnv<'_>,
    chroma: NetworkedStoreEnv<'_>,
    milvus: NetworkedStoreEnv<'_>,
) -> Result<(BoxedVectorStore, bool, String), String> {
    match choice.unwrap_or("sqlite") {
        "sqlite" => {
            #[cfg(feature = "sqlite")]
            {
                let store = memoria_core::vector_store::SqliteVectorStore::open(sqlite_path).map_err(|err| err.to_string())?;
                let label = format!("SqliteVectorStore ({}; see ADR-33)", sqlite_path.display());
                Ok((Box::new(store), false, label))
            }
            #[cfg(not(feature = "sqlite"))]
            {
                let _ = sqlite_path;
                Err("the mcp binary must be built with --features sqlite to use the default MEMORIA_VECTOR_STORE=sqlite (see ADR-38); pass MEMORIA_VECTOR_STORE=local to fall back to JSON-snapshot persistence".to_string())
            }
        }
        "local" => Ok((Box::new(load_store(json_snapshot_path)), true, "InMemoryVectorStore + JSON snapshot (ADR-11)".to_string())),
        "postgres" => {
            #[cfg(feature = "postgres")]
            {
                let url = postgres
                    .url
                    .ok_or_else(|| "MEMORIA_VECTOR_STORE=postgres requires MEMORIA_POSTGRES_URL to be set".to_string())?;
                let dimension: usize = postgres
                    .dimension
                    .ok_or_else(|| {
                        "MEMORIA_VECTOR_STORE=postgres requires MEMORIA_POSTGRES_DIMENSION to be set (Postgres's native VECTOR(N) column needs a fixed dimension; see ADR-46)".to_string()
                    })?
                    .parse()
                    .map_err(|_| "MEMORIA_POSTGRES_DIMENSION must be a positive integer".to_string())?;
                let table = postgres.name.unwrap_or("memoria_vectors");
                let store = memoria_core::vector_store::PgVectorStore::open(url, dimension, table).map_err(|err| err.to_string())?;
                let label = format!("PgVectorStore (table {table}, dimension {dimension}; see ADR-46)");
                Ok((Box::new(store), false, label))
            }
            #[cfg(not(feature = "postgres"))]
            {
                let _ = postgres;
                Err("the mcp binary must be built with --features postgres to use MEMORIA_VECTOR_STORE=postgres (see ADR-46)".to_string())
            }
        }
        "qdrant" => {
            #[cfg(feature = "qdrant")]
            {
                let url = qdrant.url.ok_or_else(|| "MEMORIA_VECTOR_STORE=qdrant requires MEMORIA_QDRANT_URL to be set".to_string())?;
                let dimension: usize = qdrant
                    .dimension
                    .ok_or_else(|| {
                        "MEMORIA_VECTOR_STORE=qdrant requires MEMORIA_QDRANT_DIMENSION to be set (Qdrant's own collection schema needs a fixed dimension; see ADR-47)".to_string()
                    })?
                    .parse()
                    .map_err(|_| "MEMORIA_QDRANT_DIMENSION must be a positive integer".to_string())?;
                let collection = qdrant.name.unwrap_or("memoria_vectors");
                let store = memoria_core::vector_store::QdrantVectorStore::open(url, dimension, collection).map_err(|err| err.to_string())?;
                let label = format!("QdrantVectorStore (collection {collection}, dimension {dimension}; see ADR-47)");
                Ok((Box::new(store), false, label))
            }
            #[cfg(not(feature = "qdrant"))]
            {
                let _ = qdrant;
                Err("the mcp binary must be built with --features qdrant to use MEMORIA_VECTOR_STORE=qdrant (see ADR-47)".to_string())
            }
        }
        "chroma" => {
            #[cfg(feature = "chroma")]
            {
                let url = chroma.url.ok_or_else(|| "MEMORIA_VECTOR_STORE=chroma requires MEMORIA_CHROMA_URL to be set".to_string())?;
                let dimension: usize = chroma
                    .dimension
                    .ok_or_else(|| {
                        "MEMORIA_VECTOR_STORE=chroma requires MEMORIA_CHROMA_DIMENSION to be set (memoria validates every vector's dimension client-side; see ADR-48)".to_string()
                    })?
                    .parse()
                    .map_err(|_| "MEMORIA_CHROMA_DIMENSION must be a positive integer".to_string())?;
                let collection = chroma.name.unwrap_or("memoria_vectors");
                let store = memoria_core::vector_store::ChromaVectorStore::open(url, dimension, collection).map_err(|err| err.to_string())?;
                let label = format!("ChromaVectorStore (collection {collection}, dimension {dimension}; see ADR-48)");
                Ok((Box::new(store), false, label))
            }
            #[cfg(not(feature = "chroma"))]
            {
                let _ = chroma;
                Err("the mcp binary must be built with --features chroma to use MEMORIA_VECTOR_STORE=chroma (see ADR-48)".to_string())
            }
        }
        "milvus" => {
            #[cfg(feature = "milvus")]
            {
                let url = milvus.url.ok_or_else(|| "MEMORIA_VECTOR_STORE=milvus requires MEMORIA_MILVUS_URL to be set".to_string())?;
                let dimension: usize = milvus
                    .dimension
                    .ok_or_else(|| {
                        "MEMORIA_VECTOR_STORE=milvus requires MEMORIA_MILVUS_DIMENSION to be set (Milvus's own collection schema needs a fixed dimension; see ADR-49)".to_string()
                    })?
                    .parse()
                    .map_err(|_| "MEMORIA_MILVUS_DIMENSION must be a positive integer".to_string())?;
                let collection = milvus.name.unwrap_or("memoria_vectors");
                let store = memoria_core::vector_store::MilvusVectorStore::open(url, dimension, collection).map_err(|err| err.to_string())?;
                let label = format!("MilvusVectorStore (collection {collection}, dimension {dimension}; see ADR-49)");
                Ok((Box::new(store), false, label))
            }
            #[cfg(not(feature = "milvus"))]
            {
                let _ = milvus;
                Err("the mcp binary must be built with --features milvus to use MEMORIA_VECTOR_STORE=milvus (see ADR-49)".to_string())
            }
        }
        other => Err(format!("unknown MEMORIA_VECTOR_STORE value {other:?} (expected \"sqlite\", \"local\", \"postgres\", \"qdrant\", \"chroma\", or \"milvus\")")),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mcp_secret = std::env::var("MEMORIA_MCP_SECRET").ok().filter(|value| !value.is_empty());

    if let Ok(server_url) = std::env::var("MEMORIA_SERVER_URL") {
        let api_key = std::env::var("MEMORIA_API_KEY").ok().filter(|value| !value.is_empty());
        eprintln!("mode: remote (server={server_url})");
        let backend = Backend::Remote(RemoteClient::new(&server_url, api_key));
        let mut server = MemoriaMcpServer::new(backend, mcp_secret, PathBuf::from("/dev/null"), PathBuf::from("/dev/null"), false);
        server.persist_history = false;
        let service = server.serve(stdio()).await?;
        service.waiting().await?;
        return Ok(());
    }

    let llm_provider_choice = std::env::var("MEMORIA_LLM_PROVIDER").ok();
    let (llm_provider, llm_label) = resolve_llm_provider(
        llm_provider_choice.as_deref(),
        std::env::var("MEMORIA_LLM_MODEL").ok(),
        std::env::var("MEMORIA_LLM_BASE_URL").ok(),
        std::env::var("MEMORIA_CANDLE_CACHE_DIR").ok(),
    )
    .map_err(|message| -> Box<dyn std::error::Error> { message.into() })?;

    let (embedding_provider, embedding_label) = resolve_embedding_provider(
        std::env::var("MEMORIA_EMBEDDING_PROVIDER").ok().as_deref(),
        std::env::var("MEMORIA_EMBEDDING_MODEL").ok(),
        std::env::var("MEMORIA_EMBEDDING_BASE_URL").ok(),
        std::env::var("MEMORIA_FASTEMBED_CACHE_DIR").ok(),
    )
    .map_err(|message| -> Box<dyn std::error::Error> { message.into() })?;

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let store_path = resolve_store_path(std::env::var("MEMORIA_STORE_PATH").ok().as_deref(), &home);
    let sqlite_path = resolve_sqlite_path(std::env::var("MEMORIA_SQLITE_PATH").ok().as_deref(), &home);
    let postgres_url = std::env::var("MEMORIA_POSTGRES_URL").ok();
    let postgres_dimension = std::env::var("MEMORIA_POSTGRES_DIMENSION").ok();
    let postgres_table = std::env::var("MEMORIA_POSTGRES_TABLE").ok();
    let qdrant_url = std::env::var("MEMORIA_QDRANT_URL").ok();
    let qdrant_dimension = std::env::var("MEMORIA_QDRANT_DIMENSION").ok();
    let qdrant_collection = std::env::var("MEMORIA_QDRANT_COLLECTION").ok();
    let chroma_url = std::env::var("MEMORIA_CHROMA_URL").ok();
    let chroma_dimension = std::env::var("MEMORIA_CHROMA_DIMENSION").ok();
    let chroma_collection = std::env::var("MEMORIA_CHROMA_COLLECTION").ok();
    let milvus_url = std::env::var("MEMORIA_MILVUS_URL").ok();
    let milvus_dimension = std::env::var("MEMORIA_MILVUS_DIMENSION").ok();
    let milvus_collection = std::env::var("MEMORIA_MILVUS_COLLECTION").ok();
    let postgres_env = NetworkedStoreEnv { url: postgres_url.as_deref(), dimension: postgres_dimension.as_deref(), name: postgres_table.as_deref() };
    let qdrant_env = NetworkedStoreEnv { url: qdrant_url.as_deref(), dimension: qdrant_dimension.as_deref(), name: qdrant_collection.as_deref() };
    let chroma_env = NetworkedStoreEnv { url: chroma_url.as_deref(), dimension: chroma_dimension.as_deref(), name: chroma_collection.as_deref() };
    let milvus_env = NetworkedStoreEnv { url: milvus_url.as_deref(), dimension: milvus_dimension.as_deref(), name: milvus_collection.as_deref() };
    let (vector_store, persist_json_snapshot, vector_store_label) = resolve_vector_store(
        std::env::var("MEMORIA_VECTOR_STORE").ok().as_deref(),
        &store_path,
        &sqlite_path,
        postgres_env,
        qdrant_env,
        chroma_env,
        milvus_env,
    )
    .map_err(|message| -> Box<dyn std::error::Error> { message.into() })?;

    eprintln!("llm provider: {llm_label}");
    eprintln!("embedding provider: {embedding_label}");
    eprintln!("vector store: {vector_store_label}");
    if persist_json_snapshot {
        eprintln!("store: {}", store_path.display());
    }
    let history_path = resolve_history_path(&store_path);
    eprintln!("history: {}", history_path.display());

    let memory = Arc::new(Memory::new(llm_provider, embedding_provider, vector_store));
    memory.load_history_snapshot(load_history(&history_path));
    let server = MemoriaMcpServer::new(Backend::Local(memory), mcp_secret, store_path, history_path, persist_json_snapshot);

    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use memoria_core::llm::{LocalSentenceLlmProvider, Message, Role};

    fn test_memory() -> SharedMemory {
        let llm: BoxedLlm = Box::new(LocalSentenceLlmProvider::new());
        let embedding: BoxedEmbedding = Box::new(LocalHashEmbeddingProvider::new());
        let vector_store: BoxedVectorStore = Box::new(InMemoryVectorStore::new());
        Arc::new(Memory::new(llm, embedding, vector_store))
    }

    fn add_fact(memory: &SharedMemory, content: &str, user_id: &str) -> String {
        let mut scope = HashMap::new();
        scope.insert("user_id".to_string(), user_id.to_string());
        let ids = memory.add(&[Message::new(Role::User, content)], scope, false).expect("add should succeed");
        ids.into_iter().next().expect("expected at least one id")
    }

    struct DescribingLlmProvider;

    impl memoria_core::llm::LlmProvider for DescribingLlmProvider {
        fn complete(&self, _messages: &[Message]) -> Result<memoria_core::llm::Completion, memoria_core::llm::LlmError> {
            Ok(memoria_core::llm::Completion { content: "A photo of a red bicycle.".to_string() })
        }
    }

    fn test_memory_with_llm(llm: BoxedLlm) -> SharedMemory {
        let embedding: BoxedEmbedding = Box::new(LocalHashEmbeddingProvider::new());
        let vector_store: BoxedVectorStore = Box::new(InMemoryVectorStore::new());
        Arc::new(Memory::new(llm, embedding, vector_store))
    }

    fn test_server(memory: SharedMemory, mcp_secret: Option<String>) -> MemoriaMcpServer {
        let mut server = MemoriaMcpServer::new(Backend::Local(memory), mcp_secret, PathBuf::from("/dev/null"), PathBuf::from("/dev/null"), false);
        server.persist_history = false;
        server
    }

    fn local_memory(server: &MemoriaMcpServer) -> &SharedMemory {
        match &server.backend {
            Backend::Local(memory) => memory,
            Backend::Remote(_) => panic!("expected a local backend in this test"),
        }
    }

    #[test]
    fn scope_from_optional_builds_only_the_provided_keys() {
        let scope = scope_from_optional(Some("alice".to_string()), None, None);
        assert_eq!(scope.len(), 1);
        assert_eq!(scope.get("user_id"), Some(&"alice".to_string()));
    }

    #[test]
    fn check_secret_passes_when_none_is_configured() {
        assert!(check_secret(None, None).is_ok());
    }

    #[test]
    fn check_secret_rejects_a_missing_value_when_one_is_configured() {
        assert!(check_secret(Some("s3cret"), None).is_err());
    }

    #[test]
    fn check_secret_rejects_a_wrong_value() {
        assert!(check_secret(Some("s3cret"), Some("wrong")).is_err());
    }

    #[test]
    fn check_secret_accepts_the_matching_value() {
        assert!(check_secret(Some("s3cret"), Some("s3cret")).is_ok());
    }

    #[tokio::test]
    async fn search_memories_finds_a_real_stored_fact_within_scope() {
        let memory = test_memory();
        add_fact(&memory, "Alice is an engineer.", "alice");
        let server = test_server(memory, None);
        let request = SearchMemoriesRequest {
            query: "engineer".to_string(),
            user_id: Some("alice".to_string()),
            agent_id: None,
            run_id: None,
            top_k: 10,
            threshold: None,
            show_expired: false,
            explain: false,
            secret: None,
        };
        let Json(SearchMemoriesResult { results }) = server.search_memories(Parameters(request)).await.expect("search should succeed");
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn search_memories_with_explain_true_returns_a_real_score_details_breakdown() {
        let memory = test_memory();
        add_fact(&memory, "Alice is an engineer.", "alice");
        let server = test_server(memory, None);
        let request = SearchMemoriesRequest {
            query: "engineer".to_string(),
            user_id: Some("alice".to_string()),
            agent_id: None,
            run_id: None,
            top_k: 10,
            threshold: None,
            show_expired: false,
            explain: true,
            secret: None,
        };
        let Json(SearchMemoriesResult { results }) = server.search_memories(Parameters(request)).await.expect("search should succeed");
        let details = results[0].score_details.as_ref().expect("explain=true must return score_details");
        assert!((details.final_score - results[0].score).abs() < 1e-6, "final_score must equal the real returned score");
    }

    #[tokio::test]
    async fn search_memories_rejects_a_missing_secret_when_one_is_configured() {
        let memory = test_memory();
        let server = test_server(memory, Some("s3cret".to_string()));
        let request = SearchMemoriesRequest {
            query: "anything".to_string(),
            user_id: Some("alice".to_string()),
            agent_id: None,
            run_id: None,
            top_k: 10,
            threshold: None,
            show_expired: false,
            explain: false,
            secret: None,
        };
        let result = server.search_memories(Parameters(request)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn get_memory_returns_a_real_stored_record_by_id() {
        let memory = test_memory();
        let id = add_fact(&memory, "Bob likes tea.", "bob");
        let server = test_server(memory, None);
        let request = GetMemoryRequest { id: id.clone(), secret: None };
        let Json(record) = server.get_memory(Parameters(request)).await.expect("get should succeed");
        assert_eq!(record.id, id);
    }

    #[tokio::test]
    async fn get_memory_returns_a_tool_error_for_an_unknown_id() {
        let memory = test_memory();
        let server = test_server(memory, None);
        let request = GetMemoryRequest { id: "never-existed".to_string(), secret: None };
        let result = server.get_memory(Parameters(request)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn get_memories_scopes_results_to_the_requested_user_only() {
        let memory = test_memory();
        add_fact(&memory, "Alice is an engineer.", "alice");
        add_fact(&memory, "Carol is a designer.", "carol");
        let server = test_server(memory, None);
        let request =
            GetMemoriesRequest { user_id: Some("alice".to_string()), agent_id: None, run_id: None, offset: 0, limit: 50, show_expired: false, secret: None };
        let Json(GetMemoriesResult { memories }) = server.get_memories(Parameters(request)).await.expect("list should succeed");
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].payload.get("user_id"), Some(&"alice".to_string()));
    }

    #[tokio::test]
    async fn get_memories_without_a_scope_is_rejected() {
        let memory = test_memory();
        add_fact(&memory, "Alice is an engineer.", "alice");
        let server = test_server(memory, None);
        let request = GetMemoriesRequest { user_id: None, agent_id: None, run_id: None, offset: 0, limit: 50, show_expired: false, secret: None };
        let result = server.get_memories(Parameters(request)).await;
        assert!(result.is_err(), "get_memories with no user_id/agent_id/run_id must be rejected, not return every record unscoped");
    }

    #[tokio::test]
    async fn memory_history_reports_real_add_and_delete_events_in_order() {
        let memory = test_memory();
        let id = add_fact(&memory, "Dana runs marathons.", "dana");
        memory.delete(&id).expect("delete should succeed");
        let server = test_server(memory, None);
        let request = MemoryHistoryRequest { id, offset: 0, limit: 100, secret: None };
        let Json(MemoryHistoryResult { entries }) = server.memory_history(Parameters(request)).await.expect("history should succeed");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].event, "added");
        assert_eq!(entries[1].event, "deleted");
    }

    #[tokio::test]
    async fn add_memory_stores_content_verbatim_when_infer_is_false() {
        let memory = test_memory();
        let server = test_server(memory, None);
        let request = AddMemoryRequest {
            content: "Erin runs a bakery.".to_string(),
            user_id: Some("erin".to_string()),
            agent_id: None,
            run_id: None,
            infer: false,
            images: Vec::new(),
            secret: None,
        };
        let Json(AddMemoryResult { ids }) = server.add_memory(Parameters(request)).await.expect("add should succeed");
        assert_eq!(ids.len(), 1);
        let stored = local_memory(&server).get(&ids[0]).expect("get should succeed").expect("expected a record");
        assert_eq!(stored.payload.get("content"), Some(&"Erin runs a bakery.".to_string()));
    }

    #[tokio::test]
    async fn add_memory_with_images_describes_it_before_storage() {
        let memory = test_memory_with_llm(Box::new(DescribingLlmProvider));
        let server = test_server(memory, None);
        let request = AddMemoryRequest {
            content: String::new(),
            user_id: Some("erin".to_string()),
            agent_id: None,
            run_id: None,
            infer: false,
            images: vec!["base64imagedata".to_string()],
            secret: None,
        };
        let Json(AddMemoryResult { ids }) = server.add_memory(Parameters(request)).await.expect("add should succeed");
        let stored = local_memory(&server).get(&ids[0]).expect("get should succeed").expect("expected a record");
        assert_eq!(
            stored.payload.get("content"),
            Some(&"A photo of a red bicycle.".to_string()),
            "an image-bearing add must be described before storage"
        );
    }

    #[tokio::test]
    async fn add_memory_without_images_is_unaffected() {
        let memory = test_memory();
        let server = test_server(memory, None);
        let request = AddMemoryRequest {
            content: "Frank likes tea now.".to_string(),
            user_id: Some("frank".to_string()),
            agent_id: None,
            run_id: None,
            infer: false,
            images: Vec::new(),
            secret: None,
        };
        let Json(AddMemoryResult { ids }) = server.add_memory(Parameters(request)).await.expect("add should succeed");
        let stored = local_memory(&server).get(&ids[0]).expect("get should succeed").expect("expected a record");
        assert_eq!(stored.payload.get("content"), Some(&"Frank likes tea now.".to_string()));
    }

    #[tokio::test]
    async fn add_memory_rejects_a_missing_secret_when_one_is_configured() {
        let memory = test_memory();
        let server = test_server(memory, Some("s3cret".to_string()));
        let request =
            AddMemoryRequest { content: "anything".to_string(), user_id: Some("erin".to_string()), agent_id: None, run_id: None, infer: false, images: Vec::new(), secret: None };
        assert!(server.add_memory(Parameters(request)).await.is_err());
    }

    #[tokio::test]
    async fn update_memory_changes_content_and_returns_the_updated_record() {
        let memory = test_memory();
        let id = add_fact(&memory, "Frank likes coffee.", "frank");
        let server = test_server(memory, None);
        let request = UpdateMemoryRequest { id: id.clone(), content: Some("Frank likes tea now.".to_string()), metadata: None, secret: None };
        let Json(record) = server.update_memory(Parameters(request)).await.expect("update should succeed");
        assert_eq!(record.payload.get("content"), Some(&"Frank likes tea now.".to_string()));
    }

    #[tokio::test]
    async fn update_memory_returns_a_tool_error_for_an_unknown_id() {
        let memory = test_memory();
        let server = test_server(memory, None);
        let request = UpdateMemoryRequest { id: "never-existed".to_string(), content: Some("x".to_string()), metadata: None, secret: None };
        assert!(server.update_memory(Parameters(request)).await.is_err());
    }

    #[tokio::test]
    async fn delete_memory_removes_a_real_record() {
        let memory = test_memory();
        let id = add_fact(&memory, "Grace paints landscapes.", "grace");
        let server = test_server(memory, None);
        let request = DeleteMemoryRequest { id: id.clone(), secret: None };
        let Json(DeleteMemoryResult { deleted }) = server.delete_memory(Parameters(request)).await.expect("delete should succeed");
        assert!(deleted);
        assert!(local_memory(&server).get(&id).expect("get should succeed").is_none());
    }

    #[tokio::test]
    async fn delete_all_memories_rejects_an_empty_scope() {
        let memory = test_memory();
        let server = test_server(memory, None);
        let request = DeleteAllMemoriesRequest { user_id: None, agent_id: None, run_id: None, secret: None };
        assert!(server.delete_all_memories(Parameters(request)).await.is_err());
    }

    #[tokio::test]
    async fn delete_all_memories_deletes_only_the_scoped_records() {
        let memory = test_memory();
        add_fact(&memory, "Henry codes in Rust.", "henry");
        add_fact(&memory, "Henry also bikes.", "henry");
        add_fact(&memory, "Iris paints.", "iris");
        let server = test_server(memory, None);
        let request = DeleteAllMemoriesRequest { user_id: Some("henry".to_string()), agent_id: None, run_id: None, secret: None };
        let Json(result) = server.delete_all_memories(Parameters(request)).await.expect("delete_all should succeed");
        assert_eq!(result.deleted, 2);
        let remaining = local_memory(&server).list_all(0, usize::MAX, true, None).expect("list should succeed");
        assert_eq!(remaining.len(), 1);
    }

    #[tokio::test]
    async fn list_entities_groups_real_records_by_scope_field() {
        let memory = test_memory();
        add_fact(&memory, "Jack is an engineer.", "jack");
        add_fact(&memory, "Jack likes hiking.", "jack");
        let server = test_server(memory, None);
        let request = ListEntitiesRequest { secret: None };
        let Json(ListEntitiesResult { entities }) = server.list_entities(Parameters(request)).await.expect("list_entities should succeed");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].entity_type, "user_id");
        assert_eq!(entities[0].entity_id, "jack");
        assert_eq!(entities[0].memory_count, 2);
    }

    #[test]
    fn write_atomically_round_trips_real_content() {
        let dir = std::env::temp_dir().join(format!("memoria-mcp-atomic-test-{}", std::process::id()));
        let path = dir.join("store.json");
        write_atomically(&path, b"hello").expect("write should succeed");
        let contents = std::fs::read(&path).expect("read should succeed");
        assert_eq!(contents, b"hello");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn add_memory_persists_a_json_snapshot_when_configured() {
        let dir = std::env::temp_dir().join(format!("memoria-mcp-snapshot-test-{}", std::process::id()));
        let store_path = dir.join("store.json");
        let history_path = dir.join("history.json");
        let memory = test_memory();
        let server = MemoriaMcpServer::new(Backend::Local(memory), None, store_path.clone(), history_path, true);
        let request =
            AddMemoryRequest { content: "Kim leads the platform team.".to_string(), user_id: Some("kim".to_string()), agent_id: None, run_id: None, infer: false, images: Vec::new(), secret: None };
        server.add_memory(Parameters(request)).await.expect("add should succeed");
        let saved = std::fs::read_to_string(&store_path).expect("snapshot file should exist");
        assert!(saved.contains("Kim leads the platform team."));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_vector_store_defaults_to_sqlite() {
        let dir = std::env::temp_dir().join(format!("memoria-mcp-default-store-test-{}", std::process::id()));
        let (_, persist_json_snapshot, label) =
            resolve_vector_store(None, &dir.join("unused.json"), &dir.join("default.db"), NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), NetworkedStoreEnv::default()).expect("expected a store");
        assert!(!persist_json_snapshot, "the sqlite default persists itself; it must not also write a JSON snapshot");
        assert!(label.contains("SqliteVectorStore"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_history_path_is_a_sibling_of_the_store_path() {
        let path = resolve_history_path(&PathBuf::from("/home/alice/.memoria/mcp-store.json"));
        assert_eq!(path, PathBuf::from("/home/alice/.memoria/history.json"));
    }

    #[test]
    fn load_history_with_no_file_present_is_empty() {
        let dir = std::env::temp_dir().join(format!("memoria-mcp-history-load-test-{}", std::process::id()));
        assert!(load_history(&dir.join("does-not-exist.json")).is_empty());
    }

    #[tokio::test]
    async fn add_memory_and_delete_memory_persist_history_when_configured() {
        let dir = std::env::temp_dir().join(format!("memoria-mcp-history-test-{}", std::process::id()));
        let history_path = dir.join("history.json");
        let memory = test_memory();
        let server = MemoriaMcpServer::new(Backend::Local(memory), None, PathBuf::from("/dev/null"), history_path.clone(), false);

        let add_request =
            AddMemoryRequest { content: "Liam manages infrastructure.".to_string(), user_id: Some("liam".to_string()), agent_id: None, run_id: None, infer: false, images: Vec::new(), secret: None };
        let Json(AddMemoryResult { ids }) = server.add_memory(Parameters(add_request)).await.expect("add should succeed");
        let id = ids.into_iter().next().expect("expected an id");
        server.delete_memory(Parameters(DeleteMemoryRequest { id: id.clone(), secret: None })).await.expect("delete should succeed");

        let saved = load_history(&history_path);
        let entries = saved.get(&id).expect("expected history for this id");
        assert_eq!(entries.len(), 2);

        let fresh_memory = test_memory();
        fresh_memory.load_history_snapshot(saved);
        let restored_entries = fresh_memory.history(&id, 0, usize::MAX).expect("history should succeed");
        assert_eq!(restored_entries.len(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_store_path_defaults_to_an_mcp_specific_file() {
        let path = resolve_store_path(None, "/home/alice");
        assert_eq!(path, PathBuf::from("/home/alice/.memoria/mcp-store.json"));
    }

    #[test]
    fn resolve_sqlite_path_defaults_to_an_mcp_specific_file() {
        let path = resolve_sqlite_path(None, "/home/alice");
        assert_eq!(path, PathBuf::from("/home/alice/.memoria/mcp-store.db"));
    }

    #[test]
    fn resolve_vector_store_unknown_choice_is_a_clear_error() {
        let dir = std::env::temp_dir();
        let result = resolve_vector_store(Some("bogus"), &dir.join("unused.json"), &dir.join("unused.db"), NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), NetworkedStoreEnv::default());
        assert!(result.is_err());
    }

    #[cfg(not(feature = "postgres"))]
    #[test]
    fn resolve_vector_store_postgres_choice_fails_clearly_without_the_feature() {
        let dir = std::env::temp_dir();
        let postgres = NetworkedStoreEnv { url: Some("postgres://x"), dimension: Some("2"), name: None };
        let result = resolve_vector_store(Some("postgres"), &dir.join("unused.json"), &dir.join("unused.db"), postgres, NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), NetworkedStoreEnv::default());
        assert!(result.is_err());
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn resolve_vector_store_postgres_choice_requires_url() {
        let dir = std::env::temp_dir();
        let postgres = NetworkedStoreEnv { url: None, dimension: Some("2"), name: None };
        let err = resolve_vector_store(Some("postgres"), &dir.join("unused.json"), &dir.join("unused.db"), postgres, NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), NetworkedStoreEnv::default()).err().expect("expected an error");
        assert!(err.contains("MEMORIA_POSTGRES_URL"), "got: {err}");
    }

    #[cfg(feature = "postgres")]
    #[test]
    fn resolve_vector_store_postgres_choice_requires_dimension() {
        let dir = std::env::temp_dir();
        let postgres = NetworkedStoreEnv { url: Some("postgres://x"), dimension: None, name: None };
        let err = resolve_vector_store(Some("postgres"), &dir.join("unused.json"), &dir.join("unused.db"), postgres, NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), NetworkedStoreEnv::default()).err().expect("expected an error");
        assert!(err.contains("MEMORIA_POSTGRES_DIMENSION"), "got: {err}");
    }

    #[cfg(feature = "postgres")]
    #[test]
    #[ignore = "requires a real Postgres+pgvector instance reachable at MEMORIA_TEST_POSTGRES_URL"]
    fn resolve_vector_store_postgres_choice_builds_and_does_not_persist_json_snapshot() {
        let url = std::env::var("MEMORIA_TEST_POSTGRES_URL").expect("MEMORIA_TEST_POSTGRES_URL must be set for this test");
        let dir = std::env::temp_dir();
        let table = format!("memoria_mcp_resolve_vector_store_test_{}", std::process::id());
        let postgres = NetworkedStoreEnv { url: Some(&url), dimension: Some("2"), name: Some(&table) };
        let (_, persist_json_snapshot, label) =
            resolve_vector_store(Some("postgres"), &dir.join("unused.json"), &dir.join("unused.db"), postgres, NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), NetworkedStoreEnv::default()).expect("expected a store");
        assert!(label.contains("PgVectorStore"), "got: {label}");
        assert!(!persist_json_snapshot, "the postgres backend persists itself; it must not also write a JSON snapshot");
    }

    #[cfg(not(feature = "qdrant"))]
    #[test]
    fn resolve_vector_store_qdrant_choice_fails_clearly_without_the_feature() {
        let dir = std::env::temp_dir();
        let qdrant = NetworkedStoreEnv { url: Some("http://x"), dimension: Some("2"), name: None };
        let result = resolve_vector_store(Some("qdrant"), &dir.join("unused.json"), &dir.join("unused.db"), NetworkedStoreEnv::default(), qdrant, NetworkedStoreEnv::default(), NetworkedStoreEnv::default());
        assert!(result.is_err());
    }

    #[cfg(feature = "qdrant")]
    #[test]
    fn resolve_vector_store_qdrant_choice_requires_url() {
        let dir = std::env::temp_dir();
        let qdrant = NetworkedStoreEnv { url: None, dimension: Some("2"), name: None };
        let err = resolve_vector_store(Some("qdrant"), &dir.join("unused.json"), &dir.join("unused.db"), NetworkedStoreEnv::default(), qdrant, NetworkedStoreEnv::default(), NetworkedStoreEnv::default()).err().expect("expected an error");
        assert!(err.contains("MEMORIA_QDRANT_URL"), "got: {err}");
    }

    #[cfg(feature = "qdrant")]
    #[test]
    fn resolve_vector_store_qdrant_choice_requires_dimension() {
        let dir = std::env::temp_dir();
        let qdrant = NetworkedStoreEnv { url: Some("http://x"), dimension: None, name: None };
        let err = resolve_vector_store(Some("qdrant"), &dir.join("unused.json"), &dir.join("unused.db"), NetworkedStoreEnv::default(), qdrant, NetworkedStoreEnv::default(), NetworkedStoreEnv::default()).err().expect("expected an error");
        assert!(err.contains("MEMORIA_QDRANT_DIMENSION"), "got: {err}");
    }

    #[cfg(feature = "qdrant")]
    #[test]
    #[ignore = "requires a real Qdrant instance reachable at MEMORIA_TEST_QDRANT_URL"]
    fn resolve_vector_store_qdrant_choice_builds_and_does_not_persist_json_snapshot() {
        let url = std::env::var("MEMORIA_TEST_QDRANT_URL").expect("MEMORIA_TEST_QDRANT_URL must be set for this test");
        let dir = std::env::temp_dir();
        let collection = format!("memoria_mcp_resolve_vector_store_test_{}", std::process::id());
        let qdrant = NetworkedStoreEnv { url: Some(&url), dimension: Some("2"), name: Some(&collection) };
        let (_, persist_json_snapshot, label) =
            resolve_vector_store(Some("qdrant"), &dir.join("unused.json"), &dir.join("unused.db"), NetworkedStoreEnv::default(), qdrant, NetworkedStoreEnv::default(), NetworkedStoreEnv::default()).expect("expected a store");
        assert!(label.contains("QdrantVectorStore"), "got: {label}");
        assert!(!persist_json_snapshot, "the qdrant backend persists itself; it must not also write a JSON snapshot");
    }

    #[cfg(not(feature = "chroma"))]
    #[test]
    fn resolve_vector_store_chroma_choice_fails_clearly_without_the_feature() {
        let dir = std::env::temp_dir();
        let chroma = NetworkedStoreEnv { url: Some("http://x"), dimension: Some("2"), name: None };
        let result = resolve_vector_store(Some("chroma"), &dir.join("unused.json"), &dir.join("unused.db"), NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), chroma, NetworkedStoreEnv::default());
        assert!(result.is_err());
    }

    #[cfg(feature = "chroma")]
    #[test]
    fn resolve_vector_store_chroma_choice_requires_url() {
        let dir = std::env::temp_dir();
        let chroma = NetworkedStoreEnv { url: None, dimension: Some("2"), name: None };
        let err =
            resolve_vector_store(Some("chroma"), &dir.join("unused.json"), &dir.join("unused.db"), NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), chroma, NetworkedStoreEnv::default()).err().expect("expected an error");
        assert!(err.contains("MEMORIA_CHROMA_URL"), "got: {err}");
    }

    #[cfg(feature = "chroma")]
    #[test]
    fn resolve_vector_store_chroma_choice_requires_dimension() {
        let dir = std::env::temp_dir();
        let chroma = NetworkedStoreEnv { url: Some("http://x"), dimension: None, name: None };
        let err =
            resolve_vector_store(Some("chroma"), &dir.join("unused.json"), &dir.join("unused.db"), NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), chroma, NetworkedStoreEnv::default()).err().expect("expected an error");
        assert!(err.contains("MEMORIA_CHROMA_DIMENSION"), "got: {err}");
    }

    #[cfg(feature = "chroma")]
    #[test]
    #[ignore = "requires a real Chroma instance reachable at MEMORIA_TEST_CHROMA_URL"]
    fn resolve_vector_store_chroma_choice_builds_and_does_not_persist_json_snapshot() {
        let url = std::env::var("MEMORIA_TEST_CHROMA_URL").expect("MEMORIA_TEST_CHROMA_URL must be set for this test");
        let dir = std::env::temp_dir();
        let collection = format!("memoria_mcp_resolve_vector_store_test_{}", std::process::id());
        let chroma = NetworkedStoreEnv { url: Some(&url), dimension: Some("2"), name: Some(&collection) };
        let (_, persist_json_snapshot, label) = resolve_vector_store(
            Some("chroma"),
            &dir.join("unused.json"),
            &dir.join("unused.db"),
            NetworkedStoreEnv::default(),
            NetworkedStoreEnv::default(),
            chroma,
            NetworkedStoreEnv::default(),
        )
        .expect("expected a store");
        assert!(label.contains("ChromaVectorStore"), "got: {label}");
        assert!(!persist_json_snapshot, "the chroma backend persists itself; it must not also write a JSON snapshot");
    }

    #[cfg(not(feature = "milvus"))]
    #[test]
    fn resolve_vector_store_milvus_choice_fails_clearly_without_the_feature() {
        let dir = std::env::temp_dir();
        let milvus = NetworkedStoreEnv { url: Some("http://x"), dimension: Some("2"), name: None };
        let result = resolve_vector_store(Some("milvus"), &dir.join("unused.json"), &dir.join("unused.db"), NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), milvus);
        assert!(result.is_err());
    }

    #[cfg(feature = "milvus")]
    #[test]
    fn resolve_vector_store_milvus_choice_requires_url() {
        let dir = std::env::temp_dir();
        let milvus = NetworkedStoreEnv { url: None, dimension: Some("2"), name: None };
        let err = resolve_vector_store(Some("milvus"), &dir.join("unused.json"), &dir.join("unused.db"), NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), milvus)
            .err()
            .expect("expected an error");
        assert!(err.contains("MEMORIA_MILVUS_URL"), "got: {err}");
    }

    #[cfg(feature = "milvus")]
    #[test]
    fn resolve_vector_store_milvus_choice_requires_dimension() {
        let dir = std::env::temp_dir();
        let milvus = NetworkedStoreEnv { url: Some("http://x"), dimension: None, name: None };
        let err = resolve_vector_store(Some("milvus"), &dir.join("unused.json"), &dir.join("unused.db"), NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), NetworkedStoreEnv::default(), milvus)
            .err()
            .expect("expected an error");
        assert!(err.contains("MEMORIA_MILVUS_DIMENSION"), "got: {err}");
    }

    #[cfg(feature = "milvus")]
    #[test]
    #[ignore = "requires a real Milvus instance reachable at MEMORIA_TEST_MILVUS_URL"]
    fn resolve_vector_store_milvus_choice_builds_and_does_not_persist_json_snapshot() {
        let url = std::env::var("MEMORIA_TEST_MILVUS_URL").expect("MEMORIA_TEST_MILVUS_URL must be set for this test");
        let dir = std::env::temp_dir();
        let collection = format!("memoria_mcp_resolve_vector_store_test_{}", std::process::id());
        let milvus = NetworkedStoreEnv { url: Some(&url), dimension: Some("2"), name: Some(&collection) };
        let (_, persist_json_snapshot, label) = resolve_vector_store(
            Some("milvus"),
            &dir.join("unused.json"),
            &dir.join("unused.db"),
            NetworkedStoreEnv::default(),
            NetworkedStoreEnv::default(),
            NetworkedStoreEnv::default(),
            milvus,
        )
        .expect("expected a store");
        assert!(label.contains("MilvusVectorStore"), "got: {label}");
        assert!(!persist_json_snapshot, "the milvus backend persists itself; it must not also write a JSON snapshot");
    }

    #[test]
    fn resolve_llm_provider_unknown_choice_is_a_clear_error() {
        assert!(resolve_llm_provider(Some("bogus"), None, None, None).is_err());
    }

    #[cfg(not(feature = "candle"))]
    #[test]
    fn resolve_llm_provider_candle_choice_fails_clearly_without_the_feature() {
        assert!(resolve_llm_provider(Some("candle"), None, None, None).is_err());
    }

    #[cfg(feature = "candle")]
    #[test]
    #[ignore = "downloads a real ~430MB GGUF model + tokenizer on first run; needs real network access"]
    fn resolve_llm_provider_candle_choice_builds_with_defaults() {
        let cache_dir = std::env::var("MEMORIA_TEST_CANDLE_CACHE_DIR").ok();
        let (_, label) = resolve_llm_provider(Some("candle"), None, None, cache_dir).expect("expected a provider");
        assert!(label.contains("qwen2.5-0.5b-instruct-q4_0"), "got: {label}");
    }

    #[test]
    fn resolve_embedding_provider_unknown_choice_is_a_clear_error() {
        assert!(resolve_embedding_provider(Some("bogus"), None, None, None).is_err());
    }

    struct CapturedRequest {
        method: String,
        path: String,
        body: String,
    }

    fn spawn_fake_server(status: u16, response_body: &str) -> (String, std::sync::mpsc::Receiver<CapturedRequest>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind should succeed");
        let addr = listener.local_addr().expect("local_addr should succeed");
        let (tx, rx) = std::sync::mpsc::channel();
        let response_body = response_body.to_string();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                let text = String::from_utf8_lossy(&buf[..n]).to_string();
                let request_line = text.lines().next().unwrap_or_default().to_string();
                let mut parts = request_line.split_whitespace();
                let method = parts.next().unwrap_or_default().to_string();
                let path = parts.next().unwrap_or_default().to_string();
                let body_start = text.find("\r\n\r\n").map_or(text.len(), |i| i + 4);
                let body = text[body_start..].to_string();
                let _ = tx.send(CapturedRequest { method, path, body });
                let response = format!(
                    "HTTP/1.1 {status} status\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                    response_body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (format!("http://{addr}"), rx)
    }

    fn spawn_routed_fake_server(routes: Vec<(&'static str, u16, &'static str)>) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind should succeed");
        let addr = listener.local_addr().expect("local_addr should succeed");
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut buf = [0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                let text = String::from_utf8_lossy(&buf[..n]).to_string();
                let path = text.lines().next().unwrap_or_default().split_whitespace().nth(1).unwrap_or_default().to_string();
                let route = routes.iter().find(|(route_path, _, _)| path.starts_with(route_path) || path.split('?').next() == Some(route_path));
                let (status, body) = route.map_or((404, ""), |(_, status, body)| (*status, *body));
                let response = format!("HTTP/1.1 {status} status\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                let _ = stream.write_all(response.as_bytes());
            }
        });
        format!("http://{addr}")
    }

    fn remote_test_server(base_url: &str, mcp_secret: Option<String>) -> MemoriaMcpServer {
        let backend = Backend::Remote(RemoteClient::new(base_url, None));
        let mut server = MemoriaMcpServer::new(backend, mcp_secret, PathBuf::from("/dev/null"), PathBuf::from("/dev/null"), false);
        server.persist_history = false;
        server
    }

    #[tokio::test]
    async fn add_memory_remote_sends_the_real_expected_request_body() {
        let (base_url, rx) = spawn_fake_server(201, r#"{"ids":["rec-1"]}"#);
        let server = remote_test_server(&base_url, None);
        let request = AddMemoryRequest { content: "hello".to_string(), user_id: Some("alice".to_string()), agent_id: None, run_id: None, infer: false, images: vec![], secret: None };
        let Json(result) = server.add_memory(Parameters(request)).await.expect("add should succeed");
        assert_eq!(result.ids, vec!["rec-1".to_string()]);
        let captured = rx.recv_timeout(std::time::Duration::from_secs(2)).expect("expected a captured request");
        assert_eq!(captured.method, "POST");
        assert_eq!(captured.path, "/memories");
        let body: serde_json::Value = serde_json::from_str(&captured.body).expect("body must be real JSON");
        assert_eq!(body["content"], "hello");
        assert_eq!(body["user_id"], "alice");
        assert_eq!(body["infer"], false);
    }

    #[tokio::test]
    async fn search_memories_remote_sends_the_real_expected_request_body_and_parses_a_real_response() {
        let (base_url, rx) = spawn_fake_server(200, r#"{"results":[{"id":"rec-1","score":0.5,"payload":{"content":"hi"},"score_details":null}]}"#);
        let server = remote_test_server(&base_url, None);
        let request = SearchMemoriesRequest {
            query: "hi".to_string(),
            user_id: Some("alice".to_string()),
            agent_id: None,
            run_id: None,
            top_k: 5,
            threshold: None,
            show_expired: false,
            explain: false,
            secret: None,
        };
        let Json(result) = server.search_memories(Parameters(request)).await.expect("search should succeed");
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].id, "rec-1");
        let captured = rx.recv_timeout(std::time::Duration::from_secs(2)).expect("expected a captured request");
        assert_eq!(captured.method, "POST");
        assert_eq!(captured.path, "/search");
        let body: serde_json::Value = serde_json::from_str(&captured.body).expect("body must be real JSON");
        assert_eq!(body["query"], "hi");
        assert_eq!(body["user_id"], "alice");
        assert_eq!(body["top_k"], 5);
    }

    #[tokio::test]
    async fn delete_all_memories_remote_rejects_an_empty_scope_without_ever_making_a_request() {
        let (base_url, rx) = spawn_fake_server(200, r#"{"deleted":0}"#);
        let server = remote_test_server(&base_url, None);
        let request = DeleteAllMemoriesRequest { user_id: None, agent_id: None, run_id: None, secret: None };
        let err = server.delete_all_memories(Parameters(request)).await.err().expect("expected a validation error");
        assert!(err.contains("requires at least one of user_id, agent_id, or run_id"), "got: {err}");
        assert!(rx.recv_timeout(std::time::Duration::from_millis(200)).is_err(), "no request should have been sent to the server");
    }

    #[tokio::test]
    async fn get_memory_remote_maps_a_non_2xx_response_to_a_real_specific_error() {
        let (base_url, _rx) = spawn_fake_server(500, r#"{"error":"boom"}"#);
        let server = remote_test_server(&base_url, None);
        let request = GetMemoryRequest { id: "rec-1".to_string(), secret: None };
        let err = server.get_memory(Parameters(request)).await.err().expect("expected an error");
        assert!(err.contains("500"), "got: {err}");
        assert!(err.contains("boom"), "got: {err}");
    }

    #[tokio::test]
    async fn get_memory_remote_returns_the_real_not_found_message_on_404() {
        let (base_url, _rx) = spawn_fake_server(404, r#"{"error":"not found: rec-1"}"#);
        let server = remote_test_server(&base_url, None);
        let request = GetMemoryRequest { id: "rec-1".to_string(), secret: None };
        let err = server.get_memory(Parameters(request)).await.err().expect("expected an error");
        assert_eq!(err, "no memory found with id rec-1");
    }

    #[test]
    fn remote_client_trims_a_trailing_slash_from_the_configured_server_url() {
        let client = RemoteClient::new("http://127.0.0.1:8080/", None);
        assert_eq!(client.base_url, "http://127.0.0.1:8080");
    }

    #[tokio::test]
    async fn delete_memory_remote_reports_a_real_error_instead_of_success_when_the_server_rejects_it() {
        let (base_url, _rx) = spawn_fake_server(404, r#"{"error":"no memory found with id rec-1"}"#);
        let server = remote_test_server(&base_url, None);
        let request = DeleteMemoryRequest { id: "rec-1".to_string(), secret: None };
        let err = server.delete_memory(Parameters(request)).await.err().expect("a failed remote delete must be a real error, not a reported success");
        assert!(err.contains("404"), "got: {err}");
    }

    #[tokio::test]
    async fn get_memories_remote_propagates_a_real_error_from_one_failed_record_fetch_instead_of_silently_shortening_the_list() {
        let base_url = spawn_routed_fake_server(vec![
            ("/memories/rec-1", 200, r#"{"id":"rec-1","vector":[],"payload":{"content":"ok"}}"#),
            ("/memories/rec-2", 500, r#"{"error":"boom"}"#),
            ("/memories", 200, r#"{"ids":["rec-1","rec-2"]}"#),
        ]);
        let server = remote_test_server(&base_url, None);
        let request = GetMemoriesRequest { user_id: Some("alice".to_string()), agent_id: None, run_id: None, offset: 0, limit: 50, show_expired: false, secret: None };
        let err = server.get_memories(Parameters(request)).await.err().expect("a real fetch failure must surface as an error, not a shorter list");
        assert!(err.contains("500") && err.contains("boom"), "got: {err}");
    }
}
