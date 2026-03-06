#![forbid(unsafe_code)]
#![allow(clippy::multiple_crate_versions)]

use memoria_core::embedding::{EmbeddingProvider, LocalHashEmbeddingProvider};
use memoria_core::filter::{FilterExpr, FilterOp, FilterValue};
use memoria_core::llm::{LlmProvider, LocalSentenceLlmProvider};
use memoria_core::memory::Memory;
use memoria_core::vector_store::{InMemoryVectorStore, VectorStore};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

type BoxedLlm = Box<dyn LlmProvider + Send + Sync>;
type BoxedEmbedding = Box<dyn EmbeddingProvider + Send + Sync>;
type BoxedVectorStore = Box<dyn VectorStore + Send + Sync>;
type SharedMemory = Arc<Memory<BoxedLlm, BoxedEmbedding, BoxedVectorStore>>;

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

fn scope_filter(scope: &HashMap<String, String>) -> Option<FilterExpr> {
    let mut clauses: Vec<FilterExpr> = scope
        .iter()
        .map(|(key, value)| FilterExpr::Field(key.clone(), FilterOp::Eq(FilterValue::String(value.clone()))))
        .collect();
    match clauses.len() {
        0 => None,
        1 => clauses.pop(),
        _ => Some(FilterExpr::And(clauses)),
    }
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

fn check_secret(configured: Option<&str>, provided: Option<&str>) -> Result<(), McpError> {
    configured.map_or(Ok(()), |expected| {
        if provided.is_some_and(|value| constant_time_eq(value.as_bytes(), expected.as_bytes())) {
            Ok(())
        } else {
            Err(McpError::invalid_params(
                "missing or incorrect secret (MEMORIA_MCP_SECRET is configured on this server; see ADR-38)",
                None,
            ))
        }
    })
}

fn core_error_to_mcp(err: &memoria_core::CoreError) -> McpError {
    match err {
        memoria_core::CoreError::NotFound(message) | memoria_core::CoreError::Validation(message) => {
            McpError::invalid_params(message.clone(), None)
        }
        other => McpError::internal_error(other.to_string(), None),
    }
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
}

impl From<memoria_core::vector_store::SearchResult> for MemoryResult {
    fn from(result: memoria_core::vector_store::SearchResult) -> Self {
        Self { id: result.id, score: result.score, payload: result.payload }
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

#[derive(Clone)]
struct MemoriaMcpServer {
    memory: SharedMemory,
    mcp_secret: Option<String>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MemoriaMcpServer {
    fn new(memory: SharedMemory, mcp_secret: Option<String>) -> Self {
        Self { memory, mcp_secret, tool_router: Self::tool_router() }
    }

    #[tool(description = "Search memories with a semantic query, optionally scoped to a user, agent, or run")]
    async fn search_memories(&self, Parameters(request): Parameters<SearchMemoriesRequest>) -> Result<Json<Vec<MemoryResult>>, McpError> {
        check_secret(self.mcp_secret.as_deref(), request.secret.as_deref())?;
        let scope = scope_from_optional(request.user_id, request.agent_id, request.run_id);
        let results = self
            .memory
            .search(&request.query, request.top_k, &scope, request.threshold, request.show_expired, None, false)
            .map_err(|err| core_error_to_mcp(&err))?;
        Ok(Json(results.into_iter().map(MemoryResult::from).collect()))
    }

    #[tool(description = "Fetch a single memory by id")]
    async fn get_memory(&self, Parameters(request): Parameters<GetMemoryRequest>) -> Result<Json<MemoryRecord>, McpError> {
        check_secret(self.mcp_secret.as_deref(), request.secret.as_deref())?;
        match self.memory.get(&request.id).map_err(|err| core_error_to_mcp(&err))? {
            Some(record) => Ok(Json(MemoryRecord::from(record))),
            None => Err(McpError::invalid_params(format!("no memory found with id {}", request.id), None)),
        }
    }

    #[tool(description = "List memories, optionally scoped to a user, agent, or run")]
    async fn get_memories(&self, Parameters(request): Parameters<GetMemoriesRequest>) -> Result<Json<Vec<MemoryRecord>>, McpError> {
        check_secret(self.mcp_secret.as_deref(), request.secret.as_deref())?;
        let scope = scope_from_optional(request.user_id, request.agent_id, request.run_id);
        let filter = scope_filter(&scope);
        let ids = self
            .memory
            .list(request.offset, request.limit, request.show_expired, filter.as_ref())
            .map_err(|err| core_error_to_mcp(&err))?;
        let records = ids
            .into_iter()
            .filter_map(|id| self.memory.get(&id).ok().flatten())
            .map(MemoryRecord::from)
            .collect();
        Ok(Json(records))
    }

    #[tool(description = "Fetch the change history (add/delete events) for a single memory")]
    async fn memory_history(&self, Parameters(request): Parameters<MemoryHistoryRequest>) -> Result<Json<Vec<HistoryEntry>>, McpError> {
        check_secret(self.mcp_secret.as_deref(), request.secret.as_deref())?;
        let entries = self
            .memory
            .history(&request.id, request.offset, request.limit)
            .map_err(|err| core_error_to_mcp(&err))?;
        Ok(Json(entries.into_iter().map(HistoryEntry::from).collect()))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemoriaMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "memoria: a local-first memory layer for AI agents. Read-only tools in this increment: \
             search_memories, get_memory, get_memories, memory_history. See ADR-38.",
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

fn resolve_llm_provider(provider_choice: Option<&str>, model: Option<String>, base_url: Option<String>) -> Result<BoxedLlm, String> {
    match provider_choice.unwrap_or("local") {
        "local" => Ok(Box::new(LocalSentenceLlmProvider::new())),
        "ollama" => {
            #[cfg(feature = "ollama")]
            {
                let model = model.unwrap_or_else(|| "qwen2.5:0.5b".to_string());
                let config = memoria_core::llm::LlmConfig { model, base_url, api_key: None, temperature: None };
                let provider = memoria_core::llm::OllamaLlmProvider::from_config(config).map_err(|err| err.to_string())?;
                Ok(Box::new(provider))
            }
            #[cfg(not(feature = "ollama"))]
            {
                let _ = (model, base_url);
                Err("MEMORIA_LLM_PROVIDER=ollama requires the mcp binary to be built with --features ollama".to_string())
            }
        }
        other => Err(format!("unknown MEMORIA_LLM_PROVIDER value {other:?} (expected \"local\" or \"ollama\")")),
    }
}

fn resolve_embedding_provider(
    provider_choice: Option<&str>,
    model: Option<String>,
    base_url: Option<String>,
    fastembed_cache_dir: Option<String>,
) -> Result<BoxedEmbedding, String> {
    match provider_choice.unwrap_or("local") {
        "local" => Ok(Box::new(LocalHashEmbeddingProvider::new())),
        "ollama" => {
            #[cfg(feature = "ollama")]
            {
                let model = model.unwrap_or_else(|| "nomic-embed-text".to_string());
                let config = memoria_core::embedding::EmbeddingConfig { model, base_url, api_key: None, dimensions: None };
                let provider = memoria_core::embedding::OllamaEmbeddingProvider::from_config(config).map_err(|err| err.to_string())?;
                Ok(Box::new(provider))
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
                let config = memoria_core::embedding::EmbeddingConfig { model, base_url: None, api_key: None, dimensions: None };
                let cache_dir = fastembed_cache_dir.map(PathBuf::from);
                let provider = memoria_core::embedding::FastEmbedEmbeddingProvider::from_config(&config, cache_dir).map_err(|err| err.to_string())?;
                Ok(Box::new(provider))
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

fn resolve_vector_store(choice: Option<&str>, json_snapshot_path: &Path, sqlite_path: &Path) -> Result<BoxedVectorStore, String> {
    match choice.unwrap_or("sqlite") {
        "sqlite" => {
            #[cfg(feature = "sqlite")]
            {
                let store = memoria_core::vector_store::SqliteVectorStore::open(sqlite_path).map_err(|err| err.to_string())?;
                Ok(Box::new(store))
            }
            #[cfg(not(feature = "sqlite"))]
            {
                let _ = sqlite_path;
                Err("the mcp binary must be built with --features sqlite to use the default MEMORIA_VECTOR_STORE=sqlite (see ADR-38); pass MEMORIA_VECTOR_STORE=local to fall back to JSON-snapshot persistence".to_string())
            }
        }
        "local" => Ok(Box::new(load_store(json_snapshot_path))),
        other => Err(format!("unknown MEMORIA_VECTOR_STORE value {other:?} (expected \"sqlite\" or \"local\")")),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm_provider_choice = std::env::var("MEMORIA_LLM_PROVIDER").ok();
    let llm_provider = resolve_llm_provider(
        llm_provider_choice.as_deref(),
        std::env::var("MEMORIA_LLM_MODEL").ok(),
        std::env::var("MEMORIA_LLM_BASE_URL").ok(),
    )
    .map_err(|message| -> Box<dyn std::error::Error> { message.into() })?;

    let embedding_provider = resolve_embedding_provider(
        std::env::var("MEMORIA_EMBEDDING_PROVIDER").ok().as_deref(),
        std::env::var("MEMORIA_EMBEDDING_MODEL").ok(),
        std::env::var("MEMORIA_EMBEDDING_BASE_URL").ok(),
        std::env::var("MEMORIA_FASTEMBED_CACHE_DIR").ok(),
    )
    .map_err(|message| -> Box<dyn std::error::Error> { message.into() })?;

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let store_path = resolve_store_path(std::env::var("MEMORIA_STORE_PATH").ok().as_deref(), &home);
    let sqlite_path = resolve_sqlite_path(std::env::var("MEMORIA_SQLITE_PATH").ok().as_deref(), &home);
    let vector_store = resolve_vector_store(std::env::var("MEMORIA_VECTOR_STORE").ok().as_deref(), &store_path, &sqlite_path)
        .map_err(|message| -> Box<dyn std::error::Error> { message.into() })?;

    let memory = Arc::new(Memory::new(llm_provider, embedding_provider, vector_store));
    let mcp_secret = std::env::var("MEMORIA_MCP_SECRET").ok().filter(|value| !value.is_empty());
    let server = MemoriaMcpServer::new(memory, mcp_secret);

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

    #[test]
    fn scope_from_optional_builds_only_the_provided_keys() {
        let scope = scope_from_optional(Some("alice".to_string()), None, None);
        assert_eq!(scope.len(), 1);
        assert_eq!(scope.get("user_id"), Some(&"alice".to_string()));
    }

    #[test]
    fn scope_filter_is_none_for_an_empty_scope() {
        assert_eq!(scope_filter(&HashMap::new()), None);
    }

    #[test]
    fn scope_filter_combines_multiple_keys_with_and() {
        let mut scope = HashMap::new();
        scope.insert("user_id".to_string(), "alice".to_string());
        scope.insert("agent_id".to_string(), "assistant".to_string());
        let filter = scope_filter(&scope).expect("expected a filter");
        match filter {
            FilterExpr::And(clauses) => assert_eq!(clauses.len(), 2),
            other => panic!("expected FilterExpr::And, got {other:?}"),
        }
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
        let server = MemoriaMcpServer::new(memory, None);
        let request = SearchMemoriesRequest {
            query: "engineer".to_string(),
            user_id: Some("alice".to_string()),
            agent_id: None,
            run_id: None,
            top_k: 10,
            threshold: None,
            show_expired: false,
            secret: None,
        };
        let Json(results) = server.search_memories(Parameters(request)).await.expect("search should succeed");
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn search_memories_rejects_a_missing_secret_when_one_is_configured() {
        let memory = test_memory();
        let server = MemoriaMcpServer::new(memory, Some("s3cret".to_string()));
        let request = SearchMemoriesRequest {
            query: "anything".to_string(),
            user_id: Some("alice".to_string()),
            agent_id: None,
            run_id: None,
            top_k: 10,
            threshold: None,
            show_expired: false,
            secret: None,
        };
        let result = server.search_memories(Parameters(request)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn get_memory_returns_a_real_stored_record_by_id() {
        let memory = test_memory();
        let id = add_fact(&memory, "Bob likes tea.", "bob");
        let server = MemoriaMcpServer::new(memory, None);
        let request = GetMemoryRequest { id: id.clone(), secret: None };
        let Json(record) = server.get_memory(Parameters(request)).await.expect("get should succeed");
        assert_eq!(record.id, id);
    }

    #[tokio::test]
    async fn get_memory_returns_a_tool_error_for_an_unknown_id() {
        let memory = test_memory();
        let server = MemoriaMcpServer::new(memory, None);
        let request = GetMemoryRequest { id: "never-existed".to_string(), secret: None };
        let result = server.get_memory(Parameters(request)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn get_memories_scopes_results_to_the_requested_user_only() {
        let memory = test_memory();
        add_fact(&memory, "Alice is an engineer.", "alice");
        add_fact(&memory, "Carol is a designer.", "carol");
        let server = MemoriaMcpServer::new(memory, None);
        let request =
            GetMemoriesRequest { user_id: Some("alice".to_string()), agent_id: None, run_id: None, offset: 0, limit: 50, show_expired: false, secret: None };
        let Json(records) = server.get_memories(Parameters(request)).await.expect("list should succeed");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].payload.get("user_id"), Some(&"alice".to_string()));
    }

    #[tokio::test]
    async fn memory_history_reports_real_add_and_delete_events_in_order() {
        let memory = test_memory();
        let id = add_fact(&memory, "Dana runs marathons.", "dana");
        memory.delete(&id).expect("delete should succeed");
        let server = MemoriaMcpServer::new(memory, None);
        let request = MemoryHistoryRequest { id, offset: 0, limit: 100, secret: None };
        let Json(entries) = server.memory_history(Parameters(request)).await.expect("history should succeed");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].event, "added");
        assert_eq!(entries[1].event, "deleted");
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
        let result = resolve_vector_store(Some("postgres"), &dir.join("unused.json"), &dir.join("unused.db"));
        assert!(result.is_err());
    }

    #[test]
    fn resolve_llm_provider_unknown_choice_is_a_clear_error() {
        assert!(resolve_llm_provider(Some("bogus"), None, None).is_err());
    }

    #[test]
    fn resolve_embedding_provider_unknown_choice_is_a_clear_error() {
        assert!(resolve_embedding_provider(Some("bogus"), None, None, None).is_err());
    }
}
