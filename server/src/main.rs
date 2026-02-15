#![forbid(unsafe_code)]

use core::embedding::LocalHashEmbeddingProvider;
use core::llm::{LocalSentenceLlmProvider, Message, Role};
use core::memory::Memory;
use core::vector_store::InMemoryVectorStore;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

struct ParsedRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn parse_request(buf: &[u8]) -> Option<ParsedRequest> {
    let mut headers = [httparse::EMPTY_HEADER; 32];
    let mut req = httparse::Request::new(&mut headers);
    let httparse::Status::Complete(header_len) = req.parse(buf).ok()? else {
        return None;
    };
    let method = req.method?.to_string();
    let path = req.path?.to_string();
    let content_length = req
        .headers
        .iter()
        .find(|h| h.name.eq_ignore_ascii_case("content-length"))
        .and_then(|h| std::str::from_utf8(h.value).ok())
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let available = buf.len().saturating_sub(header_len);
    if available < content_length {
        return None;
    }
    let body = buf[header_len..header_len + content_length].to_vec();
    Some(ParsedRequest { method, path, body })
}

#[derive(serde::Deserialize)]
struct CreateMemoryRequest {
    content: String,
    user_id: Option<String>,
    agent_id: Option<String>,
    run_id: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CreateMemoryResponse {
    ids: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ErrorResponse {
    error: String,
}

fn error_body(message: impl Into<String>) -> Vec<u8> {
    serde_json::to_vec(&ErrorResponse { error: message.into() }).unwrap_or_default()
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

fn error_response(err: &core::CoreError) -> (u16, Vec<u8>) {
    match err {
        core::CoreError::Validation(msg) => (400, error_body(msg.clone())),
        core::CoreError::NotFound(msg) => (404, error_body(msg.clone())),
        other => (500, error_body(other.to_string())),
    }
}

fn handle_create_memory<L, E, V>(memory: &Memory<L, E, V>, body: &[u8]) -> (u16, Vec<u8>)
where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    let request: CreateMemoryRequest = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(err) => return (400, error_body(format!("malformed request body: {err}"))),
    };
    let scope = scope_from_optional(request.user_id, request.agent_id, request.run_id);

    match memory.add(&[Message::new(Role::User, request.content)], scope) {
        Ok(ids) => (201, serde_json::to_vec(&CreateMemoryResponse { ids }).unwrap_or_default()),
        Err(err) => error_response(&err),
    }
}

fn handle_get_memory<L, E, V>(memory: &Memory<L, E, V>, id: &str) -> (u16, Vec<u8>)
where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    match memory.get(id) {
        Ok(Some(record)) => (200, serde_json::to_vec(&record).unwrap_or_default()),
        Ok(None) => (404, error_body(format!("not found: {id}"))),
        Err(err) => error_response(&err),
    }
}

#[derive(serde::Deserialize)]
struct SearchMemoryRequest {
    query: String,
    user_id: Option<String>,
    agent_id: Option<String>,
    run_id: Option<String>,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

const fn default_top_k() -> usize {
    10
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SearchMemoryResponse {
    results: Vec<core::vector_store::SearchResult>,
}

fn handle_search_memory<L, E, V>(memory: &Memory<L, E, V>, body: &[u8]) -> (u16, Vec<u8>)
where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    let request: SearchMemoryRequest = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(err) => return (400, error_body(format!("malformed request body: {err}"))),
    };
    let scope = scope_from_optional(request.user_id, request.agent_id, request.run_id);

    match memory.search(&request.query, request.top_k, &scope) {
        Ok(results) => (200, serde_json::to_vec(&SearchMemoryResponse { results }).unwrap_or_default()),
        Err(err) => error_response(&err),
    }
}

#[derive(serde::Deserialize)]
struct UpdateMemoryRequest {
    content: Option<String>,
    metadata: Option<HashMap<String, String>>,
}

fn handle_update_memory<L, E, V>(memory: &Memory<L, E, V>, id: &str, body: &[u8]) -> (u16, Vec<u8>)
where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    let request: UpdateMemoryRequest = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(err) => return (400, error_body(format!("malformed request body: {err}"))),
    };

    match memory.update(id, request.content.as_deref(), request.metadata) {
        Ok(()) => (200, Vec::new()),
        Err(err) => error_response(&err),
    }
}

fn route<L, E, V>(memory: &Memory<L, E, V>, req: &ParsedRequest) -> (u16, Vec<u8>)
where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    let segments: Vec<&str> = req.path.trim_matches('/').split('/').collect();
    match (req.method.as_str(), segments.as_slice()) {
        ("POST", ["memories"]) => handle_create_memory(memory, &req.body),
        ("POST", ["memories", "search"]) => handle_search_memory(memory, &req.body),
        ("GET", ["memories", id]) => handle_get_memory(memory, id),
        ("PUT", ["memories", id]) => handle_update_memory(memory, id, &req.body),
        _ => (404, error_body("not found")),
    }
}

const fn reason_phrase(status: u16) -> &'static str {
    match status {
        201 => "Created",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    }
}

fn build_response(status: u16, body: &[u8]) -> Vec<u8> {
    let mut response = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        reason_phrase(status),
        body.len()
    )
    .into_bytes();
    response.extend_from_slice(body);
    response
}

async fn handle_connection<L, E, V>(mut stream: TcpStream, memory: Arc<Memory<L, E, V>>)
where
    L: core::llm::LlmProvider + Send + Sync + 'static,
    E: core::embedding::EmbeddingProvider + Send + Sync + 'static,
    V: core::vector_store::VectorStore + Send + Sync + 'static,
{
    let mut buf = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        let Ok(n) = stream.read(&mut chunk).await else {
            return;
        };
        if n == 0 {
            return;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(req) = parse_request(&buf) {
            let (status, body) = route(&memory, &req);
            let response = build_response(status, &body);
            let _ = stream.write_all(&response).await;
            return;
        }
    }
}

async fn serve<L, E, V>(listener: TcpListener, memory: Arc<Memory<L, E, V>>)
where
    L: core::llm::LlmProvider + Send + Sync + 'static,
    E: core::embedding::EmbeddingProvider + Send + Sync + 'static,
    V: core::vector_store::VectorStore + Send + Sync + 'static,
{
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
        let memory = Arc::clone(&memory);
        tokio::spawn(handle_connection(stream, memory));
    }
}

fn main() {
    let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    runtime.block_on(async {
        let memory = Arc::new(Memory::new(
            LocalSentenceLlmProvider::new(),
            LocalHashEmbeddingProvider::new(),
            InMemoryVectorStore::new(),
        ));
        let listener = TcpListener::bind("127.0.0.1:8080").await.expect("failed to bind to 127.0.0.1:8080");
        serve(listener, memory).await;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_request_extracts_method_path_and_body() {
        let raw = b"POST /memories HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello";
        let parsed = parse_request(raw).expect("expected a parsed request");
        assert_eq!(parsed.method, "POST");
        assert_eq!(parsed.path, "/memories");
        assert_eq!(parsed.body, b"hello");
    }

    #[test]
    fn parse_request_returns_none_when_body_incomplete() {
        let raw = b"POST /memories HTTP/1.1\r\nContent-Length: 10\r\n\r\nhello";
        assert!(parse_request(raw).is_none());
    }

    #[test]
    fn parse_request_returns_none_when_headers_incomplete() {
        let raw = b"POST /memories HTTP/1.1\r\nContent-Le";
        assert!(parse_request(raw).is_none());
    }

    #[test]
    fn handle_create_memory_happy_path_returns_201_with_ids() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let body = br#"{"content":"Alice is an engineer.","user_id":"alice"}"#;
        let (status, response_body) = handle_create_memory(&memory, body);
        assert_eq!(status, 201);
        let response: CreateMemoryResponse = serde_json::from_slice(&response_body).expect("expected valid JSON");
        assert!(!response.ids.is_empty());
    }

    #[test]
    fn handle_create_memory_rejects_malformed_json_body() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (status, _) = handle_create_memory(&memory, b"not json");
        assert_eq!(status, 400);
    }

    #[test]
    fn handle_create_memory_rejects_missing_scope() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let body = br#"{"content":"no scope here"}"#;
        let (status, response_body) = handle_create_memory(&memory, body);
        assert_eq!(status, 400);
        let response: ErrorResponse = serde_json::from_slice(&response_body).expect("expected valid JSON");
        assert!(response.error.contains("user_id"));
    }

    #[test]
    fn handle_get_memory_returns_the_record_after_create() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (_, create_body) = handle_create_memory(&memory, br#"{"content":"Alice is an engineer.","user_id":"alice"}"#);
        let created: CreateMemoryResponse = serde_json::from_slice(&create_body).expect("expected valid JSON");
        let id = created.ids.first().expect("expected at least one id");

        let (status, body) = handle_get_memory(&memory, id);
        assert_eq!(status, 200);
        let record: core::vector_store::VectorRecord = serde_json::from_slice(&body).expect("expected valid JSON");
        assert_eq!(&record.id, id);
    }

    #[test]
    fn handle_get_memory_returns_404_for_unknown_id() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (status, _) = handle_get_memory(&memory, "never-existed");
        assert_eq!(status, 404);
    }

    #[test]
    fn handle_search_memory_happy_path_returns_200_with_results() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        handle_create_memory(&memory, br#"{"content":"Alice is an engineer.","user_id":"alice"}"#);

        let (status, body) = handle_search_memory(&memory, br#"{"query":"engineer","user_id":"alice"}"#);
        assert_eq!(status, 200);
        let response: SearchMemoryResponse = serde_json::from_slice(&body).expect("expected valid JSON");
        assert!(!response.results.is_empty());
    }

    #[test]
    fn handle_search_memory_rejects_missing_scope() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (status, _) = handle_search_memory(&memory, br#"{"query":"anything"}"#);
        assert_eq!(status, 400);
    }

    #[test]
    fn handle_update_memory_happy_path_returns_200() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (_, create_body) = handle_create_memory(&memory, br#"{"content":"Alice is an engineer.","user_id":"alice"}"#);
        let created: CreateMemoryResponse = serde_json::from_slice(&create_body).expect("expected valid JSON");
        let id = created.ids.first().expect("expected at least one id");

        let (status, _) = handle_update_memory(&memory, id, br#"{"content":"Alice is a senior engineer."}"#);
        assert_eq!(status, 200);

        let (_, get_body) = handle_get_memory(&memory, id);
        let record: core::vector_store::VectorRecord = serde_json::from_slice(&get_body).expect("expected valid JSON");
        assert_eq!(record.payload.get("content"), Some(&"Alice is a senior engineer.".to_string()));
    }

    #[test]
    fn handle_update_memory_returns_404_for_unknown_id() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (status, _) = handle_update_memory(&memory, "never-existed", br#"{"content":"anything"}"#);
        assert_eq!(status, 404);
    }

    #[test]
    fn route_dispatches_get_memories_id() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (_, create_body) = handle_create_memory(&memory, br#"{"content":"Alice is an engineer.","user_id":"alice"}"#);
        let created: CreateMemoryResponse = serde_json::from_slice(&create_body).expect("expected valid JSON");
        let id = created.ids.first().expect("expected at least one id").clone();

        let req = ParsedRequest { method: "GET".to_string(), path: format!("/memories/{id}"), body: Vec::new() };
        let (status, _) = route(&memory, &req);
        assert_eq!(status, 200);
    }

    #[test]
    fn route_dispatches_post_memories_search() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        handle_create_memory(&memory, br#"{"content":"Alice is an engineer.","user_id":"alice"}"#);

        let req = ParsedRequest {
            method: "POST".to_string(),
            path: "/memories/search".to_string(),
            body: br#"{"query":"engineer","user_id":"alice"}"#.to_vec(),
        };
        let (status, _) = route(&memory, &req);
        assert_eq!(status, 200);
    }

    #[test]
    fn route_dispatches_put_memories_id() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (_, create_body) = handle_create_memory(&memory, br#"{"content":"Alice is an engineer.","user_id":"alice"}"#);
        let created: CreateMemoryResponse = serde_json::from_slice(&create_body).expect("expected valid JSON");
        let id = created.ids.first().expect("expected at least one id").clone();

        let req = ParsedRequest {
            method: "PUT".to_string(),
            path: format!("/memories/{id}"),
            body: br#"{"content":"updated"}"#.to_vec(),
        };
        let (status, _) = route(&memory, &req);
        assert_eq!(status, 200);
    }

    #[test]
    fn route_dispatches_post_memories() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let req = ParsedRequest {
            method: "POST".to_string(),
            path: "/memories".to_string(),
            body: br#"{"content":"Alice is an engineer.","user_id":"alice"}"#.to_vec(),
        };
        let (status, _) = route(&memory, &req);
        assert_eq!(status, 201);
    }

    #[test]
    fn route_returns_404_for_unknown_path() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let req = ParsedRequest { method: "GET".to_string(), path: "/nonexistent".to_string(), body: Vec::new() };
        let (status, _) = route(&memory, &req);
        assert_eq!(status, 404);
    }

    #[test]
    fn build_response_has_correct_status_line_and_content_length() {
        let response = build_response(201, b"{}");
        let text = String::from_utf8(response).expect("response should be valid utf8");
        assert!(text.starts_with("HTTP/1.1 201 Created\r\n"));
        assert!(text.contains("Content-Length: 2\r\n"));
        assert!(text.ends_with("{}"));
    }

    #[test]
    fn server_handles_a_real_create_memory_request_over_tcp() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Arc::new(Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new()));
            tokio::spawn(serve(listener, memory));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            let body = br#"{"content":"Alice is an engineer.","user_id":"alice"}"#;
            let request = format!(
                "POST /memories HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            stream.write_all(request.as_bytes()).await.expect("write should succeed");
            stream.write_all(body).await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");

            assert!(response_text.starts_with("HTTP/1.1 201 Created\r\n"), "got: {response_text}");
            assert!(response_text.contains("\"ids\":["));
        });
    }

    #[test]
    fn server_returns_404_over_tcp_for_unknown_route() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Arc::new(Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new()));
            tokio::spawn(serve(listener, memory));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            stream.write_all(b"GET /nonexistent HTTP/1.1\r\nContent-Length: 0\r\n\r\n").await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");

            assert!(response_text.starts_with("HTTP/1.1 404 Not Found\r\n"), "got: {response_text}");
        });
    }
}
