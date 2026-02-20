#![forbid(unsafe_code)]

use core::embedding::LocalHashEmbeddingProvider;
use core::llm::{LocalSentenceLlmProvider, Message, Role};
use core::memory::Memory;
use core::vector_store::InMemoryVectorStore;
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const RATE_LIMIT_CAPACITY: f64 = 20.0;
const RATE_LIMIT_REFILL_PER_SEC: f64 = 5.0;

struct RateLimitBucket {
    tokens: f64,
    last_refill: Instant,
}

struct RateLimiter {
    capacity: f64,
    refill_per_sec: f64,
    buckets: Mutex<HashMap<IpAddr, RateLimitBucket>>,
}

impl RateLimiter {
    fn new(capacity: f64, refill_per_sec: f64) -> Self {
        Self { capacity, refill_per_sec, buckets: Mutex::new(HashMap::new()) }
    }

    #[allow(clippy::significant_drop_tightening)]
    fn check_and_consume(&self, key: IpAddr) -> bool {
        let now = Instant::now();
        let mut buckets = self.buckets.lock().expect("rate limiter lock poisoned");
        let bucket = buckets.entry(key).or_insert_with(|| RateLimitBucket { tokens: self.capacity, last_refill: now });
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        bucket.tokens = elapsed.mul_add(self.refill_per_sec, bucket.tokens).min(self.capacity);
        bucket.last_refill = now;
        let allowed = bucket.tokens >= 1.0;
        if allowed {
            bucket.tokens -= 1.0;
        }
        allowed
    }
}

struct ParsedRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

const MAX_REQUEST_BODY_BYTES: usize = 1_048_576;

struct ParsedHeaders {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    header_len: usize,
    content_length: usize,
}

fn parse_headers(buf: &[u8]) -> Option<ParsedHeaders> {
    let mut raw_headers = [httparse::EMPTY_HEADER; 32];
    let mut req = httparse::Request::new(&mut raw_headers);
    let httparse::Status::Complete(header_len) = req.parse(buf).ok()? else {
        return None;
    };
    let method = req.method?.to_string();
    let path = req.path?.to_string();
    let headers: Vec<(String, String)> = req
        .headers
        .iter()
        .filter_map(|h| std::str::from_utf8(h.value).ok().map(|v| (h.name.to_string(), v.to_string())))
        .collect();
    let content_length = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.parse::<usize>().ok())
        .unwrap_or(0);
    Some(ParsedHeaders { method, path, headers, header_len, content_length })
}

fn check_body_size(content_length: usize) -> Option<(u16, Vec<u8>)> {
    if content_length > MAX_REQUEST_BODY_BYTES {
        Some((413, error_body("request body too large")))
    } else {
        None
    }
}

fn parse_request(buf: &[u8]) -> Option<ParsedRequest> {
    let parsed = parse_headers(buf)?;
    let available = buf.len().saturating_sub(parsed.header_len);
    if available < parsed.content_length {
        return None;
    }
    let body = buf[parsed.header_len..parsed.header_len + parsed.content_length].to_vec();
    Some(ParsedRequest { method: parsed.method, path: parsed.path, headers: parsed.headers, body })
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

fn format_log_line(method: &str, path: &str, status: u16) -> String {
    format!("{method} {path} {status}")
}

fn handle_health() -> (u16, Vec<u8>) {
    (200, br#"{"status":"ok"}"#.to_vec())
}

fn handle_ready<L, E, V>(memory: &Memory<L, E, V>) -> (u16, Vec<u8>)
where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    match memory.health_check() {
        Ok(()) => (200, br#"{"status":"ready"}"#.to_vec()),
        Err(err) => (503, error_body(format!("not ready: {err}"))),
    }
}

fn handle_delete_memory<L, E, V>(memory: &Memory<L, E, V>, id: &str) -> (u16, Vec<u8>)
where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    match memory.delete(id) {
        Ok(()) => (200, Vec::new()),
        Err(err) => error_response(&err),
    }
}

fn resolve_auth_config(api_key_env: Option<String>, allow_no_auth_env: Option<String>) -> Result<Option<String>, String> {
    match (api_key_env, allow_no_auth_env) {
        (Some(key), _) if !key.is_empty() => Ok(Some(key)),
        (_, Some(flag)) if flag == "1" => Ok(None),
        _ => Err("refusing to start: no MEMORIA_API_KEY configured and MEMORIA_ALLOW_NO_AUTH=1 not set (ADR-14)".to_string()),
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn is_authorized(configured_token: Option<&str>, headers: &[(String, String)]) -> bool {
    let Some(expected) = configured_token else {
        return true;
    };
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        .and_then(|(_, value)| value.strip_prefix("Bearer "))
        .is_some_and(|token| constant_time_eq(token.as_bytes(), expected.as_bytes()))
}

fn parse_query(query: &str) -> HashMap<String, String> {
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ListMemoryResponse {
    ids: Vec<String>,
}

fn handle_list_memory<L, E, V>(memory: &Memory<L, E, V>, query: &str) -> (u16, Vec<u8>)
where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    let params = parse_query(query);
    let offset = params.get("offset").and_then(|v| v.parse().ok()).unwrap_or(0);
    let limit = params.get("limit").and_then(|v| v.parse().ok()).unwrap_or(100);

    match memory.list(offset, limit) {
        Ok(ids) => (200, serde_json::to_vec(&ListMemoryResponse { ids }).unwrap_or_default()),
        Err(err) => error_response(&err),
    }
}

fn route<L, E, V>(memory: &Memory<L, E, V>, req: &ParsedRequest) -> (u16, Vec<u8>)
where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    let (path_only, query) = req.path.split_once('?').unwrap_or((req.path.as_str(), ""));
    let segments: Vec<&str> = path_only.trim_matches('/').split('/').collect();
    match (req.method.as_str(), segments.as_slice()) {
        ("POST", ["memories"]) => handle_create_memory(memory, &req.body),
        ("POST", ["memories", "search"]) => handle_search_memory(memory, &req.body),
        ("GET", ["memories"]) => handle_list_memory(memory, query),
        ("GET", ["memories", id]) => handle_get_memory(memory, id),
        ("PUT", ["memories", id]) => handle_update_memory(memory, id, &req.body),
        ("DELETE", ["memories", id]) => handle_delete_memory(memory, id),
        _ => (404, error_body("not found")),
    }
}

const fn reason_phrase(status: u16) -> &'static str {
    match status {
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

fn resolve_cors_origin(env: Option<String>) -> Option<String> {
    env.filter(|origin| !origin.is_empty())
}

fn cors_allow_origin_header(configured: Option<&str>, request_origin: Option<&str>) -> Option<(String, String)> {
    let configured = configured?;
    let request_origin = request_origin?;
    if configured == request_origin {
        Some(("Access-Control-Allow-Origin".to_string(), configured.to_string()))
    } else {
        None
    }
}

fn build_response(status: u16, body: &[u8]) -> Vec<u8> {
    build_response_with_headers(status, body, &[])
}

fn build_response_with_headers(status: u16, body: &[u8], extra_headers: &[(String, String)]) -> Vec<u8> {
    let mut response = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        reason_phrase(status),
        body.len()
    );
    for (name, value) in extra_headers {
        let _ = std::fmt::Write::write_fmt(&mut response, format_args!("{name}: {value}\r\n"));
    }
    response.push_str("\r\n");
    let mut response = response.into_bytes();
    response.extend_from_slice(body);
    response
}

fn find_header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
}

async fn handle_connection<L, E, V>(
    mut stream: TcpStream,
    memory: Arc<Memory<L, E, V>>,
    token: Arc<Option<String>>,
    rate_limiter: Arc<RateLimiter>,
    cors_origin: Arc<Option<String>>,
    peer_ip: IpAddr,
) where
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
        if let Some(headers) = parse_headers(&buf) {
            if let Some((status, body)) = check_body_size(headers.content_length) {
                eprintln!("{}", format_log_line(&headers.method, &headers.path, status));
                let response = build_response(status, &body);
                let _ = stream.write_all(&response).await;
                return;
            }
        }
        if let Some(req) = parse_request(&buf) {
            let request_origin = find_header(&req.headers, "origin");
            let cors_header = cors_allow_origin_header(cors_origin.as_deref(), request_origin);

            if req.method == "OPTIONS" && request_origin.is_some() {
                let mut headers: Vec<(String, String)> = cors_header.into_iter().collect();
                headers.push(("Access-Control-Allow-Methods".to_string(), "GET, POST, PUT, DELETE".to_string()));
                headers.push(("Access-Control-Allow-Headers".to_string(), "Content-Type, Authorization".to_string()));
                eprintln!("{}", format_log_line(&req.method, &req.path, 204));
                let response = build_response_with_headers(204, b"", &headers);
                let _ = stream.write_all(&response).await;
                return;
            }

            let (status, body) = if !rate_limiter.check_and_consume(peer_ip) {
                (429, error_body("rate limit exceeded"))
            } else if req.method == "GET" && req.path == "/health" {
                handle_health()
            } else if req.method == "GET" && req.path == "/ready" {
                handle_ready(&memory)
            } else if is_authorized(token.as_deref(), &req.headers) {
                route(&memory, &req)
            } else {
                (401, error_body("unauthorized"))
            };
            eprintln!("{}", format_log_line(&req.method, &req.path, status));
            let response = build_response_with_headers(status, &body, &cors_header.into_iter().collect::<Vec<_>>());
            let _ = stream.write_all(&response).await;
            return;
        }
    }
}

async fn serve<L, E, V>(
    listener: TcpListener,
    memory: Arc<Memory<L, E, V>>,
    token: Arc<Option<String>>,
    rate_limiter: Arc<RateLimiter>,
    cors_origin: Arc<Option<String>>,
) where
    L: core::llm::LlmProvider + Send + Sync + 'static,
    E: core::embedding::EmbeddingProvider + Send + Sync + 'static,
    V: core::vector_store::VectorStore + Send + Sync + 'static,
{
    loop {
        let Ok((stream, peer_addr)) = listener.accept().await else {
            continue;
        };
        let memory = Arc::clone(&memory);
        let token = Arc::clone(&token);
        let rate_limiter = Arc::clone(&rate_limiter);
        let cors_origin = Arc::clone(&cors_origin);
        tokio::spawn(handle_connection(stream, memory, token, rate_limiter, cors_origin, peer_addr.ip()));
    }
}

fn main() {
    let api_key_env = std::env::var("MEMORIA_API_KEY").ok();
    let allow_no_auth_env = std::env::var("MEMORIA_ALLOW_NO_AUTH").ok();
    let token = match resolve_auth_config(api_key_env, allow_no_auth_env) {
        Ok(token) => Arc::new(token),
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };
    let cors_origin = Arc::new(resolve_cors_origin(std::env::var("MEMORIA_CORS_ORIGIN").ok()));

    let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    runtime.block_on(async {
        let memory = Arc::new(Memory::new(
            LocalSentenceLlmProvider::new(),
            LocalHashEmbeddingProvider::new(),
            InMemoryVectorStore::new(),
        ));
        let rate_limiter = Arc::new(RateLimiter::new(RATE_LIMIT_CAPACITY, RATE_LIMIT_REFILL_PER_SEC));
        let listener = TcpListener::bind("127.0.0.1:8080").await.expect("failed to bind to 127.0.0.1:8080");
        serve(listener, memory, token, rate_limiter, cors_origin).await;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn test_ip(last_octet: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(127, 0, 0, last_octet))
    }

    #[test]
    fn rate_limiter_allows_requests_up_to_capacity() {
        let limiter = RateLimiter::new(3.0, 0.0);
        let ip = test_ip(1);
        assert!(limiter.check_and_consume(ip));
        assert!(limiter.check_and_consume(ip));
        assert!(limiter.check_and_consume(ip));
    }

    #[test]
    fn rate_limiter_rejects_the_request_after_capacity_is_exhausted() {
        let limiter = RateLimiter::new(2.0, 0.0);
        let ip = test_ip(2);
        assert!(limiter.check_and_consume(ip));
        assert!(limiter.check_and_consume(ip));
        assert!(!limiter.check_and_consume(ip));
    }

    #[test]
    fn rate_limiter_tracks_separate_ips_independently() {
        let limiter = RateLimiter::new(1.0, 0.0);
        let first = test_ip(3);
        let second = test_ip(4);
        assert!(limiter.check_and_consume(first));
        assert!(!limiter.check_and_consume(first));
        assert!(limiter.check_and_consume(second));
    }

    #[test]
    fn rate_limiter_refills_over_time_for_sustained_traffic() {
        let limiter = RateLimiter::new(1.0, 1000.0);
        let ip = test_ip(5);
        assert!(limiter.check_and_consume(ip));
        assert!(!limiter.check_and_consume(ip));
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(limiter.check_and_consume(ip));
    }

    struct FailingLlmProvider;

    impl core::llm::LlmProvider for FailingLlmProvider {
        fn complete(&self, _messages: &[Message]) -> Result<core::llm::Completion, core::llm::LlmError> {
            Err(core::llm::LlmError::Timeout)
        }
    }

    #[test]
    fn error_response_maps_validation_to_400() {
        let (status, _) = error_response(&core::CoreError::Validation("bad input".to_string()));
        assert_eq!(status, 400);
    }

    #[test]
    fn error_response_maps_not_found_to_404() {
        let (status, _) = error_response(&core::CoreError::NotFound("missing".to_string()));
        assert_eq!(status, 404);
    }

    #[test]
    fn error_response_maps_config_to_500() {
        let (status, _) = error_response(&core::CoreError::Config("bad config".to_string()));
        assert_eq!(status, 500);
    }

    #[test]
    fn error_response_maps_provider_to_500() {
        let err: core::CoreError = core::llm::LlmError::Timeout.into();
        let (status, _) = error_response(&err);
        assert_eq!(status, 500);
    }

    #[test]
    fn format_log_line_includes_method_path_and_status() {
        assert_eq!(format_log_line("GET", "/memories", 200), "GET /memories 200");
    }

    #[test]
    fn format_log_line_handles_an_error_status_and_a_deep_path() {
        assert_eq!(format_log_line("DELETE", "/memories/rec-1-2-3", 404), "DELETE /memories/rec-1-2-3 404");
    }

    #[test]
    fn handle_ready_returns_200_when_providers_are_healthy() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (status, body) = handle_ready(&memory);
        assert_eq!(status, 200);
        assert_eq!(body, br#"{"status":"ready"}"#);
    }

    #[test]
    fn handle_ready_returns_503_when_a_provider_is_unhealthy() {
        let memory = Memory::new(FailingLlmProvider, LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (status, _body) = handle_ready(&memory);
        assert_eq!(status, 503);
    }

    #[test]
    fn handle_health_returns_200_with_ok_status() {
        let (status, body) = handle_health();
        assert_eq!(status, 200);
        assert_eq!(body, br#"{"status":"ok"}"#);
    }

    #[test]
    fn resolve_cors_origin_with_configured_origin_resolves_to_that_origin() {
        assert_eq!(resolve_cors_origin(Some("https://example.com".to_string())), Some("https://example.com".to_string()));
    }

    #[test]
    fn resolve_cors_origin_with_unset_or_empty_resolves_to_none() {
        assert_eq!(resolve_cors_origin(None), None);
        assert_eq!(resolve_cors_origin(Some(String::new())), None);
    }

    #[test]
    fn cors_allow_origin_header_matches_the_configured_origin() {
        let header = cors_allow_origin_header(Some("https://example.com"), Some("https://example.com"));
        assert_eq!(header, Some(("Access-Control-Allow-Origin".to_string(), "https://example.com".to_string())));
    }

    #[test]
    fn cors_allow_origin_header_rejects_a_mismatched_origin() {
        assert_eq!(cors_allow_origin_header(Some("https://example.com"), Some("https://evil.example")), None);
    }

    #[test]
    fn cors_allow_origin_header_is_absent_when_cors_is_not_configured() {
        assert_eq!(cors_allow_origin_header(None, Some("https://example.com")), None);
    }

    #[test]
    fn build_response_with_headers_includes_the_extra_headers() {
        let response = build_response_with_headers(200, b"{}", &[("Access-Control-Allow-Origin".to_string(), "https://example.com".to_string())]);
        let text = String::from_utf8(response).expect("response should be valid utf8");
        assert!(text.contains("Access-Control-Allow-Origin: https://example.com\r\n"), "got: {text}");
    }

    #[test]
    fn parse_headers_extracts_content_length_before_the_body_arrives() {
        let raw = b"POST /memories HTTP/1.1\r\nContent-Length: 999999\r\n\r\n";
        let parsed = parse_headers(raw).expect("expected parsed headers");
        assert_eq!(parsed.content_length, 999_999);
    }

    #[test]
    fn parse_headers_returns_none_when_headers_incomplete() {
        let raw = b"POST /memories HTTP/1.1\r\nContent-Le";
        assert!(parse_headers(raw).is_none());
    }

    #[test]
    fn check_body_size_allows_a_body_within_the_limit() {
        assert!(check_body_size(1024).is_none());
    }

    #[test]
    fn check_body_size_rejects_a_body_over_the_limit() {
        let (status, _body) = check_body_size(MAX_REQUEST_BODY_BYTES + 1).expect("expected a rejection");
        assert_eq!(status, 413);
    }

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

        let req = ParsedRequest { method: "GET".to_string(), path: format!("/memories/{id}"), headers: Vec::new(), body: Vec::new() };
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
            headers: Vec::new(),
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
            headers: Vec::new(),
            body: br#"{"content":"updated"}"#.to_vec(),
        };
        let (status, _) = route(&memory, &req);
        assert_eq!(status, 200);
    }

    #[test]
    fn handle_delete_memory_is_idempotent() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (_, create_body) = handle_create_memory(&memory, br#"{"content":"Alice is an engineer.","user_id":"alice"}"#);
        let created: CreateMemoryResponse = serde_json::from_slice(&create_body).expect("expected valid JSON");
        let id = created.ids.first().expect("expected at least one id");

        let (status, _) = handle_delete_memory(&memory, id);
        assert_eq!(status, 200);
        let (status_second, _) = handle_delete_memory(&memory, id);
        assert_eq!(status_second, 200, "deleting an already-deleted id must still succeed");

        let (get_status, _) = handle_get_memory(&memory, id);
        assert_eq!(get_status, 404);
    }

    #[test]
    fn handle_delete_memory_on_never_existing_id_still_succeeds() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (status, _) = handle_delete_memory(&memory, "never-existed");
        assert_eq!(status, 200);
    }

    #[test]
    fn resolve_auth_config_with_configured_key_resolves_to_that_key() {
        let result = resolve_auth_config(Some("secret-token".to_string()), None);
        assert_eq!(result, Ok(Some("secret-token".to_string())));
    }

    #[test]
    fn resolve_auth_config_with_explicit_override_resolves_to_no_auth() {
        let result = resolve_auth_config(None, Some("1".to_string()));
        assert_eq!(result, Ok(None));
    }

    #[test]
    fn resolve_auth_config_with_nothing_set_fails_closed() {
        let result = resolve_auth_config(None, None);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_auth_config_with_empty_key_and_no_override_fails_closed() {
        let result = resolve_auth_config(Some(String::new()), None);
        assert!(result.is_err(), "an empty MEMORIA_API_KEY must not be treated as configured");
    }

    #[test]
    fn resolve_auth_config_rejects_a_non_one_override_value() {
        let result = resolve_auth_config(None, Some("true".to_string()));
        assert!(result.is_err(), "MEMORIA_ALLOW_NO_AUTH must be exactly \"1\", not any truthy-looking string");
    }

    #[test]
    fn resolve_auth_config_prefers_a_real_key_over_the_override_if_both_are_set() {
        let result = resolve_auth_config(Some("secret".to_string()), Some("1".to_string()));
        assert_eq!(result, Ok(Some("secret".to_string())));
    }

    #[test]
    fn constant_time_eq_matches_identical_bytes() {
        assert!(constant_time_eq(b"secret", b"secret"));
    }

    #[test]
    fn constant_time_eq_rejects_different_bytes() {
        assert!(!constant_time_eq(b"secret", b"wrong-token"));
    }

    #[test]
    fn constant_time_eq_rejects_different_lengths() {
        assert!(!constant_time_eq(b"short", b"a-much-longer-value"));
    }

    #[test]
    fn is_authorized_with_no_configured_token_allows_anything() {
        assert!(is_authorized(None, &[]));
    }

    #[test]
    fn is_authorized_with_correct_bearer_token_succeeds() {
        let headers = vec![("Authorization".to_string(), "Bearer secret".to_string())];
        assert!(is_authorized(Some("secret"), &headers));
    }

    #[test]
    fn is_authorized_with_wrong_bearer_token_fails() {
        let headers = vec![("Authorization".to_string(), "Bearer wrong".to_string())];
        assert!(!is_authorized(Some("secret"), &headers));
    }

    #[test]
    fn is_authorized_with_missing_header_fails() {
        assert!(!is_authorized(Some("secret"), &[]));
    }

    #[test]
    fn is_authorized_with_header_missing_bearer_prefix_fails() {
        let headers = vec![("Authorization".to_string(), "secret".to_string())];
        assert!(!is_authorized(Some("secret"), &headers));
    }

    #[test]
    fn parse_query_extracts_key_value_pairs() {
        let params = parse_query("offset=5&limit=10");
        assert_eq!(params.get("offset"), Some(&"5".to_string()));
        assert_eq!(params.get("limit"), Some(&"10".to_string()));
    }

    #[test]
    fn parse_query_with_empty_string_is_empty() {
        assert!(parse_query("").is_empty());
    }

    #[test]
    fn handle_list_memory_returns_ids_after_create() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (_, create_body) = handle_create_memory(&memory, br#"{"content":"Alice is an engineer.","user_id":"alice"}"#);
        let created: CreateMemoryResponse = serde_json::from_slice(&create_body).expect("expected valid JSON");

        let (status, body) = handle_list_memory(&memory, "");
        assert_eq!(status, 200);
        let response: ListMemoryResponse = serde_json::from_slice(&body).expect("expected valid JSON");
        for id in &created.ids {
            assert!(response.ids.contains(id));
        }
    }

    #[test]
    fn handle_list_memory_respects_limit_query_param() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        handle_create_memory(&memory, br#"{"content":"Alice is an engineer.","user_id":"alice"}"#);

        let (status, body) = handle_list_memory(&memory, "limit=0");
        assert_eq!(status, 200);
        let response: ListMemoryResponse = serde_json::from_slice(&body).expect("expected valid JSON");
        assert!(response.ids.is_empty(), "limit=0 should return nothing, not error");
    }

    #[test]
    fn route_dispatches_delete_memories_id() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (_, create_body) = handle_create_memory(&memory, br#"{"content":"Alice is an engineer.","user_id":"alice"}"#);
        let created: CreateMemoryResponse = serde_json::from_slice(&create_body).expect("expected valid JSON");
        let id = created.ids.first().expect("expected at least one id").clone();

        let req = ParsedRequest { method: "DELETE".to_string(), path: format!("/memories/{id}"), headers: Vec::new(), body: Vec::new() };
        let (status, _) = route(&memory, &req);
        assert_eq!(status, 200);
    }

    #[test]
    fn route_dispatches_get_memories_with_query_string() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        handle_create_memory(&memory, br#"{"content":"Alice is an engineer.","user_id":"alice"}"#);

        let req = ParsedRequest { method: "GET".to_string(), path: "/memories?offset=0&limit=10".to_string(), headers: Vec::new(), body: Vec::new() };
        let (status, body) = route(&memory, &req);
        assert_eq!(status, 200);
        let response: ListMemoryResponse = serde_json::from_slice(&body).expect("expected valid JSON");
        assert!(!response.ids.is_empty());
    }

    #[test]
    fn route_dispatches_post_memories() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let req = ParsedRequest {
            method: "POST".to_string(),
            path: "/memories".to_string(),
            headers: Vec::new(),
            body: br#"{"content":"Alice is an engineer.","user_id":"alice"}"#.to_vec(),
        };
        let (status, _) = route(&memory, &req);
        assert_eq!(status, 201);
    }

    #[test]
    fn route_returns_404_for_unknown_path() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let req = ParsedRequest { method: "GET".to_string(), path: "/nonexistent".to_string(), headers: Vec::new(), body: Vec::new() };
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
            tokio::spawn(serve(listener, memory, Arc::new(None), Arc::new(RateLimiter::new(1000.0, 1000.0)), Arc::new(None)));

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
            tokio::spawn(serve(listener, memory, Arc::new(None), Arc::new(RateLimiter::new(1000.0, 1000.0)), Arc::new(None)));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            stream.write_all(b"GET /nonexistent HTTP/1.1\r\nContent-Length: 0\r\n\r\n").await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");

            assert!(response_text.starts_with("HTTP/1.1 404 Not Found\r\n"), "got: {response_text}");
        });
    }

    #[test]
    fn server_rejects_a_request_with_no_token_when_auth_is_configured() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Arc::new(Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new()));
            tokio::spawn(serve(listener, memory, Arc::new(Some("secret".to_string())), Arc::new(RateLimiter::new(1000.0, 1000.0)), Arc::new(None)));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            let body = br#"{"content":"Alice is an engineer.","user_id":"alice"}"#;
            let request = format!("POST /memories HTTP/1.1\r\nContent-Length: {}\r\n\r\n", body.len());
            stream.write_all(request.as_bytes()).await.expect("write should succeed");
            stream.write_all(body).await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");

            assert!(response_text.starts_with("HTTP/1.1 401"), "got: {response_text}");
        });
    }

    #[test]
    fn server_accepts_a_request_with_the_correct_token_when_auth_is_configured() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Arc::new(Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new()));
            tokio::spawn(serve(listener, memory, Arc::new(Some("secret".to_string())), Arc::new(RateLimiter::new(1000.0, 1000.0)), Arc::new(None)));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            let body = br#"{"content":"Alice is an engineer.","user_id":"alice"}"#;
            let request = format!(
                "POST /memories HTTP/1.1\r\nAuthorization: Bearer secret\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            stream.write_all(request.as_bytes()).await.expect("write should succeed");
            stream.write_all(body).await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");

            assert!(response_text.starts_with("HTTP/1.1 201 Created\r\n"), "got: {response_text}");
        });
    }

    #[test]
    fn server_returns_429_over_tcp_after_the_bucket_is_exhausted() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Arc::new(Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new()));
            tokio::spawn(serve(listener, memory, Arc::new(None), Arc::new(RateLimiter::new(2.0, 0.0)), Arc::new(None)));

            let mut last_response_text = String::new();
            for _ in 0..3 {
                let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
                stream.write_all(b"GET /nonexistent HTTP/1.1\r\nContent-Length: 0\r\n\r\n").await.expect("write should succeed");
                let mut response = Vec::new();
                stream.read_to_end(&mut response).await.expect("read should succeed");
                last_response_text = String::from_utf8(response).expect("response should be valid utf8");
            }

            assert!(last_response_text.starts_with("HTTP/1.1 429"), "got: {last_response_text}");
        });
    }

    #[test]
    fn server_allows_traffic_again_after_the_bucket_refills() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Arc::new(Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new()));
            tokio::spawn(serve(listener, memory, Arc::new(None), Arc::new(RateLimiter::new(1.0, 1000.0)), Arc::new(None)));

            let mut first_stream = TcpStream::connect(addr).await.expect("connect should succeed");
            first_stream.write_all(b"GET /nonexistent HTTP/1.1\r\nContent-Length: 0\r\n\r\n").await.expect("write should succeed");
            let mut first_response = Vec::new();
            first_stream.read_to_end(&mut first_response).await.expect("read should succeed");
            let first_text = String::from_utf8(first_response).expect("response should be valid utf8");
            assert!(first_text.starts_with("HTTP/1.1 404"), "got: {first_text}");

            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

            let mut second_stream = TcpStream::connect(addr).await.expect("connect should succeed");
            second_stream.write_all(b"GET /nonexistent HTTP/1.1\r\nContent-Length: 0\r\n\r\n").await.expect("write should succeed");
            let mut second_response = Vec::new();
            second_stream.read_to_end(&mut second_response).await.expect("read should succeed");
            let second_text = String::from_utf8(second_response).expect("response should be valid utf8");
            assert!(second_text.starts_with("HTTP/1.1 404"), "got: {second_text}");
        });
    }

    #[test]
    fn server_rejects_an_oversized_body_without_waiting_for_all_of_it() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Arc::new(Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new()));
            tokio::spawn(serve(listener, memory, Arc::new(None), Arc::new(RateLimiter::new(1000.0, 1000.0)), Arc::new(None)));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            let declared_length = MAX_REQUEST_BODY_BYTES + 1;
            let mut request = format!("POST /memories HTTP/1.1\r\nContent-Length: {declared_length}\r\n\r\n").into_bytes();
            request.extend_from_slice(b"only a few bytes, never the full declared body");
            stream.write_all(&request).await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");

            assert!(response_text.starts_with("HTTP/1.1 413"), "got: {response_text}");
        });
    }

    #[test]
    fn server_echoes_the_configured_origin_when_it_matches() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Arc::new(Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new()));
            let cors_origin = Arc::new(Some("https://example.com".to_string()));
            tokio::spawn(serve(listener, memory, Arc::new(None), Arc::new(RateLimiter::new(1000.0, 1000.0)), cors_origin));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            let request = "GET /nonexistent HTTP/1.1\r\nOrigin: https://example.com\r\nContent-Length: 0\r\n\r\n";
            stream.write_all(request.as_bytes()).await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");

            assert!(response_text.contains("Access-Control-Allow-Origin: https://example.com\r\n"), "got: {response_text}");
        });
    }

    #[test]
    fn server_omits_the_cors_header_for_a_mismatched_origin() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Arc::new(Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new()));
            let cors_origin = Arc::new(Some("https://example.com".to_string()));
            tokio::spawn(serve(listener, memory, Arc::new(None), Arc::new(RateLimiter::new(1000.0, 1000.0)), cors_origin));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            let request = "GET /nonexistent HTTP/1.1\r\nOrigin: https://evil.example\r\nContent-Length: 0\r\n\r\n";
            stream.write_all(request.as_bytes()).await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");

            assert!(!response_text.contains("Access-Control-Allow-Origin"), "got: {response_text}");
        });
    }

    #[test]
    fn server_answers_an_options_preflight_request_directly() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Arc::new(Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new()));
            let cors_origin = Arc::new(Some("https://example.com".to_string()));
            tokio::spawn(serve(listener, memory, Arc::new(None), Arc::new(RateLimiter::new(1000.0, 1000.0)), cors_origin));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            let request = "OPTIONS /memories HTTP/1.1\r\nOrigin: https://example.com\r\nContent-Length: 0\r\n\r\n";
            stream.write_all(request.as_bytes()).await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");

            assert!(response_text.starts_with("HTTP/1.1 204"), "got: {response_text}");
            assert!(response_text.contains("Access-Control-Allow-Methods: GET, POST, PUT, DELETE\r\n"), "got: {response_text}");
            assert!(response_text.contains("Access-Control-Allow-Headers: Content-Type, Authorization\r\n"), "got: {response_text}");
        });
    }

    #[test]
    fn server_answers_health_without_a_token_even_when_auth_is_configured() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Arc::new(Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new()));
            tokio::spawn(serve(
                listener,
                memory,
                Arc::new(Some("secret".to_string())),
                Arc::new(RateLimiter::new(1000.0, 1000.0)),
                Arc::new(None),
            ));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            stream.write_all(b"GET /health HTTP/1.1\r\nContent-Length: 0\r\n\r\n").await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");

            assert!(response_text.starts_with("HTTP/1.1 200 OK\r\n"), "got: {response_text}");
            assert!(response_text.ends_with(r#"{"status":"ok"}"#), "got: {response_text}");
        });
    }

    #[test]
    fn server_answers_ready_without_a_token_when_providers_are_healthy() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Arc::new(Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new()));
            tokio::spawn(serve(
                listener,
                memory,
                Arc::new(Some("secret".to_string())),
                Arc::new(RateLimiter::new(1000.0, 1000.0)),
                Arc::new(None),
            ));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            stream.write_all(b"GET /ready HTTP/1.1\r\nContent-Length: 0\r\n\r\n").await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");

            assert!(response_text.starts_with("HTTP/1.1 200 OK\r\n"), "got: {response_text}");
            assert!(response_text.ends_with(r#"{"status":"ready"}"#), "got: {response_text}");
        });
    }
}
