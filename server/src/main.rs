#![forbid(unsafe_code)]

mod auth_store;
mod crypto;

use core::embedding::LocalHashEmbeddingProvider;
use core::llm::{LocalSentenceLlmProvider, Message, Role};
use core::memory::Memory;
use core::vector_store::{InMemoryVectorStore, VectorRecord, VectorStore};
use std::collections::HashMap;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const RATE_LIMIT_CAPACITY: f64 = 20.0;
const RATE_LIMIT_REFILL_PER_SEC: f64 = 5.0;

pub(crate) fn write_atomically(path: &Path, data: &[u8]) -> std::io::Result<()> {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temp_path = path.with_extension(format!("json.tmp.{}.{unique}", std::process::id()));
    std::fs::write(&temp_path, data)?;
    std::fs::rename(&temp_path, path)
}

fn resolve_store_path(env_override: Option<&str>, home: &str) -> PathBuf {
    if let Some(p) = env_override {
        return PathBuf::from(p);
    }
    Path::new(home).join(".memoria").join("server-store.json")
}

fn resolve_auth_store_path(env_override: Option<&str>, home: &str) -> PathBuf {
    if let Some(p) = env_override {
        return PathBuf::from(p);
    }
    Path::new(home).join(".memoria").join("auth-store.json")
}

fn resolve_jwt_secret(env: Option<String>) -> Result<Vec<u8>, String> {
    match env {
        Some(secret) if !secret.is_empty() => Ok(secret.into_bytes()),
        _ => Err("refusing to start: no MEMORIA_JWT_SECRET configured (ADR-29)".to_string()),
    }
}

fn load_store(path: &Path) -> InMemoryVectorStore {
    let store = InMemoryVectorStore::new();
    if let Ok(data) = std::fs::read_to_string(path) {
        if let Ok(records) = serde_json::from_str::<Vec<VectorRecord>>(&data) {
            for record in records {
                let _ = store.insert(record);
            }
        }
    }
    store
}

fn save_store<L, E, V>(memory: &Memory<L, E, V>, path: &Path) -> std::io::Result<()>
where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    let ids = memory.list(0, usize::MAX).unwrap_or_default();
    let records: Vec<VectorRecord> = ids.iter().filter_map(|id| memory.get(id).ok().flatten()).collect();
    let data = serde_json::to_vec(&records).unwrap_or_default();
    write_atomically(path, &data)
}

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

const ACCESS_TOKEN_TTL_SECS: u64 = 900;

#[derive(serde::Deserialize)]
struct RegisterRequest {
    name: String,
    email: String,
    password: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct AuthTokenResponse {
    access_token: String,
    refresh_token: String,
    token_type: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SetupStatusResponse {
    setup_complete: bool,
}

fn build_access_token(user: &auth_store::User, jwt_secret: &[u8]) -> String {
    let now = auth_store::unix_now();
    let role_str = match user.role {
        auth_store::Role::Admin => "admin",
        auth_store::Role::User => "user",
    };
    let claims = serde_json::json!({"sub": user.id, "role": role_str, "iat": now, "exp": now + ACCESS_TOKEN_TTL_SECS});
    crypto::encode_jwt(&claims, jwt_secret)
}

fn issue_auth_tokens(auth_store: &auth_store::AuthStore, jwt_secret: &[u8], user: &auth_store::User) -> AuthTokenResponse {
    let access_token = build_access_token(user, jwt_secret);
    let refresh_token = auth_store.issue_refresh_token(&user.id);
    AuthTokenResponse { access_token, refresh_token, token_type: "bearer".to_string() }
}

fn handle_auth_register(auth_store: &auth_store::AuthStore, jwt_secret: &[u8], body: &[u8]) -> (u16, Vec<u8>) {
    if !auth_store::registration_is_allowed(auth_store.user_count()) {
        return (403, error_body("registration is closed: an account already exists"));
    }
    let request: RegisterRequest = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(err) => return (400, error_body(format!("malformed request body: {err}"))),
    };
    if request.name.is_empty() || request.email.is_empty() || request.password.is_empty() {
        return (400, error_body("name, email, and password must not be empty"));
    }
    let password_hash = crypto::hash_password(&request.password);
    let user = auth_store::User::new(request.name, request.email, password_hash, auth_store::Role::Admin);
    auth_store.insert_user(user.clone());
    let tokens = issue_auth_tokens(auth_store, jwt_secret, &user);
    (201, serde_json::to_vec(&tokens).unwrap_or_default())
}

fn handle_auth_setup_status(auth_store: &auth_store::AuthStore) -> (u16, Vec<u8>) {
    let setup_complete = !auth_store::registration_is_allowed(auth_store.user_count());
    (200, serde_json::to_vec(&SetupStatusResponse { setup_complete }).unwrap_or_default())
}

fn dummy_password_hash() -> &'static str {
    static DUMMY: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    DUMMY.get_or_init(|| crypto::hash_password("dummy-password-for-login-timing-safety"))
}

#[derive(serde::Deserialize)]
struct LoginRequest {
    email: String,
    password: String,
}

fn handle_auth_login(auth_store: &auth_store::AuthStore, jwt_secret: &[u8], body: &[u8]) -> (u16, Vec<u8>) {
    let request: LoginRequest = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(err) => return (400, error_body(format!("malformed request body: {err}"))),
    };
    let user = auth_store.find_user_by_email(&request.email);
    let password_to_check = user.as_ref().map_or_else(|| dummy_password_hash().to_string(), |u| u.password_hash.clone());
    let password_ok = crypto::verify_password(&request.password, &password_to_check);
    let Some(user) = user.filter(|_| password_ok) else {
        return (401, error_body("invalid email or password"));
    };
    let tokens = issue_auth_tokens(auth_store, jwt_secret, &user);
    (200, serde_json::to_vec(&tokens).unwrap_or_default())
}

#[derive(serde::Deserialize)]
struct RefreshRequest {
    refresh_token: String,
}

fn handle_auth_refresh(auth_store: &auth_store::AuthStore, jwt_secret: &[u8], body: &[u8]) -> (u16, Vec<u8>) {
    let request: RefreshRequest = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(err) => return (400, error_body(format!("malformed request body: {err}"))),
    };
    let (new_jti, user_id) = match auth_store.rotate_refresh_token(&request.refresh_token) {
        Ok(rotated) => rotated,
        Err(auth_store::RefreshError::Unknown | auth_store::RefreshError::AlreadyUsed) => {
            return (401, error_body("invalid or already-used refresh token"));
        }
    };
    let Some(user) = auth_store.find_user_by_id(&user_id) else {
        return (401, error_body("invalid or already-used refresh token"));
    };
    let access_token = build_access_token(&user, jwt_secret);
    let response = AuthTokenResponse { access_token, refresh_token: new_jti, token_type: "bearer".to_string() };
    (200, serde_json::to_vec(&response).unwrap_or_default())
}

fn current_user_from_headers(
    auth_store: &auth_store::AuthStore,
    jwt_secret: &[u8],
    headers: &[(String, String)],
) -> Option<auth_store::User> {
    let token = bearer_token(headers)?;
    let claims = crypto::decode_jwt(token, jwt_secret, auth_store::unix_now()).ok()?;
    let sub = claims.get("sub")?.as_str()?;
    auth_store.find_user_by_id(sub)
}

#[derive(serde::Serialize, serde::Deserialize)]
struct UserProfileResponse {
    id: String,
    name: String,
    email: String,
    role: String,
    onboarding_complete: bool,
}

impl From<&auth_store::User> for UserProfileResponse {
    fn from(user: &auth_store::User) -> Self {
        let role = match user.role {
            auth_store::Role::Admin => "admin",
            auth_store::Role::User => "user",
        };
        Self {
            id: user.id.clone(),
            name: user.name.clone(),
            email: user.email.clone(),
            role: role.to_string(),
            onboarding_complete: user.onboarding_complete,
        }
    }
}

fn handle_auth_me_get(auth_store: &auth_store::AuthStore, jwt_secret: &[u8], headers: &[(String, String)]) -> (u16, Vec<u8>) {
    let Some(user) = current_user_from_headers(auth_store, jwt_secret, headers) else {
        return (401, error_body("a valid access token identifying a real user is required"));
    };
    (200, serde_json::to_vec(&UserProfileResponse::from(&user)).unwrap_or_default())
}

#[derive(serde::Deserialize)]
struct UpdateProfileRequest {
    name: Option<String>,
    email: Option<String>,
}

fn handle_auth_me_patch(
    auth_store: &auth_store::AuthStore,
    jwt_secret: &[u8],
    headers: &[(String, String)],
    body: &[u8],
) -> (u16, Vec<u8>) {
    let Some(user) = current_user_from_headers(auth_store, jwt_secret, headers) else {
        return (401, error_body("a valid access token identifying a real user is required"));
    };
    let request: UpdateProfileRequest = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(err) => return (400, error_body(format!("malformed request body: {err}"))),
    };
    let Some(updated) = auth_store.update_profile(&user.id, request.name, request.email) else {
        return (404, error_body("user not found"));
    };
    (200, serde_json::to_vec(&UserProfileResponse::from(&updated)).unwrap_or_default())
}

#[derive(serde::Deserialize)]
struct ChangePasswordRequest {
    current_password: String,
    new_password: String,
}

fn handle_auth_change_password(
    auth_store: &auth_store::AuthStore,
    jwt_secret: &[u8],
    headers: &[(String, String)],
    body: &[u8],
) -> (u16, Vec<u8>) {
    let Some(user) = current_user_from_headers(auth_store, jwt_secret, headers) else {
        return (401, error_body("a valid access token identifying a real user is required"));
    };
    let request: ChangePasswordRequest = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(err) => return (400, error_body(format!("malformed request body: {err}"))),
    };
    if !crypto::verify_password(&request.current_password, &user.password_hash) {
        return (401, error_body("current password is incorrect"));
    }
    if request.new_password.is_empty() {
        return (400, error_body("new_password must not be empty"));
    }
    let new_hash = crypto::hash_password(&request.new_password);
    auth_store.update_password_hash(&user.id, new_hash);
    (200, serde_json::to_vec(&UserProfileResponse::from(&user)).unwrap_or_default())
}

fn handle_auth_onboarding_complete(
    auth_store: &auth_store::AuthStore,
    jwt_secret: &[u8],
    headers: &[(String, String)],
) -> (u16, Vec<u8>) {
    let Some(user) = current_user_from_headers(auth_store, jwt_secret, headers) else {
        return (401, error_body("a valid access token identifying a real user is required"));
    };
    auth_store.mark_onboarding_complete(&user.id);
    let Some(updated) = auth_store.find_user_by_id(&user.id) else {
        return (404, error_body("user not found"));
    };
    (200, serde_json::to_vec(&UserProfileResponse::from(&updated)).unwrap_or_default())
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
    threshold: Option<f32>,
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

    match memory.search(&request.query, request.top_k, &scope, request.threshold) {
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

fn bearer_token(headers: &[(String, String)]) -> Option<&str> {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        .and_then(|(_, value)| value.strip_prefix("Bearer "))
}

fn is_authorized(
    admin_key: Option<&str>,
    auth_store: &auth_store::AuthStore,
    jwt_secret: &[u8],
    headers: &[(String, String)],
) -> bool {
    let Some(admin_key) = admin_key else {
        return true;
    };
    if let Some(token) = bearer_token(headers) {
        if constant_time_eq(token.as_bytes(), admin_key.as_bytes()) {
            return true;
        }
        if crypto::decode_jwt(token, jwt_secret, auth_store::unix_now()).is_ok() {
            return true;
        }
    }
    if let Some(key) = find_header(headers, "x-api-key") {
        let key_hash = crypto::sha256_hex(key.as_bytes());
        if auth_store.find_active_api_key_by_hash(&key_hash).is_some() {
            return true;
        }
    }
    false
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

fn is_successful_mutation(method: &str, path: &str, status: u16) -> bool {
    let (path_only, _query) = path.split_once('?').unwrap_or((path, ""));
    let segments: Vec<&str> = path_only.trim_matches('/').split('/').collect();
    matches!(
        (method, segments.as_slice(), status),
        ("POST", ["memories"], 201) | ("PUT" | "DELETE", ["memories", _], 200)
    )
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

fn resolve_llm_provider(
    provider_choice: Option<&str>,
    model: Option<String>,
    base_url: Option<String>,
) -> Result<(Box<dyn core::llm::LlmProvider + Send + Sync>, String), String> {
    match provider_choice.unwrap_or("local") {
        "local" => Ok((
            Box::new(LocalSentenceLlmProvider::new()),
            "LocalSentenceLlmProvider (local, non-AI; see ADR-12)".to_string(),
        )),
        "ollama" => {
            #[cfg(feature = "ollama")]
            {
                let model = model.unwrap_or_else(|| "qwen2.5:0.5b".to_string());
                let resolved_base_url = base_url.clone().unwrap_or_else(|| "http://localhost:11434".to_string());
                let config = core::llm::LlmConfig { model: model.clone(), base_url, api_key: None, temperature: None };
                let provider = core::llm::OllamaLlmProvider::from_config(config).map_err(|err| err.to_string())?;
                let label = format!("OllamaLlmProvider (model={model}, base_url={resolved_base_url}; see ADR-25)");
                Ok((Box::new(provider), label))
            }
            #[cfg(not(feature = "ollama"))]
            {
                let _ = (model, base_url);
                Err("MEMORIA_LLM_PROVIDER=ollama requires the server binary to be built with --features ollama".to_string())
            }
        }
        other => Err(format!("unknown MEMORIA_LLM_PROVIDER value {other:?} (expected \"local\" or \"ollama\")")),
    }
}

fn resolve_embedding_provider(
    provider_choice: Option<&str>,
    model: Option<String>,
    base_url: Option<String>,
) -> Result<(Box<dyn core::embedding::EmbeddingProvider + Send + Sync>, String), String> {
    match provider_choice.unwrap_or("local") {
        "local" => Ok((
            Box::new(LocalHashEmbeddingProvider::new()),
            "LocalHashEmbeddingProvider (local, non-AI; see ADR-12)".to_string(),
        )),
        "ollama" => {
            #[cfg(feature = "ollama")]
            {
                let model = model.unwrap_or_else(|| "nomic-embed-text".to_string());
                let resolved_base_url = base_url.clone().unwrap_or_else(|| "http://localhost:11434".to_string());
                let config = core::embedding::EmbeddingConfig { model: model.clone(), base_url, api_key: None, dimensions: None };
                let provider = core::embedding::OllamaEmbeddingProvider::from_config(config).map_err(|err| err.to_string())?;
                let label = format!("OllamaEmbeddingProvider (model={model}, base_url={resolved_base_url}; see ADR-24)");
                Ok((Box::new(provider), label))
            }
            #[cfg(not(feature = "ollama"))]
            {
                let _ = (model, base_url);
                Err("MEMORIA_EMBEDDING_PROVIDER=ollama requires the server binary to be built with --features ollama".to_string())
            }
        }
        other => Err(format!("unknown MEMORIA_EMBEDDING_PROVIDER value {other:?} (expected \"local\" or \"ollama\")")),
    }
}

fn validate_cors_origin(origin: Option<String>) -> Result<Option<String>, String> {
    if origin.as_deref() == Some("*") {
        return Err(
            "MEMORIA_CORS_ORIGIN must not be \"*\" -- CORS is restricted to exactly one configured origin, never a wildcard (see ADR-17)"
                .to_string(),
        );
    }
    Ok(origin)
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

struct ServerState<L, E, V>
where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    memory: Memory<L, E, V>,
    token: Option<String>,
    rate_limiter: RateLimiter,
    cors_origin: Option<String>,
    store_path: PathBuf,
    auth_store: auth_store::AuthStore,
    auth_store_path: PathBuf,
    jwt_secret: Vec<u8>,
}

fn dispatch_request<L, E, V>(state: &ServerState<L, E, V>, req: &ParsedRequest, peer_ip: IpAddr) -> (u16, Vec<u8>)
where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    if !state.rate_limiter.check_and_consume(peer_ip) {
        return (429, error_body("rate limit exceeded"));
    }
    if req.method == "GET" && req.path == "/health" {
        return handle_health();
    }
    if req.method == "GET" && req.path == "/ready" {
        return handle_ready(&state.memory);
    }
    if req.method == "POST" && req.path == "/auth/register" {
        let result = handle_auth_register(&state.auth_store, &state.jwt_secret, &req.body);
        if result.0 == 201 {
            let _ = state.auth_store.save(&state.auth_store_path);
        }
        return result;
    }
    if req.method == "GET" && req.path == "/auth/setup-status" {
        return handle_auth_setup_status(&state.auth_store);
    }
    if req.method == "POST" && req.path == "/auth/login" {
        let result = handle_auth_login(&state.auth_store, &state.jwt_secret, &req.body);
        if result.0 == 200 {
            let _ = state.auth_store.save(&state.auth_store_path);
        }
        return result;
    }
    if req.method == "POST" && req.path == "/auth/refresh" {
        let result = handle_auth_refresh(&state.auth_store, &state.jwt_secret, &req.body);
        if result.0 == 200 {
            let _ = state.auth_store.save(&state.auth_store_path);
        }
        return result;
    }
    if !is_authorized(state.token.as_deref(), &state.auth_store, &state.jwt_secret, &req.headers) {
        return (401, error_body("unauthorized"));
    }
    if req.method == "GET" && req.path == "/auth/me" {
        return handle_auth_me_get(&state.auth_store, &state.jwt_secret, &req.headers);
    }
    if req.method == "PATCH" && req.path == "/auth/me" {
        let result = handle_auth_me_patch(&state.auth_store, &state.jwt_secret, &req.headers, &req.body);
        if result.0 == 200 {
            let _ = state.auth_store.save(&state.auth_store_path);
        }
        return result;
    }
    if req.method == "POST" && req.path == "/auth/change-password" {
        let result = handle_auth_change_password(&state.auth_store, &state.jwt_secret, &req.headers, &req.body);
        if result.0 == 200 {
            let _ = state.auth_store.save(&state.auth_store_path);
        }
        return result;
    }
    if req.method == "POST" && req.path == "/auth/onboarding-complete" {
        let result = handle_auth_onboarding_complete(&state.auth_store, &state.jwt_secret, &req.headers);
        if result.0 == 200 {
            let _ = state.auth_store.save(&state.auth_store_path);
        }
        return result;
    }
    let result = route(&state.memory, req);
    if is_successful_mutation(&req.method, &req.path, result.0) {
        let _ = save_store(&state.memory, &state.store_path);
    }
    result
}

async fn handle_connection<L, E, V>(mut stream: TcpStream, state: Arc<ServerState<L, E, V>>, peer_ip: IpAddr)
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
            let cors_header = cors_allow_origin_header(state.cors_origin.as_deref(), request_origin);

            if req.method == "OPTIONS" && request_origin.is_some() {
                let mut headers: Vec<(String, String)> = cors_header.into_iter().collect();
                headers.push(("Access-Control-Allow-Methods".to_string(), "GET, POST, PUT, PATCH, DELETE".to_string()));
                headers.push(("Access-Control-Allow-Headers".to_string(), "Content-Type, Authorization, X-API-Key".to_string()));
                eprintln!("{}", format_log_line(&req.method, &req.path, 204));
                let response = build_response_with_headers(204, b"", &headers);
                let _ = stream.write_all(&response).await;
                return;
            }

            let blocking_state = Arc::clone(&state);
            let (status, body) = tokio::task::spawn_blocking(move || {
                let (status, body) = dispatch_request(&blocking_state, &req, peer_ip);
                eprintln!("{}", format_log_line(&req.method, &req.path, status));
                (status, body)
            })
            .await
            .expect("request-handling task should not panic");
            let response = build_response_with_headers(status, &body, &cors_header.into_iter().collect::<Vec<_>>());
            let _ = stream.write_all(&response).await;
            return;
        }
    }
}

async fn serve<L, E, V>(
    listener: TcpListener,
    state: Arc<ServerState<L, E, V>>,
    mut shutdown: impl std::future::Future<Output = ()> + Unpin,
) where
    L: core::llm::LlmProvider + Send + Sync + 'static,
    E: core::embedding::EmbeddingProvider + Send + Sync + 'static,
    V: core::vector_store::VectorStore + Send + Sync + 'static,
{
    let mut in_flight = tokio::task::JoinSet::new();
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let Ok((stream, peer_addr)) = accepted else {
                    continue;
                };
                let state = Arc::clone(&state);
                in_flight.spawn(handle_connection(stream, state, peer_addr.ip()));
            }
            () = &mut shutdown => {
                break;
            }
        }
    }
    while in_flight.join_next().await.is_some() {}
}

fn main() {
    let api_key_env = std::env::var("MEMORIA_API_KEY").ok();
    let allow_no_auth_env = std::env::var("MEMORIA_ALLOW_NO_AUTH").ok();
    let token = match resolve_auth_config(api_key_env, allow_no_auth_env) {
        Ok(token) => token,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };
    let cors_origin = match validate_cors_origin(resolve_cors_origin(std::env::var("MEMORIA_CORS_ORIGIN").ok())) {
        Ok(origin) => origin,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };

    let llm_provider_choice = std::env::var("MEMORIA_LLM_PROVIDER").ok();
    let (llm_provider, llm_label) = match resolve_llm_provider(
        llm_provider_choice.as_deref(),
        std::env::var("MEMORIA_LLM_MODEL").ok(),
        std::env::var("MEMORIA_LLM_BASE_URL").ok(),
    ) {
        Ok(resolved) => resolved,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };
    let embedding_provider_choice = std::env::var("MEMORIA_EMBEDDING_PROVIDER").ok();
    let (embedding_provider, embedding_label) = match resolve_embedding_provider(
        embedding_provider_choice.as_deref(),
        std::env::var("MEMORIA_EMBEDDING_MODEL").ok(),
        std::env::var("MEMORIA_EMBEDDING_BASE_URL").ok(),
    ) {
        Ok(resolved) => resolved,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };
    eprintln!("llm provider: {llm_label}");
    eprintln!("embedding provider: {embedding_label}");

    let jwt_secret = match resolve_jwt_secret(std::env::var("MEMORIA_JWT_SECRET").ok()) {
        Ok(secret) => secret,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let store_path = resolve_store_path(std::env::var("MEMORIA_STORE_PATH").ok().as_deref(), &home);
    eprintln!("store: {}", store_path.display());
    let store = load_store(&store_path);

    let auth_store_path = resolve_auth_store_path(std::env::var("MEMORIA_AUTH_STORE_PATH").ok().as_deref(), &home);
    eprintln!("auth store: {}", auth_store_path.display());
    let auth_store = auth_store::AuthStore::load(&auth_store_path);

    let rate_limiter = RateLimiter::new(RATE_LIMIT_CAPACITY, RATE_LIMIT_REFILL_PER_SEC);
    let memory = Memory::new(llm_provider, embedding_provider, store);
    let state_outliving_the_runtime =
        Arc::new(ServerState { memory, token, rate_limiter, cors_origin, store_path, auth_store, auth_store_path, jwt_secret });

    let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    runtime.block_on(async {
        let state = Arc::clone(&state_outliving_the_runtime);
        let listener = TcpListener::bind("127.0.0.1:8080").await.expect("failed to bind to 127.0.0.1:8080");
        let shutdown = Box::pin(async {
            let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install SIGTERM handler");
            tokio::select! {
                _ = sigterm.recv() => {}
                _ = tokio::signal::ctrl_c() => {}
            }
        });
        serve(listener, state, shutdown).await;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn test_ip(last_octet: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(127, 0, 0, last_octet))
    }

    fn test_store_path() -> PathBuf {
        std::env::temp_dir().join("memoria-server-tests-scratch-store.json")
    }

    fn test_auth_store_path() -> PathBuf {
        std::env::temp_dir().join(format!("memoria-server-tests-scratch-auth-store-{}.json", std::process::id()))
    }

    fn test_jwt_secret() -> Vec<u8> {
        b"test-jwt-secret".to_vec()
    }

    fn test_state<L, E, V>(
        memory: Memory<L, E, V>,
        token: Option<String>,
        rate_limiter: RateLimiter,
        cors_origin: Option<String>,
        store_path: PathBuf,
    ) -> Arc<ServerState<L, E, V>>
    where
        L: core::llm::LlmProvider,
        E: core::embedding::EmbeddingProvider,
        V: core::vector_store::VectorStore,
    {
        Arc::new(ServerState {
            memory,
            token,
            rate_limiter,
            cors_origin,
            store_path,
            auth_store: auth_store::AuthStore::new(),
            auth_store_path: test_auth_store_path(),
            jwt_secret: test_jwt_secret(),
        })
    }

    async fn post_over_tcp(addr: std::net::SocketAddr, path: &str, body: &[u8]) -> (u16, Vec<u8>) {
        let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
        let request = format!("POST {path} HTTP/1.1\r\nContent-Length: {}\r\n\r\n", body.len());
        stream.write_all(request.as_bytes()).await.expect("write should succeed");
        stream.write_all(body).await.expect("write should succeed");
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.expect("read should succeed");
        let text = String::from_utf8(response).expect("response should be valid utf8");
        let status: u16 = text.split_whitespace().nth(1).expect("a status line").parse().expect("a numeric status");
        let json_start = text.find("\r\n\r\n").expect("a header/body separator") + 4;
        (status, text.as_bytes()[json_start..].to_vec())
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
    fn write_atomically_creates_the_file_with_the_given_contents() {
        let dir = std::env::temp_dir().join(format!("memoria-server-atomic-test-{}-1", std::process::id()));
        let path = dir.join("store.json");
        write_atomically(&path, b"hello").expect("write should succeed");
        assert_eq!(std::fs::read(&path).expect("file should exist"), b"hello");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_atomically_leaves_no_temp_file_behind() {
        let dir = std::env::temp_dir().join(format!("memoria-server-atomic-test-{}-2", std::process::id()));
        let path = dir.join("store.json");
        write_atomically(&path, b"hello").expect("write should succeed");
        let entries: Vec<_> = std::fs::read_dir(&dir).expect("dir should exist").filter_map(Result::ok).collect();
        assert_eq!(entries.len(), 1, "expected exactly the target file, no leftover temp file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_atomically_overwrites_an_existing_file() {
        let dir = std::env::temp_dir().join(format!("memoria-server-atomic-test-{}-3", std::process::id()));
        let path = dir.join("store.json");
        write_atomically(&path, b"first").expect("write should succeed");
        write_atomically(&path, b"second").expect("write should succeed");
        assert_eq!(std::fs::read(&path).expect("file should exist"), b"second");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_atomically_never_corrupts_the_file_under_concurrent_writers() {
        let dir = std::env::temp_dir().join(format!("memoria-server-atomic-test-{}-4", std::process::id()));
        let path = Arc::new(dir.join("store.json"));
        let handles: Vec<_> = (0..20)
            .map(|i| {
                let path = Arc::clone(&path);
                std::thread::spawn(move || {
                    let payload = format!("payload-from-writer-{i}");
                    write_atomically(&path, payload.as_bytes()).expect("write should succeed");
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("writer thread should not panic");
        }
        let final_contents = std::fs::read_to_string(&*path).expect("file should exist");
        assert!(
            final_contents.starts_with("payload-from-writer-"),
            "final file content must be exactly one writer's complete payload, got: {final_contents:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_store_path_defaults_to_a_server_specific_file_distinct_from_the_cli() {
        let path = resolve_store_path(None, "/home/alice");
        assert_eq!(path, PathBuf::from("/home/alice/.memoria/server-store.json"));
    }

    #[test]
    fn resolve_store_path_env_override_wins() {
        let path = resolve_store_path(Some("/custom/path.json"), "/home/alice");
        assert_eq!(path, PathBuf::from("/custom/path.json"));
    }

    #[test]
    fn resolve_auth_store_path_defaults_to_a_file_distinct_from_the_vector_store() {
        let path = resolve_auth_store_path(None, "/home/alice");
        assert_eq!(path, PathBuf::from("/home/alice/.memoria/auth-store.json"));
    }

    #[test]
    fn resolve_auth_store_path_env_override_wins() {
        let path = resolve_auth_store_path(Some("/custom/auth.json"), "/home/alice");
        assert_eq!(path, PathBuf::from("/custom/auth.json"));
    }

    #[test]
    fn resolve_jwt_secret_with_a_configured_value_resolves_to_that_value() {
        let secret = resolve_jwt_secret(Some("real-secret".to_string())).expect("a configured secret must resolve");
        assert_eq!(secret, b"real-secret".to_vec());
    }

    #[test]
    fn resolve_jwt_secret_with_nothing_set_fails_closed() {
        assert!(resolve_jwt_secret(None).is_err());
    }

    #[test]
    fn resolve_jwt_secret_with_an_empty_value_fails_closed() {
        assert!(resolve_jwt_secret(Some(String::new())).is_err(), "an empty MEMORIA_JWT_SECRET must not be treated as configured");
    }

    #[test]
    fn is_successful_mutation_true_for_create() {
        assert!(is_successful_mutation("POST", "/memories", 201));
    }

    #[test]
    fn is_successful_mutation_true_for_update() {
        assert!(is_successful_mutation("PUT", "/memories/rec-1", 200));
    }

    #[test]
    fn is_successful_mutation_true_for_delete() {
        assert!(is_successful_mutation("DELETE", "/memories/rec-1", 200));
    }

    #[test]
    fn is_successful_mutation_false_for_search() {
        assert!(!is_successful_mutation("POST", "/memories/search", 200));
    }

    #[test]
    fn is_successful_mutation_false_for_get_and_list() {
        assert!(!is_successful_mutation("GET", "/memories", 200));
        assert!(!is_successful_mutation("GET", "/memories/rec-1", 200));
    }

    #[test]
    fn is_successful_mutation_false_when_the_status_indicates_failure() {
        assert!(!is_successful_mutation("POST", "/memories", 400));
        assert!(!is_successful_mutation("PUT", "/memories/rec-1", 404));
        assert!(!is_successful_mutation("DELETE", "/memories/rec-1", 500));
    }

    #[test]
    fn load_store_with_no_file_present_is_empty() {
        let dir = std::env::temp_dir().join(format!("memoria-server-load-test-{}", std::process::id()));
        let path = dir.join("does-not-exist.json");
        let store = load_store(&path);
        assert!(store.list(0, usize::MAX).expect("list should succeed").is_empty());
    }

    #[test]
    fn save_then_load_round_trips_records() {
        let dir = std::env::temp_dir().join(format!("memoria-server-roundtrip-test-{}", std::process::id()));
        let path = dir.join("store.json");
        let memory = Arc::new(Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new()));
        memory.add(&[Message::new(Role::User, "Alice is an engineer.".to_string())], scope_from_optional(Some("alice".to_string()), None, None)).expect("add should succeed");

        save_store(&memory, &path).expect("save should succeed");
        let reloaded = load_store(&path);
        assert_eq!(reloaded.list(0, usize::MAX).expect("list should succeed").len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_llm_provider_defaults_to_local() {
        let (_, label) = resolve_llm_provider(None, None, None).expect("expected a provider");
        assert!(label.contains("LocalSentenceLlmProvider"), "got: {label}");
    }

    #[test]
    fn resolve_llm_provider_rejects_an_unknown_choice() {
        assert!(resolve_llm_provider(Some("bogus"), None, None).is_err());
    }

    #[test]
    fn resolve_embedding_provider_defaults_to_local() {
        let (_, label) = resolve_embedding_provider(None, None, None).expect("expected a provider");
        assert!(label.contains("LocalHashEmbeddingProvider"), "got: {label}");
    }

    #[test]
    fn resolve_embedding_provider_rejects_an_unknown_choice() {
        assert!(resolve_embedding_provider(Some("bogus"), None, None).is_err());
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn resolve_llm_provider_ollama_choice_builds_with_defaults() {
        let (_, label) = resolve_llm_provider(Some("ollama"), None, None).expect("expected a provider");
        assert!(label.contains("qwen2.5:0.5b"), "got: {label}");
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn resolve_embedding_provider_ollama_choice_builds_with_defaults() {
        let (_, label) = resolve_embedding_provider(Some("ollama"), None, None).expect("expected a provider");
        assert!(label.contains("nomic-embed-text"), "got: {label}");
    }

    #[cfg(not(feature = "ollama"))]
    #[test]
    fn resolve_llm_provider_ollama_choice_fails_clearly_without_the_feature() {
        assert!(resolve_llm_provider(Some("ollama"), None, None).is_err());
    }

    #[test]
    fn validate_cors_origin_accepts_a_real_origin() {
        assert_eq!(validate_cors_origin(Some("https://example.com".to_string())), Ok(Some("https://example.com".to_string())));
    }

    #[test]
    fn validate_cors_origin_accepts_none() {
        assert_eq!(validate_cors_origin(None), Ok(None));
    }

    #[test]
    fn validate_cors_origin_rejects_a_literal_wildcard() {
        assert!(validate_cors_origin(Some("*".to_string())).is_err());
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
    fn handle_auth_register_happy_path_returns_201_with_real_tokens() {
        let store = auth_store::AuthStore::new();
        let body = br#"{"name":"Alice","email":"alice@example.com","password":"correct horse battery staple"}"#;
        let (status, response_body) = handle_auth_register(&store, b"jwt-secret", body);
        assert_eq!(status, 201);
        let response: AuthTokenResponse = serde_json::from_slice(&response_body).expect("expected valid JSON");
        assert_eq!(response.token_type, "bearer");
        assert!(crypto::decode_jwt(&response.access_token, b"jwt-secret", auth_store::unix_now()).is_ok());
        assert!(!response.refresh_token.is_empty());
    }

    #[test]
    fn handle_auth_register_rejects_a_second_registration() {
        let store = auth_store::AuthStore::new();
        let body = br#"{"name":"Alice","email":"alice@example.com","password":"password-one"}"#;
        handle_auth_register(&store, b"jwt-secret", body);
        let second_body = br#"{"name":"Bob","email":"bob@example.com","password":"password-two"}"#;
        let (status, _) = handle_auth_register(&store, b"jwt-secret", second_body);
        assert_eq!(status, 403);
    }

    #[test]
    fn handle_auth_register_rejects_malformed_json_body() {
        let store = auth_store::AuthStore::new();
        let (status, _) = handle_auth_register(&store, b"jwt-secret", b"not json");
        assert_eq!(status, 400);
    }

    #[test]
    fn handle_auth_register_rejects_an_empty_password() {
        let store = auth_store::AuthStore::new();
        let body = br#"{"name":"Alice","email":"alice@example.com","password":""}"#;
        let (status, _) = handle_auth_register(&store, b"jwt-secret", body);
        assert_eq!(status, 400);
    }

    #[test]
    fn handle_auth_setup_status_reflects_real_store_state() {
        let store = auth_store::AuthStore::new();
        let (_, before) = handle_auth_setup_status(&store);
        let before: SetupStatusResponse = serde_json::from_slice(&before).expect("expected valid JSON");
        assert!(!before.setup_complete);

        store.insert_user(auth_store::User::new(
            "Alice".to_string(),
            "alice@example.com".to_string(),
            "hash".to_string(),
            auth_store::Role::Admin,
        ));
        let (_, after) = handle_auth_setup_status(&store);
        let after: SetupStatusResponse = serde_json::from_slice(&after).expect("expected valid JSON");
        assert!(after.setup_complete);
    }

    #[test]
    fn handle_auth_login_happy_path_returns_200_with_real_tokens() {
        let store = auth_store::AuthStore::new();
        seed_user_cheaply(&store, "correct horse battery staple");

        let login_body = br#"{"email":"alice@example.com","password":"correct horse battery staple"}"#;
        let (status, response_body) = handle_auth_login(&store, b"jwt-secret", login_body);
        assert_eq!(status, 200);
        let response: AuthTokenResponse = serde_json::from_slice(&response_body).expect("expected valid JSON");
        assert!(crypto::decode_jwt(&response.access_token, b"jwt-secret", auth_store::unix_now()).is_ok());
    }

    #[test]
    fn handle_auth_login_rejects_the_wrong_password() {
        let store = auth_store::AuthStore::new();
        seed_user_cheaply(&store, "correct horse battery staple");

        let login_body = br#"{"email":"alice@example.com","password":"wrong password"}"#;
        let (status, _) = handle_auth_login(&store, b"jwt-secret", login_body);
        assert_eq!(status, 401);
    }

    #[test]
    fn handle_auth_login_rejects_an_unknown_email_taking_the_same_code_path_as_a_wrong_password() {
        let store = auth_store::AuthStore::new();
        let login_body = br#"{"email":"nobody@example.com","password":"anything"}"#;
        let (status, response_body) = handle_auth_login(&store, b"jwt-secret", login_body);
        assert_eq!(status, 401);
        let response: ErrorResponse = serde_json::from_slice(&response_body).expect("expected valid JSON");
        assert_eq!(response.error, "invalid email or password");
    }

    #[test]
    fn handle_auth_login_rejects_malformed_json_body() {
        let store = auth_store::AuthStore::new();
        let (status, _) = handle_auth_login(&store, b"jwt-secret", b"not json");
        assert_eq!(status, 400);
    }

    #[test]
    fn handle_auth_refresh_happy_path_rotates_and_returns_a_new_access_token() {
        let store = auth_store::AuthStore::new();
        let tokens = seed_user_cheaply_and_issue_tokens(&store, "correct horse battery staple");

        let refresh_body = serde_json::to_vec(&serde_json::json!({"refresh_token": tokens.refresh_token})).expect("valid JSON");
        let (status, response_body) = handle_auth_refresh(&store, b"jwt-secret", &refresh_body);
        assert_eq!(status, 200);
        let refreshed: AuthTokenResponse = serde_json::from_slice(&response_body).expect("expected valid JSON");
        assert_ne!(refreshed.refresh_token, tokens.refresh_token);
        assert!(crypto::decode_jwt(&refreshed.access_token, b"jwt-secret", auth_store::unix_now()).is_ok());
    }

    #[test]
    fn handle_auth_refresh_rejects_reuse_of_an_already_rotated_token() {
        let store = auth_store::AuthStore::new();
        let tokens = seed_user_cheaply_and_issue_tokens(&store, "correct horse battery staple");

        let refresh_body = serde_json::to_vec(&serde_json::json!({"refresh_token": tokens.refresh_token})).expect("valid JSON");
        handle_auth_refresh(&store, b"jwt-secret", &refresh_body);
        let (status, _) = handle_auth_refresh(&store, b"jwt-secret", &refresh_body);
        assert_eq!(status, 401);
    }

    #[test]
    fn handle_auth_refresh_rejects_an_unknown_token() {
        let store = auth_store::AuthStore::new();
        let refresh_body = br#"{"refresh_token":"never-issued"}"#;
        let (status, _) = handle_auth_refresh(&store, b"jwt-secret", refresh_body);
        assert_eq!(status, 401);
    }

    #[test]
    fn handle_auth_refresh_rejects_malformed_json_body() {
        let store = auth_store::AuthStore::new();
        let (status, _) = handle_auth_refresh(&store, b"jwt-secret", b"not json");
        assert_eq!(status, 400);
    }

    const CHEAP_NON_PRODUCTION_ITERATIONS_FOR_TEST_SETUP: u32 = 10;

    fn seed_user_cheaply(store: &auth_store::AuthStore, password: &str) -> auth_store::User {
        let password_hash = crypto::hash_password_with_iterations(password, CHEAP_NON_PRODUCTION_ITERATIONS_FOR_TEST_SETUP);
        let user =
            auth_store::User::new("Alice".to_string(), "alice@example.com".to_string(), password_hash, auth_store::Role::Admin);
        store.insert_user(user.clone());
        user
    }

    fn seed_user_cheaply_and_issue_tokens(store: &auth_store::AuthStore, password: &str) -> AuthTokenResponse {
        let user = seed_user_cheaply(store, password);
        issue_auth_tokens(store, b"jwt-secret", &user)
    }

    fn seed_user_cheaply_and_get_access_token(store: &auth_store::AuthStore, password: &str) -> String {
        seed_user_cheaply_and_issue_tokens(store, password).access_token
    }

    fn bearer_headers(token: &str) -> Vec<(String, String)> {
        vec![("Authorization".to_string(), format!("Bearer {token}"))]
    }

    #[test]
    fn handle_auth_me_get_returns_the_real_authenticated_users_profile() {
        let store = auth_store::AuthStore::new();
        let token = seed_user_cheaply_and_get_access_token(&store, "correct horse battery staple");
        let (status, body) = handle_auth_me_get(&store, b"jwt-secret", &bearer_headers(&token));
        assert_eq!(status, 200);
        let profile: UserProfileResponse = serde_json::from_slice(&body).expect("expected valid JSON");
        assert_eq!(profile.email, "alice@example.com");
        assert_eq!(profile.role, "admin");
    }

    #[test]
    fn handle_auth_me_get_rejects_a_missing_token() {
        let store = auth_store::AuthStore::new();
        let (status, _) = handle_auth_me_get(&store, b"jwt-secret", &[]);
        assert_eq!(status, 401);
    }

    #[test]
    fn handle_auth_me_get_rejects_a_token_signed_by_a_different_secret() {
        let store = auth_store::AuthStore::new();
        let token = seed_user_cheaply_and_get_access_token(&store, "correct horse battery staple");
        let (status, _) = handle_auth_me_get(&store, b"a-different-secret", &bearer_headers(&token));
        assert_eq!(status, 401);
    }

    #[test]
    fn handle_auth_me_patch_updates_the_real_stored_name() {
        let store = auth_store::AuthStore::new();
        let token = seed_user_cheaply_and_get_access_token(&store, "correct horse battery staple");
        let body = br#"{"name":"Alicia"}"#;
        let (status, response) = handle_auth_me_patch(&store, b"jwt-secret", &bearer_headers(&token), body);
        assert_eq!(status, 200);
        let profile: UserProfileResponse = serde_json::from_slice(&response).expect("expected valid JSON");
        assert_eq!(profile.name, "Alicia");
        assert_eq!(profile.email, "alice@example.com", "email must be unchanged when not provided");
    }

    #[test]
    fn handle_auth_me_patch_rejects_a_missing_token() {
        let store = auth_store::AuthStore::new();
        let (status, _) = handle_auth_me_patch(&store, b"jwt-secret", &[], br#"{"name":"X"}"#);
        assert_eq!(status, 401);
    }

    #[test]
    fn handle_auth_change_password_then_login_succeeds_with_the_new_password() {
        let store = auth_store::AuthStore::new();
        let token = seed_user_cheaply_and_get_access_token(&store, "correct horse battery staple");
        let change_body = br#"{"current_password":"correct horse battery staple","new_password":"a brand new password"}"#;
        let (status, _) = handle_auth_change_password(&store, b"jwt-secret", &bearer_headers(&token), change_body);
        assert_eq!(status, 200);

        let login_body = br#"{"email":"alice@example.com","password":"a brand new password"}"#;
        let (login_status, _) = handle_auth_login(&store, b"jwt-secret", login_body);
        assert_eq!(login_status, 200);

        let old_login_body = br#"{"email":"alice@example.com","password":"correct horse battery staple"}"#;
        let (old_login_status, _) = handle_auth_login(&store, b"jwt-secret", old_login_body);
        assert_eq!(old_login_status, 401, "the old password must no longer work");
    }

    #[test]
    fn handle_auth_change_password_rejects_the_wrong_current_password() {
        let store = auth_store::AuthStore::new();
        let token = seed_user_cheaply_and_get_access_token(&store, "correct horse battery staple");
        let body = br#"{"current_password":"wrong password","new_password":"a brand new password"}"#;
        let (status, _) = handle_auth_change_password(&store, b"jwt-secret", &bearer_headers(&token), body);
        assert_eq!(status, 401);
    }

    #[test]
    fn handle_auth_change_password_rejects_an_empty_new_password() {
        let store = auth_store::AuthStore::new();
        let token = seed_user_cheaply_and_get_access_token(&store, "correct horse battery staple");
        let body = br#"{"current_password":"correct horse battery staple","new_password":""}"#;
        let (status, _) = handle_auth_change_password(&store, b"jwt-secret", &bearer_headers(&token), body);
        assert_eq!(status, 400);
    }

    #[test]
    fn handle_auth_onboarding_complete_flips_the_real_flag() {
        let store = auth_store::AuthStore::new();
        let token = seed_user_cheaply_and_get_access_token(&store, "correct horse battery staple");
        let (status, response) = handle_auth_onboarding_complete(&store, b"jwt-secret", &bearer_headers(&token));
        assert_eq!(status, 200);
        let profile: UserProfileResponse = serde_json::from_slice(&response).expect("expected valid JSON");
        assert!(profile.onboarding_complete);
    }

    #[test]
    fn handle_auth_onboarding_complete_rejects_a_missing_token() {
        let store = auth_store::AuthStore::new();
        let (status, _) = handle_auth_onboarding_complete(&store, b"jwt-secret", &[]);
        assert_eq!(status, 401);
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
    fn handle_search_memory_honors_a_threshold() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        handle_create_memory(&memory, br#"{"content":"Alice is an engineer.","user_id":"alice"}"#);

        let (status, body) = handle_search_memory(&memory, br#"{"query":"engineer","user_id":"alice","threshold":0.0}"#);
        assert_eq!(status, 200);
        let response: SearchMemoryResponse = serde_json::from_slice(&body).expect("expected valid JSON");
        assert!(response.results.is_empty(), "an unreachably strict threshold should filter out the hash-based placeholder's match");
    }

    #[test]
    fn handle_search_memory_rejects_a_negative_threshold() {
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let (status, _) = handle_search_memory(&memory, br#"{"query":"anything","user_id":"alice","threshold":-1.0}"#);
        assert_eq!(status, 400);
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
    fn is_authorized_with_no_configured_admin_key_allows_anything() {
        let store = auth_store::AuthStore::new();
        assert!(is_authorized(None, &store, b"jwt-secret", &[]));
    }

    #[test]
    fn is_authorized_with_correct_admin_bearer_token_succeeds() {
        let store = auth_store::AuthStore::new();
        let headers = vec![("Authorization".to_string(), "Bearer secret".to_string())];
        assert!(is_authorized(Some("secret"), &store, b"jwt-secret", &headers));
    }

    #[test]
    fn is_authorized_with_wrong_bearer_token_fails() {
        let store = auth_store::AuthStore::new();
        let headers = vec![("Authorization".to_string(), "Bearer wrong".to_string())];
        assert!(!is_authorized(Some("secret"), &store, b"jwt-secret", &headers));
    }

    #[test]
    fn is_authorized_with_missing_header_fails() {
        let store = auth_store::AuthStore::new();
        assert!(!is_authorized(Some("secret"), &store, b"jwt-secret", &[]));
    }

    #[test]
    fn is_authorized_with_header_missing_bearer_prefix_fails() {
        let store = auth_store::AuthStore::new();
        let headers = vec![("Authorization".to_string(), "secret".to_string())];
        assert!(!is_authorized(Some("secret"), &store, b"jwt-secret", &headers));
    }

    #[test]
    fn is_authorized_with_a_real_valid_jwt_succeeds() {
        let store = auth_store::AuthStore::new();
        let claims = serde_json::json!({"sub": "user-1", "exp": auth_store::unix_now() + 900});
        let token = crypto::encode_jwt(&claims, b"jwt-secret");
        let headers = vec![("Authorization".to_string(), format!("Bearer {token}"))];
        assert!(is_authorized(Some("admin-secret"), &store, b"jwt-secret", &headers));
    }

    #[test]
    fn is_authorized_with_an_expired_jwt_fails() {
        let store = auth_store::AuthStore::new();
        let claims = serde_json::json!({"sub": "user-1", "exp": 0});
        let token = crypto::encode_jwt(&claims, b"jwt-secret");
        let headers = vec![("Authorization".to_string(), format!("Bearer {token}"))];
        assert!(!is_authorized(Some("admin-secret"), &store, b"jwt-secret", &headers));
    }

    #[test]
    fn is_authorized_with_a_jwt_signed_by_a_different_secret_fails() {
        let store = auth_store::AuthStore::new();
        let claims = serde_json::json!({"sub": "user-1", "exp": auth_store::unix_now() + 900});
        let token = crypto::encode_jwt(&claims, b"a-different-secret");
        let headers = vec![("Authorization".to_string(), format!("Bearer {token}"))];
        assert!(!is_authorized(Some("admin-secret"), &store, b"jwt-secret", &headers));
    }

    #[test]
    fn is_authorized_with_a_real_active_api_key_succeeds() {
        let store = auth_store::AuthStore::new();
        let key_hash = crypto::sha256_hex(b"a-real-api-key");
        store.insert_api_key(auth_store::ApiKey::new("user-1".to_string(), "ci".to_string(), key_hash, "prefix".to_string()));
        let headers = vec![("X-API-Key".to_string(), "a-real-api-key".to_string())];
        assert!(is_authorized(Some("admin-secret"), &store, b"jwt-secret", &headers));
    }

    #[test]
    fn is_authorized_with_a_revoked_api_key_fails() {
        let store = auth_store::AuthStore::new();
        let key_hash = crypto::sha256_hex(b"a-real-api-key");
        let mut key = auth_store::ApiKey::new("user-1".to_string(), "ci".to_string(), key_hash, "prefix".to_string());
        key.revoked_at = Some(auth_store::unix_now());
        store.insert_api_key(key);
        let headers = vec![("X-API-Key".to_string(), "a-real-api-key".to_string())];
        assert!(!is_authorized(Some("admin-secret"), &store, b"jwt-secret", &headers));
    }

    #[test]
    fn is_authorized_with_an_unknown_api_key_fails() {
        let store = auth_store::AuthStore::new();
        let headers = vec![("X-API-Key".to_string(), "never-issued".to_string())];
        assert!(!is_authorized(Some("admin-secret"), &store, b"jwt-secret", &headers));
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
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let state = test_state(memory, None, RateLimiter::new(1000.0, 1000.0), None, test_store_path());
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

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
    fn server_handles_a_real_registration_over_tcp_even_with_auth_configured() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let state = test_state(
                memory,
                Some("a-real-configured-bearer-token".to_string()),
                RateLimiter::new(1000.0, 1000.0),
                None,
                test_store_path(),
            );
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            let body = br#"{"name":"Alice","email":"alice@example.com","password":"correct horse battery staple"}"#;
            let request = format!("POST /auth/register HTTP/1.1\r\nContent-Length: {}\r\n\r\n", body.len());
            stream.write_all(request.as_bytes()).await.expect("write should succeed");
            stream.write_all(body).await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");
            assert!(response_text.starts_with("HTTP/1.1 201 Created\r\n"), "got: {response_text}");
            assert!(response_text.contains("\"access_token\""));
        });
    }

    #[test]
    fn server_handles_a_real_register_login_refresh_flow_over_tcp() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let state = test_state(memory, None, RateLimiter::new(1000.0, 1000.0), None, test_store_path());
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

            let register_body = br#"{"name":"Alice","email":"alice@example.com","password":"correct horse battery staple"}"#;
            let (register_status, _) = post_over_tcp(addr, "/auth/register", register_body).await;
            assert_eq!(register_status, 201);

            let login_body = br#"{"email":"alice@example.com","password":"correct horse battery staple"}"#;
            let (login_status, login_response) = post_over_tcp(addr, "/auth/login", login_body).await;
            assert_eq!(login_status, 200);
            let logged_in: AuthTokenResponse = serde_json::from_slice(&login_response).expect("valid JSON");
            let decoded = crypto::decode_jwt(&logged_in.access_token, b"test-jwt-secret", auth_store::unix_now())
                .expect("a real login's JWT must decode and verify");
            assert_eq!(decoded["role"], "admin");

            let refresh_body =
                serde_json::to_vec(&serde_json::json!({"refresh_token": logged_in.refresh_token})).expect("valid JSON");
            let (refresh_status, refresh_response) = post_over_tcp(addr, "/auth/refresh", &refresh_body).await;
            assert_eq!(refresh_status, 200);
            let refreshed: AuthTokenResponse = serde_json::from_slice(&refresh_response).expect("valid JSON");
            assert_ne!(refreshed.refresh_token, logged_in.refresh_token);

            let (reuse_status, _) = post_over_tcp(addr, "/auth/refresh", &refresh_body).await;
            assert_eq!(reuse_status, 401, "a refresh token must not work a second time after it has already rotated");
        });
    }

    #[test]
    fn server_persists_a_real_create_request_to_disk_and_a_fresh_server_reloads_it() {
        let dir = std::env::temp_dir().join(format!("memoria-server-e2e-persist-test-{}", std::process::id()));
        let path = Arc::new(dir.join("store.json"));

        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let state = test_state(memory, None, RateLimiter::new(1000.0, 1000.0), None, (*path).clone());
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            let body = br#"{"content":"Bob is learning to play the guitar.","user_id":"bob"}"#;
            let request = format!("POST /memories HTTP/1.1\r\nContent-Length: {}\r\n\r\n", body.len());
            stream.write_all(request.as_bytes()).await.expect("write should succeed");
            stream.write_all(body).await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");
            assert!(response_text.starts_with("HTTP/1.1 201 Created\r\n"), "got: {response_text}");
        });

        let independently_reloaded_store = load_store(&path);
        assert_eq!(independently_reloaded_store.list(0, usize::MAX).expect("list should succeed").len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn server_returns_404_over_tcp_for_unknown_route() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let state = test_state(memory, None, RateLimiter::new(1000.0, 1000.0), None, test_store_path());
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

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
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let state = test_state(memory, Some("secret".to_string()), RateLimiter::new(1000.0, 1000.0), None, test_store_path());
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

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
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let state = test_state(memory, Some("secret".to_string()), RateLimiter::new(1000.0, 1000.0), None, test_store_path());
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

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
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let state = test_state(memory, None, RateLimiter::new(2.0, 0.0), None, test_store_path());
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

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
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let state = test_state(memory, None, RateLimiter::new(1.0, 1000.0), None, test_store_path());
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

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
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let state = test_state(memory, None, RateLimiter::new(1000.0, 1000.0), None, test_store_path());
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

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
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let cors_origin = Some("https://example.com".to_string());
            let state = test_state(memory, None, RateLimiter::new(1000.0, 1000.0), cors_origin, test_store_path());
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

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
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let cors_origin = Some("https://example.com".to_string());
            let state = test_state(memory, None, RateLimiter::new(1000.0, 1000.0), cors_origin, test_store_path());
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

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
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let cors_origin = Some("https://example.com".to_string());
            let state = test_state(memory, None, RateLimiter::new(1000.0, 1000.0), cors_origin, test_store_path());
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            let request = "OPTIONS /memories HTTP/1.1\r\nOrigin: https://example.com\r\nContent-Length: 0\r\n\r\n";
            stream.write_all(request.as_bytes()).await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");

            assert!(response_text.starts_with("HTTP/1.1 204"), "got: {response_text}");
            assert!(response_text.contains("Access-Control-Allow-Methods: GET, POST, PUT, PATCH, DELETE\r\n"), "got: {response_text}");
            assert!(
                response_text.contains("Access-Control-Allow-Headers: Content-Type, Authorization, X-API-Key\r\n"),
                "got: {response_text}"
            );
        });
    }

    #[test]
    fn server_answers_health_without_a_token_even_when_auth_is_configured() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let state = test_state(memory, Some("secret".to_string()), RateLimiter::new(1000.0, 1000.0), None, test_store_path());
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

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
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let state = test_state(memory, Some("secret".to_string()), RateLimiter::new(1000.0, 1000.0), None, test_store_path());
            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            stream.write_all(b"GET /ready HTTP/1.1\r\nContent-Length: 0\r\n\r\n").await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");

            assert!(response_text.starts_with("HTTP/1.1 200 OK\r\n"), "got: {response_text}");
            assert!(response_text.ends_with(r#"{"status":"ready"}"#), "got: {response_text}");
        });
    }

    #[test]
    fn server_drains_an_in_flight_request_before_shutdown_completes() {
        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
            let state = test_state(memory, None, RateLimiter::new(1000.0, 1000.0), None, test_store_path());
            let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
            let shutdown = Box::pin(async {
                let _ = shutdown_rx.await;
            });
            let serve_handle = tokio::spawn(serve(listener, state, shutdown));
            tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            let body = br#"{"content":"Alice is an engineer.","user_id":"alice"}"#;
            let request_head = format!("POST /memories HTTP/1.1\r\nContent-Length: {}\r\n\r\n", body.len());
            stream.write_all(request_head.as_bytes()).await.expect("write should succeed");

            shutdown_tx.send(()).expect("shutdown receiver should still be alive");
            tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

            stream.write_all(body).await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");
            let response_text = String::from_utf8(response).expect("response should be valid utf8");
            assert!(response_text.starts_with("HTTP/1.1 201 Created\r\n"), "got: {response_text}");

            tokio::time::timeout(tokio::time::Duration::from_secs(2), serve_handle)
                .await
                .expect("serve() should return promptly once the in-flight connection finishes")
                .expect("serve() task should not panic");
        });
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn server_does_not_panic_when_a_blocking_provider_is_called_from_the_async_runtime() {
        let llm_config = core::llm::LlmConfig {
            model: "qwen2.5:0.5b".to_string(),
            base_url: Some("http://127.0.0.1:1".to_string()),
            api_key: None,
            temperature: None,
        };
        let llm = core::llm::OllamaLlmProvider::from_config(llm_config).expect("valid config should construct");
        let memory = Memory::new(llm, LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let state_outliving_the_runtime = test_state(memory, None, RateLimiter::new(1000.0, 1000.0), None, test_store_path());

        let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        runtime.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind should succeed");
            let addr = listener.local_addr().expect("local_addr should succeed");
            let state = Arc::clone(&state_outliving_the_runtime);

            tokio::spawn(serve(listener, state, Box::pin(std::future::pending())));

            let mut stream = TcpStream::connect(addr).await.expect("connect should succeed");
            let body = br#"{"content":"hi","user_id":"u1"}"#;
            let request = format!("POST /memories HTTP/1.1\r\nContent-Length: {}\r\n\r\n", body.len());
            stream.write_all(request.as_bytes()).await.expect("write should succeed");
            stream.write_all(body).await.expect("write should succeed");

            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.expect("read should succeed");

            assert!(!response.is_empty(), "expected a real HTTP response, got none -- the request-handling task likely panicked");
            let response_text = String::from_utf8_lossy(&response);
            assert!(response_text.starts_with("HTTP/1.1 5"), "got: {response_text}");
        });
    }
}
