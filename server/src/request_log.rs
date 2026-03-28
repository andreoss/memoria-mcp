use serde::{Deserialize, Serialize};
use std::sync::Mutex;

const MAX_ENTRIES: usize = 1000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestLogEntry {
    pub method: String,
    pub path: String,
    pub status: u16,
    pub latency_ms: u64,
    pub auth_kind: String,
    pub created_at: u64,
}

pub struct RequestLogStore {
    entries: Mutex<Vec<RequestLogEntry>>,
}

impl RequestLogStore {
    pub const fn new() -> Self {
        Self { entries: Mutex::new(Vec::new()) }
    }

    pub fn append(&self, entry: RequestLogEntry) {
        let mut entries = self.entries.lock().expect("request log lock poisoned");
        entries.push(entry);
        if entries.len() > MAX_ENTRIES {
            let overflow = entries.len() - MAX_ENTRIES;
            entries.drain(0..overflow);
        }
    }

    pub fn all(&self) -> Vec<RequestLogEntry> {
        self.entries.lock().expect("request log lock poisoned").clone()
    }

    // Test-only: nothing in the served request path reads the log's length.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.lock().expect("request log lock poisoned").len()
    }
}

impl Default for RequestLogStore {
    fn default() -> Self {
        Self::new()
    }
}

pub fn classify_auth_kind(headers: &[(String, String)]) -> &'static str {
    let has_bearer = headers
        .iter()
        .any(|(name, value)| name.eq_ignore_ascii_case("authorization") && value.starts_with("Bearer "));
    if has_bearer {
        return "bearer";
    }
    let has_api_key = headers.iter().any(|(name, _)| name.eq_ignore_ascii_case("x-api-key"));
    if has_api_key {
        return "api_key";
    }
    "none"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(method: &str, path: &str, status: u16) -> RequestLogEntry {
        RequestLogEntry {
            method: method.to_string(),
            path: path.to_string(),
            status,
            latency_ms: 0,
            auth_kind: "none".to_string(),
            created_at: 0,
        }
    }

    #[test]
    fn append_then_all_returns_the_real_entries_in_order() {
        let store = RequestLogStore::new();
        store.append(entry("GET", "/health", 200));
        store.append(entry("POST", "/memories", 201));
        let all = store.all();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].path, "/health");
        assert_eq!(all[1].path, "/memories");
    }

    #[test]
    fn len_reflects_real_appends() {
        let store = RequestLogStore::new();
        assert_eq!(store.len(), 0);
        store.append(entry("GET", "/health", 200));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn append_drops_the_oldest_entries_once_the_cap_is_exceeded() {
        let store = RequestLogStore::new();
        for i in 0..(MAX_ENTRIES + 10) {
            store.append(entry("GET", "/health", 200));
            let _ = i;
        }
        assert_eq!(store.len(), MAX_ENTRIES, "the log must never grow past its real cap");
    }

    #[test]
    fn classify_auth_kind_detects_a_real_bearer_header() {
        let headers = vec![("Authorization".to_string(), "Bearer sometoken".to_string())];
        assert_eq!(classify_auth_kind(&headers), "bearer");
    }

    #[test]
    fn classify_auth_kind_detects_a_real_api_key_header() {
        let headers = vec![("X-API-Key".to_string(), "somekey".to_string())];
        assert_eq!(classify_auth_kind(&headers), "api_key");
    }

    #[test]
    fn classify_auth_kind_prefers_bearer_when_both_are_present() {
        let headers =
            vec![("Authorization".to_string(), "Bearer sometoken".to_string()), ("X-API-Key".to_string(), "somekey".to_string())];
        assert_eq!(classify_auth_kind(&headers), "bearer");
    }

    #[test]
    fn classify_auth_kind_with_neither_header_is_none() {
        assert_eq!(classify_auth_kind(&[]), "none");
    }

    #[test]
    fn classify_auth_kind_ignores_a_malformed_authorization_header() {
        let headers = vec![("Authorization".to_string(), "sometoken".to_string())];
        assert_eq!(classify_auth_kind(&headers), "none");
    }
}
