#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn next_id(prefix: &str) -> String {
    let n = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let nanos =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let pid = std::process::id();
    format!("{prefix}-{nanos}-{pid}-{n}")
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    Admin,
    User,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub name: String,
    pub email: String,
    pub password_hash: String,
    pub role: Role,
    pub created_at: u64,
}

impl User {
    pub fn new(name: String, email: String, password_hash: String, role: Role) -> Self {
        Self { id: next_id("user"), name, email, password_hash, role, created_at: unix_now() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: String,
    pub user_id: String,
    pub label: String,
    pub key_hash: String,
    pub key_prefix: String,
    pub created_at: u64,
    pub revoked_at: Option<u64>,
}

impl ApiKey {
    pub fn new(user_id: String, label: String, key_hash: String, key_prefix: String) -> Self {
        Self { id: next_id("key"), user_id, label, key_hash, key_prefix, created_at: unix_now(), revoked_at: None }
    }

    pub const fn is_active(&self) -> bool {
        self.revoked_at.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshTokenRecord {
    pub jti: String,
    pub user_id: String,
    pub issued_at: u64,
    pub revoked_at: Option<u64>,
}

impl RefreshTokenRecord {
    pub fn new(user_id: String) -> Self {
        Self { jti: next_id("jti"), user_id, issued_at: unix_now(), revoked_at: None }
    }

    pub const fn is_active(&self) -> bool {
        self.revoked_at.is_none()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum RefreshError {
    Unknown,
    AlreadyUsed,
}

pub const fn registration_is_allowed(existing_user_count: usize) -> bool {
    existing_user_count == 0
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct AuthSnapshot {
    users: Vec<User>,
    api_keys: Vec<ApiKey>,
    refresh_tokens: Vec<RefreshTokenRecord>,
}

pub struct AuthStore {
    users: Mutex<Vec<User>>,
    api_keys: Mutex<Vec<ApiKey>>,
    refresh_tokens: Mutex<Vec<RefreshTokenRecord>>,
}

impl AuthStore {
    pub const fn new() -> Self {
        Self { users: Mutex::new(Vec::new()), api_keys: Mutex::new(Vec::new()), refresh_tokens: Mutex::new(Vec::new()) }
    }

    pub fn load(path: &Path) -> Self {
        let snapshot = std::fs::read_to_string(path)
            .ok()
            .and_then(|data| serde_json::from_str::<AuthSnapshot>(&data).ok())
            .unwrap_or_default();
        Self {
            users: Mutex::new(snapshot.users),
            api_keys: Mutex::new(snapshot.api_keys),
            refresh_tokens: Mutex::new(snapshot.refresh_tokens),
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let snapshot = AuthSnapshot {
            users: self.users.lock().expect("auth store users lock poisoned").clone(),
            api_keys: self.api_keys.lock().expect("auth store api_keys lock poisoned").clone(),
            refresh_tokens: self.refresh_tokens.lock().expect("auth store refresh_tokens lock poisoned").clone(),
        };
        let data = serde_json::to_vec(&snapshot).unwrap_or_default();
        crate::write_atomically(path, &data)
    }

    pub fn issue_refresh_token(&self, user_id: &str) -> String {
        let record = RefreshTokenRecord::new(user_id.to_string());
        let jti = record.jti.clone();
        self.refresh_tokens.lock().expect("auth store refresh_tokens lock poisoned").push(record);
        jti
    }

    pub fn rotate_refresh_token(&self, presented_jti: &str) -> Result<String, RefreshError> {
        let mut tokens = self.refresh_tokens.lock().expect("auth store refresh_tokens lock poisoned");
        let Some(existing) = tokens.iter_mut().find(|t| t.jti == presented_jti) else {
            return Err(RefreshError::Unknown);
        };
        if !existing.is_active() {
            return Err(RefreshError::AlreadyUsed);
        }
        existing.revoked_at = Some(unix_now());
        let user_id = existing.user_id.clone();
        let new_record = RefreshTokenRecord::new(user_id);
        let new_jti = new_record.jti.clone();
        tokens.push(new_record);
        drop(tokens);
        Ok(new_jti)
    }

    pub fn insert_user(&self, user: User) {
        self.users.lock().expect("auth store users lock poisoned").push(user);
    }

    pub fn user_count(&self) -> usize {
        self.users.lock().expect("auth store users lock poisoned").len()
    }

    pub fn find_user_by_email(&self, email: &str) -> Option<User> {
        self.users
            .lock()
            .expect("auth store users lock poisoned")
            .iter()
            .find(|u| u.email.eq_ignore_ascii_case(email))
            .cloned()
    }

    pub fn find_user_by_id(&self, id: &str) -> Option<User> {
        self.users.lock().expect("auth store users lock poisoned").iter().find(|u| u.id == id).cloned()
    }

    pub fn insert_api_key(&self, key: ApiKey) {
        self.api_keys.lock().expect("auth store api_keys lock poisoned").push(key);
    }

    pub fn find_api_keys_by_user(&self, user_id: &str) -> Vec<ApiKey> {
        self.api_keys
            .lock()
            .expect("auth store api_keys lock poisoned")
            .iter()
            .filter(|k| k.user_id == user_id)
            .cloned()
            .collect()
    }
}

impl Default for AuthStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("memoria-auth-store-test-{}-{label}", std::process::id()))
    }

    #[test]
    #[allow(clippy::needless_collect)]
    fn next_id_produces_distinct_ids_for_concurrent_callers() {
        let handles: Vec<std::thread::JoinHandle<String>> = (0..20).map(|_| std::thread::spawn(|| next_id("x"))).collect();
        let ids: Vec<String> = handles.into_iter().map(|h| h.join().expect("thread should not panic")).collect();
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "every generated id must be distinct");
    }

    #[test]
    fn insert_user_then_find_by_email_is_case_insensitive() {
        let store = AuthStore::new();
        store.insert_user(User::new("Alice".to_string(), "Alice@Example.com".to_string(), "hash".to_string(), Role::User));
        let found = store.find_user_by_email("alice@example.com").expect("case-insensitive match");
        assert_eq!(found.name, "Alice");
    }

    #[test]
    fn find_user_by_email_returns_none_for_an_unknown_email() {
        let store = AuthStore::new();
        assert!(store.find_user_by_email("nobody@example.com").is_none());
    }

    #[test]
    fn user_count_reflects_real_inserts() {
        let store = AuthStore::new();
        assert_eq!(store.user_count(), 0);
        store.insert_user(User::new("A".to_string(), "a@x.com".to_string(), "h".to_string(), Role::Admin));
        assert_eq!(store.user_count(), 1);
    }

    #[test]
    fn a_new_api_key_is_active_and_a_revoked_one_is_not() {
        let mut key = ApiKey::new("user-1".to_string(), "label".to_string(), "hash".to_string(), "prefix".to_string());
        assert!(key.is_active());
        key.revoked_at = Some(unix_now());
        assert!(!key.is_active());
    }

    #[test]
    fn find_api_keys_by_user_only_returns_that_users_keys() {
        let store = AuthStore::new();
        store.insert_api_key(ApiKey::new("user-1".to_string(), "a".to_string(), "h1".to_string(), "p1".to_string()));
        store.insert_api_key(ApiKey::new("user-2".to_string(), "b".to_string(), "h2".to_string(), "p2".to_string()));
        let keys = store.find_api_keys_by_user("user-1");
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].label, "a");
    }

    #[test]
    fn load_with_no_file_present_is_empty() {
        let store = AuthStore::load(&scratch_path("missing"));
        assert_eq!(store.user_count(), 0);
    }

    #[test]
    fn save_then_load_round_trips_a_real_user_api_key_and_refresh_token() {
        let path = scratch_path("round-trip");
        let store = AuthStore::new();
        store.insert_user(User::new("Bob".to_string(), "bob@example.com".to_string(), "hash".to_string(), Role::Admin));
        store.insert_api_key(ApiKey::new("user-1".to_string(), "ci key".to_string(), "keyhash".to_string(), "km_ab".to_string()));
        let jti = store.issue_refresh_token("user-1");
        store.save(&path).expect("save should succeed");

        let reloaded = AuthStore::load(&path);
        assert_eq!(reloaded.user_count(), 1);
        let user = reloaded.find_user_by_email("bob@example.com").expect("reloaded user must be found");
        assert_eq!(user.role, Role::Admin);
        let keys = reloaded.find_api_keys_by_user("user-1");
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].key_prefix, "km_ab");
        assert!(
            reloaded.rotate_refresh_token(&jti).is_ok(),
            "a refresh token issued before save must still rotate cleanly after a reload"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rotate_refresh_token_succeeds_once_and_returns_a_new_jti() {
        let store = AuthStore::new();
        let jti = store.issue_refresh_token("user-1");
        let new_jti = store.rotate_refresh_token(&jti).expect("first rotation must succeed");
        assert_ne!(jti, new_jti, "rotation must issue a genuinely new token, not reuse the old one");
    }

    #[test]
    fn rotate_refresh_token_rejects_reuse_of_an_already_rotated_token() {
        let store = AuthStore::new();
        let jti = store.issue_refresh_token("user-1");
        store.rotate_refresh_token(&jti).expect("first rotation must succeed");
        assert_eq!(store.rotate_refresh_token(&jti), Err(RefreshError::AlreadyUsed));
    }

    #[test]
    fn rotate_refresh_token_rejects_an_unknown_jti() {
        let store = AuthStore::new();
        assert_eq!(store.rotate_refresh_token("never-issued"), Err(RefreshError::Unknown));
    }

    #[test]
    fn rotate_refresh_token_can_chain_through_several_real_rotations() {
        let store = AuthStore::new();
        let mut jti = store.issue_refresh_token("user-1");
        for _ in 0..5 {
            jti = store.rotate_refresh_token(&jti).expect("each rotation in the chain must succeed");
        }
        assert!(store.rotate_refresh_token(&jti).is_ok(), "the final token in the chain must still be usable once");
    }

    #[test]
    fn registration_is_allowed_only_before_the_first_user_exists() {
        assert!(registration_is_allowed(0));
        assert!(!registration_is_allowed(1));
        assert!(!registration_is_allowed(2));
    }

    #[test]
    fn save_never_corrupts_the_file_under_concurrent_writers() {
        use std::sync::Arc;
        let path = Arc::new(scratch_path("concurrent"));
        let handles: Vec<_> = (0..20)
            .map(|i| {
                let path = Arc::clone(&path);
                std::thread::spawn(move || {
                    let store = AuthStore::new();
                    store.insert_user(User::new(
                        format!("writer-{i}"),
                        format!("writer-{i}@example.com"),
                        "hash".to_string(),
                        Role::User,
                    ));
                    store.save(&path).expect("save should succeed");
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("writer thread should not panic");
        }
        let reloaded = AuthStore::load(&path);
        assert_eq!(reloaded.user_count(), 1, "the final file must be exactly one writer's complete snapshot");
        let _ = std::fs::remove_file(&*path);
    }
}
