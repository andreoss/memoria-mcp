use crate::embedding::{EmbeddingConfig, EmbeddingProvider};
use crate::llm::{extract_facts, LlmConfig, LlmProvider, Message, Role};
use crate::vector_store::{VectorRecord, VectorStore, VectorStoreConfig};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

static NEXT_RECORD_ID: AtomicU64 = AtomicU64::new(0);

fn next_record_id() -> String {
    let n = NEXT_RECORD_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let pid = std::process::id();
    format!("rec-{nanos}-{pid}-{n}")
}

fn has_scope_id(scope: &HashMap<String, String>) -> bool {
    scope.keys().any(|k| k == "user_id" || k == "agent_id" || k == "run_id")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HistoryEvent {
    Added,
    Deleted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryEntry {
    pub event: HistoryEvent,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryConfig {
    pub llm: LlmConfig,
    pub embedding: EmbeddingConfig,
    pub vector_store: VectorStoreConfig,
}

impl MemoryConfig {
    #[allow(clippy::missing_errors_doc)]
    pub fn validate(&self) -> Result<(), crate::CoreError> {
        self.llm.validate()?;
        self.embedding.validate()?;
        self.vector_store.validate()?;
        Ok(())
    }
}

pub struct Memory<L, E, V>
where
    L: LlmProvider,
    E: EmbeddingProvider,
    V: VectorStore,
{
    llm: L,
    embedding: E,
    vector_store: V,
    history: Mutex<HashMap<String, Vec<HistoryEntry>>>,
    max_metadata_bytes: Option<usize>,
    max_content_length: Option<usize>,
    max_top_k: Option<usize>,
}

impl<L, E, V> Memory<L, E, V>
where
    L: LlmProvider,
    E: EmbeddingProvider,
    V: VectorStore,
{
    #[must_use]
    pub fn new(llm: L, embedding: E, vector_store: V) -> Self {
        Self {
            llm,
            embedding,
            vector_store,
            history: Mutex::new(HashMap::new()),
            max_metadata_bytes: None,
            max_content_length: None,
            max_top_k: None,
        }
    }

    #[must_use]
    pub const fn with_max_top_k(mut self, limit: usize) -> Self {
        self.max_top_k = Some(limit);
        self
    }

    #[must_use]
    pub const fn with_max_content_length(mut self, limit: usize) -> Self {
        self.max_content_length = Some(limit);
        self
    }

    #[must_use]
    pub const fn with_max_metadata_bytes(mut self, limit: usize) -> Self {
        self.max_metadata_bytes = Some(limit);
        self
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        scope: &HashMap<String, String>,
        threshold: Option<f32>,
    ) -> Result<Vec<crate::vector_store::SearchResult>, crate::CoreError> {
        if top_k == 0 {
            return Err(crate::CoreError::Validation("top_k must be greater than zero".to_string()));
        }
        if let Some(max_top_k) = self.max_top_k {
            if top_k > max_top_k {
                return Err(crate::CoreError::Validation(format!(
                    "top_k of {top_k} exceeds the configured ceiling of {max_top_k}"
                )));
            }
        }
        if !has_scope_id(scope) {
            return Err(crate::CoreError::Validation("scope must contain user_id, agent_id, or run_id".to_string()));
        }
        if let Some(threshold) = threshold {
            if threshold < 0.0 {
                return Err(crate::CoreError::Validation(format!("threshold must not be negative, got {threshold}")));
            }
        }
        let vector = self.embedding.embed(query)?;
        self.vector_store.search(&vector, top_k, scope, threshold).map_err(From::from)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn update(
        &self,
        id: &str,
        content: Option<&str>,
        metadata: Option<std::collections::HashMap<String, String>>,
    ) -> Result<(), crate::CoreError> {
        if metadata.as_ref().is_some_and(|m| m.keys().any(|k| k == "user_id" || k == "agent_id" || k == "run_id")) {
            return Err(crate::CoreError::Validation("metadata update must not change scope fields (user_id, agent_id, run_id)".to_string()));
        }
        let Some(mut record) = self.vector_store.get(id)? else {
            return Err(crate::CoreError::NotFound(format!("record {id} not found")));
        };
        if let Some(content) = content {
            record.vector = self.embedding.embed(content)?;
            record.payload.insert("content".to_string(), content.to_string());
        }
        if let Some(metadata_map) = metadata {
            for (key, value) in metadata_map {
                record.payload.insert(key, value);
            }
        }
        self.vector_store.update(record)?;
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn get(&self, id: &str) -> Result<Option<VectorRecord>, crate::CoreError> {
        self.vector_store.get(id).map_err(From::from)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn list(&self, offset: usize, limit: usize) -> Result<Vec<String>, crate::CoreError> {
        self.vector_store.list(offset, limit).map_err(From::from)
    }

    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn delete(&self, id: &str) -> Result<(), crate::CoreError> {
        let content = self.vector_store.get(id)?.and_then(|r| r.payload.get("content").cloned());
        self.vector_store.delete(id)?;
        if let Some(content) = content {
            self.history.lock().expect("lock poisoned").entry(id.to_string()).or_default().push(HistoryEntry {
                event: HistoryEvent::Deleted,
                content,
            });
        }
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn reset(&self, scope: &HashMap<String, String>) -> Result<(), crate::CoreError> {
        let ids = self.vector_store.list(0, usize::MAX)?;
        for id in ids {
            let Some(record) = self.vector_store.get(&id)? else {
                continue;
            };
            let matches = scope.iter().all(|(k, v)| record.payload.get(k) == Some(v));
            if matches {
                self.vector_store.delete(&id)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn health_check(&self) -> Result<(), crate::CoreError> {
        self.llm.complete(&[Message::new(crate::llm::Role::User, "health check")])?;
        self.embedding.embed("health check")?;
        self.vector_store.list(0, 0)?;
        Ok(())
    }

    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn history(&self, id: &str, offset: usize, limit: usize) -> Result<Vec<HistoryEntry>, crate::CoreError> {
        let entries = self.history.lock().expect("lock poisoned").get(id).cloned().unwrap_or_default();
        Ok(entries.into_iter().skip(offset).take(limit).collect())
    }

    #[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value, clippy::missing_panics_doc)]
    pub fn add(
        &self,
        messages: &[Message],
        scope: HashMap<String, String>,
        infer: bool,
    ) -> Result<Vec<String>, crate::CoreError> {
        if !has_scope_id(&scope) {
            return Err(crate::CoreError::Validation("scope must contain user_id, agent_id, or run_id".to_string()));
        }
        if let Some(max_bytes) = self.max_metadata_bytes {
            let total_bytes: usize = scope.iter().map(|(k, v)| k.len() + v.len()).sum();
            if total_bytes > max_bytes {
                return Err(crate::CoreError::Validation(format!(
                    "metadata payload of {total_bytes} bytes exceeds the configured limit of {max_bytes} bytes"
                )));
            }
        }
        for message in messages {
            if message.content.is_empty() {
                return Err(crate::CoreError::Validation("message content must not be empty".to_string()));
            }
            if let Some(max_len) = self.max_content_length {
                if message.content.len() > max_len {
                    return Err(crate::CoreError::Validation(format!(
                        "message content of {} bytes exceeds the configured limit of {max_len} bytes",
                        message.content.len()
                    )));
                }
            }
        }
        let items: Vec<(String, Option<Role>)> = if infer {
            extract_facts(&self.llm, messages)?.into_iter().map(|fact| (fact, None)).collect()
        } else {
            messages.iter().map(|m| (m.content.clone(), Some(m.role))).collect()
        };
        let mut ids = Vec::new();
        for (content, role) in &items {
            let vector = self.embedding.embed(content)?;
            if let Ok(results) = self.vector_store.search(&vector, 100, &scope, None) {
                if results.iter().any(|r| r.payload.get("content") == Some(content)) {
                    continue;
                }
            }
            let id = next_record_id();
            let mut payload = scope.clone();
            if !payload.contains_key("content") {
                payload.insert("content".to_string(), content.clone());
            }
            if let Some(role) = role {
                payload.insert("role".to_string(), role.as_str().to_string());
            }
            let record = VectorRecord::new(id.clone(), vector, payload);
            self.vector_store.insert(record)?;
            self.history.lock().expect("lock poisoned").entry(id.clone()).or_default().push(HistoryEntry {
                event: HistoryEvent::Added,
                content: content.clone(),
            });
            ids.push(id);
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EmbeddingConfig;
    use crate::llm::{LlmConfig, Role};
    use crate::test_support::{EchoLlmProvider, FakeEmbeddingProvider, FakeLlmProvider, VecVectorStore};
    use crate::vector_store::{InMemoryVectorStore, VectorStoreConfig};

    fn scope() -> HashMap<String, String> {
        HashMap::from([("user_id".to_string(), "alice".to_string())])
    }

    #[test]
    fn test_add_single_message_creates_one_or_more_records() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.\nBob lives in Berlin.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer and Bob lives in Berlin.")];
        let ids = memory.add(&messages, scope(), true).expect("add should succeed");
        assert!(
            !ids.is_empty(),
            "add should return at least one record id"
        );

        let id = ids.first().expect("expected at least one id");
        let record = memory
            .vector_store
            .get(id)
            .expect("get should succeed")
            .expect("record should exist in the store");
        assert_eq!(&record.id, id);
    }

    #[test]
    fn test_add_with_infer_false_stores_raw_content_verbatim_not_extracted_facts() {
        let llm = FakeLlmProvider::with_facts("a completely different extracted fact");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "The user is allergic to shellfish.")];
        let ids = memory.add(&messages, scope(), false).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");
        let record = memory.vector_store.get(id).expect("get should succeed").expect("record should exist");
        assert_eq!(
            record.payload.get("content"),
            Some(&"The user is allergic to shellfish.".to_string()),
            "infer=false must store the raw message content verbatim, never the LLM's extracted fact"
        );
    }

    #[test]
    fn test_add_with_infer_false_tags_the_payload_with_the_real_message_role() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::Assistant, "Sure, I can help with that.")];
        let ids = memory.add(&messages, scope(), false).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");
        let record = memory.vector_store.get(id).expect("get should succeed").expect("record should exist");
        assert_eq!(record.payload.get("role"), Some(&"assistant".to_string()));
    }

    #[test]
    fn test_add_with_infer_false_creates_one_record_per_message_not_per_fact() {
        let llm = FakeLlmProvider::with_facts("one\ntwo\nthree");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "First message."), Message::new(Role::User, "Second message.")];
        let ids = memory.add(&messages, scope(), false).expect("add should succeed");
        assert_eq!(ids.len(), 2, "infer=false must produce exactly one record per input message");
    }

    #[test]
    fn test_add_with_infer_true_still_calls_extract_facts_unchanged() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.\nBob lives in Berlin.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer and Bob lives in Berlin.")];
        let ids = memory.add(&messages, scope(), true).expect("add should succeed");
        assert_eq!(ids.len(), 2, "infer=true's existing extraction behavior must be unchanged");
        let contents: Vec<String> = ids
            .iter()
            .map(|id| memory.vector_store.get(id).unwrap().unwrap().payload.get("content").unwrap().clone())
            .collect();
        assert!(contents.contains(&"Alice is an engineer.".to_string()));
        assert!(contents.contains(&"Bob lives in Berlin.".to_string()));
    }

    #[test]
    fn test_add_with_infer_true_never_tags_a_role_on_the_payload() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let ids = memory.add(&messages, scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");
        let record = memory.vector_store.get(id).expect("get should succeed").expect("record should exist");
        assert_eq!(record.payload.get("role"), None, "infer=true's facts have no single originating role to tag");
    }

    #[test]
    fn test_add_multiple_messages_in_one_call() {
        let llm = FakeLlmProvider::with_facts("The project started in 2021.\nThe team uses Rust.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [
            Message::new(Role::User, "When did the project start?"),
            Message::new(Role::Assistant, "In 2021, I believe."),
            Message::new(Role::User, "What language does the team use?"),
            Message::new(Role::Assistant, "They use Rust."),
        ];
        let ids = memory.add(&messages, scope(), true).expect("add should succeed");
        assert!(
            !ids.is_empty(),
            "add should return at least one record id for multiple messages"
        );

        for id in &ids {
            let record = memory
                .vector_store
                .get(id)
                .expect("get should succeed")
                .expect("record should exist in the store");
            assert_eq!(&record.id, id);
        }
    }

    #[test]
    fn test_add_scope_is_stored_in_payload() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let s = scope();
        let ids = memory.add(&messages, s.clone(), true).expect("add should succeed");
        assert!(!ids.is_empty());

        let query = vec![1.0_f32, 1.0, 1.0, 1.0];
        let results = memory
            .vector_store
            .search(&query, ids.len(), &s, None)
            .expect("search should succeed");
        assert!(
            !results.is_empty(),
            "search with scope filter should return records"
        );

        for result in &results {
            for (key, value) in &s {
                assert_eq!(
                    result.payload.get(key),
                    Some(value),
                    "scope entry {key} should be present in the record payload"
                );
            }
        }
    }

    #[test]
    fn test_add_scope_with_user_id() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let s = HashMap::from([("user_id".to_string(), "alice".to_string())]);
        let ids = memory.add(&messages, s, true).expect("add should succeed");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_add_scope_with_agent_id() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let s = HashMap::from([("agent_id".to_string(), "bot1".to_string())]);
        let ids = memory.add(&messages, s, true).expect("add should succeed");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_add_scope_with_run_id() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let s = HashMap::from([("run_id".to_string(), "run-123".to_string())]);
        let ids = memory.add(&messages, s, true).expect("add should succeed");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_add_empty_scope_is_rejected() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let s = HashMap::new();
        let result = memory.add(&messages, s, true);
        assert!(matches!(result, Err(crate::CoreError::Validation(_))));
    }

    #[test]
    fn test_add_scope_with_non_scope_key_is_rejected() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let s = HashMap::from([("source".to_string(), "x".to_string())]);
        let result = memory.add(&messages, s, true);
        assert!(matches!(result, Err(crate::CoreError::Validation(_))));
    }

    #[test]
    fn test_add_scope_with_multiple_keys_one_is_scope_id() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let s = HashMap::from([("source".to_string(), "x".to_string()), ("user_id".to_string(), "alice".to_string())]);
        let ids = memory.add(&messages, s, true).expect("add should succeed");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_add_metadata_is_generic() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let s = HashMap::from([("user_id".to_string(), "alice".to_string()), ("source".to_string(), "chat_import".to_string())]);
        let ids = memory.add(&messages, s, true).expect("add should succeed");
        assert!(!ids.is_empty());

        let id = ids.first().expect("expected at least one id");
        let record = memory
            .vector_store
            .get(id)
            .expect("get should succeed")
            .expect("record should exist in the store");
        let payload = record.payload;
        assert_eq!(payload.get("user_id"), Some(&"alice".to_string()), "user_id should be in payload");
        assert_eq!(payload.get("source"), Some(&"chat_import".to_string()), "source should be in payload");
    }

    #[test]
    fn test_add_dedup_same_fact_same_scope() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let s = scope();
        let ids1 = memory.add(&messages, s.clone(), true).expect("add should succeed");
        let ids2 = memory.add(&messages, s, true).expect("add should succeed");
        assert!(ids2.is_empty(), "second add should return empty ids for duplicate");
        let record = memory.vector_store.get(&ids1[0]).expect("get should succeed").expect("record should exist");
        let payload = record.payload;
        assert_eq!(payload.get("content"), Some(&"Alice is an engineer.".to_string()), "content should be stored in payload");
    }

    #[test]
    fn test_add_dedup_same_fact_different_scope() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let s1 = HashMap::from([("user_id".to_string(), "alice".to_string())]);
        let s2 = HashMap::from([("user_id".to_string(), "bob".to_string())]);
        let ids1 = memory.add(&messages, s1, true).expect("add should succeed");
        let ids2 = memory.add(&messages, s2, true).expect("add should succeed");
        assert!(!ids2.is_empty(), "second add should return an id for different scope");
        let unique_ids: std::collections::HashSet<String> = ids1.into_iter().chain(ids2).collect();
        assert_eq!(unique_ids.len(), 2, "same fact in different scopes should result in two records");
    }

    #[test]
    fn test_add_rejects_empty_message_content() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [
            Message::new(Role::User, "Alice is an engineer."),
            Message::new(Role::User, ""),
        ];
        let result = memory.add(&messages, scope(), true);
        assert!(matches!(result, Err(crate::CoreError::Validation(_))));
    }

    #[test]
    fn test_add_all_messages_have_content_still_works() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [
            Message::new(Role::User, "Alice is an engineer."),
            Message::new(Role::User, "Bob lives in Berlin."),
        ];
        let ids = memory.add(&messages, scope(), true).expect("add should succeed");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_search_returns_matching_memories() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let _ = memory.add(&messages, scope(), true).expect("add should succeed");

        let query = "Alice engineer";
        let results = memory.search(query, 10, &scope(), None).expect("search should succeed");
        assert!(
            !results.is_empty(),
            "search should return matching memories"
        );
        assert!(
            results
                .iter()
                .any(|r| r.payload.get("content").map(String::as_str) == Some("Alice is an engineer.")),
            "search results should contain the added fact"
        );
    }

    #[test]
    fn test_search_respects_top_k() {
        let llm = FakeLlmProvider::with_facts("Fact one.\nFact two.\nFact three.\nFact four.\nFact five.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Five facts.")];
        let _ = memory.add(&messages, scope(), true).expect("add should succeed");

        for i in 1..=5 {
            let fact = format!("Fact {i}.");
            let s = HashMap::from([("user_id".to_string(), "alice".to_string())]);
            memory.add(&[Message::new(Role::User, &fact)], s, true).expect("add should succeed");
        }

        let query = "fact";
        let results = memory.search(query, 3, &scope(), None).expect("search should succeed");
        assert!(
            results.len() <= 3,
            "search should return at most top_k results, got {}",
            results.len()
        );
    }

    #[test]
    fn test_search_scoped_to_user() {
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(FakeLlmProvider::new(), embedding.clone(), store);

        let alice_scope = HashMap::from([("user_id".to_string(), "alice".to_string())]);
        let bob_scope = HashMap::from([("user_id".to_string(), "bob".to_string())]);

        let alice_vector = embedding.embed("Alice is an engineer.").expect("embed should succeed");
        let bob_vector = embedding.embed("Bob is a designer.").expect("embed should succeed");

        let mut alice_payload = alice_scope.clone();
        alice_payload.insert("content".to_string(), "Alice is an engineer.".to_string());
        let alice_record = VectorRecord::new("rec-1".to_string(), alice_vector, alice_payload);
        memory.vector_store.insert(alice_record).expect("insert should succeed");

        let mut bob_payload = bob_scope.clone();
        bob_payload.insert("content".to_string(), "Bob is a designer.".to_string());
        let bob_record = VectorRecord::new("rec-2".to_string(), bob_vector, bob_payload);
        memory.vector_store.insert(bob_record).expect("insert should succeed");

        let results = memory.search("engineer", 10, &alice_scope, None).expect("search should succeed");
        assert_eq!(
            results.len(),
            1,
            "should only return alice's records when scoped to alice"
        );
        assert_eq!(
            results[0].payload.get("content"),
            Some(&"Alice is an engineer.".to_string()),
            "returned record should be alice's"
        );

        let results = memory.search("designer", 10, &bob_scope, None).expect("search should succeed");
        assert_eq!(
            results.len(),
            1,
            "should only return bob's records when scoped to bob"
        );
        assert_eq!(
            results[0].payload.get("content"),
            Some(&"Bob is a designer.".to_string()),
            "returned record should be bob's"
        );
    }

    #[test]
    fn test_search_no_matches_returns_empty_ok() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 10, &scope(), None);
        assert!(matches!(result, Ok(vec) if vec.is_empty()));
    }

    #[test]
    fn test_search_rejects_zero_top_k() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 0, &scope(), None);
        assert!(matches!(result, Err(crate::CoreError::Validation(_))));
    }

    #[test]
    fn test_search_rejects_a_negative_threshold() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 10, &scope(), Some(-0.1));
        assert!(matches!(result, Err(crate::CoreError::Validation(_))));
    }

    #[test]
    fn test_search_accepts_a_zero_threshold() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 10, &scope(), Some(0.0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_search_accepts_positive_top_k() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 1, &scope(), None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_update_content_changes_stored_content() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let ids = memory.add(&messages, scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");

        memory
            .update(id, Some("Alice is a senior engineer."), None)
            .expect("update should succeed");

        let record = memory
            .vector_store
            .get(id)
            .expect("get should succeed")
            .expect("record should exist in the store");
        assert_eq!(
            record.payload.get("content"),
            Some(&"Alice is a senior engineer.".to_string()),
            "content should be updated to new value"
        );
    }

    #[test]
    fn test_update_missing_id_returns_not_found() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.update("does-not-exist", None, None);
        assert!(matches!(result, Err(crate::CoreError::NotFound(_))));
    }

    #[test]
    fn test_update_metadata_merges_new_key() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let ids = memory.add(&messages, scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");

        memory.update(id, None, Some(HashMap::from([("source".to_string(), "chat_import".to_string())]))).expect("update should succeed");

        let record = memory.vector_store.get(id).expect("get should succeed").expect("record should exist");
        let payload = record.payload;
        assert_eq!(payload.get("user_id"), Some(&"alice".to_string()), "user_id should still be present");
        assert_eq!(payload.get("source"), Some(&"chat_import".to_string()), "source should be merged");
    }

    #[test]
    fn test_update_metadata_overwrites_existing_key() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let ids = memory
            .add(&messages, HashMap::from([("user_id".to_string(), "alice".to_string()), ("source".to_string(), "chat_import".to_string())]), true)
            .expect("add should succeed");
        let id = ids.first().expect("expected at least one id");

        let original_content = memory.vector_store.get(id).expect("get should succeed").expect("record should exist").payload.get("content").cloned();

        memory.update(id, None, Some(HashMap::from([("source".to_string(), "manual_edit".to_string())]))).expect("update should succeed");

        let record = memory.vector_store.get(id).expect("get should succeed").expect("record should exist");
        let payload = record.payload;
        assert_eq!(payload.get("source"), Some(&"manual_edit".to_string()), "source should be overwritten to manual_edit");
        assert_eq!(payload.get("content"), original_content.as_ref(), "content should remain unchanged");
    }

    #[test]
    fn test_update_rejects_changing_user_id() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let ids = memory.add(&messages, scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");

        let result = memory.update(id, None, Some(HashMap::from([("user_id".to_string(), "mallory".to_string())])));
        assert!(matches!(result, Err(crate::CoreError::Validation(_))));
    }

    #[test]
    fn test_update_rejects_changing_agent_id_or_run_id() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let ids = memory.add(&messages, scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");

        let result = memory.update(id, None, Some(HashMap::from([("agent_id".to_string(), "other-bot".to_string())])));
        assert!(matches!(result, Err(crate::CoreError::Validation(_))));
    }

    #[test]
    fn test_delete_removes_record() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let ids = memory.add(&messages, scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");

        memory.delete(id).expect("delete should succeed");

        let record = memory.vector_store.get(id).expect("get should succeed");
        assert_eq!(record, None, "deleted record should no longer be found");
    }

    #[test]
    fn test_delete_is_idempotent_when_called_twice() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let ids = memory.add(&messages, scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");

        memory.delete(id).expect("first delete should succeed");
        let second = memory.delete(id);
        assert!(second.is_ok(), "deleting an already-deleted id must not error");
    }

    #[test]
    fn test_delete_nonexistent_id_is_idempotent() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.delete("never-inserted");
        assert!(result.is_ok(), "deleting an id that was never inserted must not error");
    }

    #[test]
    fn test_reset_clears_all_memories_for_scope() {
        let llm = FakeLlmProvider::with_facts("Fact one.\nFact two.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Two facts.")];
        let ids = memory.add(&messages, scope(), true).expect("add should succeed");
        assert!(ids.len() >= 2, "expected at least two records for this test to be meaningful");

        memory.reset(&scope()).expect("reset should succeed");

        for id in &ids {
            let record = memory.vector_store.get(id).expect("get should succeed");
            assert_eq!(record, None, "reset should have deleted every record in scope");
        }
    }

    #[test]
    fn test_reset_leaves_other_scopes_untouched() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let alice_scope = HashMap::from([("user_id".to_string(), "alice".to_string())]);
        let bob_scope = HashMap::from([("user_id".to_string(), "bob".to_string())]);

        let alice_ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], alice_scope.clone(), true).expect("add should succeed");
        let bob_ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], bob_scope, true).expect("add should succeed");

        memory.reset(&alice_scope).expect("reset should succeed");

        for id in &alice_ids {
            assert_eq!(memory.vector_store.get(id).expect("get should succeed"), None, "alice's records should be gone");
        }
        for id in &bob_ids {
            assert!(memory.vector_store.get(id).expect("get should succeed").is_some(), "bob's records should be untouched");
        }
    }

    #[test]
    fn test_reset_with_no_matching_scope_is_noop() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");

        let other_scope = HashMap::from([("user_id".to_string(), "nobody-here".to_string())]);
        memory.reset(&other_scope).expect("reset with no matches should not error");

        for id in &ids {
            assert!(memory.vector_store.get(id).expect("get should succeed").is_some(), "unrelated scope's records should be untouched");
        }
    }

    #[test]
    fn test_history_returns_entry_after_add() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");

        let entries = memory.history(id, 0, usize::MAX).expect("history should succeed");
        assert_eq!(entries.len(), 1, "expected exactly one history entry after a single add");
        assert_eq!(entries[0].event, HistoryEvent::Added);
        assert_eq!(entries[0].content, "Alice is an engineer.");
    }

    #[test]
    fn test_history_for_unknown_id_returns_empty() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let entries = memory.history("never-added", 0, usize::MAX).expect("history should succeed");
        assert!(entries.is_empty(), "history for an id that was never added should be empty, not an error");
    }

    #[test]
    fn test_history_retains_deleted_entry_after_delete() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");

        memory.delete(id).expect("delete should succeed");

        let entries = memory.history(id, 0, usize::MAX).expect("history should succeed");
        assert_eq!(entries.len(), 2, "expected an Added entry followed by a Deleted entry");
        assert_eq!(entries[0].event, HistoryEvent::Added);
        assert_eq!(entries[1].event, HistoryEvent::Deleted);
        assert_eq!(entries[1].content, "Alice is an engineer.");
    }

    #[test]
    fn test_deleting_nonexistent_id_adds_no_history_entry() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        memory.delete("never-added").expect("delete should succeed");

        let entries = memory.history("never-added", 0, usize::MAX).expect("history should succeed");
        assert!(entries.is_empty(), "deleting an id that was never added should not create a history entry");
    }

    #[test]
    fn test_history_pagination_respects_offset_and_limit() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");
        memory.delete(id).expect("delete should succeed");

        let first_page = memory.history(id, 0, 1).expect("history should succeed");
        assert_eq!(first_page.len(), 1);
        assert_eq!(first_page[0].event, HistoryEvent::Added);

        let second_page = memory.history(id, 1, 1).expect("history should succeed");
        assert_eq!(second_page.len(), 1);
        assert_eq!(second_page[0].event, HistoryEvent::Deleted);

        let whole_log = memory.history(id, 0, 100).expect("history should succeed");
        assert_eq!(whole_log.len(), 2, "a limit larger than the log should return everything, not error");
    }

    #[test]
    fn test_history_pagination_offset_beyond_log_returns_empty() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");

        let page = memory.history(id, 10, 5).expect("history should succeed");
        assert!(page.is_empty(), "an offset past the end of the log should return empty, not error");
    }

    #[test]
    fn test_concurrent_add_operations_across_scopes_do_not_corrupt_state() {
        const THREAD_COUNT: usize = 8;

        let llm = FakeLlmProvider::with_facts("Fact one.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        std::thread::scope(|s| {
            for i in 0..THREAD_COUNT {
                let memory_ref = &memory;
                s.spawn(move || {
                    let thread_scope = HashMap::from([("user_id".to_string(), format!("user-{i}"))]);
                    memory_ref.add(&[Message::new(Role::User, "Fact one.")], thread_scope, true).expect("add should succeed");
                });
            }
        });

        let ids = memory.vector_store.list(0, usize::MAX).expect("list should succeed");
        assert_eq!(ids.len(), THREAD_COUNT, "each thread's distinct scope should have produced exactly one record, none lost or duplicated");
    }

    #[test]
    fn test_concurrent_delete_operations_do_not_lose_deletes() {
        let llm = FakeLlmProvider::with_facts("Fact one.\nFact two.\nFact three.\nFact four.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Four facts.")], scope(), true).expect("add should succeed");
        assert!(ids.len() >= 4, "expected at least four records for this test to be meaningful");

        std::thread::scope(|s| {
            for id in &ids {
                let memory_ref = &memory;
                s.spawn(move || {
                    memory_ref.delete(id).expect("delete should succeed");
                });
            }
        });

        for id in &ids {
            assert_eq!(
                memory.vector_store.get(id).expect("get should succeed"),
                None,
                "every concurrently-deleted record should be gone, none left behind"
            );
        }
    }

    #[test]
    fn test_concurrent_update_and_delete_resolve_deterministically() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id").clone();

        let update_result = std::thread::scope(|s| {
            let memory_ref = &memory;
            let id_ref = &id;
            let handle = s.spawn(move || memory_ref.update(id_ref, Some("Alice is a senior engineer."), None));
            memory.delete(&id).expect("delete should succeed");
            handle.join().expect("update thread should not panic")
        });

        assert_eq!(
            memory.vector_store.get(&id).expect("get should succeed"),
            None,
            "record should be gone regardless of which operation the scheduler ran first"
        );
        assert!(
            matches!(update_result, Ok(()) | Err(crate::CoreError::NotFound(_))),
            "update racing a delete on the same id must resolve to one of two well-defined outcomes, never a corrupt state: got {update_result:?}"
        );
    }

    #[test]
    fn memory_config_with_all_valid_sub_configs_passes_validation() {
        let config = MemoryConfig {
            llm: LlmConfig { model: "llama3".to_string(), base_url: None, api_key: None, temperature: None },
            embedding: EmbeddingConfig { model: "nomic-embed-text".to_string(), base_url: None, api_key: None, dimensions: None },
            vector_store: VectorStoreConfig { collection_name: "memories".to_string(), url: None, api_key: None, dimension: None },
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn memory_config_with_invalid_llm_sub_config_is_rejected() {
        let config = MemoryConfig {
            llm: LlmConfig { model: String::new(), base_url: None, api_key: None, temperature: None },
            embedding: EmbeddingConfig { model: "nomic-embed-text".to_string(), base_url: None, api_key: None, dimensions: None },
            vector_store: VectorStoreConfig { collection_name: "memories".to_string(), url: None, api_key: None, dimension: None },
        };
        assert!(matches!(config.validate(), Err(crate::CoreError::Config(_))));
    }

    #[test]
    fn memory_config_with_invalid_embedding_sub_config_is_rejected() {
        let config = MemoryConfig {
            llm: LlmConfig { model: "llama3".to_string(), base_url: None, api_key: None, temperature: None },
            embedding: EmbeddingConfig { model: String::new(), base_url: None, api_key: None, dimensions: None },
            vector_store: VectorStoreConfig { collection_name: "memories".to_string(), url: None, api_key: None, dimension: None },
        };
        assert!(matches!(config.validate(), Err(crate::CoreError::Config(_))));
    }

    #[test]
    fn memory_config_with_invalid_vector_store_sub_config_is_rejected() {
        let config = MemoryConfig {
            llm: LlmConfig { model: "llama3".to_string(), base_url: None, api_key: None, temperature: None },
            embedding: EmbeddingConfig { model: "nomic-embed-text".to_string(), base_url: None, api_key: None, dimensions: None },
            vector_store: VectorStoreConfig { collection_name: String::new(), url: None, api_key: None, dimension: None },
        };
        assert!(matches!(config.validate(), Err(crate::CoreError::Config(_))));
    }

    #[test]
    fn test_end_to_end_multi_user_isolation() {
        let llm = FakeLlmProvider::with_facts("I am an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let alice_scope = HashMap::from([("user_id".to_string(), "alice".to_string())]);
        let bob_scope = HashMap::from([("user_id".to_string(), "bob".to_string())]);

        memory.add(&[Message::new(Role::User, "I am an engineer.")], alice_scope.clone(), true).expect("add should succeed");
        memory.add(&[Message::new(Role::User, "I am an engineer.")], bob_scope.clone(), true).expect("add should succeed");

        let alice_results = memory.search("engineer", 10, &alice_scope, None).expect("search should succeed");
        let bob_results = memory.search("engineer", 10, &bob_scope, None).expect("search should succeed");

        assert_eq!(alice_results.len(), 1, "alice should see exactly her own memory, even though bob added identical content");
        assert_eq!(bob_results.len(), 1, "bob should see exactly his own memory, even though alice added identical content");
        assert_eq!(alice_results[0].payload.get("user_id"), Some(&"alice".to_string()));
        assert_eq!(bob_results[0].payload.get("user_id"), Some(&"bob".to_string()));
        assert_ne!(
            alice_results[0].id, bob_results[0].id,
            "alice and bob must end up with distinct records despite adding identical content"
        );
    }

    #[test]
    fn test_end_to_end_multi_agent_isolation() {
        let llm = FakeLlmProvider::with_facts("I am an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let scheduler_scope = HashMap::from([("agent_id".to_string(), "scheduler-bot".to_string())]);
        let support_scope = HashMap::from([("agent_id".to_string(), "support-bot".to_string())]);

        memory.add(&[Message::new(Role::User, "I am an engineer.")], scheduler_scope.clone(), true).expect("add should succeed");
        memory.add(&[Message::new(Role::User, "I am an engineer.")], support_scope.clone(), true).expect("add should succeed");

        let scheduler_results = memory.search("engineer", 10, &scheduler_scope, None).expect("search should succeed");
        let support_results = memory.search("engineer", 10, &support_scope, None).expect("search should succeed");

        assert_eq!(scheduler_results.len(), 1, "scheduler-bot should see exactly its own memory");
        assert_eq!(support_results.len(), 1, "support-bot should see exactly its own memory");
        assert_eq!(scheduler_results[0].payload.get("agent_id"), Some(&"scheduler-bot".to_string()));
        assert_eq!(support_results[0].payload.get("agent_id"), Some(&"support-bot".to_string()));
        assert_ne!(
            scheduler_results[0].id, support_results[0].id,
            "agent scoping must isolate the same way user scoping does"
        );
    }

    #[test]
    fn test_end_to_end_reset_then_search_returns_nothing() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");
        let before = memory.search("engineer", 10, &scope(), None).expect("search should succeed");
        assert!(!before.is_empty(), "expected a result before reset for this test to be meaningful");

        memory.reset(&scope()).expect("reset should succeed");

        let after = memory.search("engineer", 10, &scope(), None).expect("search should succeed");
        assert!(after.is_empty(), "search after reset should return nothing");
    }

    #[test]
    fn test_end_to_end_update_then_search_reflects_the_update() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");

        memory.update(id, Some("Alice is a senior engineer."), None).expect("update should succeed");

        let results = memory.search("senior engineer", 10, &scope(), None).expect("search should succeed");
        assert!(
            results.iter().any(|r| r.payload.get("content") == Some(&"Alice is a senior engineer.".to_string())),
            "search after update should reflect the new content"
        );
        assert!(
            !results.iter().any(|r| r.payload.get("content") == Some(&"Alice is an engineer.".to_string())),
            "search should not return the pre-update content"
        );
    }

    #[test]
    fn test_add_rejects_metadata_over_configured_size_limit() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store).with_max_metadata_bytes(10);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let result = memory.add(&messages, scope(), true);
        assert!(
            matches!(result, Err(crate::CoreError::Validation(_))),
            "scope() alone (\"user_id\" + \"alice\") is well over 10 bytes and should be rejected"
        );
    }

    #[test]
    fn test_add_accepts_metadata_within_configured_size_limit() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store).with_max_metadata_bytes(1000);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let result = memory.add(&messages, scope(), true);
        assert!(result.is_ok(), "scope() is well under 1000 bytes and should be accepted");
    }

    #[test]
    fn test_add_with_no_configured_limit_accepts_any_size() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let large_value = "x".repeat(10_000);
        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let s = HashMap::from([("user_id".to_string(), "alice".to_string()), ("notes".to_string(), large_value)]);
        let result = memory.add(&messages, s, true);
        assert!(result.is_ok(), "with no configured limit, metadata size should never be rejected");
    }

    #[test]
    fn test_add_wraps_dimension_mismatch_as_provider_error() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        store
            .insert(VectorRecord::new("seed".to_string(), vec![1.0, 2.0, 3.0], HashMap::new()))
            .expect("seed insert should succeed");
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let result = memory.add(&messages, scope(), true);

        assert!(
            matches!(result, Err(crate::CoreError::Provider { .. })),
            "a dimension mismatch from the vector store must surface through Memory::add as CoreError::Provider, not panic or a different variant"
        );
    }

    #[test]
    fn test_add_with_zero_facts_extracted_succeeds_with_no_ids() {
        let llm = FakeLlmProvider::with_response("");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Just saying hello, nothing memorable.")];
        let ids = memory.add(&messages, scope(), true).expect("zero extracted facts should not be an error");
        assert!(ids.is_empty(), "no facts extracted should mean no records inserted, not an error");
    }

    #[test]
    fn test_add_deduplicates_duplicate_facts_within_one_call() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.\nAlice is an engineer.\nBob lives in Berlin.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer, and again, Alice is an engineer. Bob lives in Berlin.")];
        let ids = memory.add(&messages, scope(), true).expect("add should succeed");

        assert_eq!(
            ids.len(), 2,
            "the LLM returned the same fact twice within one call; only two distinct records should be created"
        );
    }

    #[test]
    fn test_search_rejects_empty_scope() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 10, &HashMap::new(), None);
        assert!(
            matches!(result, Err(crate::CoreError::Validation(_))),
            "search with no scope-identifying key must be rejected, not silently return every scope's records"
        );
    }

    #[test]
    fn test_search_rejects_scope_with_no_scope_id_key() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let non_scope_filter = HashMap::from([("source".to_string(), "chat_import".to_string())]);
        let result = memory.search("anything", 10, &non_scope_filter, None);
        assert!(matches!(result, Err(crate::CoreError::Validation(_))));
    }

    #[test]
    fn test_update_with_no_fields_provided_is_a_noop_not_an_error() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");
        let before = memory.vector_store.get(id).expect("get should succeed").expect("record should exist");

        let result = memory.update(id, None, None);
        assert!(result.is_ok(), "update with no fields provided should be a no-op, not an error");

        let after = memory.vector_store.get(id).expect("get should succeed").expect("record should exist");
        assert_eq!(before, after, "update with nothing to change should leave the record byte-for-byte identical");
    }

    #[test]
    fn test_delete_requires_exact_id_match_not_a_prefix_match() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.\nBob lives in Berlin.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer and Bob lives in Berlin.")], scope(), true).expect("add should succeed");
        assert!(ids.len() >= 2, "expected at least two records for this test to be meaningful");
        let first_id = &ids[0];
        let common_prefix = &first_id[..first_id.len() - 1];

        memory.delete(common_prefix).expect("deleting a nonexistent id should be idempotent, not an error");

        for id in &ids {
            assert!(
                memory.vector_store.get(id).expect("get should succeed").is_some(),
                "deleting by a prefix of one id must not remove any record, including the one it's a prefix of"
            );
        }
    }

    #[test]
    fn test_add_rejects_content_over_configured_length_limit() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store).with_max_content_length(10);

        let messages = [Message::new(Role::User, "This message is way longer than ten characters.")];
        let result = memory.add(&messages, scope(), true);
        assert!(matches!(result, Err(crate::CoreError::Validation(_))));
    }

    #[test]
    fn test_add_accepts_content_within_configured_length_limit() {
        let llm = FakeLlmProvider::with_facts("Short fact.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store).with_max_content_length(1000);

        let messages = [Message::new(Role::User, "Short.")];
        let result = memory.add(&messages, scope(), true);
        assert!(result.is_ok(), "content well under the configured limit should be accepted");
    }

    #[test]
    fn test_add_with_no_configured_length_limit_accepts_any_length() {
        let llm = FakeLlmProvider::with_facts("A fact.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let long_message = "x".repeat(10_000);
        let messages = [Message::new(Role::User, long_message)];
        let result = memory.add(&messages, scope(), true);
        assert!(result.is_ok(), "with no configured limit, message length should never be rejected");
    }

    #[test]
    fn test_add_wraps_llm_backend_failure_as_provider_error() {
        let llm = FakeLlmProvider::failing();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let result = memory.add(&messages, scope(), true);
        assert!(matches!(result, Err(crate::CoreError::Provider { .. })));
    }

    #[test]
    fn test_add_wraps_llm_timeout_as_provider_error() {
        let llm = FakeLlmProvider::timing_out();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let result = memory.add(&messages, scope(), true);
        assert!(matches!(result, Err(crate::CoreError::Provider { .. })));
    }

    #[test]
    fn test_add_wraps_embedding_backend_failure_as_provider_error() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::failing();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let result = memory.add(&messages, scope(), true);
        assert!(matches!(result, Err(crate::CoreError::Provider { .. })));
    }

    #[test]
    fn test_search_wraps_embedding_timeout_as_provider_error() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::timing_out();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 10, &scope(), None);
        assert!(matches!(result, Err(crate::CoreError::Provider { .. })));
    }

    #[test]
    fn test_search_rejects_top_k_over_configured_ceiling() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store).with_max_top_k(50);

        let result = memory.search("anything", 51, &scope(), None);
        assert!(matches!(result, Err(crate::CoreError::Validation(_))));
    }

    #[test]
    fn test_search_accepts_top_k_at_configured_ceiling() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store).with_max_top_k(50);

        let result = memory.search("anything", 50, &scope(), None);
        assert!(result.is_ok(), "top_k exactly at the configured ceiling should be accepted");
    }

    #[test]
    fn test_search_with_no_configured_ceiling_accepts_any_top_k() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 1_000_000, &scope(), None);
        assert!(result.is_ok(), "with no configured ceiling, top_k should never be rejected for being too large");
    }

    #[test]
    fn test_memory_works_unchanged_with_a_structurally_different_vector_store() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = VecVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");
        assert!(!ids.is_empty());

        let results = memory.search("engineer", 10, &scope(), None).expect("search should succeed");
        assert!(results.iter().any(|r| r.payload.get("content") == Some(&"Alice is an engineer.".to_string())));

        let id = ids.first().expect("expected at least one id");
        memory.update(id, Some("Alice is a senior engineer."), None).expect("update should succeed");
        memory.delete(id).expect("delete should succeed");
        assert_eq!(memory.vector_store.get(id).expect("get should succeed"), None);
    }

    #[test]
    fn test_memory_works_unchanged_with_a_structurally_different_llm_provider() {
        let llm = EchoLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");
        assert!(!ids.is_empty(), "EchoLlmProvider should still produce at least one fact (extract_facts wraps the conversation before echoing it)");

        let results = memory.search("Alice", 10, &scope(), None).expect("search should succeed");
        assert!(!results.is_empty(), "search should find something after adding via a different LLM provider");
    }

    #[test]
    fn test_health_check_succeeds_with_healthy_providers() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        assert!(memory.health_check().is_ok());
    }

    #[test]
    fn test_health_check_reports_llm_failure() {
        let llm = FakeLlmProvider::failing();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        assert!(matches!(memory.health_check(), Err(crate::CoreError::Provider { .. })));
    }

    #[test]
    fn test_health_check_reports_embedding_failure() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::failing();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        assert!(matches!(memory.health_check(), Err(crate::CoreError::Provider { .. })));
    }

    #[test]
    fn test_get_returns_the_record_after_add() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");

        let record = memory.get(id).expect("get should succeed").expect("record should exist");
        assert_eq!(&record.id, id);
        assert_eq!(record.payload.get("content"), Some(&"Alice is an engineer.".to_string()));
    }

    #[test]
    fn test_get_returns_none_for_unknown_id() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        assert_eq!(memory.get("never-added").expect("get should succeed"), None);
    }

    #[test]
    fn test_list_returns_all_ids_after_add() {
        let llm = FakeLlmProvider::with_facts("Fact one.\nFact two.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Two facts.")], scope(), true).expect("add should succeed");

        let listed = memory.list(0, usize::MAX).expect("list should succeed");
        for id in &ids {
            assert!(listed.contains(id), "list should include every id add returned");
        }
    }

    #[test]
    fn test_list_respects_offset_and_limit() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let listed = memory.list(0, 0).expect("list should succeed");
        assert!(listed.is_empty(), "a zero limit should return nothing, not error");
    }

    #[test]
    fn test_next_record_id_differs_across_a_counter_reset() {
        let before = next_record_id();
        NEXT_RECORD_ID.store(0, Ordering::Relaxed);
        let after = next_record_id();
        assert_ne!(
            before, after,
            "an id generated after the in-process counter resets to 0 (simulating a fresh process loading a persisted store) must not collide with one generated before the reset"
        );
    }
}
