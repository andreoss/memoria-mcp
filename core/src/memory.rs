use crate::embedding::{EmbeddingConfig, EmbeddingProvider};
use crate::llm::{extract_facts, LlmConfig, LlmProvider, Message};
use crate::vector_store::{VectorRecord, VectorStore, VectorStoreConfig};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

static NEXT_RECORD_ID: AtomicU64 = AtomicU64::new(0);

fn next_record_id() -> String {
    let n = NEXT_RECORD_ID.fetch_add(1, Ordering::Relaxed);
    format!("rec-{n}")
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
        }
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        scope: &HashMap<String, String>,
    ) -> Result<Vec<crate::vector_store::SearchResult>, crate::CoreError> {
        if top_k == 0 {
            return Err(crate::CoreError::Validation("top_k must be greater than zero".to_string()));
        }
        let vector = self.embedding.embed(query)?;
        self.vector_store.search(&vector, top_k, scope).map_err(From::from)
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

    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn history(&self, id: &str) -> Result<Vec<HistoryEntry>, crate::CoreError> {
        Ok(self.history.lock().expect("lock poisoned").get(id).cloned().unwrap_or_default())
    }

    #[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value, clippy::missing_panics_doc)]
    pub fn add(
        &self,
        messages: &[Message],
        scope: HashMap<String, String>,
    ) -> Result<Vec<String>, crate::CoreError> {
        let has_scope_id = scope.keys().any(|k| *k == "user_id" || *k == "agent_id" || *k == "run_id");
        if !has_scope_id {
            return Err(crate::CoreError::Validation("scope must contain user_id, agent_id, or run_id".to_string()));
        }
        for message in messages {
            if message.content.is_empty() {
                return Err(crate::CoreError::Validation("message content must not be empty".to_string()));
            }
        }
        let facts = extract_facts(&self.llm, messages)?;
        let mut ids = Vec::new();
        for fact in &facts {
            let vector = self.embedding.embed(fact)?;
            if let Ok(results) = self.vector_store.search(&vector, 100, &scope) {
                if results.iter().any(|r| r.payload.get("content") == Some(fact)) {
                    continue;
                }
            }
            let id = next_record_id();
            let mut payload = scope.clone();
            if !payload.contains_key("content") {
                payload.insert("content".to_string(), fact.clone());
            }
            let record = VectorRecord::new(id.clone(), vector, payload);
            self.vector_store.insert(record)?;
            self.history.lock().expect("lock poisoned").entry(id.clone()).or_default().push(HistoryEntry {
                event: HistoryEvent::Added,
                content: fact.clone(),
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
    use crate::test_support::{FakeEmbeddingProvider, FakeLlmProvider};
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
        let ids = memory.add(&messages, scope()).expect("add should succeed");
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
        let ids = memory.add(&messages, scope()).expect("add should succeed");
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
        let ids = memory.add(&messages, s.clone()).expect("add should succeed");
        assert!(!ids.is_empty());

        let query = vec![1.0_f32, 1.0, 1.0, 1.0];
        let results = memory
            .vector_store
            .search(&query, ids.len(), &s)
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
        let ids = memory.add(&messages, s).expect("add should succeed");
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
        let ids = memory.add(&messages, s).expect("add should succeed");
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
        let ids = memory.add(&messages, s).expect("add should succeed");
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
        let result = memory.add(&messages, s);
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
        let result = memory.add(&messages, s);
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
        let ids = memory.add(&messages, s).expect("add should succeed");
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
        let ids = memory.add(&messages, s).expect("add should succeed");
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
        let ids1 = memory.add(&messages, s.clone()).expect("add should succeed");
        let ids2 = memory.add(&messages, s).expect("add should succeed");
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
        let ids1 = memory.add(&messages, s1).expect("add should succeed");
        let ids2 = memory.add(&messages, s2).expect("add should succeed");
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
        let result = memory.add(&messages, scope());
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
        let ids = memory.add(&messages, scope()).expect("add should succeed");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_search_returns_matching_memories() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let _ = memory.add(&messages, scope()).expect("add should succeed");

        let query = "Alice engineer";
        let results = memory.search(query, 10, &scope()).expect("search should succeed");
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
        let _ = memory.add(&messages, scope()).expect("add should succeed");

        for i in 1..=5 {
            let fact = format!("Fact {i}.");
            let s = HashMap::from([("user_id".to_string(), "alice".to_string())]);
            memory.add(&[Message::new(Role::User, &fact)], s).expect("add should succeed");
        }

        let query = "fact";
        let results = memory.search(query, 3, &scope()).expect("search should succeed");
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

        let results = memory.search("engineer", 10, &alice_scope).expect("search should succeed");
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

        let results = memory.search("designer", 10, &bob_scope).expect("search should succeed");
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

        let result = memory.search("anything", 10, &scope());
        assert!(matches!(result, Ok(vec) if vec.is_empty()));
    }

    #[test]
    fn test_search_rejects_zero_top_k() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 0, &scope());
        assert!(matches!(result, Err(crate::CoreError::Validation(_))));
    }

    #[test]
    fn test_search_accepts_positive_top_k() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 1, &scope());
        assert!(result.is_ok());
    }

    #[test]
    fn test_update_content_changes_stored_content() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let ids = memory.add(&messages, scope()).expect("add should succeed");
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
        let ids = memory.add(&messages, scope()).expect("add should succeed");
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
            .add(&messages, HashMap::from([("user_id".to_string(), "alice".to_string()), ("source".to_string(), "chat_import".to_string())]))
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
        let ids = memory.add(&messages, scope()).expect("add should succeed");
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
        let ids = memory.add(&messages, scope()).expect("add should succeed");
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
        let ids = memory.add(&messages, scope()).expect("add should succeed");
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
        let ids = memory.add(&messages, scope()).expect("add should succeed");
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
        let ids = memory.add(&messages, scope()).expect("add should succeed");
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

        let alice_ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], alice_scope.clone()).expect("add should succeed");
        let bob_ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], bob_scope).expect("add should succeed");

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

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope()).expect("add should succeed");

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

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope()).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");

        let entries = memory.history(id).expect("history should succeed");
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

        let entries = memory.history("never-added").expect("history should succeed");
        assert!(entries.is_empty(), "history for an id that was never added should be empty, not an error");
    }

    #[test]
    fn test_history_retains_deleted_entry_after_delete() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope()).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");

        memory.delete(id).expect("delete should succeed");

        let entries = memory.history(id).expect("history should succeed");
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

        let entries = memory.history("never-added").expect("history should succeed");
        assert!(entries.is_empty(), "deleting an id that was never added should not create a history entry");
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
                    memory_ref.add(&[Message::new(Role::User, "Fact one.")], thread_scope).expect("add should succeed");
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

        let ids = memory.add(&[Message::new(Role::User, "Four facts.")], scope()).expect("add should succeed");
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

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope()).expect("add should succeed");
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

        memory.add(&[Message::new(Role::User, "I am an engineer.")], alice_scope.clone()).expect("add should succeed");
        memory.add(&[Message::new(Role::User, "I am an engineer.")], bob_scope.clone()).expect("add should succeed");

        let alice_results = memory.search("engineer", 10, &alice_scope).expect("search should succeed");
        let bob_results = memory.search("engineer", 10, &bob_scope).expect("search should succeed");

        assert_eq!(alice_results.len(), 1, "alice should see exactly her own memory, even though bob added identical content");
        assert_eq!(bob_results.len(), 1, "bob should see exactly his own memory, even though alice added identical content");
        assert_eq!(alice_results[0].payload.get("user_id"), Some(&"alice".to_string()));
        assert_eq!(bob_results[0].payload.get("user_id"), Some(&"bob".to_string()));
        assert_ne!(
            alice_results[0].id, bob_results[0].id,
            "alice and bob must end up with distinct records despite adding identical content"
        );
    }
}
