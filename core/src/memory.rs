use crate::embedding::EmbeddingProvider;
use crate::llm::{extract_facts, LlmProvider, Message};
use crate::vector_store::{VectorRecord, VectorStore};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_RECORD_ID: AtomicU64 = AtomicU64::new(0);

fn next_record_id() -> String {
    let n = NEXT_RECORD_ID.fetch_add(1, Ordering::Relaxed);
    format!("rec-{n}")
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
}

impl<L, E, V> Memory<L, E, V>
where
    L: LlmProvider,
    E: EmbeddingProvider,
    V: VectorStore,
{
    #[must_use]
    pub const fn new(llm: L, embedding: E, vector_store: V) -> Self {
        Self {
            llm,
            embedding,
            vector_store,
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

    #[allow(clippy::missing_errors_doc)]
    pub fn delete(&self, id: &str) -> Result<(), crate::CoreError> {
        self.vector_store.delete(id).map_err(From::from)
    }

    #[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value)]
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
            ids.push(id);
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;
    use crate::test_support::{FakeEmbeddingProvider, FakeLlmProvider};
    use crate::vector_store::InMemoryVectorStore;

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
}
