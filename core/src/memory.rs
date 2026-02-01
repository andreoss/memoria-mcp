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

    #[allow(clippy::missing_errors_doc, clippy::needless_pass_by_value)]
    pub fn add(
        &self,
        messages: &[Message],
        scope: HashMap<String, String>,
    ) -> Result<Vec<String>, crate::CoreError> {
        let facts = extract_facts(&self.llm, messages)?;
        let mut ids = Vec::with_capacity(facts.len());
        for fact in &facts {
            let vector = self.embedding.embed(fact)?;
            let id = next_record_id();
            let payload = scope.clone();
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
    use crate::llm::{Completion, LlmError, Role};
    use crate::vector_store::InMemoryVectorStore;

    struct FakeLlmProvider {
        response: String,
    }

    impl FakeLlmProvider {
        #[must_use]
        fn with_facts(response: impl Into<String>) -> Self {
            Self {
                response: response.into(),
            }
        }
    }

    impl LlmProvider for FakeLlmProvider {
        fn complete(&self, messages: &[Message]) -> Result<Completion, LlmError> {
            if messages.is_empty() {
                return Err(LlmError::EmptyMessages);
            }
            Ok(Completion::new(self.response.clone()))
        }
    }

    struct FakeEmbeddingProvider;

    impl FakeEmbeddingProvider {
        #[must_use]
        fn new() -> Self {
            Self
        }
    }

    impl EmbeddingProvider for FakeEmbeddingProvider {
        fn embed(&self, text: &str) -> Result<Vec<f32>, crate::embedding::EmbeddingError> {
            if text.is_empty() {
                return Err(crate::embedding::EmbeddingError::EmptyInput);
            }
            let bytes = text.as_bytes();
            let dim: u8 = 4;
            let mut vector = Vec::with_capacity(usize::from(dim));
            for d in 0u8..dim {
                let mut acc = f32::from(d);
                let mut idx: u8 = 0;
                for &b in bytes {
                    let w = f32::from(idx % dim) + 1.0;
                    acc = acc.mul_add(f32::from(b), w);
                    idx = idx.wrapping_add(1);
                }
                vector.push(acc);
            }
            Ok(vector)
        }
    }

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
}
