use crate::embedding::{EmbeddingError, EmbeddingProvider};
use crate::llm::{Completion, LlmError, LlmProvider, Message};
use crate::reranker::{RerankError, Reranker};
use crate::vector_store::{SearchResult, VectorRecord, VectorStore, VectorStoreError};
use std::collections::HashMap;
use std::sync::Mutex;

pub struct FakeLlmProvider {
    fail_with_backend: bool,
    fail_with_timeout: bool,
    fail_with_malformed: Option<String>,
    fail_with_auth: bool,
    raw_response: Option<String>,
}

impl FakeLlmProvider {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            fail_with_backend: false,
            fail_with_timeout: false,
            fail_with_malformed: None,
            fail_with_auth: false,
            raw_response: None,
        }
    }

    #[must_use]
    pub(crate) fn failing() -> Self {
        Self {
            fail_with_backend: true,
            fail_with_timeout: false,
            fail_with_malformed: None,
            fail_with_auth: false,
            raw_response: None,
        }
    }

    #[must_use]
    pub(crate) fn timing_out() -> Self {
        Self {
            fail_with_backend: false,
            fail_with_timeout: true,
            fail_with_malformed: None,
            fail_with_auth: false,
            raw_response: None,
        }
    }

    #[must_use]
    pub(crate) fn returning_malformed(reason: impl Into<String>) -> Self {
        Self {
            fail_with_backend: false,
            fail_with_timeout: false,
            fail_with_malformed: Some(reason.into()),
            fail_with_auth: false,
            raw_response: None,
        }
    }

    #[must_use]
    pub(crate) fn unauthorized() -> Self {
        Self {
            fail_with_backend: false,
            fail_with_timeout: false,
            fail_with_malformed: None,
            fail_with_auth: true,
            raw_response: None,
        }
    }

    #[must_use]
    pub(crate) fn with_response(content: impl Into<String>) -> Self {
        Self {
            fail_with_backend: false,
            fail_with_timeout: false,
            fail_with_malformed: None,
            fail_with_auth: false,
            raw_response: Some(content.into()),
        }
    }

    #[must_use]
    pub(crate) fn with_facts(content: impl Into<String>) -> Self {
        Self::with_response(content)
    }
}

impl LlmProvider for FakeLlmProvider {
    fn complete(&self, messages: &[Message]) -> Result<Completion, LlmError> {
        if self.fail_with_backend {
            return Err(LlmError::Backend("fake backend failure".to_string()));
        }
        if self.fail_with_timeout {
            return Err(LlmError::Timeout);
        }
        if let Some(reason) = &self.fail_with_malformed {
            return Err(LlmError::Malformed(reason.clone()));
        }
        if self.fail_with_auth {
            return Err(LlmError::AuthFailure);
        }
        if messages.is_empty() {
            return Err(LlmError::EmptyMessages);
        }
        let content = self
            .raw_response
            .clone()
            .unwrap_or_else(|| {
                messages
                    .iter()
                    .map(|m| m.content.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            });
        Ok(Completion::new(content))
    }
}

pub struct EchoLlmProvider;

impl EchoLlmProvider {
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self
    }
}

impl LlmProvider for EchoLlmProvider {
    fn complete(&self, messages: &[Message]) -> Result<Completion, LlmError> {
        if messages.is_empty() {
            return Err(LlmError::EmptyMessages);
        }
        let content = messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n");
        Ok(Completion::new(content))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FakeEmbeddingProvider {
    fail_with_backend: bool,
    fail_with_timeout: bool,
}

impl FakeEmbeddingProvider {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            fail_with_backend: false,
            fail_with_timeout: false,
        }
    }

    #[must_use]
    pub(crate) fn failing() -> Self {
        Self {
            fail_with_backend: true,
            fail_with_timeout: false,
        }
    }

    #[must_use]
    pub(crate) fn timing_out() -> Self {
        Self {
            fail_with_backend: false,
            fail_with_timeout: true,
        }
    }
}

impl EmbeddingProvider for FakeEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if self.fail_with_backend {
            return Err(EmbeddingError::Backend("fake backend failure".to_string()));
        }
        if self.fail_with_timeout {
            return Err(EmbeddingError::Timeout);
        }
        crate::embedding::LocalHashEmbeddingProvider::new().embed(text)
    }
}

pub struct VecVectorStore {
    records: Mutex<Vec<VectorRecord>>,
}

impl VecVectorStore {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            records: Mutex::new(Vec::new()),
        }
    }
}

impl VectorStore for VecVectorStore {
    fn insert(&self, record: VectorRecord) -> Result<(), VectorStoreError> {
        let mut records = self.records.lock().expect("lock poisoned");
        if let Some(existing) = records.first() {
            let expected = existing.vector.len();
            let actual = record.vector.len();
            if expected != actual {
                return Err(VectorStoreError::DimensionMismatch { expected, actual });
            }
        }
        records.push(record);
        drop(records);
        Ok(())
    }

    fn search(
        &self,
        vector: &[f32],
        top_k: usize,
        filters: &HashMap<String, String>,
        threshold: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut scored: Vec<SearchResult> = self
            .records
            .lock()
            .expect("lock poisoned")
            .iter()
            .filter(|r| filters.iter().all(|(k, v)| r.payload.get(k).is_some_and(|pv| pv == v)))
            .map(|r| {
                let score = r.vector.iter().zip(vector.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, |acc, d| acc + d);
                SearchResult { id: r.id.clone(), score, payload: r.payload.clone() }
            })
            .collect();
        if let Some(threshold) = threshold {
            scored.retain(|result| result.score <= threshold);
        }
        scored.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    fn get(&self, id: &str) -> Result<Option<VectorRecord>, VectorStoreError> {
        Ok(self.records.lock().expect("lock poisoned").iter().find(|r| r.id == id).cloned())
    }

    fn update(&self, record: VectorRecord) -> Result<(), VectorStoreError> {
        let mut records = self.records.lock().expect("lock poisoned");
        records.iter_mut().find(|r| r.id == record.id).map_or(Err(VectorStoreError::NotFound), |existing| {
            *existing = record;
            Ok(())
        })
    }

    fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        self.records.lock().expect("lock poisoned").retain(|r| r.id != id);
        Ok(())
    }

    fn list(&self, offset: usize, limit: usize) -> Result<Vec<String>, VectorStoreError> {
        Ok(self.records.lock().expect("lock poisoned").iter().map(|r| r.id.clone()).skip(offset).take(limit).collect())
    }

    fn reset(&self) -> Result<(), VectorStoreError> {
        self.records.lock().expect("lock poisoned").clear();
        Ok(())
    }
}

pub struct FixedKeywordSearchVectorStore {
    inner: crate::vector_store::InMemoryVectorStore,
    keyword_results: Vec<SearchResult>,
}

impl FixedKeywordSearchVectorStore {
    #[must_use]
    pub(crate) fn new(keyword_results: Vec<SearchResult>) -> Self {
        Self { inner: crate::vector_store::InMemoryVectorStore::new(), keyword_results }
    }
}

impl VectorStore for FixedKeywordSearchVectorStore {
    fn insert(&self, record: VectorRecord) -> Result<(), VectorStoreError> {
        self.inner.insert(record)
    }

    fn search(&self, vector: &[f32], top_k: usize, filters: &HashMap<String, String>, threshold: Option<f32>) -> Result<Vec<SearchResult>, VectorStoreError> {
        self.inner.search(vector, top_k, filters, threshold)
    }

    fn get(&self, id: &str) -> Result<Option<VectorRecord>, VectorStoreError> {
        self.inner.get(id)
    }

    fn update(&self, record: VectorRecord) -> Result<(), VectorStoreError> {
        self.inner.update(record)
    }

    fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        self.inner.delete(id)
    }

    fn list(&self, offset: usize, limit: usize) -> Result<Vec<String>, VectorStoreError> {
        self.inner.list(offset, limit)
    }

    fn reset(&self) -> Result<(), VectorStoreError> {
        self.inner.reset()
    }

    fn keyword_search(&self, _query: &str, _top_k: usize, _filters: &HashMap<String, String>) -> Result<Option<Vec<SearchResult>>, VectorStoreError> {
        Ok(Some(self.keyword_results.clone()))
    }
}

pub struct FakeRerankerProvider {
    fail_with_backend: bool,
}

impl FakeRerankerProvider {
    #[must_use]
    pub(crate) fn failing() -> Self {
        Self { fail_with_backend: true }
    }
}

impl Reranker for FakeRerankerProvider {
    fn rerank(&self, query: &str, results: Vec<SearchResult>) -> Result<Vec<SearchResult>, RerankError> {
        if self.fail_with_backend {
            return Err(RerankError::Backend("fake backend failure".to_string()));
        }
        crate::reranker::LocalOverlapReranker::new().rerank(query, results).map_err(|_| RerankError::Backend("unreachable".to_string()))
    }
}
