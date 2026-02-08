use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;

#[derive(Clone, Debug, PartialEq)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: HashMap<String, String>,
}

impl VectorRecord {
    #[must_use]
    pub fn new(id: impl Into<String>, vector: Vec<f32>, payload: HashMap<String, String>) -> Self {
        Self {
            id: id.into(),
            vector,
            payload,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub payload: HashMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VectorStoreError {
    NotFound,
    DimensionMismatch { expected: usize, actual: usize },
    Backend(String),
}

impl fmt::Display for VectorStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "record not found"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "vector dimension mismatch: expected {expected}, got {actual}")
            }
            Self::Backend(reason) => write!(f, "backend error: {reason}"),
        }
    }
}

impl std::error::Error for VectorStoreError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorStoreConfig {
    pub collection_name: String,
    pub url: Option<String>,
    pub api_key: Option<String>,
    pub dimension: Option<usize>,
}

impl VectorStoreConfig {
    #[allow(clippy::missing_errors_doc)]
    pub fn validate(&self) -> Result<(), crate::CoreError> {
        if self.collection_name.trim().is_empty() {
            return Err(crate::CoreError::Config("collection_name must not be empty".to_string()));
        }
        if self.dimension == Some(0) {
            return Err(crate::CoreError::Config("dimension must be greater than zero".to_string()));
        }
        Ok(())
    }
}

pub trait VectorStore {
    #[allow(clippy::missing_errors_doc)]
    fn insert(&self, record: VectorRecord) -> Result<(), VectorStoreError>;

    #[allow(clippy::missing_errors_doc)]
    fn search(
        &self,
        vector: &[f32],
        top_k: usize,
        filters: &HashMap<String, String>,
    ) -> Result<Vec<SearchResult>, VectorStoreError>;

    #[allow(clippy::missing_errors_doc)]
    fn get(&self, id: &str) -> Result<Option<VectorRecord>, VectorStoreError>;

    #[allow(clippy::missing_errors_doc)]
    fn update(&self, record: VectorRecord) -> Result<(), VectorStoreError>;

    #[allow(clippy::missing_errors_doc)]
    fn delete(&self, id: &str) -> Result<(), VectorStoreError>;

    #[allow(clippy::missing_errors_doc)]
    fn list(&self, offset: usize, limit: usize) -> Result<Vec<String>, VectorStoreError>;

    #[allow(clippy::missing_errors_doc)]
    fn reset(&self) -> Result<(), VectorStoreError>;
}

pub trait VectorStoreContractTests: VectorStore {
    fn contract_insert_then_get_round_trips(&self) {
        let record = VectorRecord::new("a", vec![1.0, 2.0], HashMap::new());
        self.insert(record.clone()).expect("insert should succeed");
        let fetched = self.get("a").expect("get should succeed");
        assert_eq!(fetched, Some(record), "inserted record should round-trip");
    }

    fn contract_delete_then_get_returns_none(&self) {
        let record = VectorRecord::new("b", vec![3.0, 4.0], HashMap::new());
        self.insert(record).expect("insert should succeed");
        self.delete("b").expect("delete should succeed");
        let fetched = self.get("b").expect("get should succeed");
        assert_eq!(fetched, None, "deleted record should not be found");
    }

    fn contract_reset_clears_everything(&self) {
        let record = VectorRecord::new("c", vec![5.0, 6.0], HashMap::new());
        self.insert(record).expect("insert should succeed");
        self.reset().expect("reset should succeed");
        let listed = self.list(0, usize::MAX).expect("list should succeed");
        assert!(listed.is_empty(), "reset should clear all records");
    }

    fn contract_search_respects_top_k(&self) {
        let records = [
            VectorRecord::new("a", vec![1.0, 0.0], HashMap::new()),
            VectorRecord::new("b", vec![0.0, 1.0], HashMap::new()),
            VectorRecord::new("c", vec![1.0, 1.0], HashMap::new()),
            VectorRecord::new("d", vec![2.0, 0.0], HashMap::new()),
        ];
        for record in records {
            self.insert(record).expect("insert should succeed");
        }
        let top_k = 2;
        let results = self
            .search(&[0.0, 0.0], top_k, &HashMap::new())
            .expect("search should succeed");
        assert_eq!(
            results.len(),
            top_k,
            "search should return exactly top_k results"
        );
    }

    fn contract_update_then_get_reflects_change(&self) {
        let original = VectorRecord::new("a", vec![1.0, 2.0], HashMap::new());
        self.insert(original).expect("insert should succeed");
        let updated = VectorRecord::new("a", vec![3.0, 4.0], HashMap::new());
        self.update(updated.clone()).expect("update should succeed");
        let fetched = self.get("a").expect("get should succeed");
        assert_eq!(fetched, Some(updated), "updated record should be reflected");
    }

    fn contract_delete_nonexistent_is_idempotent(&self) {
        let result = self.delete("never-inserted");
        assert_eq!(
            result,
            Ok(()),
            "deleting a record that was never inserted must be idempotent, not an error"
        );
    }

    fn contract_update_nonexistent_returns_not_found(&self) {
        let record = VectorRecord::new("missing", vec![1.0, 2.0], HashMap::new());
        let result = self.update(record);
        assert_eq!(
            result,
            Err(VectorStoreError::NotFound),
            "updating a nonexistent record should return NotFound"
        );
    }

    fn contract_list_returns_all_inserted_ids(&self) {
        let records = [
            VectorRecord::new("a", vec![1.0, 0.0], HashMap::new()),
            VectorRecord::new("b", vec![0.0, 1.0], HashMap::new()),
            VectorRecord::new("c", vec![1.0, 1.0], HashMap::new()),
        ];
        for record in records {
            self.insert(record).expect("insert should succeed");
        }
        let listed = self.list(0, usize::MAX).expect("list should succeed");
        let mut listed = listed;
        listed.sort();
        assert_eq!(
            listed,
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            "list should return every inserted id"
        );
    }

    fn contract_list_on_empty_store_returns_empty(&self) {
        let listed = self.list(0, usize::MAX).expect("list should succeed on empty store");
        assert!(
            listed.is_empty(),
            "list on an empty store should return an empty Vec, not an error"
        );
    }

    fn contract_insert_rejects_mismatched_dimension(&self) {
        let first = VectorRecord::new("a", vec![1.0, 2.0], HashMap::new());
        self.insert(first).expect("first insert should establish dimension");
        let wrong = VectorRecord::new("b", vec![1.0], HashMap::new());
        let result = self.insert(wrong);
        assert_eq!(
            result,
            Err(VectorStoreError::DimensionMismatch {
                expected: 2,
                actual: 1
            }),
            "insert with a mismatched vector dimension must be rejected"
        );
    }

    fn contract_list_pagination_respects_offset_and_limit(&self) {
        let records = [
            VectorRecord::new("a", vec![1.0, 0.0], HashMap::new()),
            VectorRecord::new("b", vec![0.0, 1.0], HashMap::new()),
            VectorRecord::new("c", vec![1.0, 1.0], HashMap::new()),
            VectorRecord::new("d", vec![2.0, 0.0], HashMap::new()),
            VectorRecord::new("e", vec![3.0, 0.0], HashMap::new()),
        ];
        for record in records {
            self.insert(record).expect("insert should succeed");
        }
        let first_page = self.list(0, 2).expect("list should succeed");
        assert_eq!(
            first_page.len(),
            2,
            "list with a small limit should return at most that many ids"
        );
        let second_page = self.list(2, 2).expect("list should succeed");
        assert_eq!(
            second_page.len(),
            2,
            "list with an offset should skip the first page"
        );
        assert!(
            first_page.iter().all(|id| !second_page.contains(id)),
            "offset results must differ from the first page"
        );
    }

    fn contract_search_orders_by_score(&self) {
        let records = [
            VectorRecord::new("a", vec![1.0, 0.0], HashMap::new()),
            VectorRecord::new("b", vec![0.0, 1.0], HashMap::new()),
            VectorRecord::new("c", vec![5.0, 5.0], HashMap::new()),
        ];
        for record in records {
            self.insert(record).expect("insert should succeed");
        }
        let results = self
            .search(&[1.1, 0.0], 3, &HashMap::new())
            .expect("search should succeed");
        assert_eq!(results.first().expect("result").id, "a", "closest vector should come first");
    }

    fn contract_search_filters_by_metadata_key(&self) {
        let alice = VectorRecord::new(
            "alice-1",
            vec![1.0, 0.0],
            HashMap::from([("user_id".to_string(), "alice".to_string())]),
        );
        let bob = VectorRecord::new(
            "bob-1",
            vec![1.0, 0.0],
            HashMap::from([("user_id".to_string(), "bob".to_string())]),
        );
        self.insert(alice).expect("insert should succeed");
        self.insert(bob).expect("insert should succeed");

        let filters = HashMap::from([("user_id".to_string(), "alice".to_string())]);
        let results = self
            .search(&[0.0, 0.0], 10, &filters)
            .expect("search should succeed");
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, vec!["alice-1"], "only alice's record should match");
        assert!(
            !ids.contains(&"bob-1"),
            "bob's record must be filtered out"
        );
    }

    fn contract_search_filters_by_agent_id(&self) {
        let a1 = VectorRecord::new(
            "agent-a-1",
            vec![1.0, 0.0],
            HashMap::from([("agent_id".to_string(), "agent-a".to_string())]),
        );
        let a2 = VectorRecord::new(
            "agent-b-1",
            vec![1.0, 0.0],
            HashMap::from([("agent_id".to_string(), "agent-b".to_string())]),
        );
        self.insert(a1).expect("insert should succeed");
        self.insert(a2).expect("insert should succeed");

        let filters = HashMap::from([("agent_id".to_string(), "agent-a".to_string())]);
        let results = self
            .search(&[0.0, 0.0], 10, &filters)
            .expect("search should succeed");
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, vec!["agent-a-1"], "only agent-a's record should match");
        assert!(
            !ids.contains(&"agent-b-1"),
            "agent-b's record must be filtered out"
        );
    }

    fn contract_search_filters_by_run_id(&self) {
        let r1 = VectorRecord::new(
            "run-1-1",
            vec![1.0, 0.0],
            HashMap::from([("run_id".to_string(), "run-1".to_string())]),
        );
        let r2 = VectorRecord::new(
            "run-2-1",
            vec![1.0, 0.0],
            HashMap::from([("run_id".to_string(), "run-2".to_string())]),
        );
        self.insert(r1).expect("insert should succeed");
        self.insert(r2).expect("insert should succeed");

        let filters = HashMap::from([("run_id".to_string(), "run-1".to_string())]);
        let results = self
            .search(&[0.0, 0.0], 10, &filters)
            .expect("search should succeed");
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, vec!["run-1-1"], "only run-1's record should match");
        assert!(
            !ids.contains(&"run-2-1"),
            "run-2's record must be filtered out"
        );
    }
}


impl<T: VectorStore + ?Sized> VectorStoreContractTests for T {}

pub struct InMemoryVectorStore {
    records: Mutex<HashMap<String, VectorRecord>>,
}

impl InMemoryVectorStore {
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorStore for InMemoryVectorStore {
    fn insert(&self, record: VectorRecord) -> Result<(), VectorStoreError> {
        let mut records = self.records.lock().expect("lock poisoned");
        if let Some(existing) = records.values().next() {
            let expected = existing.vector.len();
            let actual = record.vector.len();
            if expected != actual {
                drop(records);
                return Err(VectorStoreError::DimensionMismatch { expected, actual });
            }
        }
        records.insert(record.id.clone(), record);
        drop(records);
        Ok(())
    }

    fn search(
        &self,
        vector: &[f32],
        top_k: usize,
        filters: &HashMap<String, String>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let records = self.records.lock().expect("lock poisoned");
        let mut scored: Vec<SearchResult> = records
            .values()
            .filter(|r| {
                filters
                    .iter()
                    .all(|(k, v)| r.payload.get(k).is_some_and(|pv| pv == v))
            })
            .map(|r| {
                let score = r
                    .vector
                    .iter()
                    .zip(vector.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, |acc, d| acc + d);
                SearchResult {
                    id: r.id.clone(),
                    score,
                    payload: r.payload.clone(),
                }
            })
            .collect();
        drop(records);
        scored.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    fn get(&self, id: &str) -> Result<Option<VectorRecord>, VectorStoreError> {
        Ok(self.records.lock().expect("lock poisoned").get(id).cloned())
    }

    fn update(&self, record: VectorRecord) -> Result<(), VectorStoreError> {
        let mut records = self.records.lock().expect("lock poisoned");
        if records.contains_key(&record.id) {
            records.insert(record.id.clone(), record);
            drop(records);
            Ok(())
        } else {
            drop(records);
            Err(VectorStoreError::NotFound)
        }
    }

    fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        self.records.lock().expect("lock poisoned").remove(id);
        Ok(())
    }

    fn list(&self, offset: usize, limit: usize) -> Result<Vec<String>, VectorStoreError> {
        Ok(self
            .records
            .lock()
            .expect("lock poisoned")
            .keys()
            .skip(offset)
            .take(limit)
            .cloned()
            .collect())
    }

    fn reset(&self) -> Result<(), VectorStoreError> {
        self.records.lock().expect("lock poisoned").clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{InMemoryVectorStore, VectorStoreConfig, VectorStoreContractTests};

    #[test]
    fn in_memory_store_passes_insert_then_get_contract() {
        InMemoryVectorStore::new().contract_insert_then_get_round_trips();
    }

    #[test]
    fn in_memory_store_passes_delete_then_get_contract() {
        InMemoryVectorStore::new().contract_delete_then_get_returns_none();
    }

    #[test]
    fn in_memory_store_passes_reset_contract() {
        InMemoryVectorStore::new().contract_reset_clears_everything();
    }

    #[test]
    fn in_memory_store_passes_search_respects_top_k_contract() {
        InMemoryVectorStore::new().contract_search_respects_top_k();
    }

    #[test]
    fn in_memory_store_passes_search_orders_by_score_contract() {
        InMemoryVectorStore::new().contract_search_orders_by_score();
    }

    #[test]
    fn in_memory_store_passes_search_filters_by_metadata_key_contract() {
        InMemoryVectorStore::new().contract_search_filters_by_metadata_key();
    }

    #[test]
    fn in_memory_store_passes_search_filters_by_agent_id_contract() {
        InMemoryVectorStore::new().contract_search_filters_by_agent_id();
    }

    #[test]
    fn in_memory_store_passes_search_filters_by_run_id_contract() {
        InMemoryVectorStore::new().contract_search_filters_by_run_id();
    }

    #[test]
    fn in_memory_store_passes_update_then_get_contract() {
        InMemoryVectorStore::new().contract_update_then_get_reflects_change();
    }

    #[test]
    fn in_memory_store_passes_update_nonexistent_contract() {
        InMemoryVectorStore::new().contract_update_nonexistent_returns_not_found();
    }

    #[test]
    fn in_memory_store_passes_delete_nonexistent_is_idempotent_contract() {
        InMemoryVectorStore::new().contract_delete_nonexistent_is_idempotent();
    }

    #[test]
    fn in_memory_store_passes_list_returns_all_inserted_ids_contract() {
        InMemoryVectorStore::new().contract_list_returns_all_inserted_ids();
    }

    #[test]
    fn in_memory_store_passes_list_on_empty_store_returns_empty_contract() {
        InMemoryVectorStore::new().contract_list_on_empty_store_returns_empty();
    }

    #[test]
    fn in_memory_store_passes_insert_rejects_mismatched_dimension_contract() {
        InMemoryVectorStore::new().contract_insert_rejects_mismatched_dimension();
    }

    #[test]
    fn in_memory_store_passes_list_pagination_respects_offset_and_limit_contract() {
        InMemoryVectorStore::new().contract_list_pagination_respects_offset_and_limit();
    }

    #[test]
    fn config_with_valid_collection_name_passes_validation() {
        let config = VectorStoreConfig {
            collection_name: "memories".to_string(),
            url: None,
            api_key: None,
            dimension: None,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_with_empty_collection_name_is_rejected() {
        let config = VectorStoreConfig {
            collection_name: String::new(),
            url: None,
            api_key: None,
            dimension: None,
        };
        assert!(matches!(config.validate(), Err(crate::CoreError::Config(_))));
    }

    #[test]
    fn config_with_whitespace_only_collection_name_is_rejected() {
        let config = VectorStoreConfig {
            collection_name: "   ".to_string(),
            url: None,
            api_key: None,
            dimension: None,
        };
        assert!(matches!(config.validate(), Err(crate::CoreError::Config(_))));
    }

    #[test]
    fn config_with_positive_dimension_passes_validation() {
        let config = VectorStoreConfig {
            collection_name: "memories".to_string(),
            url: None,
            api_key: None,
            dimension: Some(768),
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_with_zero_dimension_is_rejected() {
        let config = VectorStoreConfig {
            collection_name: "memories".to_string(),
            url: None,
            api_key: None,
            dimension: Some(0),
        };
        assert!(matches!(config.validate(), Err(crate::CoreError::Config(_))));
    }
}
