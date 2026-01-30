use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: Vec<u8>,
}

impl VectorRecord {
    #[must_use]
    pub fn new(id: impl Into<String>, vector: Vec<f32>, payload: Vec<u8>) -> Self {
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
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VectorStoreError {
    NotFound,
    Backend(String),
}

impl fmt::Display for VectorStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "record not found"),
            Self::Backend(reason) => write!(f, "backend error: {reason}"),
        }
    }
}

impl std::error::Error for VectorStoreError {}

pub trait VectorStore {
    #[allow(clippy::missing_errors_doc)]
    fn insert(&self, record: VectorRecord) -> Result<(), VectorStoreError>;

    #[allow(clippy::missing_errors_doc)]
    fn search(&self, vector: &[f32], top_k: usize) -> Result<Vec<SearchResult>, VectorStoreError>;

    #[allow(clippy::missing_errors_doc)]
    fn get(&self, id: &str) -> Result<Option<VectorRecord>, VectorStoreError>;

    #[allow(clippy::missing_errors_doc)]
    fn update(&self, record: VectorRecord) -> Result<(), VectorStoreError>;

    #[allow(clippy::missing_errors_doc)]
    fn delete(&self, id: &str) -> Result<(), VectorStoreError>;

    #[allow(clippy::missing_errors_doc)]
    fn list(&self) -> Result<Vec<String>, VectorStoreError>;

    #[allow(clippy::missing_errors_doc)]
    fn reset(&self) -> Result<(), VectorStoreError>;
}

pub trait VectorStoreContractTests: VectorStore {
    fn contract_insert_then_get_round_trips(&self) {
        let record = VectorRecord::new("a", vec![1.0, 2.0], vec![9, 9]);
        self.insert(record.clone()).expect("insert should succeed");
        let fetched = self.get("a").expect("get should succeed");
        assert_eq!(fetched, Some(record), "inserted record should round-trip");
    }

    fn contract_delete_then_get_returns_none(&self) {
        let record = VectorRecord::new("b", vec![3.0, 4.0], vec![8, 8]);
        self.insert(record).expect("insert should succeed");
        self.delete("b").expect("delete should succeed");
        let fetched = self.get("b").expect("get should succeed");
        assert_eq!(fetched, None, "deleted record should not be found");
    }

    fn contract_reset_clears_everything(&self) {
        let record = VectorRecord::new("c", vec![5.0, 6.0], vec![7, 7]);
        self.insert(record).expect("insert should succeed");
        self.reset().expect("reset should succeed");
        let listed = self.list().expect("list should succeed");
        assert!(listed.is_empty(), "reset should clear all records");
    }

    fn contract_search_respects_top_k(&self) {
        let records = [
            VectorRecord::new("a", vec![1.0, 0.0], vec![1]),
            VectorRecord::new("b", vec![0.0, 1.0], vec![2]),
            VectorRecord::new("c", vec![1.0, 1.0], vec![3]),
            VectorRecord::new("d", vec![2.0, 0.0], vec![4]),
        ];
        for record in records {
            self.insert(record).expect("insert should succeed");
        }
        let top_k = 2;
        let results = self
            .search(&[0.0, 0.0], top_k)
            .expect("search should succeed");
        assert_eq!(
            results.len(),
            top_k,
            "search should return exactly top_k results"
        );
    }

    fn contract_update_then_get_reflects_change(&self) {
        let original = VectorRecord::new("a", vec![1.0, 2.0], vec![9, 9]);
        self.insert(original).expect("insert should succeed");
        let updated = VectorRecord::new("a", vec![3.0, 4.0], vec![8, 8]);
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
        let record = VectorRecord::new("missing", vec![1.0, 2.0], vec![9, 9]);
        let result = self.update(record);
        assert_eq!(
            result,
            Err(VectorStoreError::NotFound),
            "updating a nonexistent record should return NotFound"
        );
    }

    fn contract_list_returns_all_inserted_ids(&self) {
        let records = [
            VectorRecord::new("a", vec![1.0, 0.0], vec![1]),
            VectorRecord::new("b", vec![0.0, 1.0], vec![2]),
            VectorRecord::new("c", vec![1.0, 1.0], vec![3]),
        ];
        for record in records {
            self.insert(record).expect("insert should succeed");
        }
        let listed = self.list().expect("list should succeed");
        let mut listed = listed;
        listed.sort();
        assert_eq!(
            listed,
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            "list should return every inserted id"
        );
    }

    fn contract_list_on_empty_store_returns_empty(&self) {
        let listed = self.list().expect("list should succeed on empty store");
        assert!(
            listed.is_empty(),
            "list on an empty store should return an empty Vec, not an error"
        );
    }

    fn contract_search_orders_by_score(&self) {
        let records = [
            VectorRecord::new("a", vec![1.0, 0.0], vec![1]),
            VectorRecord::new("b", vec![0.0, 1.0], vec![2]),
            VectorRecord::new("c", vec![5.0, 5.0], vec![3]),
        ];
        for record in records {
            self.insert(record).expect("insert should succeed");
        }
        let results = self
            .search(&[1.1, 0.0], 3)
            .expect("search should succeed");
        assert_eq!(results.first().expect("result").id, "a", "closest vector should come first");
    }
}

impl<T: VectorStore + ?Sized> VectorStoreContractTests for T {}

use std::collections::HashMap;
use std::sync::Mutex;

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
        self.records
            .lock()
            .expect("lock poisoned")
            .insert(record.id.clone(), record);
        Ok(())
    }

    fn search(&self, vector: &[f32], top_k: usize) -> Result<Vec<SearchResult>, VectorStoreError> {
        let records = self.records.lock().expect("lock poisoned");
        let mut scored: Vec<SearchResult> = records
            .values()
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

    fn list(&self) -> Result<Vec<String>, VectorStoreError> {
        Ok(self
            .records
            .lock()
            .expect("lock poisoned")
            .keys()
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
    use super::{InMemoryVectorStore, VectorStoreContractTests};

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
}
