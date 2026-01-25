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
}

impl<T: VectorStore + ?Sized> VectorStoreContractTests for T {}

#[cfg(test)]
mod tests {
    use super::{
        SearchResult, VectorRecord, VectorStore, VectorStoreContractTests, VectorStoreError,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct FakeVectorStore {
        records: Mutex<HashMap<String, VectorRecord>>,
    }

    impl FakeVectorStore {
        #[must_use]
        fn new() -> Self {
            Self {
                records: Mutex::new(HashMap::new()),
            }
        }
    }

    impl VectorStore for FakeVectorStore {
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

    #[test]
    fn fake_store_passes_insert_then_get_contract() {
        FakeVectorStore::new().contract_insert_then_get_round_trips();
    }

    #[test]
    fn fake_store_passes_delete_then_get_contract() {
        FakeVectorStore::new().contract_delete_then_get_returns_none();
    }

    #[test]
    fn fake_store_passes_reset_contract() {
        FakeVectorStore::new().contract_reset_clears_everything();
    }
}
