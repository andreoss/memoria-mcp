use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
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

#[derive(Clone, Debug, PartialEq, Eq)]
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
