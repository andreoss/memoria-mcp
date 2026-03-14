use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        threshold: Option<f32>,
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

    #[allow(clippy::missing_errors_doc)]
    fn keyword_search(&self, _query: &str, _top_k: usize, _filters: &HashMap<String, String>) -> Result<Option<Vec<SearchResult>>, VectorStoreError> {
        Ok(None)
    }
}

impl<T: VectorStore + ?Sized> VectorStore for Box<T> {
    fn insert(&self, record: VectorRecord) -> Result<(), VectorStoreError> {
        self.as_ref().insert(record)
    }

    fn search(
        &self,
        vector: &[f32],
        top_k: usize,
        filters: &HashMap<String, String>,
        threshold: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        self.as_ref().search(vector, top_k, filters, threshold)
    }

    fn get(&self, id: &str) -> Result<Option<VectorRecord>, VectorStoreError> {
        self.as_ref().get(id)
    }

    fn update(&self, record: VectorRecord) -> Result<(), VectorStoreError> {
        self.as_ref().update(record)
    }

    fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        self.as_ref().delete(id)
    }

    fn list(&self, offset: usize, limit: usize) -> Result<Vec<String>, VectorStoreError> {
        self.as_ref().list(offset, limit)
    }

    fn reset(&self) -> Result<(), VectorStoreError> {
        self.as_ref().reset()
    }

    fn keyword_search(&self, query: &str, top_k: usize, filters: &HashMap<String, String>) -> Result<Option<Vec<SearchResult>>, VectorStoreError> {
        self.as_ref().keyword_search(query, top_k, filters)
    }
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
            .search(&[0.0, 0.0], top_k, &HashMap::new(), None)
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
            .search(&[1.1, 0.0], 3, &HashMap::new(), None)
            .expect("search should succeed");
        assert_eq!(results.first().expect("result").id, "a", "closest vector should come first");
    }

    fn contract_search_respects_threshold(&self) {
        let records = [
            VectorRecord::new("close", vec![1.0, 0.0], HashMap::new()),
            VectorRecord::new("far", vec![50.0, 50.0], HashMap::new()),
        ];
        for record in records {
            self.insert(record).expect("insert should succeed");
        }
        let results = self
            .search(&[1.0, 0.0], 10, &HashMap::new(), Some(1.0))
            .expect("search should succeed");
        assert_eq!(results.len(), 1, "only the record within the score threshold should be returned");
        assert_eq!(results[0].id, "close");
    }

    fn contract_search_with_no_threshold_returns_everything_up_to_top_k(&self) {
        let records = [
            VectorRecord::new("close", vec![1.0, 0.0], HashMap::new()),
            VectorRecord::new("far", vec![50.0, 50.0], HashMap::new()),
        ];
        for record in records {
            self.insert(record).expect("insert should succeed");
        }
        let results = self
            .search(&[1.0, 0.0], 10, &HashMap::new(), None)
            .expect("search should succeed");
        assert_eq!(results.len(), 2, "an unset threshold should not filter anything");
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
            .search(&[0.0, 0.0], 10, &filters, None)
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
            .search(&[0.0, 0.0], 10, &filters, None)
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
            .search(&[0.0, 0.0], 10, &filters, None)
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
        threshold: Option<f32>,
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
        if let Some(threshold) = threshold {
            scored.retain(|result| result.score <= threshold);
        }
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

#[cfg(feature = "sqlite")]
use rusqlite::OptionalExtension as _;

#[cfg(feature = "sqlite")]
pub struct SqliteVectorStore {
    conn: Mutex<rusqlite::Connection>,
}

#[cfg(feature = "sqlite")]
impl SqliteVectorStore {
    #[allow(clippy::missing_errors_doc)]
    pub fn open(path: &std::path::Path) -> Result<Self, VectorStoreError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        }
        let conn = rusqlite::Connection::open(path).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, vector TEXT NOT NULL, payload TEXT NOT NULL)",
            [],
        )
        .map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(id UNINDEXED, content)", [])
            .map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    fn sync_fts(conn: &rusqlite::Connection, id: &str, content: &str) -> Result<(), VectorStoreError> {
        conn.execute("DELETE FROM records_fts WHERE id = ?1", rusqlite::params![id]).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        if !content.is_empty() {
            conn.execute("INSERT INTO records_fts (id, content) VALUES (?1, ?2)", rusqlite::params![id, content])
                .map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        }
        Ok(())
    }

    fn fts_match_expression(query: &str) -> String {
        query
            .split_whitespace()
            .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" OR ")
    }

    fn decode_record(id: String, vector_json: &str, payload_json: &str) -> Result<VectorRecord, VectorStoreError> {
        let vector: Vec<f32> = serde_json::from_str(vector_json).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        let payload: HashMap<String, String> = serde_json::from_str(payload_json).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        Ok(VectorRecord { id, vector, payload })
    }

    fn all_records(conn: &rusqlite::Connection) -> Result<Vec<VectorRecord>, VectorStoreError> {
        let mut stmt = conn
            .prepare("SELECT id, vector, payload FROM records")
            .map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let vector: String = row.get(1)?;
                let payload: String = row.get(2)?;
                Ok((id, vector, payload))
            })
            .map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        let mut records = Vec::new();
        for row in rows {
            let (id, vector, payload) = row.map_err(|err| VectorStoreError::Backend(err.to_string()))?;
            records.push(Self::decode_record(id, &vector, &payload)?);
        }
        Ok(records)
    }
}

#[cfg(feature = "sqlite")]
impl VectorStore for SqliteVectorStore {
    fn insert(&self, record: VectorRecord) -> Result<(), VectorStoreError> {
        let conn = self.conn.lock().expect("lock poisoned");
        let existing_vector: Option<String> = conn
            .query_row("SELECT vector FROM records LIMIT 1", [], |row| row.get(0))
            .optional()
            .map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        if let Some(existing_vector) = existing_vector {
            let existing: Vec<f32> = serde_json::from_str(&existing_vector).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
            if existing.len() != record.vector.len() {
                return Err(VectorStoreError::DimensionMismatch { expected: existing.len(), actual: record.vector.len() });
            }
        }
        let vector_json = serde_json::to_string(&record.vector).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        let payload_json = serde_json::to_string(&record.payload).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        let result = conn
            .execute(
                "INSERT INTO records (id, vector, payload) VALUES (?1, ?2, ?3)
             ON CONFLICT(id) DO UPDATE SET vector = excluded.vector, payload = excluded.payload",
                rusqlite::params![record.id, vector_json, payload_json],
            )
            .map(|_| ())
            .map_err(|err| VectorStoreError::Backend(err.to_string()));
        result?;
        let content = record.payload.get("content").map_or("", String::as_str);
        let fts_result = Self::sync_fts(&conn, &record.id, content);
        drop(conn);
        fts_result
    }

    fn search(
        &self,
        vector: &[f32],
        top_k: usize,
        filters: &HashMap<String, String>,
        threshold: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let conn = self.conn.lock().expect("lock poisoned");
        let records = Self::all_records(&conn)?;
        drop(conn);
        let mut scored: Vec<SearchResult> = records
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
        let conn = self.conn.lock().expect("lock poisoned");
        let row: Result<Option<(String, String)>, VectorStoreError> = conn
            .query_row("SELECT vector, payload FROM records WHERE id = ?1", rusqlite::params![id], |row| {
                let vector: String = row.get(0)?;
                let payload: String = row.get(1)?;
                Ok((vector, payload))
            })
            .optional()
            .map_err(|err| VectorStoreError::Backend(err.to_string()));
        drop(conn);
        match row? {
            Some((vector, payload)) => Ok(Some(Self::decode_record(id.to_string(), &vector, &payload)?)),
            None => Ok(None),
        }
    }

    fn update(&self, record: VectorRecord) -> Result<(), VectorStoreError> {
        let conn = self.conn.lock().expect("lock poisoned");
        let vector_json = serde_json::to_string(&record.vector).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        let payload_json = serde_json::to_string(&record.payload).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        let affected = conn
            .execute("UPDATE records SET vector = ?2, payload = ?3 WHERE id = ?1", rusqlite::params![record.id, vector_json, payload_json])
            .map_err(|err| VectorStoreError::Backend(err.to_string()));
        if affected? == 0 {
            drop(conn);
            return Err(VectorStoreError::NotFound);
        }
        let content = record.payload.get("content").map_or("", String::as_str);
        let fts_result = Self::sync_fts(&conn, &record.id, content);
        drop(conn);
        fts_result
    }

    fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        let conn = self.conn.lock().expect("lock poisoned");
        let result = conn.execute("DELETE FROM records WHERE id = ?1", rusqlite::params![id]).map(|_| ()).map_err(|err| VectorStoreError::Backend(err.to_string()));
        result?;
        let fts_result = conn.execute("DELETE FROM records_fts WHERE id = ?1", rusqlite::params![id]).map(|_| ()).map_err(|err| VectorStoreError::Backend(err.to_string()));
        drop(conn);
        fts_result
    }

    fn list(&self, offset: usize, limit: usize) -> Result<Vec<String>, VectorStoreError> {
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare("SELECT id FROM records ORDER BY id LIMIT ?1 OFFSET ?2")
            .map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let offset = i64::try_from(offset).unwrap_or(i64::MAX);
        let rows = stmt
            .query_map(rusqlite::params![limit, offset], |row| row.get::<_, String>(0))
            .map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        let mut ids = Vec::new();
        for row in rows {
            ids.push(row.map_err(|err| VectorStoreError::Backend(err.to_string()))?);
        }
        drop(stmt);
        drop(conn);
        Ok(ids)
    }

    fn reset(&self) -> Result<(), VectorStoreError> {
        let conn = self.conn.lock().expect("lock poisoned");
        let result = conn.execute("DELETE FROM records", []).map(|_| ()).map_err(|err| VectorStoreError::Backend(err.to_string()));
        result?;
        let fts_result = conn.execute("DELETE FROM records_fts", []).map(|_| ()).map_err(|err| VectorStoreError::Backend(err.to_string()));
        drop(conn);
        fts_result
    }

    fn keyword_search(&self, query: &str, top_k: usize, filters: &HashMap<String, String>) -> Result<Option<Vec<SearchResult>>, VectorStoreError> {
        let match_expression = Self::fts_match_expression(query);
        if match_expression.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let conn = self.conn.lock().expect("lock poisoned");
        let mut stmt = conn
            .prepare("SELECT id, bm25(records_fts) FROM records_fts WHERE records_fts MATCH ?1 ORDER BY bm25(records_fts)")
            .map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        let rows = stmt
            .query_map(rusqlite::params![match_expression], |row| {
                let id: String = row.get(0)?;
                let score: f64 = row.get(1)?;
                Ok((id, score))
            })
            .map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        let mut matches = Vec::new();
        for row in rows {
            matches.push(row.map_err(|err| VectorStoreError::Backend(err.to_string()))?);
        }
        drop(stmt);

        let mut results = Vec::new();
        for (id, score) in matches {
            let record_row: Option<String> = conn
                .query_row("SELECT payload FROM records WHERE id = ?1", rusqlite::params![id], |row| row.get(0))
                .optional()
                .map_err(|err| VectorStoreError::Backend(err.to_string()))?;
            let Some(payload_json) = record_row else { continue };
            let payload: HashMap<String, String> = serde_json::from_str(&payload_json).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
            if !filters.iter().all(|(k, v)| payload.get(k).is_some_and(|pv| pv == v)) {
                continue;
            }
            #[allow(clippy::cast_possible_truncation)]
            let score = score as f32;
            results.push(SearchResult { id, score, payload });
            if results.len() >= top_k {
                break;
            }
        }
        drop(conn);
        Ok(Some(results))
    }
}

#[cfg(feature = "postgres")]
pub struct PgVectorStore {
    client: Mutex<postgres::Client>,
    dimension: usize,
    table: String,
}

#[cfg(feature = "postgres")]
impl PgVectorStore {
    #[allow(clippy::missing_errors_doc)]
    pub fn open(connection_string: &str, dimension: usize, table: &str) -> Result<Self, VectorStoreError> {
        if !table.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || table.is_empty() {
            return Err(VectorStoreError::Backend(format!("invalid table name: {table}")));
        }
        let mut client = postgres::Client::connect(connection_string, postgres::NoTls).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        client.execute("CREATE EXTENSION IF NOT EXISTS vector", &[]).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        client
            .execute(
                &format!("CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, vector VECTOR({dimension}) NOT NULL, payload JSONB NOT NULL)"),
                &[],
            )
            .map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        Ok(Self { client: Mutex::new(client), dimension, table: table.to_string() })
    }

    fn decode_payload(value: serde_json::Value) -> Result<HashMap<String, String>, VectorStoreError> {
        serde_json::from_value(value).map_err(|err| VectorStoreError::Backend(err.to_string()))
    }

    fn filters_to_jsonb(filters: &HashMap<String, String>) -> serde_json::Value {
        serde_json::Value::Object(filters.iter().map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone()))).collect())
    }
}

#[cfg(feature = "postgres")]
impl VectorStore for PgVectorStore {
    fn insert(&self, record: VectorRecord) -> Result<(), VectorStoreError> {
        if record.vector.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch { expected: self.dimension, actual: record.vector.len() });
        }
        let vector = pgvector::Vector::from(record.vector);
        let payload = serde_json::to_value(&record.payload).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        let mut client = self.client.lock().expect("lock poisoned");
        client
            .execute(
                &format!(
                    "INSERT INTO {} (id, vector, payload) VALUES ($1, $2, $3) \
                     ON CONFLICT (id) DO UPDATE SET vector = excluded.vector, payload = excluded.payload",
                    self.table
                ),
                &[&record.id, &vector, &payload],
            )
            .map(|_| ())
            .map_err(|err| VectorStoreError::Backend(err.to_string()))
    }

    fn search(&self, vector: &[f32], top_k: usize, filters: &HashMap<String, String>, threshold: Option<f32>) -> Result<Vec<SearchResult>, VectorStoreError> {
        let query_vector = pgvector::Vector::from(vector.to_vec());
        let filters_json = Self::filters_to_jsonb(filters);
        let mut client = self.client.lock().expect("lock poisoned");
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let limit = top_k.min(i64::MAX as usize) as i64;
        let rows = if let Some(threshold) = threshold {
            client
                .query(
                    &format!(
                        "SELECT id, vector <-> $1 AS distance, payload FROM {} \
                         WHERE payload @> $2::jsonb AND vector <-> $1 <= $3 ORDER BY distance LIMIT $4",
                        self.table
                    ),
                    &[&query_vector, &filters_json, &f64::from(threshold), &limit],
                )
                .map_err(|err| VectorStoreError::Backend(err.to_string()))?
        } else {
            client
                .query(
                    &format!("SELECT id, vector <-> $1 AS distance, payload FROM {} WHERE payload @> $2::jsonb ORDER BY distance LIMIT $3", self.table),
                    &[&query_vector, &filters_json, &limit],
                )
                .map_err(|err| VectorStoreError::Backend(err.to_string()))?
        };
        drop(client);
        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            let id: String = row.get(0);
            #[allow(clippy::cast_possible_truncation)]
            let score = row.get::<_, f64>(1) as f32;
            let payload = Self::decode_payload(row.get(2))?;
            results.push(SearchResult { id, score, payload });
        }
        Ok(results)
    }

    fn get(&self, id: &str) -> Result<Option<VectorRecord>, VectorStoreError> {
        let mut client = self.client.lock().expect("lock poisoned");
        let row = client
            .query_opt(&format!("SELECT vector, payload FROM {} WHERE id = $1", self.table), &[&id])
            .map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        drop(client);
        match row {
            Some(row) => {
                let vector: pgvector::Vector = row.get(0);
                let payload = Self::decode_payload(row.get(1))?;
                Ok(Some(VectorRecord { id: id.to_string(), vector: vector.to_vec(), payload }))
            }
            None => Ok(None),
        }
    }

    fn update(&self, record: VectorRecord) -> Result<(), VectorStoreError> {
        let vector = pgvector::Vector::from(record.vector);
        let payload = serde_json::to_value(&record.payload).map_err(|err| VectorStoreError::Backend(err.to_string()))?;
        let mut client = self.client.lock().expect("lock poisoned");
        let affected = client
            .execute(&format!("UPDATE {} SET vector = $2, payload = $3 WHERE id = $1", self.table), &[&record.id, &vector, &payload])
            .map_err(|err| VectorStoreError::Backend(err.to_string()));
        drop(client);
        if affected? == 0 {
            return Err(VectorStoreError::NotFound);
        }
        Ok(())
    }

    fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        let mut client = self.client.lock().expect("lock poisoned");
        client.execute(&format!("DELETE FROM {} WHERE id = $1", self.table), &[&id]).map(|_| ()).map_err(|err| VectorStoreError::Backend(err.to_string()))
    }

    fn list(&self, offset: usize, limit: usize) -> Result<Vec<String>, VectorStoreError> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let limit = limit.min(i64::MAX as usize) as i64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let offset = offset.min(i64::MAX as usize) as i64;
        let mut client = self.client.lock().expect("lock poisoned");
        let rows = client
            .query(&format!("SELECT id FROM {} ORDER BY id LIMIT $1 OFFSET $2", self.table), &[&limit, &offset])
            .map_err(|err| VectorStoreError::Backend(err.to_string()));
        drop(client);
        Ok(rows?.into_iter().map(|row| row.get(0)).collect())
    }

    fn reset(&self) -> Result<(), VectorStoreError> {
        let mut client = self.client.lock().expect("lock poisoned");
        client.execute(&format!("DELETE FROM {}", self.table), &[]).map(|_| ()).map_err(|err| VectorStoreError::Backend(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::{InMemoryVectorStore, VectorStore, VectorStoreConfig, VectorStoreContractTests};
    use crate::test_support::VecVectorStore;
    use std::collections::HashMap;

    #[test]
    fn in_memory_store_passes_insert_then_get_contract() {
        InMemoryVectorStore::new().contract_insert_then_get_round_trips();
    }

    #[test]
    fn in_memory_store_keyword_search_is_not_supported_by_default() {
        let store = InMemoryVectorStore::new();
        let result = store.keyword_search("anything", 10, &HashMap::new()).expect("the default keyword_search must not error");
        assert!(result.is_none(), "InMemoryVectorStore has no native full-text engine");
    }

    #[test]
    fn vec_store_passes_insert_then_get_contract() {
        VecVectorStore::new().contract_insert_then_get_round_trips();
    }

    #[test]
    fn vec_store_passes_delete_then_get_contract() {
        VecVectorStore::new().contract_delete_then_get_returns_none();
    }

    #[test]
    fn vec_store_passes_reset_contract() {
        VecVectorStore::new().contract_reset_clears_everything();
    }

    #[test]
    fn vec_store_passes_search_respects_top_k_contract() {
        VecVectorStore::new().contract_search_respects_top_k();
    }

    #[test]
    fn vec_store_passes_search_orders_by_score_contract() {
        VecVectorStore::new().contract_search_orders_by_score();
    }

    #[test]
    fn vec_store_passes_search_respects_threshold_contract() {
        VecVectorStore::new().contract_search_respects_threshold();
    }

    #[test]
    fn vec_store_passes_search_with_no_threshold_returns_everything_contract() {
        VecVectorStore::new().contract_search_with_no_threshold_returns_everything_up_to_top_k();
    }

    #[test]
    fn vec_store_passes_search_filters_by_agent_id_contract() {
        VecVectorStore::new().contract_search_filters_by_agent_id();
    }

    #[test]
    fn vec_store_passes_update_nonexistent_contract() {
        VecVectorStore::new().contract_update_nonexistent_returns_not_found();
    }

    #[test]
    fn vec_store_passes_delete_nonexistent_is_idempotent_contract() {
        VecVectorStore::new().contract_delete_nonexistent_is_idempotent();
    }

    #[test]
    fn vec_store_passes_insert_rejects_mismatched_dimension_contract() {
        VecVectorStore::new().contract_insert_rejects_mismatched_dimension();
    }

    #[test]
    fn vec_store_passes_list_pagination_respects_offset_and_limit_contract() {
        VecVectorStore::new().contract_list_pagination_respects_offset_and_limit();
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
    fn in_memory_store_passes_search_respects_threshold_contract() {
        InMemoryVectorStore::new().contract_search_respects_threshold();
    }

    #[test]
    fn in_memory_store_passes_search_with_no_threshold_returns_everything_contract() {
        InMemoryVectorStore::new().contract_search_with_no_threshold_returns_everything_up_to_top_k();
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

    #[cfg(feature = "sqlite")]
    mod sqlite_tests {
        use super::super::{SqliteVectorStore, VectorRecord, VectorStore, VectorStoreContractTests};
        use std::collections::HashMap;

        fn temp_db_path(name: &str) -> std::path::PathBuf {
            std::env::temp_dir().join(format!("memoria-sqlite-vector-store-test-{name}-{}.db", std::process::id()))
        }

        struct TempStore {
            store: SqliteVectorStore,
            path: std::path::PathBuf,
        }

        impl Drop for TempStore {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.path);
            }
        }

        impl std::ops::Deref for TempStore {
            type Target = SqliteVectorStore;
            fn deref(&self) -> &Self::Target {
                &self.store
            }
        }

        fn temp_store(name: &str) -> TempStore {
            let path = temp_db_path(name);
            let _ = std::fs::remove_file(&path);
            let store = SqliteVectorStore::open(&path).expect("open should succeed");
            TempStore { store, path }
        }

        #[test]
        fn sqlite_store_passes_insert_then_get_contract() {
            temp_store("insert-then-get").contract_insert_then_get_round_trips();
        }

        #[test]
        fn sqlite_store_passes_delete_then_get_contract() {
            temp_store("delete-then-get").contract_delete_then_get_returns_none();
        }

        #[test]
        fn sqlite_store_passes_reset_contract() {
            temp_store("reset").contract_reset_clears_everything();
        }

        #[test]
        fn sqlite_store_passes_search_respects_top_k_contract() {
            temp_store("search-top-k").contract_search_respects_top_k();
        }

        #[test]
        fn sqlite_store_passes_search_orders_by_score_contract() {
            temp_store("search-orders").contract_search_orders_by_score();
        }

        #[test]
        fn sqlite_store_passes_search_respects_threshold_contract() {
            temp_store("search-threshold").contract_search_respects_threshold();
        }

        #[test]
        fn sqlite_store_passes_search_with_no_threshold_returns_everything_contract() {
            temp_store("search-no-threshold").contract_search_with_no_threshold_returns_everything_up_to_top_k();
        }

        #[test]
        fn sqlite_store_passes_search_filters_by_metadata_key_contract() {
            temp_store("search-filters-metadata").contract_search_filters_by_metadata_key();
        }

        #[test]
        fn sqlite_store_passes_search_filters_by_agent_id_contract() {
            temp_store("search-filters-agent").contract_search_filters_by_agent_id();
        }

        #[test]
        fn sqlite_store_passes_search_filters_by_run_id_contract() {
            temp_store("search-filters-run").contract_search_filters_by_run_id();
        }

        #[test]
        fn sqlite_store_passes_update_then_get_contract() {
            temp_store("update-then-get").contract_update_then_get_reflects_change();
        }

        #[test]
        fn sqlite_store_passes_update_nonexistent_contract() {
            temp_store("update-nonexistent").contract_update_nonexistent_returns_not_found();
        }

        #[test]
        fn sqlite_store_passes_delete_nonexistent_is_idempotent_contract() {
            temp_store("delete-nonexistent").contract_delete_nonexistent_is_idempotent();
        }

        #[test]
        fn sqlite_store_passes_list_returns_all_inserted_ids_contract() {
            temp_store("list-all").contract_list_returns_all_inserted_ids();
        }

        #[test]
        fn sqlite_store_passes_list_on_empty_store_returns_empty_contract() {
            temp_store("list-empty").contract_list_on_empty_store_returns_empty();
        }

        #[test]
        fn sqlite_store_passes_insert_rejects_mismatched_dimension_contract() {
            temp_store("dimension-mismatch").contract_insert_rejects_mismatched_dimension();
        }

        #[test]
        fn sqlite_store_passes_list_pagination_respects_offset_and_limit_contract() {
            temp_store("pagination").contract_list_pagination_respects_offset_and_limit();
        }

        #[test]
        fn records_survive_reopening_the_same_database_file() {
            let path = temp_db_path("durability");
            let _ = std::fs::remove_file(&path);
            {
                let store = SqliteVectorStore::open(&path).expect("open should succeed");
                store
                    .insert(VectorRecord::new("rec-1", vec![1.0, 2.0, 3.0], HashMap::from([("user_id".to_string(), "alice".to_string())])))
                    .expect("insert should succeed");
            }
            let reopened = SqliteVectorStore::open(&path).expect("reopen should succeed");
            let record = reopened.get("rec-1").expect("get should succeed").expect("record should survive reopening the file");
            assert_eq!(record.vector, vec![1.0, 2.0, 3.0]);
            assert_eq!(record.payload.get("user_id"), Some(&"alice".to_string()));
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn open_creates_the_database_file_if_it_does_not_exist() {
            let path = temp_db_path("create-on-open");
            let _ = std::fs::remove_file(&path);
            assert!(!path.exists(), "the file should not exist before open");
            let _store = SqliteVectorStore::open(&path).expect("open should succeed");
            assert!(path.exists(), "open should create the database file");
            let _ = std::fs::remove_file(&path);
        }

        fn record_with_content(id: &str, content: &str, scope: &[(&str, &str)]) -> VectorRecord {
            let mut payload: HashMap<String, String> = scope.iter().map(|(k, v)| ((*k).to_string(), (*v).to_string())).collect();
            payload.insert("content".to_string(), content.to_string());
            VectorRecord::new(id, vec![0.0], payload)
        }

        #[test]
        fn sqlite_store_keyword_search_finds_a_real_match_by_content() {
            let store = temp_store("keyword-match");
            store.insert(record_with_content("a", "the quick brown fox jumps", &[])).expect("insert should succeed");
            store.insert(record_with_content("b", "a lazy dog sleeps all day", &[])).expect("insert should succeed");

            let results = store.keyword_search("fox", 10, &HashMap::new()).expect("keyword_search should succeed").expect("sqlite store must support keyword_search");
            let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
            assert_eq!(ids, vec!["a"], "only the record containing the query term should match");
        }

        #[test]
        fn sqlite_store_keyword_search_ranks_the_better_match_first() {
            let store = temp_store("keyword-rank");
            store.insert(record_with_content("weak", "rust is mentioned once here", &[])).expect("insert should succeed");
            store.insert(record_with_content("strong", "rust rust rust programming in rust", &[])).expect("insert should succeed");

            let results = store.keyword_search("rust", 10, &HashMap::new()).expect("keyword_search should succeed").expect("sqlite store must support keyword_search");
            assert_eq!(results[0].id, "strong", "the record with stronger term frequency should rank first, got: {results:?}");
        }

        #[test]
        fn sqlite_store_keyword_search_respects_filters() {
            let store = temp_store("keyword-filters");
            store.insert(record_with_content("alice-rec", "engineer working on rust", &[("user_id", "alice")])).expect("insert should succeed");
            store.insert(record_with_content("bob-rec", "engineer working on rust", &[("user_id", "bob")])).expect("insert should succeed");

            let filters = HashMap::from([("user_id".to_string(), "alice".to_string())]);
            let results = store.keyword_search("engineer", 10, &filters).expect("keyword_search should succeed").expect("sqlite store must support keyword_search");
            let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
            assert_eq!(ids, vec!["alice-rec"], "keyword_search must respect the same scope filters as search");
        }

        #[test]
        fn sqlite_store_keyword_search_respects_top_k() {
            let store = temp_store("keyword-top-k");
            for i in 0..5 {
                store.insert(record_with_content(&format!("rec-{i}"), "rust rust rust", &[])).expect("insert should succeed");
            }
            let results = store.keyword_search("rust", 2, &HashMap::new()).expect("keyword_search should succeed").expect("sqlite store must support keyword_search");
            assert_eq!(results.len(), 2);
        }

        #[test]
        fn sqlite_store_keyword_search_treats_special_characters_as_literal_terms_not_fts5_syntax() {
            let store = temp_store("keyword-special-chars");
            store.insert(record_with_content("a", "rust programming language", &[])).expect("insert should succeed");

            let result = store.keyword_search("rust\" OR \"*", 10, &HashMap::new());
            assert!(result.is_ok(), "special FTS5 syntax characters in the query must never cause a backend error, got: {result:?}");
        }

        #[test]
        fn sqlite_store_keyword_search_stays_in_sync_after_update() {
            let store = temp_store("keyword-sync-update");
            let record = record_with_content("a", "original content about gardening", &[]);
            store.insert(record.clone()).expect("insert should succeed");

            let mut updated = record;
            updated.payload.insert("content".to_string(), "updated content about astronomy".to_string());
            store.update(updated).expect("update should succeed");

            let old_term_results = store.keyword_search("gardening", 10, &HashMap::new()).expect("keyword_search should succeed").expect("sqlite store must support keyword_search");
            assert!(old_term_results.is_empty(), "the old content's term must no longer match after an update");

            let new_term_results = store.keyword_search("astronomy", 10, &HashMap::new()).expect("keyword_search should succeed").expect("sqlite store must support keyword_search");
            assert_eq!(new_term_results.len(), 1, "the updated content's term must match");
        }

        #[test]
        fn sqlite_store_keyword_search_stays_in_sync_after_delete() {
            let store = temp_store("keyword-sync-delete");
            store.insert(record_with_content("a", "ephemeral content", &[])).expect("insert should succeed");
            store.delete("a").expect("delete should succeed");

            let results = store.keyword_search("ephemeral", 10, &HashMap::new()).expect("keyword_search should succeed").expect("sqlite store must support keyword_search");
            assert!(results.is_empty(), "a deleted record must not appear in keyword_search results");
        }

        #[test]
        fn sqlite_store_keyword_search_stays_in_sync_after_reset() {
            let store = temp_store("keyword-sync-reset");
            store.insert(record_with_content("a", "content before reset", &[])).expect("insert should succeed");
            store.reset().expect("reset should succeed");

            let results = store.keyword_search("content", 10, &HashMap::new()).expect("keyword_search should succeed").expect("sqlite store must support keyword_search");
            assert!(results.is_empty(), "reset must clear the keyword index too");
        }

        #[test]
        fn boxed_dyn_vector_store_forwards_keyword_search_to_the_real_sqlite_implementation() {
            let path = temp_db_path("keyword-box-forward");
            let _ = std::fs::remove_file(&path);
            let store = SqliteVectorStore::open(&path).expect("open should succeed");
            store.insert(record_with_content("a", "boxed dyn dispatch test", &[])).expect("insert should succeed");
            let boxed: Box<dyn VectorStore> = Box::new(store);

            let results = boxed.keyword_search("dispatch", 10, &HashMap::new()).expect("keyword_search should succeed");
            assert!(results.is_some(), "Box<dyn VectorStore> must forward to the real implementation, not silently fall back to the default None");
            assert_eq!(results.expect("checked above").len(), 1);

            drop(boxed);
            let _ = std::fs::remove_file(&path);
        }
    }

    #[cfg(feature = "postgres")]
    mod postgres_tests {
        use super::super::{PgVectorStore, VectorRecord, VectorStore, VectorStoreContractTests};
        use std::collections::HashMap;

        fn test_url() -> Option<String> {
            std::env::var("MEMORIA_TEST_POSTGRES_URL").ok()
        }

        struct TempStore {
            store: PgVectorStore,
        }

        impl Drop for TempStore {
            fn drop(&mut self) {
                let _ = self.store.reset();
            }
        }

        impl std::ops::Deref for TempStore {
            type Target = PgVectorStore;
            fn deref(&self) -> &Self::Target {
                &self.store
            }
        }

        fn temp_store(name: &str) -> Option<TempStore> {
            let url = test_url()?;
            let table = format!("memoria_pg_test_{name}_{}", std::process::id());
            let store = PgVectorStore::open(&url, 2, &table).expect("open should succeed");
            store.reset().expect("reset should succeed");
            Some(TempStore { store })
        }

        macro_rules! pg_contract_test {
            ($test_name:ident, $slug:literal, $contract:ident) => {
                #[test]
                #[ignore = "requires a real Postgres+pgvector instance reachable at MEMORIA_TEST_POSTGRES_URL"]
                fn $test_name() {
                    let Some(store) = temp_store($slug) else {
                        panic!("MEMORIA_TEST_POSTGRES_URL must be set to run this test");
                    };
                    store.$contract();
                }
            };
        }

        pg_contract_test!(pg_store_passes_insert_then_get_contract, "insert_then_get", contract_insert_then_get_round_trips);
        pg_contract_test!(pg_store_passes_delete_then_get_contract, "delete_then_get", contract_delete_then_get_returns_none);
        pg_contract_test!(pg_store_passes_reset_contract, "reset", contract_reset_clears_everything);
        pg_contract_test!(pg_store_passes_search_respects_top_k_contract, "search_top_k", contract_search_respects_top_k);
        pg_contract_test!(pg_store_passes_search_orders_by_score_contract, "search_orders", contract_search_orders_by_score);
        pg_contract_test!(pg_store_passes_search_respects_threshold_contract, "search_threshold", contract_search_respects_threshold);
        pg_contract_test!(
            pg_store_passes_search_with_no_threshold_returns_everything_contract,
            "search_no_threshold",
            contract_search_with_no_threshold_returns_everything_up_to_top_k
        );
        pg_contract_test!(pg_store_passes_search_filters_by_metadata_key_contract, "search_filters_metadata", contract_search_filters_by_metadata_key);
        pg_contract_test!(pg_store_passes_search_filters_by_agent_id_contract, "search_filters_agent", contract_search_filters_by_agent_id);
        pg_contract_test!(pg_store_passes_search_filters_by_run_id_contract, "search_filters_run", contract_search_filters_by_run_id);
        pg_contract_test!(pg_store_passes_update_then_get_contract, "update_then_get", contract_update_then_get_reflects_change);
        pg_contract_test!(pg_store_passes_update_nonexistent_contract, "update_nonexistent", contract_update_nonexistent_returns_not_found);
        pg_contract_test!(pg_store_passes_delete_nonexistent_is_idempotent_contract, "delete_nonexistent", contract_delete_nonexistent_is_idempotent);
        pg_contract_test!(pg_store_passes_list_returns_all_inserted_ids_contract, "list_all", contract_list_returns_all_inserted_ids);
        pg_contract_test!(pg_store_passes_list_on_empty_store_returns_empty_contract, "list_empty", contract_list_on_empty_store_returns_empty);
        pg_contract_test!(pg_store_passes_insert_rejects_mismatched_dimension_contract, "dimension_mismatch", contract_insert_rejects_mismatched_dimension);
        pg_contract_test!(pg_store_passes_list_pagination_respects_offset_and_limit_contract, "pagination", contract_list_pagination_respects_offset_and_limit);

        #[test]
        #[ignore = "requires a real Postgres+pgvector instance reachable at MEMORIA_TEST_POSTGRES_URL"]
        fn records_survive_reopening_the_same_table() {
            let Some(url) = test_url() else {
                panic!("MEMORIA_TEST_POSTGRES_URL must be set to run this test");
            };
            let table = format!("memoria_pg_test_durability_{}", std::process::id());
            {
                let store = PgVectorStore::open(&url, 3, &table).expect("open should succeed");
                store.reset().expect("reset should succeed");
                store
                    .insert(VectorRecord::new("rec-1", vec![1.0, 2.0, 3.0], HashMap::from([("user_id".to_string(), "alice".to_string())])))
                    .expect("insert should succeed");
            }
            let reopened = PgVectorStore::open(&url, 3, &table).expect("reopen should succeed");
            let record = reopened.get("rec-1").expect("get should succeed").expect("record should survive reopening the store");
            assert_eq!(record.vector, vec![1.0, 2.0, 3.0]);
            assert_eq!(record.payload.get("user_id"), Some(&"alice".to_string()));
            reopened.reset().expect("cleanup reset should succeed");
        }

        #[test]
        #[ignore = "requires a real Postgres+pgvector instance reachable at MEMORIA_TEST_POSTGRES_URL"]
        fn open_rejects_an_unsafe_table_name() {
            let Some(url) = test_url() else {
                panic!("MEMORIA_TEST_POSTGRES_URL must be set to run this test");
            };
            let result = PgVectorStore::open(&url, 2, "records; DROP TABLE records;--");
            assert!(result.is_err(), "an unsafe table name must be rejected before it ever reaches a SQL statement");
        }
    }
}
