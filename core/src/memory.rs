use crate::embedding::{EmbeddingConfig, EmbeddingProvider};
use crate::entity::extract_entities;
use crate::llm::{describe_image, extract_facts, summarize_procedure, LlmConfig, LlmProvider, Message, Role};
use crate::vector_store::{VectorRecord, VectorStore, VectorStoreConfig};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

const ENTITY_RECORD_KIND_KEY: &str = "__memoria_record_kind";
const ENTITY_RECORD_KIND_VALUE: &str = "entity";
const ENTITY_TYPE_KEY: &str = "entity_type";
const ENTITY_TEXT_KEY: &str = "entity_text";
const LINKED_MEMORY_IDS_KEY: &str = "linked_memory_ids";
const ENTITY_BOOST_FRACTION: f32 = 0.15;
const ENTITY_MATCHES_PER_QUERY_ENTITY: usize = 5;

fn is_entity_record(payload: &HashMap<String, String>) -> bool {
    payload.get(ENTITY_RECORD_KIND_KEY).map(String::as_str) == Some(ENTITY_RECORD_KIND_VALUE)
}

static NEXT_RECORD_ID: AtomicU64 = AtomicU64::new(0);

fn next_record_id() -> String {
    let n = NEXT_RECORD_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let pid = std::process::id();
    format!("rec-{nanos}-{pid}-{n}")
}

fn next_entity_record_id() -> String {
    let n = NEXT_RECORD_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let pid = std::process::id();
    format!("entity-{nanos}-{pid}-{n}")
}

fn has_scope_id(scope: &HashMap<String, String>) -> bool {
    scope.keys().any(|k| k == "user_id" || k == "agent_id" || k == "run_id")
}

fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, u32::try_from(m).unwrap_or(1), u32::try_from(d).unwrap_or(1))
}

fn unix_seconds_to_ymd_string(seconds: u64) -> String {
    let days_since_epoch = i64::try_from(seconds / 86_400).unwrap_or(0);
    let (y, m, d) = civil_from_days(days_since_epoch);
    format!("{y:04}-{m:02}-{d:02}")
}

fn today_ymd_string() -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    unix_seconds_to_ymd_string(now)
}

fn is_expired(payload: &HashMap<String, String>, today: &str) -> bool {
    payload.get("expiration_date").is_some_and(|expiration_date| expiration_date.as_str() < today)
}

fn combine_with_keyword_scores(results: &mut [crate::vector_store::SearchResult], keyword_results: &[crate::vector_store::SearchResult], explain: bool) {
    let semantic_min = results.iter().map(|r| r.score).fold(f32::INFINITY, f32::min);
    let semantic_max = results.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);
    let keyword_min = keyword_results.iter().map(|r| r.score).fold(f32::INFINITY, f32::min);
    let keyword_max = keyword_results.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);
    let keyword_scores: HashMap<&str, f32> = keyword_results.iter().map(|r| (r.id.as_str(), r.score)).collect();

    for result in results.iter_mut() {
        let normalized_semantic = crate::reranker::normalize_distance(result.score, semantic_min, semantic_max);
        let raw_keyword = keyword_scores.get(result.id.as_str()).copied();
        let normalized_keyword = raw_keyword.map_or(1.0, |score| crate::reranker::normalize_distance(score, keyword_min, keyword_max));
        let combined = f32::midpoint(normalized_semantic, normalized_keyword);
        if explain {
            if let Some(details) = result.score_details.as_mut() {
                details.bm25_score = raw_keyword;
                details.raw_score = combined;
                details.final_score = combined;
            }
        }
        result.score = combined;
    }
    results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HistoryEvent {
    Added,
    Deleted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HistoryEntry {
    pub event: HistoryEvent,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EntitySummary {
    pub entity_type: String,
    pub entity_id: String,
    pub memory_count: usize,
}

const ENTITY_SCOPE_FIELDS: [&str; 3] = ["user_id", "agent_id", "run_id"];

/// A full provider triple, for a consumer that wants to validate configuration
/// up front rather than one provider at a time.
///
/// This type is deliberately *not* used by this workspace's own `cli`/`server`/
/// `mcp` binaries: each provider validates its own config inside `from_config`
/// before building anything (`ADR-09`, closed), and a binary may mix a
/// configured provider with a config-free local one, so there is no point in
/// the startup path where all three configs exist together. It is kept as
/// library surface for downstream consumers of `memoria-core`, not as this
/// project's own startup entry point.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryConfig {
    pub llm: LlmConfig,
    pub embedding: EmbeddingConfig,
    pub vector_store: VectorStoreConfig,
}

impl MemoryConfig {
    /// Validates all three sub-configs, short-circuiting on the first failure.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::Config` from whichever sub-config fails first, in
    /// LLM, embedding, vector-store order.
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
    reranker: Option<Box<dyn crate::reranker::Reranker + Send + Sync>>,
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
            reranker: None,
            history: Mutex::new(HashMap::new()),
            max_metadata_bytes: None,
            max_content_length: None,
            max_top_k: None,
        }
    }

    #[must_use]
    pub fn with_reranker(mut self, reranker: impl crate::reranker::Reranker + Send + Sync + 'static) -> Self {
        self.reranker = Some(Box::new(reranker));
        self
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
    #[allow(clippy::fn_params_excessive_bools)]
    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        scope: &HashMap<String, String>,
        threshold: Option<f32>,
        show_expired: bool,
        filters: Option<&crate::filter::FilterExpr>,
        rerank: bool,
        explain: bool,
    ) -> Result<Vec<crate::vector_store::SearchResult>, crate::CoreError> {
        if rerank && self.reranker.is_none() {
            return Err(crate::CoreError::Validation("rerank was requested but no reranker is configured".to_string()));
        }
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
        let mut results = self.vector_store.search(&vector, usize::MAX, scope, threshold)?;
        results.retain(|r| !is_entity_record(&r.payload));
        if !show_expired {
            let today = today_ymd_string();
            results.retain(|r| !is_expired(&r.payload, &today));
        }
        if let Some(filters) = filters {
            results.retain(|r| crate::filter::evaluate(filters, &r.payload));
        }
        if explain {
            for result in &mut results {
                result.score_details = Some(crate::vector_store::ScoreDetails {
                    semantic_score: result.score,
                    bm25_score: None,
                    entity_boost: None,
                    raw_score: result.score,
                    final_score: result.score,
                });
            }
        }
        if let Some(keyword_results) = self.vector_store.keyword_search(query, usize::MAX, scope)? {
            if !keyword_results.is_empty() {
                combine_with_keyword_scores(&mut results, &keyword_results, explain);
            }
        }
        self.apply_entity_boost(query, scope, &mut results, explain)?;
        results.truncate(top_k);
        if rerank {
            if let Some(reranker) = &self.reranker {
                results = reranker.rerank(query, results)?;
            }
        }
        Ok(results)
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

    fn list_ids(
        &self,
        scope: Option<&HashMap<String, String>>,
        offset: usize,
        limit: usize,
        show_expired: bool,
        filters: Option<&crate::filter::FilterExpr>,
    ) -> Result<Vec<String>, crate::CoreError> {
        let today = today_ymd_string();
        let all_ids = self.vector_store.list(0, usize::MAX)?;
        let surviving: Vec<String> = all_ids
            .into_iter()
            .filter(|id| {
                let Some(record) = self.vector_store.get(id).ok().flatten() else {
                    return true;
                };
                let scope_matches = scope.is_none_or(|s| s.iter().all(|(k, v)| record.payload.get(k) == Some(v)));
                !is_entity_record(&record.payload)
                    && scope_matches
                    && (show_expired || !is_expired(&record.payload, &today))
                    && filters.is_none_or(|f| crate::filter::evaluate(f, &record.payload))
            })
            .collect();
        Ok(surviving.into_iter().skip(offset).take(limit).collect())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn list(
        &self,
        scope: &HashMap<String, String>,
        offset: usize,
        limit: usize,
        show_expired: bool,
        filters: Option<&crate::filter::FilterExpr>,
    ) -> Result<Vec<String>, crate::CoreError> {
        if !has_scope_id(scope) {
            return Err(crate::CoreError::Validation("scope must contain user_id, agent_id, or run_id".to_string()));
        }
        self.list_ids(Some(scope), offset, limit, show_expired, filters)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn list_all(
        &self,
        offset: usize,
        limit: usize,
        show_expired: bool,
        filters: Option<&crate::filter::FilterExpr>,
    ) -> Result<Vec<String>, crate::CoreError> {
        self.list_ids(None, offset, limit, show_expired, filters)
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
    pub fn reset(&self, scope: &HashMap<String, String>, filters: Option<&crate::filter::FilterExpr>) -> Result<usize, crate::CoreError> {
        let ids = self.vector_store.list(0, usize::MAX)?;
        let mut deleted = 0;
        for id in ids {
            let Some(record) = self.vector_store.get(&id)? else {
                continue;
            };
            let scope_matches = scope.iter().all(|(k, v)| record.payload.get(k) == Some(v));
            let filter_matches = filters.is_none_or(|f| crate::filter::evaluate(f, &record.payload));
            if scope_matches && filter_matches {
                self.delete(&id)?;
                deleted += 1;
            }
        }
        Ok(deleted)
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

    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn history_snapshot(&self) -> HashMap<String, Vec<HistoryEntry>> {
        self.history.lock().expect("lock poisoned").clone()
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn load_history_snapshot(&self, snapshot: HashMap<String, Vec<HistoryEntry>>) {
        *self.history.lock().expect("lock poisoned") = snapshot;
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn list_entities(&self) -> Result<Vec<EntitySummary>, crate::CoreError> {
        let ids = self.list_all(0, usize::MAX, true, None)?;
        let mut counts: HashMap<(String, String), usize> = HashMap::new();
        for id in ids {
            let Some(record) = self.get(&id)? else {
                continue;
            };
            for field in ENTITY_SCOPE_FIELDS {
                if let Some(value) = record.payload.get(field) {
                    *counts.entry((field.to_string(), value.clone())).or_insert(0) += 1;
                }
            }
        }
        let mut entities: Vec<EntitySummary> = counts
            .into_iter()
            .map(|((entity_type, entity_id), memory_count)| EntitySummary { entity_type, entity_id, memory_count })
            .collect();
        entities.sort_by(|a, b| (&a.entity_type, &a.entity_id).cmp(&(&b.entity_type, &b.entity_id)));
        Ok(entities)
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
        let resolved_messages: Vec<Message> = messages
            .iter()
            .map(|message| {
                if message.images.is_empty() {
                    Ok(message.clone())
                } else {
                    describe_image(&self.llm, message.images.clone()).map(|description| Message::new(message.role, description))
                }
            })
            .collect::<Result<Vec<Message>, crate::llm::LlmError>>()?;
        let messages = resolved_messages.as_slice();
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
        if scope.get("memory_type").map(String::as_str) == Some("procedural_memory") && scope.contains_key("agent_id") {
            let summary = summarize_procedure(&self.llm, messages)?;
            if summary.is_empty() {
                return Err(crate::CoreError::Validation(
                    "the LLM returned no content for the procedural memory summary -- the model may have declined the request or returned an empty response".to_string(),
                ));
            }
            let vector = self.embedding.embed(&summary)?;
            let id = next_record_id();
            let mut payload = scope;
            payload.insert("content".to_string(), summary.clone());
            let record = VectorRecord::new(id.clone(), vector, payload);
            self.vector_store.insert(record)?;
            self.history
                .lock()
                .expect("lock poisoned")
                .entry(id.clone())
                .or_default()
                .push(HistoryEntry { event: HistoryEvent::Added, content: summary });
            return Ok(vec![id]);
        }
        let items: Vec<(String, Option<Role>)> = if infer {
            extract_facts(&self.llm, messages)?.into_iter().map(|fact| (fact, None)).collect()
        } else {
            messages.iter().map(|m| (m.content.clone(), Some(m.role))).collect()
        };
        let mut ids = Vec::new();
        // T1695: one batch call rather than one round trip per extracted fact. The
        // default trait implementation is still a per-text loop, so a provider
        // without a batch endpoint behaves exactly as before.
        let contents: Vec<&str> = items.iter().map(|(content, _)| content.as_str()).collect();
        let vectors = self.embedding.embed_batch(&contents)?;
        for ((content, role), vector) in items.iter().zip(vectors) {
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
            self.link_entities(&scope, &id, content)?;
            ids.push(id);
        }
        Ok(ids)
    }

    fn find_existing_entity_record(
        &self,
        scope: &HashMap<String, String>,
        entity_type: &str,
        entity_text: &str,
    ) -> Result<Option<VectorRecord>, crate::CoreError> {
        let ids = self.vector_store.list(0, usize::MAX)?;
        for id in ids {
            let Some(record) = self.vector_store.get(&id)? else {
                continue;
            };
            if is_entity_record(&record.payload)
                && scope.iter().all(|(k, v)| record.payload.get(k) == Some(v))
                && record.payload.get(ENTITY_TYPE_KEY).map(String::as_str) == Some(entity_type)
                && record.payload.get(ENTITY_TEXT_KEY).map(String::as_str) == Some(entity_text)
            {
                return Ok(Some(record));
            }
        }
        Ok(None)
    }

    fn link_entities(&self, scope: &HashMap<String, String>, memory_id: &str, content: &str) -> Result<(), crate::CoreError> {
        for (entity_type, entity_text) in extract_entities(content) {
            let entity_type = match entity_type {
                crate::entity::EntityType::Proper => "proper",
                crate::entity::EntityType::Quoted => "quoted",
            };
            if let Some(mut record) = self.find_existing_entity_record(scope, entity_type, &entity_text)? {
                let mut linked: Vec<String> =
                    record.payload.get(LINKED_MEMORY_IDS_KEY).map(|s| s.split(',').map(String::from).collect()).unwrap_or_default();
                if !linked.iter().any(|id| id == memory_id) {
                    linked.push(memory_id.to_string());
                    record.payload.insert(LINKED_MEMORY_IDS_KEY.to_string(), linked.join(","));
                    self.vector_store.update(record)?;
                }
            } else {
                let vector = self.embedding.embed(&entity_text)?;
                let mut payload = scope.clone();
                payload.insert(ENTITY_RECORD_KIND_KEY.to_string(), ENTITY_RECORD_KIND_VALUE.to_string());
                payload.insert(ENTITY_TYPE_KEY.to_string(), entity_type.to_string());
                payload.insert(ENTITY_TEXT_KEY.to_string(), entity_text.clone());
                payload.insert(LINKED_MEMORY_IDS_KEY.to_string(), memory_id.to_string());
                let record = VectorRecord::new(next_entity_record_id(), vector, payload);
                self.vector_store.insert(record)?;
            }
        }
        Ok(())
    }

    fn apply_entity_boost(
        &self,
        query: &str,
        scope: &HashMap<String, String>,
        results: &mut [crate::vector_store::SearchResult],
        explain: bool,
    ) -> Result<(), crate::CoreError> {
        if results.is_empty() {
            return Ok(());
        }
        let query_entities = extract_entities(query);
        if query_entities.is_empty() {
            return Ok(());
        }
        let mut boosted_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (entity_type, entity_text) in query_entities {
            let entity_type = match entity_type {
                crate::entity::EntityType::Proper => "proper",
                crate::entity::EntityType::Quoted => "quoted",
            };
            let vector = self.embedding.embed(&entity_text)?;
            let mut entity_filters = scope.clone();
            entity_filters.insert(ENTITY_RECORD_KIND_KEY.to_string(), ENTITY_RECORD_KIND_VALUE.to_string());
            entity_filters.insert(ENTITY_TYPE_KEY.to_string(), entity_type.to_string());
            let matches = self.vector_store.search(&vector, ENTITY_MATCHES_PER_QUERY_ENTITY, &entity_filters, None)?;
            for matched in matches {
                if let Some(linked) = matched.payload.get(LINKED_MEMORY_IDS_KEY) {
                    boosted_ids.extend(linked.split(',').map(String::from));
                }
            }
        }
        if boosted_ids.is_empty() {
            return Ok(());
        }
        let mut boosted_any = false;
        for result in results.iter_mut() {
            if boosted_ids.contains(&result.id) {
                let factor = 1.0 - ENTITY_BOOST_FRACTION;
                let new_score = result.score * factor;
                if explain {
                    if let Some(details) = result.score_details.as_mut() {
                        details.entity_boost = Some(factor);
                        details.final_score = new_score;
                    }
                }
                result.score = new_score;
                boosted_any = true;
            }
        }
        if boosted_any {
            results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EmbeddingConfig;
    use crate::filter::{FilterExpr, FilterOp, FilterValue};
    use crate::llm::{LlmConfig, Role};
    use crate::test_support::{EchoLlmProvider, FakeEmbeddingProvider, FakeLlmProvider, VecVectorStore};
    use crate::vector_store::{InMemoryVectorStore, VectorStoreConfig};

    fn scope() -> HashMap<String, String> {
        HashMap::from([("user_id".to_string(), "alice".to_string())])
    }

    #[test]
    fn unix_seconds_to_ymd_string_matches_real_reference_dates() {
        assert_eq!(unix_seconds_to_ymd_string(0), "1970-01-01");
        assert_eq!(unix_seconds_to_ymd_string(86400), "1970-01-02");
        assert_eq!(unix_seconds_to_ymd_string(1_609_459_200), "2021-01-01");
        assert_eq!(unix_seconds_to_ymd_string(1_700_000_000), "2023-11-14");
        assert_eq!(unix_seconds_to_ymd_string(1_787_500_000), "2026-08-23");
        assert_eq!(unix_seconds_to_ymd_string(946_684_800), "2000-01-01");
        assert_eq!(unix_seconds_to_ymd_string(1_735_689_599), "2024-12-31");
    }

    #[test]
    fn unix_seconds_to_ymd_string_handles_a_real_leap_day() {
        assert_eq!(unix_seconds_to_ymd_string(951_782_400), "2000-02-29", "the year 2000 is a leap year (divisible by 400)");
    }

    #[test]
    fn is_expired_true_for_a_past_date() {
        let payload = HashMap::from([("expiration_date".to_string(), "2020-01-01".to_string())]);
        assert!(is_expired(&payload, "2026-08-24"));
    }

    #[test]
    fn is_expired_false_for_a_future_date() {
        let payload = HashMap::from([("expiration_date".to_string(), "2030-01-01".to_string())]);
        assert!(!is_expired(&payload, "2026-08-24"));
    }

    #[test]
    fn is_expired_false_for_the_current_date() {
        let payload = HashMap::from([("expiration_date".to_string(), "2026-08-24".to_string())]);
        assert!(!is_expired(&payload, "2026-08-24"), "a record expiring today has not yet expired");
    }

    #[test]
    fn is_expired_false_when_no_expiration_date_is_set() {
        let payload = HashMap::new();
        assert!(!is_expired(&payload, "2026-08-24"));
    }

    #[test]
    fn test_search_excludes_an_expired_record_by_default() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let expired_scope =
            HashMap::from([("user_id".to_string(), "alice".to_string()), ("expiration_date".to_string(), "2000-01-01".to_string())]);
        memory.add(&[Message::new(Role::User, "An expired fact.")], expired_scope, false).expect("add should succeed");

        let results = memory.search("fact", 10, &scope(), None, false, None, false, false).expect("search should succeed");
        assert!(
            !results.iter().any(|r| r.payload.get("content").map(String::as_str) == Some("An expired fact.")),
            "an expired record must not appear in search results by default"
        );
    }

    #[test]
    fn test_search_includes_an_expired_record_with_show_expired_true() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let expired_scope =
            HashMap::from([("user_id".to_string(), "alice".to_string()), ("expiration_date".to_string(), "2000-01-01".to_string())]);
        memory.add(&[Message::new(Role::User, "An expired fact.")], expired_scope, false).expect("add should succeed");

        let results = memory.search("fact", 10, &scope(), None, true, None, false, false).expect("search should succeed");
        assert!(
            results.iter().any(|r| r.payload.get("content").map(String::as_str) == Some("An expired fact.")),
            "show_expired=true must still surface an expired record"
        );
    }

    #[test]
    fn test_search_includes_a_non_expired_future_dated_record_by_default() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let future_scope =
            HashMap::from([("user_id".to_string(), "alice".to_string()), ("expiration_date".to_string(), "2099-01-01".to_string())]);
        memory.add(&[Message::new(Role::User, "A future fact.")], future_scope, false).expect("add should succeed");

        let results = memory.search("fact", 10, &scope(), None, false, None, false, false).expect("search should succeed");
        assert!(
            results.iter().any(|r| r.payload.get("content").map(String::as_str) == Some("A future fact.")),
            "a record expiring in the future must still appear with show_expired=false"
        );
    }

    #[test]
    fn test_search_excluding_expired_records_does_not_shrink_top_k_below_available_valid_results() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        for i in 1..=3 {
            let expired_scope = HashMap::from([
                ("user_id".to_string(), "alice".to_string()),
                ("expiration_date".to_string(), "2000-01-01".to_string()),
            ]);
            memory.add(&[Message::new(Role::User, format!("Expired fact {i}."))], expired_scope, false).expect("add should succeed");
        }
        for i in 1..=3 {
            let s = HashMap::from([("user_id".to_string(), "alice".to_string())]);
            memory.add(&[Message::new(Role::User, format!("Valid fact {i}."))], s, false).expect("add should succeed");
        }

        let results = memory.search("fact", 3, &scope(), None, false, None, false, false).expect("search should succeed");
        assert_eq!(results.len(), 3, "top_k=3 must return 3 real results, not fewer because expired ones were counted against the limit");
        assert!(results.iter().all(|r| r.payload.get("content").is_some_and(|c| c.starts_with("Valid fact"))));
    }

    #[test]
    fn test_list_excludes_an_expired_record_by_default() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let expired_scope =
            HashMap::from([("user_id".to_string(), "alice".to_string()), ("expiration_date".to_string(), "2000-01-01".to_string())]);
        let ids = memory.add(&[Message::new(Role::User, "An expired fact.")], expired_scope, false).expect("add should succeed");
        let expired_id = ids.first().expect("expected an id").clone();

        let listed = memory.list(&scope(), 0, 100, false, None).expect("list should succeed");
        assert!(!listed.contains(&expired_id), "an expired record must not appear in list results by default");
    }

    #[test]
    fn test_list_includes_an_expired_record_with_show_expired_true() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let expired_scope =
            HashMap::from([("user_id".to_string(), "alice".to_string()), ("expiration_date".to_string(), "2000-01-01".to_string())]);
        let ids = memory.add(&[Message::new(Role::User, "An expired fact.")], expired_scope, false).expect("add should succeed");
        let expired_id = ids.first().expect("expected an id").clone();

        let listed = memory.list(&scope(), 0, 100, true, None).expect("list should succeed");
        assert!(listed.contains(&expired_id), "show_expired=true must still surface an expired record in list");
    }

    #[test]
    fn test_search_applies_a_real_filter_on_top_of_scope() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let engineering_scope =
            HashMap::from([("user_id".to_string(), "alice".to_string()), ("category".to_string(), "engineering".to_string())]);
        let sales_scope =
            HashMap::from([("user_id".to_string(), "alice".to_string()), ("category".to_string(), "sales".to_string())]);
        memory.add(&[Message::new(Role::User, "An engineering fact.")], engineering_scope, false).expect("add should succeed");
        memory.add(&[Message::new(Role::User, "A sales fact.")], sales_scope, false).expect("add should succeed");

        let filter = FilterExpr::Field("category".to_string(), FilterOp::Eq(FilterValue::String("engineering".to_string())));
        let results = memory.search("fact", 10, &scope(), None, true, Some(&filter), false, false).expect("search should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].payload.get("content"), Some(&"An engineering fact.".to_string()));
    }

    #[test]
    fn test_search_filters_do_not_shrink_top_k_below_available_matching_results() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        for i in 1..=3 {
            let s = HashMap::from([("user_id".to_string(), "alice".to_string()), ("category".to_string(), "sales".to_string())]);
            memory.add(&[Message::new(Role::User, format!("Sales fact {i}."))], s, false).expect("add should succeed");
        }
        for i in 1..=3 {
            let s = HashMap::from([("user_id".to_string(), "alice".to_string()), ("category".to_string(), "engineering".to_string())]);
            memory.add(&[Message::new(Role::User, format!("Engineering fact {i}."))], s, false).expect("add should succeed");
        }

        let filter = FilterExpr::Field("category".to_string(), FilterOp::Eq(FilterValue::String("engineering".to_string())));
        let results = memory.search("fact", 3, &scope(), None, true, Some(&filter), false, false).expect("search should succeed");
        assert_eq!(results.len(), 3, "top_k=3 must return 3 real matches, not fewer because non-matching records were counted against the limit");
        assert!(results.iter().all(|r| r.payload.get("content").is_some_and(|c| c.starts_with("Engineering fact"))));
    }

    #[test]
    fn test_search_with_no_filter_is_unaffected() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);
        memory.add(&[Message::new(Role::User, "A fact.")], scope(), false).expect("add should succeed");

        let results = memory.search("fact", 10, &scope(), None, true, None, false, false).expect("search should succeed");
        assert_eq!(results.len(), 1, "an absent filter must not change existing search behavior");
    }

    #[test]
    fn test_list_applies_a_real_filter() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let engineering_scope =
            HashMap::from([("user_id".to_string(), "alice".to_string()), ("category".to_string(), "engineering".to_string())]);
        let sales_scope =
            HashMap::from([("user_id".to_string(), "alice".to_string()), ("category".to_string(), "sales".to_string())]);
        let engineering_ids =
            memory.add(&[Message::new(Role::User, "An engineering fact.")], engineering_scope, false).expect("add should succeed");
        let sales_ids = memory.add(&[Message::new(Role::User, "A sales fact.")], sales_scope, false).expect("add should succeed");

        let filter = FilterExpr::Field("category".to_string(), FilterOp::Eq(FilterValue::String("engineering".to_string())));
        let listed = memory.list(&scope(), 0, 100, true, Some(&filter)).expect("list should succeed");
        assert!(listed.contains(engineering_ids.first().expect("expected an id")));
        assert!(!listed.contains(sales_ids.first().expect("expected an id")));
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
    fn test_add_with_procedural_memory_type_and_agent_id_creates_one_summarized_record() {
        let llm = FakeLlmProvider::with_response("1. Called the API. Result: 200 OK.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Call the API."), Message::new(Role::Assistant, "Called it, got 200 OK.")];
        let mut scope = HashMap::new();
        scope.insert("agent_id".to_string(), "agent-1".to_string());
        scope.insert("memory_type".to_string(), "procedural_memory".to_string());

        let ids = memory.add(&messages, scope, true).expect("add should succeed");
        assert_eq!(ids.len(), 1, "procedural memory must always produce exactly one summarized record, not one per message or fact");
        let record = memory.vector_store.get(&ids[0]).unwrap().unwrap();
        assert_eq!(record.payload.get("content"), Some(&"1. Called the API. Result: 200 OK.".to_string()));
        assert_eq!(record.payload.get("memory_type"), Some(&"procedural_memory".to_string()));
        assert_eq!(record.payload.get("agent_id"), Some(&"agent-1".to_string()));
    }

    #[test]
    fn test_add_with_procedural_memory_rejects_an_empty_llm_summary() {
        let llm = FakeLlmProvider::with_response("");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Call the API.")];
        let mut scope = HashMap::new();
        scope.insert("agent_id".to_string(), "agent-1".to_string());
        scope.insert("memory_type".to_string(), "procedural_memory".to_string());

        let result = memory.add(&messages, scope, true);
        assert!(
            matches!(result, Err(crate::CoreError::Validation(_))),
            "an empty LLM summary must be a real, clear error, not a silently-created empty record"
        );
    }

    #[test]
    fn test_add_with_procedural_memory_type_but_no_agent_id_falls_back_to_normal_extraction() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let mut scope = HashMap::new();
        scope.insert("user_id".to_string(), "alice".to_string());
        scope.insert("memory_type".to_string(), "procedural_memory".to_string());

        let ids = memory.add(&messages, scope, true).expect("add should succeed");
        let record = memory.vector_store.get(&ids[0]).unwrap().unwrap();
        assert_eq!(
            record.payload.get("content"),
            Some(&"Alice is an engineer.".to_string()),
            "without agent_id, memory_type=procedural_memory must not trigger summarization"
        );
    }

    #[test]
    fn test_add_with_an_image_message_describes_it_via_the_llm_before_storage() {
        let llm = FakeLlmProvider::with_response("A photo of a red bicycle.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::with_images(Role::User, "", vec!["base64imagedata".to_string()])];
        let ids = memory.add(&messages, scope(), false).expect("add should succeed");
        assert_eq!(ids.len(), 1);
        let record = memory.vector_store.get(&ids[0]).unwrap().unwrap();
        assert_eq!(
            record.payload.get("content"),
            Some(&"A photo of a red bicycle.".to_string()),
            "an image-bearing message's content must be replaced by the LLM's real description before storage"
        );
    }

    #[test]
    fn test_add_creates_a_linked_entity_record_for_a_real_proper_noun() {
        let llm = FakeLlmProvider::with_response("unused");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Bob Smith works at the office.")];
        let ids = memory.add(&messages, scope(), false).expect("add should succeed");
        let memory_id = &ids[0];

        let all_ids = memory.vector_store.list(0, usize::MAX).expect("list should succeed");
        let entity_record = all_ids
            .iter()
            .filter_map(|id| memory.vector_store.get(id).ok().flatten())
            .find(|r| is_entity_record(&r.payload))
            .expect("a real entity record should have been created for the proper noun");
        assert_eq!(entity_record.payload.get(ENTITY_TYPE_KEY), Some(&"proper".to_string()));
        assert_eq!(entity_record.payload.get(ENTITY_TEXT_KEY), Some(&"Bob Smith".to_string()));
        assert_eq!(entity_record.payload.get(LINKED_MEMORY_IDS_KEY), Some(memory_id));
        assert_eq!(entity_record.payload.get("user_id"), Some(&"alice".to_string()), "the entity record must carry the same scope");
    }

    #[test]
    fn test_add_twice_with_the_same_entity_links_both_memory_ids_to_one_entity_record() {
        let llm = FakeLlmProvider::with_response("unused");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let first = memory
            .add(&[Message::new(Role::User, "Bob Smith works at the office.")], scope(), false)
            .expect("add should succeed");
        let second = memory
            .add(&[Message::new(Role::User, "Bob Smith left early today.")], scope(), false)
            .expect("add should succeed");

        let all_ids = memory.vector_store.list(0, usize::MAX).expect("list should succeed");
        let entity_records: Vec<VectorRecord> =
            all_ids.iter().filter_map(|id| memory.vector_store.get(id).ok().flatten()).filter(|r| is_entity_record(&r.payload)).collect();
        assert_eq!(entity_records.len(), 1, "the same entity mentioned twice must dedup into exactly one entity record");
        let linked: Vec<&str> = entity_records[0].payload.get(LINKED_MEMORY_IDS_KEY).unwrap().split(',').collect();
        assert!(linked.contains(&first[0].as_str()));
        assert!(linked.contains(&second[0].as_str()));
    }

    #[test]
    fn test_entity_records_never_appear_in_search_results() {
        let llm = FakeLlmProvider::with_response("unused");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        memory.add(&[Message::new(Role::User, "Bob Smith works at the office.")], scope(), false).expect("add should succeed");
        let results = memory.search("Bob Smith", 100, &scope(), None, true, None, false, false).expect("search should succeed");
        assert!(
            results.iter().all(|r| !is_entity_record(&r.payload)),
            "an entity record must never be returned as a real search result"
        );
    }

    #[test]
    fn test_entity_records_never_appear_in_list_results() {
        let llm = FakeLlmProvider::with_response("unused");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Bob Smith works at the office.")], scope(), false).expect("add should succeed");
        let listed = memory.list(&scope(), 0, 100, true, None).expect("list should succeed");
        assert_eq!(listed, ids, "list must return only the real memory record, never the entity record created alongside it");
    }

    #[test]
    fn test_entity_records_are_not_counted_by_list_entities() {
        let llm = FakeLlmProvider::with_response("unused");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        memory.add(&[Message::new(Role::User, "Bob Smith works at the office.")], scope(), false).expect("add should succeed");
        let entities = memory.list_entities().expect("list_entities should succeed");
        let alice = entities.iter().find(|e| e.entity_id == "alice").expect("alice entity");
        assert_eq!(alice.memory_count, 1, "the entity record's own scope fields must not inflate list_entities' real memory count");
    }

    #[test]
    fn test_apply_entity_boost_lowers_the_score_of_a_record_linked_to_a_matching_entity() {
        let llm = FakeLlmProvider::with_response("unused");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let bob_ids = memory.add(&[Message::new(Role::User, "Bob Smith works at the office.")], scope(), false).expect("add should succeed");
        let unrelated_ids = memory.add(&[Message::new(Role::User, "The weather was nice today.")], scope(), false).expect("add should succeed");

        let mut results = vec![
            crate::vector_store::SearchResult { id: unrelated_ids[0].clone(), score: 10.0, payload: HashMap::new(), score_details: None },
            crate::vector_store::SearchResult { id: bob_ids[0].clone(), score: 10.5, payload: HashMap::new(), score_details: None },
        ];
        memory.apply_entity_boost("Tell me about Bob Smith", &scope(), &mut results, false).expect("boost should succeed");

        let bob_result = results.iter().find(|r| r.id == bob_ids[0]).expect("bob's record");
        assert!(bob_result.score < 10.5, "a record linked to a matching query entity must have its score lowered (lower is better), got {}", bob_result.score);
        assert_eq!(results[0].id, bob_ids[0], "for two near-tied candidates, the boosted record must now rank first");
    }

    #[test]
    fn test_apply_entity_boost_explain_true_records_the_real_boost_factor() {
        let llm = FakeLlmProvider::with_response("unused");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let bob_ids = memory.add(&[Message::new(Role::User, "Bob Smith works at the office.")], scope(), false).expect("add should succeed");
        let unrelated_ids = memory.add(&[Message::new(Role::User, "The weather was nice today.")], scope(), false).expect("add should succeed");

        let mut results = vec![
            crate::vector_store::SearchResult {
                id: unrelated_ids[0].clone(),
                score: 10.0,
                payload: HashMap::new(),
                score_details: Some(crate::vector_store::ScoreDetails { semantic_score: 10.0, bm25_score: None, entity_boost: None, raw_score: 10.0, final_score: 10.0 }),
            },
            crate::vector_store::SearchResult {
                id: bob_ids[0].clone(),
                score: 10.5,
                payload: HashMap::new(),
                score_details: Some(crate::vector_store::ScoreDetails { semantic_score: 10.5, bm25_score: None, entity_boost: None, raw_score: 10.5, final_score: 10.5 }),
            },
        ];
        memory.apply_entity_boost("Tell me about Bob Smith", &scope(), &mut results, true).expect("boost should succeed");

        let bob_result = results.iter().find(|r| r.id == bob_ids[0]).expect("bob's record");
        let bob_details = bob_result.score_details.as_ref().expect("score_details must still be present after boosting");
        assert_eq!(bob_details.entity_boost, Some(1.0 - ENTITY_BOOST_FRACTION), "entity_boost must record the real multiplicative factor applied");
        assert!((bob_details.final_score - bob_result.score).abs() < 1e-6, "final_score must equal the real post-boost score");
        assert!((bob_details.raw_score - 10.5).abs() < 1e-6, "raw_score must stay at the pre-boost value");

        let unrelated_result = results.iter().find(|r| r.id == unrelated_ids[0]).expect("unrelated record");
        assert_eq!(unrelated_result.score_details.as_ref().expect("score_details must still be present").entity_boost, None, "an unboosted result must record no entity_boost");
    }

    #[test]
    fn test_apply_entity_boost_is_a_no_op_when_the_query_has_no_entities() {
        let llm = FakeLlmProvider::with_response("unused");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);
        memory.add(&[Message::new(Role::User, "Bob Smith works at the office.")], scope(), false).expect("add should succeed");

        let mut results = vec![crate::vector_store::SearchResult { id: "some-id".to_string(), score: 5.0, payload: HashMap::new(), score_details: None }];
        memory.apply_entity_boost("just a plain lowercase query", &scope(), &mut results, false).expect("boost should succeed");
        assert!((results[0].score - 5.0).abs() < 1e-6, "a query with no extractable entities must leave scores unchanged, got {}", results[0].score);
    }

    #[test]
    fn test_apply_entity_boost_is_a_no_op_when_no_stored_entity_matches() {
        let llm = FakeLlmProvider::with_response("unused");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let mut results = vec![crate::vector_store::SearchResult { id: "some-id".to_string(), score: 5.0, payload: HashMap::new(), score_details: None }];
        memory.apply_entity_boost("Nobody Special mentioned here", &scope(), &mut results, false).expect("boost should succeed");
        assert!((results[0].score - 5.0).abs() < 1e-6, "a query entity with no matching stored entity record must leave scores unchanged, got {}", results[0].score);
    }

    #[test]
    fn test_add_with_a_text_only_message_never_calls_describe_image() {
        let llm = FakeLlmProvider::with_response("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let messages = [Message::new(Role::User, "Alice is an engineer.")];
        let ids = memory.add(&messages, scope(), false).expect("add should succeed");
        let record = memory.vector_store.get(&ids[0]).unwrap().unwrap();
        assert_eq!(
            record.payload.get("content"),
            Some(&"Alice is an engineer.".to_string()),
            "a message with no images must be stored as-is, not routed through image description"
        );
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
        let results = memory.search(query, 10, &scope(), None, true, None, false, false).expect("search should succeed");
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
        let results = memory.search(query, 3, &scope(), None, true, None, false, false).expect("search should succeed");
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

        let results = memory.search("engineer", 10, &alice_scope, None, true, None, false, false).expect("search should succeed");
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

        let results = memory.search("designer", 10, &bob_scope, None, true, None, false, false).expect("search should succeed");
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

        let result = memory.search("anything", 10, &scope(), None, true, None, false, false);
        assert!(matches!(result, Ok(vec) if vec.is_empty()));
    }

    #[test]
    fn test_search_rejects_zero_top_k() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 0, &scope(), None, true, None, false, false);
        assert!(matches!(result, Err(crate::CoreError::Validation(_))));
    }

    #[test]
    fn test_search_rejects_a_negative_threshold() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 10, &scope(), Some(-0.1), true, None, false, false);
        assert!(matches!(result, Err(crate::CoreError::Validation(_))));
    }

    #[test]
    fn test_search_rejects_rerank_when_no_reranker_is_configured() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 10, &scope(), None, true, None, true, false);
        assert!(matches!(result, Err(crate::CoreError::Validation(_))), "rerank=true with no configured reranker must be a validation error, not a silent no-op");
    }

    #[test]
    fn test_search_with_rerank_true_invokes_the_configured_reranker() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store).with_reranker(crate::test_support::FakeRerankerProvider::failing());

        let result = memory.search("anything", 10, &scope(), None, true, None, true, false);
        assert!(matches!(result, Err(crate::CoreError::Provider { .. })), "rerank=true must actually invoke the configured reranker, got: {result:?}");
    }

    #[test]
    fn test_search_with_rerank_false_never_invokes_the_configured_reranker() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store).with_reranker(crate::test_support::FakeRerankerProvider::failing());
        memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");

        let result = memory.search("engineer", 10, &scope(), None, true, None, false, false);
        assert!(result.is_ok(), "rerank=false must not invoke a configured reranker even if it would fail: {result:?}");
    }

    fn search_result_with_content(id: &str, distance: f32, content: &str, record_scope: &HashMap<String, String>) -> crate::vector_store::SearchResult {
        let mut payload = record_scope.clone();
        payload.insert("content".to_string(), content.to_string());
        crate::vector_store::SearchResult { id: id.to_string(), score: distance, payload, score_details: None }
    }

    #[test]
    fn combine_with_keyword_scores_promotes_a_semantically_middling_result_with_a_much_stronger_keyword_match() {
        let mut results = vec![
            search_result_with_content("best_semantic", 100.0, "a", &scope()),
            search_result_with_content("best_keyword", 150.0, "b", &scope()),
            search_result_with_content("worst_both", 200.0, "c", &scope()),
        ];
        let keyword_results = vec![
            search_result_with_content("best_semantic", -1.0, "a", &scope()),
            search_result_with_content("best_keyword", -10.0, "b", &scope()),
            search_result_with_content("worst_both", -3.0, "c", &scope()),
        ];
        combine_with_keyword_scores(&mut results, &keyword_results, false);
        assert_eq!(results[0].id, "best_keyword", "a strong enough keyword match should be able to promote a semantically-middling result above the semantically-best one, got: {results:?}");
    }

    #[test]
    fn combine_with_keyword_scores_treats_an_unmatched_candidate_as_the_worst_case() {
        let mut results = vec![
            search_result_with_content("matched", 150.0, "matched", &scope()),
            search_result_with_content("unmatched", 150.0, "unmatched", &scope()),
        ];
        let keyword_results = vec![search_result_with_content("matched", -3.0, "matched", &scope())];
        combine_with_keyword_scores(&mut results, &keyword_results, false);
        assert_eq!(results[0].id, "matched", "a real keyword match should outrank a candidate with no keyword match at all, given equal semantic scores");
    }

    #[test]
    fn combine_with_keyword_scores_explain_false_produces_identical_scores_to_before_score_details_existed() {
        let mut with_explain = vec![
            search_result_with_content("a", 100.0, "a", &scope()),
            search_result_with_content("b", 150.0, "b", &scope()),
        ];
        let mut without_explain = with_explain.clone();
        let keyword_results = vec![search_result_with_content("a", -1.0, "a", &scope()), search_result_with_content("b", -10.0, "b", &scope())];
        combine_with_keyword_scores(&mut with_explain, &keyword_results, true);
        combine_with_keyword_scores(&mut without_explain, &keyword_results, false);
        let with_scores: Vec<f32> = with_explain.iter().map(|r| r.score).collect();
        let without_scores: Vec<f32> = without_explain.iter().map(|r| r.score).collect();
        assert_eq!(with_scores, without_scores, "explain must never change the actual combined score or sort order");
        let with_ids: Vec<&str> = with_explain.iter().map(|r| r.id.as_str()).collect();
        let without_ids: Vec<&str> = without_explain.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(with_ids, without_ids, "explain must never change the actual sort order");
        assert!(without_explain.iter().all(|r| r.score_details.is_none()), "explain=false must never populate score_details");
    }

    #[test]
    fn combine_with_keyword_scores_explain_true_records_the_real_component_scores() {
        let mut results = vec![search_result_with_content("matched", 150.0, "matched", &scope()), search_result_with_content("unmatched", 150.0, "unmatched", &scope())];
        for result in &mut results {
            result.score_details = Some(crate::vector_store::ScoreDetails {
                semantic_score: result.score,
                bm25_score: None,
                entity_boost: None,
                raw_score: result.score,
                final_score: result.score,
            });
        }
        let keyword_results = vec![search_result_with_content("matched", -3.0, "matched", &scope())];
        combine_with_keyword_scores(&mut results, &keyword_results, true);
        let matched = results.iter().find(|r| r.id == "matched").expect("matched result");
        let matched_details = matched.score_details.as_ref().expect("explain=true must populate score_details");
        assert!((matched_details.semantic_score - 150.0).abs() < 1e-6, "semantic_score must be the pre-normalization original, got {}", matched_details.semantic_score);
        assert_eq!(matched_details.bm25_score, Some(-3.0), "bm25_score must be the real raw keyword score for a matched id");
        assert!((matched_details.raw_score - matched.score).abs() < 1e-6, "raw_score must track the combined score at this stage");
        assert!((matched_details.final_score - matched.score).abs() < 1e-6, "final_score must track the combined score at this stage");
        let unmatched = results.iter().find(|r| r.id == "unmatched").expect("unmatched result");
        assert_eq!(unmatched.score_details.as_ref().expect("explain=true must populate score_details").bm25_score, None, "bm25_score must be None for a candidate with no keyword match");
    }

    #[test]
    fn test_search_combines_keyword_results_when_the_vector_store_supports_it() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let keyword_results = vec![search_result_with_content("b", -5.0, "rust programming", &scope())];
        let store = crate::test_support::FixedKeywordSearchVectorStore::new(keyword_results);
        let memory = Memory::new(llm, embedding, store);
        memory.add(&[Message::new(Role::User, "irrelevant content here")], scope(), false).expect("add should succeed");
        memory.add(&[Message::new(Role::User, "rust programming")], scope(), false).expect("add should succeed");

        let results = memory.search("query", 10, &scope(), None, true, None, false, false).expect("search should succeed");
        let rust_result = results.iter().find(|r| r.payload.get("content") == Some(&"rust programming".to_string())).expect("expected the rust programming record");
        let other_result = results.iter().find(|r| r.payload.get("content") == Some(&"irrelevant content here".to_string())).expect("expected the irrelevant record");
        assert!(rust_result.score < other_result.score, "the record with a real keyword match should have a better (lower) combined score, got: {results:?}");
    }

    #[test]
    fn test_search_is_unaffected_when_keyword_search_returns_an_empty_result() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = crate::test_support::FixedKeywordSearchVectorStore::new(Vec::new());
        let memory = Memory::new(llm, embedding, store);
        memory.add(&[Message::new(Role::User, "some content")], scope(), false).expect("add should succeed");

        let results = memory.search("query", 10, &scope(), None, true, None, false, false).expect("search should succeed");
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 1.0, "an empty keyword_search result must not trigger normalization/combination at all; expected the raw, unnormalized semantic distance, got {}", results[0].score);
    }

    #[test]
    fn test_search_explain_false_never_populates_score_details() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let keyword_results = vec![search_result_with_content("b", -5.0, "rust programming", &scope())];
        let store = crate::test_support::FixedKeywordSearchVectorStore::new(keyword_results);
        let memory = Memory::new(llm, embedding, store);
        memory.add(&[Message::new(Role::User, "rust programming")], scope(), false).expect("add should succeed");

        let results = memory.search("query", 10, &scope(), None, true, None, false, false).expect("search should succeed");
        assert!(results.iter().all(|r| r.score_details.is_none()), "explain=false must never populate score_details, even when keyword combination runs");
    }

    #[test]
    fn test_search_explain_true_with_a_keyword_match_records_the_real_breakdown() {
        let embedding = FakeEmbeddingProvider::new();
        let keyword_results = vec![search_result_with_content("rec-1", -5.0, "rust programming", &scope())];
        let store = crate::test_support::FixedKeywordSearchVectorStore::new(keyword_results);
        let memory = Memory::new(FakeLlmProvider::new(), embedding.clone(), store);
        let vector = embedding.embed("rust programming").expect("embed should succeed");
        let mut payload = scope();
        payload.insert("content".to_string(), "rust programming".to_string());
        memory.vector_store.insert(VectorRecord::new("rec-1".to_string(), vector, payload)).expect("insert should succeed");

        let results = memory.search("query", 10, &scope(), None, true, None, false, true).expect("search should succeed");
        let result = &results[0];
        let details = result.score_details.as_ref().expect("explain=true must populate score_details");
        assert_eq!(details.bm25_score, Some(-5.0), "a real keyword match by id must produce the real raw bm25_score");
        assert!((details.final_score - result.score).abs() < 1e-6, "final_score must equal the real returned score");
    }

    #[test]
    fn test_search_explain_true_with_neither_keyword_nor_entity_match_reports_an_unmodified_breakdown() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);
        memory.add(&[Message::new(Role::User, "plain lowercase content")], scope(), false).expect("add should succeed");

        let results = memory.search("plain lowercase content", 10, &scope(), None, true, None, false, true).expect("search should succeed");
        let details = results[0].score_details.as_ref().expect("explain=true must populate score_details even with no keyword/entity signal");
        assert_eq!(details.bm25_score, None, "no keyword_search support means bm25_score must be None");
        assert_eq!(details.entity_boost, None, "no matching query entity means entity_boost must be None");
        assert!((details.semantic_score - details.raw_score).abs() < 1e-6, "with no combination stages, semantic_score, raw_score, and final_score must all be equal");
        assert!((details.raw_score - details.final_score).abs() < 1e-6, "with no combination stages, semantic_score, raw_score, and final_score must all be equal");
    }

    #[test]
    fn test_search_explain_true_with_an_entity_boost_records_the_real_factor() {
        let llm = FakeLlmProvider::with_response("unused");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);
        memory.add(&[Message::new(Role::User, "Bob Smith works at the office.")], scope(), false).expect("add should succeed");
        memory.add(&[Message::new(Role::User, "The weather was nice today.")], scope(), false).expect("add should succeed");

        let results = memory.search("Tell me about Bob Smith", 10, &scope(), None, true, None, false, true).expect("search should succeed");
        let boosted = results.iter().find(|r| r.payload.get("content") == Some(&"Bob Smith works at the office.".to_string())).expect("bob's record");
        let details = boosted.score_details.as_ref().expect("explain=true must populate score_details");
        assert_eq!(details.entity_boost, Some(1.0 - ENTITY_BOOST_FRACTION), "a real entity-linked result must record the real boost factor");
    }

    #[test]
    fn test_search_accepts_a_zero_threshold() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 10, &scope(), Some(0.0), true, None, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_search_accepts_positive_top_k() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 1, &scope(), None, true, None, false, false);
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

        memory.reset(&scope(), None).expect("reset should succeed");

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

        memory.reset(&alice_scope, None).expect("reset should succeed");

        for id in &alice_ids {
            assert_eq!(memory.vector_store.get(id).expect("get should succeed"), None, "alice's records should be gone");
        }
        for id in &bob_ids {
            assert!(memory.vector_store.get(id).expect("get should succeed").is_some(), "bob's records should be untouched");
        }
    }

    #[test]
    fn test_reset_with_a_real_filter_deletes_only_matching_records() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let engineering_scope =
            HashMap::from([("user_id".to_string(), "alice".to_string()), ("category".to_string(), "engineering".to_string())]);
        let sales_scope =
            HashMap::from([("user_id".to_string(), "alice".to_string()), ("category".to_string(), "sales".to_string())]);
        let engineering_ids =
            memory.add(&[Message::new(Role::User, "An engineering fact.")], engineering_scope, false).expect("add should succeed");
        let sales_ids = memory.add(&[Message::new(Role::User, "A sales fact.")], sales_scope, false).expect("add should succeed");

        let filter = FilterExpr::Field("category".to_string(), FilterOp::Eq(FilterValue::String("engineering".to_string())));
        let deleted = memory.reset(&scope(), Some(&filter)).expect("reset should succeed");
        assert_eq!(deleted, 1);

        for id in &engineering_ids {
            assert_eq!(memory.vector_store.get(id).expect("get should succeed"), None, "the matching record must be gone");
        }
        for id in &sales_ids {
            assert!(memory.vector_store.get(id).expect("get should succeed").is_some(), "the non-matching record must survive");
        }
    }

    #[test]
    fn test_reset_with_an_empty_scope_and_no_filter_deletes_everything() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let alice_scope = HashMap::from([("user_id".to_string(), "alice".to_string())]);
        let bob_scope = HashMap::from([("user_id".to_string(), "bob".to_string())]);
        let alice_ids = memory.add(&[Message::new(Role::User, "Alice's fact.")], alice_scope, false).expect("add should succeed");
        let bob_ids = memory.add(&[Message::new(Role::User, "Bob's fact.")], bob_scope, false).expect("add should succeed");

        let deleted = memory.reset(&HashMap::new(), None).expect("reset should succeed");
        assert_eq!(deleted, 2);
        for id in alice_ids.iter().chain(bob_ids.iter()) {
            assert_eq!(memory.vector_store.get(id).expect("get should succeed"), None, "an empty scope with no filter must wipe everything");
        }
    }

    #[test]
    fn test_reset_returns_the_real_count_of_deleted_records() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);
        for i in 1..=3 {
            let s = HashMap::from([("user_id".to_string(), "alice".to_string())]);
            memory.add(&[Message::new(Role::User, format!("Fact {i}."))], s, false).expect("add should succeed");
        }
        let deleted = memory.reset(&scope(), None).expect("reset should succeed");
        assert_eq!(deleted, 3);
    }

    #[test]
    fn test_reset_records_a_deleted_history_entry_for_each_record_it_removes() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);
        let ids = memory.add(&[Message::new(Role::User, "A fact worth remembering.")], scope(), false).expect("add should succeed");

        memory.reset(&scope(), None).expect("reset should succeed");

        let entries = memory.history(&ids[0], 0, 10).expect("history should succeed");
        assert_eq!(entries.len(), 2, "reset must record a Deleted entry alongside the earlier Added one");
        assert_eq!(entries[0].event, HistoryEvent::Added);
        assert_eq!(entries[1].event, HistoryEvent::Deleted);
    }

    #[test]
    fn test_reset_with_no_matching_scope_is_noop() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");

        let other_scope = HashMap::from([("user_id".to_string(), "nobody-here".to_string())]);
        memory.reset(&other_scope, None).expect("reset with no matches should not error");

        for id in &ids {
            assert!(memory.vector_store.get(id).expect("get should succeed").is_some(), "unrelated scope's records should be untouched");
        }
    }

    #[test]
    fn test_list_entities_groups_real_records_by_scope_field() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        memory
            .add(
                &[Message::new(Role::User, "Alice is an engineer.")],
                HashMap::from([("user_id".to_string(), "alice".to_string())]),
                false,
            )
            .expect("add should succeed");
        memory
            .add(
                &[Message::new(Role::User, "Alice likes tea.")],
                HashMap::from([("user_id".to_string(), "alice".to_string())]),
                false,
            )
            .expect("add should succeed");
        memory
            .add(
                &[Message::new(Role::User, "Bob is a designer.")],
                HashMap::from([("user_id".to_string(), "bob".to_string())]),
                false,
            )
            .expect("add should succeed");
        memory
            .add(&[Message::new(Role::User, "Task one.")], HashMap::from([("run_id".to_string(), "run-1".to_string())]), false)
            .expect("add should succeed");

        let entities = memory.list_entities().expect("list_entities should succeed");
        let alice = entities.iter().find(|e| e.entity_type == "user_id" && e.entity_id == "alice").expect("alice entity");
        assert_eq!(alice.memory_count, 2);
        let bob = entities.iter().find(|e| e.entity_type == "user_id" && e.entity_id == "bob").expect("bob entity");
        assert_eq!(bob.memory_count, 1);
        let run = entities.iter().find(|e| e.entity_type == "run_id" && e.entity_id == "run-1").expect("run entity");
        assert_eq!(run.memory_count, 1);
    }

    #[test]
    fn test_list_entities_with_no_records_is_empty() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let entities = memory.list_entities().expect("list_entities should succeed");
        assert!(entities.is_empty());
    }

    #[test]
    fn test_history_snapshot_is_empty_for_a_fresh_memory() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        assert!(memory.history_snapshot().is_empty());
    }

    #[test]
    fn test_history_snapshot_reflects_real_added_and_deleted_entries() {
        let llm = FakeLlmProvider::with_facts("Alice is an engineer.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let ids = memory.add(&[Message::new(Role::User, "Alice is an engineer.")], scope(), true).expect("add should succeed");
        let id = ids.first().expect("expected at least one id");
        memory.delete(id).expect("delete should succeed");

        let snapshot = memory.history_snapshot();
        assert_eq!(snapshot.get(id).expect("expected an entry for this id").len(), 2);
    }

    #[test]
    fn test_load_history_snapshot_replaces_existing_state_and_history_reads_it_back() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let mut snapshot = HashMap::new();
        snapshot.insert("restored-id".to_string(), vec![HistoryEntry { event: HistoryEvent::Added, content: "restored content".to_string() }]);
        memory.load_history_snapshot(snapshot);

        let entries = memory.history("restored-id", 0, usize::MAX).expect("history should succeed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].content, "restored content");
    }

    #[test]
    fn test_load_history_snapshot_round_trips_through_history_snapshot() {
        let llm = FakeLlmProvider::with_facts("Bob likes tea.");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let source = Memory::new(llm, embedding, store);
        source.add(&[Message::new(Role::User, "Bob likes tea.")], scope(), true).expect("add should succeed");

        let llm2 = FakeLlmProvider::new();
        let embedding2 = FakeEmbeddingProvider::new();
        let store2 = InMemoryVectorStore::new();
        let restored = Memory::new(llm2, embedding2, store2);
        restored.load_history_snapshot(source.history_snapshot());

        assert_eq!(restored.history_snapshot(), source.history_snapshot());
    }

    #[test]
    fn test_list_entities_is_sorted_by_type_then_id() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        memory
            .add(&[Message::new(Role::User, "one")], HashMap::from([("user_id".to_string(), "zed".to_string())]), false)
            .expect("add should succeed");
        memory
            .add(&[Message::new(Role::User, "two")], HashMap::from([("user_id".to_string(), "alice".to_string())]), false)
            .expect("add should succeed");

        let entities = memory.list_entities().expect("list_entities should succeed");
        assert_eq!(entities[0].entity_id, "alice");
        assert_eq!(entities[1].entity_id, "zed");
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

        let alice_results = memory.search("engineer", 10, &alice_scope, None, true, None, false, false).expect("search should succeed");
        let bob_results = memory.search("engineer", 10, &bob_scope, None, true, None, false, false).expect("search should succeed");

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

        let scheduler_results = memory.search("engineer", 10, &scheduler_scope, None, true, None, false, false).expect("search should succeed");
        let support_results = memory.search("engineer", 10, &support_scope, None, true, None, false, false).expect("search should succeed");

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
        let before = memory.search("engineer", 10, &scope(), None, true, None, false, false).expect("search should succeed");
        assert!(!before.is_empty(), "expected a result before reset for this test to be meaningful");

        memory.reset(&scope(), None).expect("reset should succeed");

        let after = memory.search("engineer", 10, &scope(), None, true, None, false, false).expect("search should succeed");
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

        let results = memory.search("senior engineer", 10, &scope(), None, true, None, false, false).expect("search should succeed");
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

        let result = memory.search("anything", 10, &HashMap::new(), None, true, None, false, false);
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
        let result = memory.search("anything", 10, &non_scope_filter, None, true, None, false, false);
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

        let result = memory.search("anything", 10, &scope(), None, true, None, false, false);
        assert!(matches!(result, Err(crate::CoreError::Provider { .. })));
    }

    #[test]
    fn test_search_rejects_top_k_over_configured_ceiling() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store).with_max_top_k(50);

        let result = memory.search("anything", 51, &scope(), None, true, None, false, false);
        assert!(matches!(result, Err(crate::CoreError::Validation(_))));
    }

    #[test]
    fn test_search_accepts_top_k_at_configured_ceiling() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store).with_max_top_k(50);

        let result = memory.search("anything", 50, &scope(), None, true, None, false, false);
        assert!(result.is_ok(), "top_k exactly at the configured ceiling should be accepted");
    }

    #[test]
    fn test_search_with_no_configured_ceiling_accepts_any_top_k() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.search("anything", 1_000_000, &scope(), None, true, None, false, false);
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

        let results = memory.search("engineer", 10, &scope(), None, true, None, false, false).expect("search should succeed");
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

        let results = memory.search("Alice", 10, &scope(), None, true, None, false, false).expect("search should succeed");
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

        let listed = memory.list(&scope(), 0, usize::MAX, true, None).expect("list should succeed");
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

        let listed = memory.list(&scope(), 0, 0, true, None).expect("list should succeed");
        assert!(listed.is_empty(), "a zero limit should return nothing, not error");
    }

    #[test]
    fn test_list_without_a_scope_id_is_rejected() {
        let llm = FakeLlmProvider::new();
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let result = memory.list(&HashMap::new(), 0, usize::MAX, true, None);
        assert!(
            matches!(result, Err(crate::CoreError::Validation(_))),
            "list with no user_id/agent_id/run_id must be rejected, not return every record unscoped"
        );
    }

    #[test]
    fn test_list_only_returns_records_within_the_requested_scope() {
        let llm = FakeLlmProvider::with_facts("irrelevant");
        let embedding = FakeEmbeddingProvider::new();
        let store = InMemoryVectorStore::new();
        let memory = Memory::new(llm, embedding, store);

        let alice_scope = HashMap::from([("user_id".to_string(), "alice".to_string())]);
        let bob_scope = HashMap::from([("user_id".to_string(), "bob".to_string())]);
        let alice_ids = memory.add(&[Message::new(Role::User, "Alice's fact.")], alice_scope.clone(), false).expect("add should succeed");
        let bob_ids = memory.add(&[Message::new(Role::User, "Bob's fact.")], bob_scope, false).expect("add should succeed");

        let listed = memory.list(&alice_scope, 0, usize::MAX, true, None).expect("list should succeed");
        assert!(listed.contains(alice_ids.first().expect("expected an id")));
        assert!(
            !listed.contains(bob_ids.first().expect("expected an id")),
            "listing alice's scope must not surface bob's record"
        );
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

    #[cfg(all(feature = "sqlite", feature = "fastembed", feature = "candle"))]
    mod real_pipeline_tests {
        use super::*;
        use crate::llm::{CandleLlmProvider, LocalSentenceLlmProvider};
        use crate::reranker::LlmReranker;
        use crate::vector_store::SqliteVectorStore;

        struct TempDbPath(std::path::PathBuf);

        impl Drop for TempDbPath {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }

        fn temp_store() -> (SqliteVectorStore, TempDbPath) {
            let path = std::env::temp_dir()
                .join(format!("memoria-real-pipeline-test-{}.db", std::process::id()));
            let _ = std::fs::remove_file(&path);
            let store = SqliteVectorStore::open(&path).expect("open should succeed");
            (store, TempDbPath(path))
        }

        #[test]
        #[ignore = "downloads a real ~188MB fastembed model and a real ~430MB candle GGUF model on first run; needs real network access"]
        fn real_hybrid_search_and_real_rerank_surface_the_genuinely_relevant_record_first() {
            let embedding_cache = std::env::var("MEMORIA_TEST_FASTEMBED_CACHE_DIR").ok().map(std::path::PathBuf::from);
            let embedding_config =
                EmbeddingConfig { model: "all-MiniLM-L6-v2".to_string(), base_url: None, api_key: None, dimensions: None };
            let embedding = crate::embedding::FastEmbedEmbeddingProvider::from_config(&embedding_config, embedding_cache)
                .expect("valid embedding config should construct");

            let llm_cache = std::env::var("MEMORIA_TEST_CANDLE_CACHE_DIR").ok().map(std::path::PathBuf::from);
            let llm_config = LlmConfig { model: "qwen2.5-0.5b-instruct-q4_0".to_string(), base_url: None, api_key: None, temperature: None };
            let reranker_llm =
                CandleLlmProvider::from_config(&llm_config, llm_cache).expect("valid llm config should construct");

            let (store, _temp_db_path) = temp_store();
            let memory = Memory::new(LocalSentenceLlmProvider::new(), embedding, store)
                .with_reranker(LlmReranker::new(reranker_llm));

            memory
                .add(
                    &[Message::new(Role::User, "The cloud division's quarterly revenue exceeded expectations.")],
                    scope(),
                    false,
                )
                .expect("add should succeed");
            memory
                .add(&[Message::new(Role::User, "Bob prefers tea over coffee in the mornings.")], scope(), false)
                .expect("add should succeed");

            let results = memory
                .search("cloud division quarterly revenue", 2, &scope(), None, false, None, true, false)
                .expect("search should succeed");

            assert_eq!(results.len(), 2, "both records should be returned");
            assert!(
                results[0].payload.get("content").is_some_and(|c| c.contains("cloud division")),
                "the genuinely relevant record, surfaced through real sqlite hybrid search and re-scored by a real candle-backed LlmReranker, should rank first, got: {results:?}"
            );
        }
    }

    #[cfg(feature = "postgres")]
    mod postgres_hybrid_search_tests {
        use super::*;
        use crate::embedding::LocalHashEmbeddingProvider;
        use crate::llm::LocalSentenceLlmProvider;
        use crate::vector_store::PgVectorStore;

        fn test_url() -> Option<String> {
            std::env::var("MEMORIA_TEST_POSTGRES_URL").ok()
        }

        #[test]
        #[ignore = "requires a real Postgres+pgvector instance reachable at MEMORIA_TEST_POSTGRES_URL"]
        fn real_postgres_hybrid_search_promotes_a_much_stronger_keyword_match() {
            let Some(url) = test_url() else {
                panic!("MEMORIA_TEST_POSTGRES_URL must be set to run this test");
            };
            let table = format!("memoria_pg_hybrid_search_test_{}", std::process::id());
            let store = PgVectorStore::open(&url, 8, &table).expect("open should succeed");
            store.reset().expect("reset should succeed");
            let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), store);

            memory
                .add(&[Message::new(Role::User, "keyword needle keyword needle keyword needle")], scope(), false)
                .expect("add should succeed");
            memory.add(&[Message::new(Role::User, "totally unrelated filler content here")], scope(), false).expect("add should succeed");

            let results = memory.search("needle", 2, &scope(), None, false, None, false, false).expect("search should succeed");

            assert_eq!(results.len(), 2, "both records should be returned");
            assert!(
                results[0].payload.get("content").is_some_and(|c| c.contains("needle")),
                "a real Postgres tsvector keyword match should surface the matching record first through Memory::search's hybrid combination, got: {results:?}"
            );

            memory.vector_store.reset().expect("cleanup reset should succeed");
        }
    }
}
