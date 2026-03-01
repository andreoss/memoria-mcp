use crate::vector_store::SearchResult;
use std::collections::HashSet;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RerankError {
    Backend(String),
}

impl fmt::Display for RerankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(reason) => write!(f, "backend error: {reason}"),
        }
    }
}

impl std::error::Error for RerankError {}

pub trait Reranker {
    #[allow(clippy::missing_errors_doc)]
    fn rerank(&self, query: &str, results: Vec<SearchResult>) -> Result<Vec<SearchResult>, RerankError>;
}

impl<T: Reranker + ?Sized> Reranker for Box<T> {
    fn rerank(&self, query: &str, results: Vec<SearchResult>) -> Result<Vec<SearchResult>, RerankError> {
        self.as_ref().rerank(query, results)
    }
}

fn tokenize(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn lexical_distance(query_tokens: &HashSet<String>, content: &str) -> f32 {
    if query_tokens.is_empty() {
        return 1.0;
    }
    let content_tokens = tokenize(content);
    let overlap = query_tokens.intersection(&content_tokens).count();
    #[allow(clippy::cast_precision_loss)]
    let overlap_ratio = overlap as f32 / query_tokens.len() as f32;
    1.0 - overlap_ratio
}

pub struct LocalOverlapReranker;

impl LocalOverlapReranker {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for LocalOverlapReranker {
    fn default() -> Self {
        Self::new()
    }
}

fn normalize_distance(score: f32, min_score: f32, max_score: f32) -> f32 {
    let range = max_score - min_score;
    if range > 0.0 {
        (score - min_score) / range
    } else {
        0.0
    }
}

impl Reranker for LocalOverlapReranker {
    fn rerank(&self, query: &str, results: Vec<SearchResult>) -> Result<Vec<SearchResult>, RerankError> {
        let query_tokens = tokenize(query);
        let min_score = results.iter().map(|r| r.score).fold(f32::INFINITY, f32::min);
        let max_score = results.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);
        let mut rescored: Vec<SearchResult> = results
            .into_iter()
            .map(|mut result| {
                let normalized_semantic = normalize_distance(result.score, min_score, max_score);
                let content = result.payload.get("content").map_or("", String::as_str);
                let lexical = lexical_distance(&query_tokens, content);
                result.score = f32::midpoint(normalized_semantic, lexical);
                result
            })
            .collect();
        rescored.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(rescored)
    }
}

pub trait RerankerContractTests: Reranker {
    fn contract_empty_input_is_empty_output(&self) {
        let result = self.rerank("query", Vec::new()).expect("expected a successful rerank");
        assert!(result.is_empty(), "reranking an empty input must produce an empty output");
    }

    fn contract_preserves_the_same_set_of_ids(&self) {
        let mut payload_a = std::collections::HashMap::new();
        payload_a.insert("content".to_string(), "the quick brown fox".to_string());
        let mut payload_b = std::collections::HashMap::new();
        payload_b.insert("content".to_string(), "a lazy dog sleeps".to_string());
        let input = vec![
            SearchResult { id: "a".to_string(), score: 0.5, payload: payload_a },
            SearchResult { id: "b".to_string(), score: 0.5, payload: payload_b },
        ];
        let output = self.rerank("fox", input).expect("expected a successful rerank");
        let mut ids: Vec<&str> = output.iter().map(|r| r.id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec!["a", "b"], "reranking must not add or drop candidates");
    }

    fn contract_rejects_backend_error(&self) {
        let result = self.rerank("query", Vec::new());
        assert!(matches!(result, Err(RerankError::Backend(_))), "expected a backend error");
    }
}

impl<T: Reranker + ?Sized> RerankerContractTests for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::FakeRerankerProvider;
    use std::collections::HashMap;

    fn result_with_content(id: &str, score: f32, content: &str) -> SearchResult {
        let mut payload = HashMap::new();
        payload.insert("content".to_string(), content.to_string());
        SearchResult { id: id.to_string(), score, payload }
    }

    #[test]
    fn local_overlap_reranker_passes_the_shared_contract() {
        LocalOverlapReranker::new().contract_empty_input_is_empty_output();
        LocalOverlapReranker::new().contract_preserves_the_same_set_of_ids();
    }

    #[test]
    fn local_overlap_reranker_ranks_lexically_closer_content_first() {
        let reranker = LocalOverlapReranker::new();
        let input = vec![
            result_with_content("unrelated", 0.5, "cooking pasta recipes"),
            result_with_content("relevant", 0.5, "I love rust programming"),
        ];
        let output = reranker.rerank("rust programming", input).expect("expected a successful rerank");
        assert_eq!(output[0].id, "relevant", "the lexically closer result should be ranked first");
    }

    #[test]
    fn local_overlap_reranker_is_deterministic() {
        let reranker = LocalOverlapReranker::new();
        let input = || {
            vec![
                result_with_content("a", 0.3, "alpha beta gamma"),
                result_with_content("b", 0.4, "delta epsilon zeta"),
            ]
        };
        let first = reranker.rerank("alpha", input()).expect("expected a successful rerank");
        let second = reranker.rerank("alpha", input()).expect("expected a successful rerank");
        assert_eq!(first, second, "reranking must be deterministic for the same input");
    }

    #[test]
    fn local_overlap_reranker_handles_an_empty_query() {
        let reranker = LocalOverlapReranker::new();
        let input = vec![result_with_content("a", 0.2, "some content")];
        let output = reranker.rerank("", input).expect("expected a successful rerank");
        assert_eq!(output.len(), 1, "an empty query must not drop candidates");
    }

    #[test]
    fn fake_reranker_provider_contract_rejects_backend_error() {
        FakeRerankerProvider::failing().contract_rejects_backend_error();
    }
}
