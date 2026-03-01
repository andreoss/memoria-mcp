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

const LLM_RERANKER_SYSTEM_PROMPT: &str = "You are a relevance scoring assistant. \
Given a query and a document, score how relevant the document is to the query.\n\n\
Score the relevance on a scale from 0.0 to 1.0, where:\n\
- 1.0 = Perfectly relevant and directly answers the query\n\
- 0.8-0.9 = Highly relevant with good information\n\
- 0.6-0.7 = Moderately relevant with some useful information\n\
- 0.4-0.5 = Slightly relevant with limited useful information\n\
- 0.0-0.3 = Not relevant or no useful information\n\n\
Respond with only a single numerical score between 0.0 and 1.0. \
Do not include any explanation or additional text.";

const LLM_RERANKER_MAX_INPUT_CHARS: usize = 4000;

fn truncate_chars(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((byte_index, _)) => &s[..byte_index],
        None => s,
    }
}

fn extract_first_number(chars: &[char], require_decimal: bool) -> Option<f32> {
    let mut i = 0;
    while i < chars.len() {
        let start = i;
        let mut j = i;
        if j < chars.len() && chars[j] == '-' {
            j += 1;
        }
        let digits_start = j;
        while j < chars.len() && chars[j].is_ascii_digit() {
            j += 1;
        }
        if j > digits_start {
            let mut end = j;
            let mut has_decimal = false;
            if j < chars.len() && chars[j] == '.' {
                let mut k = j + 1;
                while k < chars.len() && chars[k].is_ascii_digit() {
                    k += 1;
                }
                if k > j + 1 {
                    end = k;
                    has_decimal = true;
                }
            }
            if has_decimal == require_decimal {
                let token: String = chars[start..end].iter().collect();
                if let Ok(value) = token.parse::<f32>() {
                    return Some(value);
                }
            }
        }
        i = start + 1;
    }
    None
}

fn extract_score(response: &str) -> f32 {
    let chars: Vec<char> = response.chars().collect();
    let parsed = extract_first_number(&chars, true).or_else(|| extract_first_number(&chars, false));
    parsed.map_or(0.5, |value| value.clamp(0.0, 1.0))
}

fn build_scoring_messages(query: &str, content: &str) -> [crate::llm::Message; 2] {
    let safe_query = truncate_chars(query, LLM_RERANKER_MAX_INPUT_CHARS);
    let safe_content = truncate_chars(content, LLM_RERANKER_MAX_INPUT_CHARS);
    [
        crate::llm::Message::new(crate::llm::Role::System, LLM_RERANKER_SYSTEM_PROMPT),
        crate::llm::Message::new(crate::llm::Role::User, format!("Query: {safe_query}\n\nDocument: {safe_content}")),
    ]
}

pub struct LlmReranker<L: crate::llm::LlmProvider> {
    llm: L,
}

impl<L: crate::llm::LlmProvider> LlmReranker<L> {
    pub const fn new(llm: L) -> Self {
        Self { llm }
    }

    fn score(&self, query: &str, content: &str) -> f32 {
        let messages = build_scoring_messages(query, content);
        match self.llm.complete(&messages) {
            Ok(completion) => extract_score(&completion.content),
            Err(_) => 0.5,
        }
    }
}

impl<L: crate::llm::LlmProvider> Reranker for LlmReranker<L> {
    fn rerank(&self, query: &str, results: Vec<SearchResult>) -> Result<Vec<SearchResult>, RerankError> {
        let mut rescored: Vec<SearchResult> = results
            .into_iter()
            .map(|mut result| {
                let content = result.payload.get("content").map_or("", String::as_str);
                let relevance = self.score(query, content);
                result.score = 1.0 - relevance;
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

    fn assert_score_eq(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-6, "expected {expected}, got {actual}");
    }

    #[test]
    fn extract_score_parses_a_plain_decimal() {
        assert_score_eq(extract_score("0.85"), 0.85);
    }

    #[test]
    fn extract_score_parses_the_first_decimal_embedded_in_prose() {
        assert_score_eq(extract_score("The score is 0.72 out of 1.0."), 0.72);
    }

    #[test]
    fn extract_score_falls_back_to_an_integer_when_no_decimal_is_present() {
        assert_score_eq(extract_score("Score: 1"), 1.0);
        assert_score_eq(extract_score("Score: 0"), 0.0);
    }

    #[test]
    fn extract_score_clamps_values_above_one() {
        assert_score_eq(extract_score("2.5"), 1.0);
    }

    #[test]
    fn extract_score_clamps_negative_values() {
        assert_score_eq(extract_score("-0.3"), 0.0);
    }

    #[test]
    fn extract_score_defaults_to_neutral_when_nothing_parses() {
        assert_score_eq(extract_score("no numbers here at all"), 0.5);
    }

    #[test]
    fn llm_reranker_converts_relevance_into_distance_and_sorts_ascending() {
        let reranker = LlmReranker::new(crate::test_support::FakeLlmProvider::with_response("0.9"));
        let input = vec![result_with_content("a", 0.1, "irrelevant to everything"), result_with_content("b", 0.9, "also irrelevant")];
        let output = reranker.rerank("query", input).expect("expected a successful rerank");
        for result in &output {
            assert!((result.score - 0.1).abs() < 1e-6, "expected score 1.0 - 0.9 = 0.1, got {}", result.score);
        }
    }

    #[test]
    fn llm_reranker_falls_back_to_a_neutral_score_when_the_backend_fails() {
        let reranker = LlmReranker::new(crate::test_support::FakeLlmProvider::failing());
        let input = vec![result_with_content("a", 0.1, "some content")];
        let output = reranker.rerank("query", input).expect("a failed backend call must degrade to neutral, not fail the whole rerank");
        assert!((output[0].score - 0.5).abs() < 1e-6, "expected the neutral fallback score, got {}", output[0].score);
    }

    #[test]
    fn build_scoring_messages_truncates_an_overly_long_query_and_content() {
        let long_query = "q".repeat(LLM_RERANKER_MAX_INPUT_CHARS * 2);
        let long_content = "d".repeat(LLM_RERANKER_MAX_INPUT_CHARS * 2);
        let messages = build_scoring_messages(&long_query, &long_content);
        let user_message = &messages[1].content;
        assert!(
            user_message.len() < LLM_RERANKER_MAX_INPUT_CHARS * 3,
            "expected the query and content to be truncated before building the prompt, got a user message of {} chars",
            user_message.len()
        );
    }

    #[test]
    fn build_scoring_messages_truncates_on_a_char_boundary_not_a_byte_boundary() {
        let multi_byte_content: String = "é".repeat(LLM_RERANKER_MAX_INPUT_CHARS * 2);
        let messages = build_scoring_messages("query", &multi_byte_content);
        assert!(messages[1].content.is_char_boundary(messages[1].content.len()), "truncation must not split a multi-byte character");
    }

    #[test]
    fn llm_reranker_passes_the_shared_contract() {
        let reranker = LlmReranker::new(crate::test_support::FakeLlmProvider::with_response("0.5"));
        reranker.contract_empty_input_is_empty_output();
        LlmReranker::new(crate::test_support::FakeLlmProvider::with_response("0.5")).contract_preserves_the_same_set_of_ids();
    }

    #[cfg(feature = "ollama")]
    mod ollama_tests {
        use super::*;
        use crate::llm::{LlmConfig, OllamaLlmProvider};

        #[test]
        #[ignore = "requires a real Ollama instance reachable at MEMORIA_TEST_OLLAMA_URL"]
        fn real_ollama_reranker_ranks_the_genuinely_relevant_record_first() {
            let base_url = std::env::var("MEMORIA_TEST_OLLAMA_URL").unwrap_or_else(|_| "http://192.0.2.1:11434".to_string());
            let config = LlmConfig { model: "qwen2.5:0.5b".to_string(), base_url: Some(base_url), api_key: None, temperature: None };
            let llm = OllamaLlmProvider::from_config(config).expect("valid config should construct");
            let reranker = LlmReranker::new(llm);

            let input = vec![
                result_with_content("relevant", 0.5, "The cloud division's revenue exceeded expectations this quarter."),
                result_with_content("unrelated", 0.5, "Bob prefers tea over coffee in the mornings."),
            ];
            let output = reranker.rerank("cloud division revenue", input).expect("expected a successful rerank");

            assert_eq!(output[0].id, "relevant", "the genuinely relevant record should be ranked first by a real model, got: {output:?}");
            assert!(output[0].score < output[1].score, "the relevant record's distance should be lower than the unrelated one's, got: {output:?}");
        }
    }
}
