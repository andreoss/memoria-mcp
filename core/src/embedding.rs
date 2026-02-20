use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmbeddingError {
    EmptyInput,
    Backend(String),
    Timeout,
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "no text provided"),
            Self::Backend(reason) => write!(f, "backend error: {reason}"),
            Self::Timeout => write!(f, "backend timed out before responding"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

pub trait EmbeddingProvider {
    #[allow(clippy::missing_errors_doc)]
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    #[allow(clippy::missing_errors_doc)]
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        texts.iter().map(|text| self.embed(text)).collect()
    }
}

pub struct LocalHashEmbeddingProvider;

impl LocalHashEmbeddingProvider {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for LocalHashEmbeddingProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingProvider for LocalHashEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        const DIM: u8 = 8;
        const MODULUS: u32 = 1_000_003;
        if text.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        let bytes = text.as_bytes();
        let mut vector = Vec::with_capacity(usize::from(DIM));
        for d in 0..DIM {
            let mut acc = u32::from(d);
            let mut idx: u8 = 0;
            for &b in bytes {
                let weight = u32::from(idx % DIM) + 1;
                acc = acc.wrapping_mul(u32::from(b) + 1).wrapping_add(weight) % MODULUS;
                idx = idx.wrapping_add(1);
            }
            #[allow(clippy::cast_precision_loss)]
            vector.push(acc as f32);
        }
        Ok(vector)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingConfig {
    pub model: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub dimensions: Option<usize>,
}

impl EmbeddingConfig {
    #[allow(clippy::missing_errors_doc)]
    pub fn validate(&self) -> Result<(), crate::CoreError> {
        if self.model.trim().is_empty() {
            return Err(crate::CoreError::Config("model must not be empty".to_string()));
        }
        if self.dimensions == Some(0) {
            return Err(crate::CoreError::Config("dimensions must be greater than zero".to_string()));
        }
        Ok(())
    }
}

pub trait EmbeddingContractTests: EmbeddingProvider {
    fn contract_happy_path(&self) {
        let result = self.embed("hello");
        assert!(result.is_ok(), "expected a successful embedding");
    }

    fn contract_rejects_empty_string(&self) {
        let result = self.embed("");
        assert!(
            matches!(result, Err(EmbeddingError::EmptyInput)),
            "expected empty string input to be rejected"
        );
    }

    fn contract_rejects_backend_error(&self) {
        let result = self.embed("hello");
        assert!(
            matches!(result, Err(EmbeddingError::Backend(_))),
            "expected a backend error"
        );
    }

    fn contract_rejects_timeout(&self) {
        let result = self.embed("hello");
        assert!(
            matches!(result, Err(EmbeddingError::Timeout)),
            "expected a timeout error"
        );
    }

    fn contract_embed_is_deterministic(&self) {
        let first = self.embed("hello").expect("expected a successful embedding");
        let second = self.embed("hello").expect("expected a successful embedding");
        assert_eq!(
            first, second,
            "embedding must be deterministic for the same input"
        );
    }

    fn contract_embed_batch_matches_individual_calls(&self) {
        let texts = ["alpha", "beta", "gamma"];
        let batch = self
            .embed_batch(&texts)
            .expect("expected a successful batch embedding");
        let individual: Vec<Vec<f32>> = texts
            .iter()
            .map(|text| self.embed(text).expect("expected a successful embedding"))
            .collect();
        assert_eq!(
            batch, individual,
            "batch embedding must match individual embed() calls"
        );
    }

    fn contract_embed_batch_empty_list_returns_empty(&self) {
        let batch = self
            .embed_batch(&[])
            .expect("expected a successful empty batch embedding");
        assert!(
            batch.is_empty(),
            "embedding an empty batch must return an empty vector"
        );
    }
}

impl<T: EmbeddingProvider + ?Sized> EmbeddingContractTests for T {}

#[cfg(feature = "ollama")]
pub struct OllamaEmbeddingProvider {
    client: reqwest::blocking::Client,
    base_url: String,
    model: String,
}

#[cfg(feature = "ollama")]
impl OllamaEmbeddingProvider {
    #[allow(clippy::missing_errors_doc)]
    pub fn from_config(config: EmbeddingConfig) -> Result<Self, crate::CoreError> {
        config.validate()?;
        Ok(Self {
            client: reqwest::blocking::Client::new(),
            base_url: config.base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
            model: config.model,
        })
    }
}

#[cfg(feature = "ollama")]
fn build_embeddings_request(model: &str, prompt: &str) -> serde_json::Value {
    serde_json::json!({"model": model, "prompt": prompt})
}

#[cfg(feature = "ollama")]
#[allow(clippy::cast_possible_truncation)]
fn parse_embeddings_response(json: &serde_json::Value) -> Result<Vec<f32>, EmbeddingError> {
    json.get("embedding")
        .and_then(serde_json::Value::as_array)
        .map(|values| values.iter().filter_map(serde_json::Value::as_f64).map(|v| v as f32).collect())
        .ok_or_else(|| EmbeddingError::Backend("missing embedding field in response".to_string()))
}

#[cfg(feature = "ollama")]
impl EmbeddingProvider for OllamaEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        if text.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        let request = build_embeddings_request(&self.model, text);
        let response = self
            .client
            .post(format!("{}/api/embeddings", self.base_url))
            .json(&request)
            .send()
            .map_err(|err| {
                if err.is_timeout() {
                    EmbeddingError::Timeout
                } else {
                    EmbeddingError::Backend(err.to_string())
                }
            })?;

        if !response.status().is_success() {
            return Err(EmbeddingError::Backend(format!("HTTP {}", response.status())));
        }

        let json: serde_json::Value = response.json().map_err(|err| EmbeddingError::Backend(err.to_string()))?;
        parse_embeddings_response(&json)
    }
}

#[cfg(test)]
mod tests {
    use super::{EmbeddingConfig, EmbeddingContractTests, EmbeddingError, EmbeddingProvider, LocalHashEmbeddingProvider};
    use crate::test_support::FakeEmbeddingProvider;

    #[test]
    fn empty_input_display_is_sensible() {
        let err = EmbeddingError::EmptyInput;
        assert_eq!(err.to_string(), "no text provided");
    }

    #[test]
    fn backend_display_is_sensible() {
        let err = EmbeddingError::Backend("boom".to_string());
        assert_eq!(err.to_string(), "backend error: boom");
    }

    #[test]
    fn timeout_display_is_sensible() {
        let err = EmbeddingError::Timeout;
        assert_eq!(err.to_string(), "backend timed out before responding");
    }

    #[test]
    fn fake_provider_passes_happy_path_contract() {
        FakeEmbeddingProvider::new().contract_happy_path();
    }

    #[test]
    fn fake_provider_passes_error_contract() {
        FakeEmbeddingProvider::new().contract_rejects_empty_string();
    }

    #[test]
    fn local_hash_provider_passes_happy_path_contract() {
        LocalHashEmbeddingProvider::new().contract_happy_path();
    }

    #[test]
    fn local_hash_provider_passes_error_contract() {
        LocalHashEmbeddingProvider::new().contract_rejects_empty_string();
    }

    #[test]
    fn local_hash_provider_passes_deterministic_contract() {
        LocalHashEmbeddingProvider::new().contract_embed_is_deterministic();
    }

    #[test]
    fn local_hash_provider_passes_batch_matches_individual_contract() {
        LocalHashEmbeddingProvider::new().contract_embed_batch_matches_individual_calls();
    }

    #[test]
    fn local_hash_provider_never_produces_non_finite_components() {
        let provider = LocalHashEmbeddingProvider::new();
        for text in ["Alice is an engineer.", "Bob lives in Berlin.", "a", &"x".repeat(500)] {
            let vector = provider.embed(text).expect("embed should succeed");
            assert!(
                vector.iter().all(|component| component.is_finite()),
                "embedding for {text:?} contained a non-finite component: {vector:?}"
            );
        }
    }

    #[test]
    fn fake_provider_passes_backend_error_contract() {
        FakeEmbeddingProvider::failing().contract_rejects_backend_error();
    }

    #[test]
    fn fake_provider_passes_timeout_contract() {
        FakeEmbeddingProvider::timing_out().contract_rejects_timeout();
    }

    #[test]
    fn fake_provider_passes_deterministic_contract() {
        FakeEmbeddingProvider::new().contract_embed_is_deterministic();
    }

    #[test]
    fn fake_provider_passes_batch_matches_individual_contract() {
        FakeEmbeddingProvider::new().contract_embed_batch_matches_individual_calls();
    }

    #[test]
    fn fake_provider_passes_batch_empty_contract() {
        FakeEmbeddingProvider::new().contract_embed_batch_empty_list_returns_empty();
    }

    #[test]
    fn config_with_valid_model_passes_validation() {
        let config = EmbeddingConfig {
            model: "nomic-embed-text".to_string(),
            base_url: None,
            api_key: None,
            dimensions: None,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_with_empty_model_is_rejected() {
        let config = EmbeddingConfig {
            model: String::new(),
            base_url: None,
            api_key: None,
            dimensions: None,
        };
        assert!(matches!(config.validate(), Err(crate::CoreError::Config(_))));
    }

    #[test]
    fn config_with_whitespace_only_model_is_rejected() {
        let config = EmbeddingConfig {
            model: "   ".to_string(),
            base_url: None,
            api_key: None,
            dimensions: None,
        };
        assert!(matches!(config.validate(), Err(crate::CoreError::Config(_))));
    }

    #[test]
    fn config_with_positive_dimensions_passes_validation() {
        let config = EmbeddingConfig {
            model: "nomic-embed-text".to_string(),
            base_url: None,
            api_key: None,
            dimensions: Some(768),
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_with_zero_dimensions_is_rejected() {
        let config = EmbeddingConfig {
            model: "nomic-embed-text".to_string(),
            base_url: None,
            api_key: None,
            dimensions: Some(0),
        };
        assert!(matches!(config.validate(), Err(crate::CoreError::Config(_))));
    }

    #[cfg(feature = "ollama")]
    mod ollama_tests {
        use super::super::{build_embeddings_request, parse_embeddings_response, OllamaEmbeddingProvider};
        use super::*;

        #[test]
        fn build_embeddings_request_has_the_expected_shape() {
            let request = build_embeddings_request("nomic-embed-text", "hello world");
            assert_eq!(request["model"], "nomic-embed-text");
            assert_eq!(request["prompt"], "hello world");
        }

        #[test]
        fn parse_embeddings_response_extracts_the_vector() {
            let json = serde_json::json!({"embedding": [0.1, 0.2, 0.3]});
            let vector = parse_embeddings_response(&json).expect("expected a vector");
            assert_eq!(vector.len(), 3);
        }

        #[test]
        fn parse_embeddings_response_rejects_a_missing_embedding_field() {
            let json = serde_json::json!({"not_embedding": []});
            assert!(matches!(parse_embeddings_response(&json), Err(EmbeddingError::Backend(_))));
        }

        #[test]
        fn from_config_rejects_an_invalid_config_before_building_the_client() {
            let config = EmbeddingConfig { model: String::new(), base_url: None, api_key: None, dimensions: None };
            assert!(matches!(OllamaEmbeddingProvider::from_config(config), Err(crate::CoreError::Config(_))));
        }

        #[test]
        fn contract_rejects_backend_error_against_an_unreachable_host() {
            let config = EmbeddingConfig {
                model: "nomic-embed-text".to_string(),
                base_url: Some("http://127.0.0.1:1".to_string()),
                api_key: None,
                dimensions: None,
            };
            let provider = OllamaEmbeddingProvider::from_config(config).expect("valid config should construct");
            provider.contract_rejects_backend_error();
        }

        #[test]
        fn rejects_empty_input_without_a_network_call() {
            let config = EmbeddingConfig {
                model: "nomic-embed-text".to_string(),
                base_url: Some("http://127.0.0.1:1".to_string()),
                api_key: None,
                dimensions: None,
            };
            let provider = OllamaEmbeddingProvider::from_config(config).expect("valid config should construct");
            assert!(matches!(provider.embed(""), Err(EmbeddingError::EmptyInput)));
        }

        #[test]
        #[ignore = "requires a real Ollama instance reachable at MEMORIA_TEST_OLLAMA_URL"]
        fn real_ollama_produces_a_real_semantic_embedding() {
            let base_url = std::env::var("MEMORIA_TEST_OLLAMA_URL").unwrap_or_else(|_| "http://192.0.2.1:11434".to_string());
            let config = EmbeddingConfig {
                model: "nomic-embed-text".to_string(),
                base_url: Some(base_url),
                api_key: None,
                dimensions: None,
            };
            let provider = OllamaEmbeddingProvider::from_config(config).expect("valid config should construct");
            provider.contract_happy_path();

            let related_a = provider.embed("Alice works as a nurse at the downtown hospital.").expect("embed should succeed");
            let related_b = provider.embed("Alice recently started a new nursing position at the hospital.").expect("embed should succeed");
            let unrelated = provider.embed("The weather in Paris was cold and rainy yesterday.").expect("embed should succeed");

            let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
            let norm = |a: &[f32]| -> f32 { a.iter().map(|x| x * x).sum::<f32>().sqrt() };
            let cosine = |a: &[f32], b: &[f32]| dot(a, b) / (norm(a) * norm(b));

            let related_similarity = cosine(&related_a, &related_b);
            let unrelated_similarity = cosine(&related_a, &unrelated);
            assert!(
                related_similarity > unrelated_similarity,
                "related sentences ({related_similarity}) should be more similar than unrelated ones ({unrelated_similarity})"
            );
        }
    }
}
