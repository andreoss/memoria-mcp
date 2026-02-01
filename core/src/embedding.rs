use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmbeddingError {
    EmptyInput,
    Backend(String),
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "no text provided"),
            Self::Backend(reason) => write!(f, "backend error: {reason}"),
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

#[cfg(test)]
mod tests {
    use super::{EmbeddingContractTests, EmbeddingError, EmbeddingProvider};

    struct FakeEmbeddingProvider {
        fail_with_backend: bool,
    }

    impl FakeEmbeddingProvider {
        #[must_use]
        fn new() -> Self {
            Self {
                fail_with_backend: false,
            }
        }

        #[must_use]
        fn failing() -> Self {
            Self {
                fail_with_backend: true,
            }
        }
    }

    impl EmbeddingProvider for FakeEmbeddingProvider {
        fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            if self.fail_with_backend {
                return Err(EmbeddingError::Backend("fake backend failure".to_string()));
            }
            if text.is_empty() {
                return Err(EmbeddingError::EmptyInput);
            }
            let digest: f32 = text
                .bytes()
                .fold(0.0, |acc, b| acc.mul_add(31.0, f32::from(b)));
            let dim = text.len().max(1);
            Ok(vec![digest; dim])
        }
    }

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
    fn fake_provider_passes_happy_path_contract() {
        FakeEmbeddingProvider::new().contract_happy_path();
    }

    #[test]
    fn fake_provider_passes_error_contract() {
        FakeEmbeddingProvider::new().contract_rejects_empty_string();
    }

    #[test]
    fn fake_provider_passes_backend_error_contract() {
        FakeEmbeddingProvider::failing().contract_rejects_backend_error();
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
}
