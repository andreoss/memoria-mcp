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
}

impl<T: EmbeddingProvider + ?Sized> EmbeddingContractTests for T {}

#[cfg(test)]
mod tests {
    use super::{EmbeddingContractTests, EmbeddingError, EmbeddingProvider};

    struct FakeEmbeddingProvider;

    impl EmbeddingProvider for FakeEmbeddingProvider {
        fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            if text.is_empty() {
                return Err(EmbeddingError::EmptyInput);
            }
            Ok(vec![0.0, 1.0, 0.0])
        }
    }

    #[test]
    fn fake_provider_passes_happy_path_contract() {
        FakeEmbeddingProvider.contract_happy_path();
    }

    #[test]
    fn fake_provider_passes_error_contract() {
        FakeEmbeddingProvider.contract_rejects_empty_string();
    }
}
