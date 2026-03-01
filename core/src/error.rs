use crate::embedding::EmbeddingError;
use crate::llm::LlmError;
use crate::reranker::RerankError;
use crate::vector_store::VectorStoreError;
use std::fmt;

#[derive(Debug)]
pub enum CoreError {
    NotFound(String),
    Validation(String),
    Config(String),
    Provider {
        message: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(message) => write!(f, "not found: {message}"),
            Self::Validation(message) => write!(f, "validation error: {message}"),
            Self::Config(message) => write!(f, "config error: {message}"),
            Self::Provider { message, .. } => write!(f, "provider error: {message}"),
        }
    }
}

impl std::error::Error for CoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl From<VectorStoreError> for CoreError {
    fn from(err: VectorStoreError) -> Self {
        match err {
            VectorStoreError::NotFound => Self::NotFound("vector store record".to_string()),
            other @ VectorStoreError::DimensionMismatch { expected, actual } => Self::Provider {
                message: format!(
                    "vector dimension mismatch: expected {expected}, got {actual}"
                ),
                source: Box::new(other),
            },
            other @ VectorStoreError::Backend(_) => Self::Provider {
                message: other.to_string(),
                source: Box::new(other),
            },
        }
    }
}

impl From<LlmError> for CoreError {
    fn from(err: LlmError) -> Self {
        Self::Provider {
            message: err.to_string(),
            source: Box::new(err),
        }
    }
}

impl From<EmbeddingError> for CoreError {
    fn from(err: EmbeddingError) -> Self {
        Self::Provider {
            message: err.to_string(),
            source: Box::new(err),
        }
    }
}

impl From<RerankError> for CoreError {
    fn from(err: RerankError) -> Self {
        Self::Provider {
            message: err.to_string(),
            source: Box::new(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestSourceError(String);

    impl fmt::Display for TestSourceError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl std::error::Error for TestSourceError {}

    #[test]
    fn not_found_display_is_sensible() {
        let err = CoreError::NotFound("memory 42".to_string());
        assert_eq!(err.to_string(), "not found: memory 42");
    }

    #[test]
    fn validation_display_is_sensible() {
        let err = CoreError::Validation("empty content".to_string());
        assert_eq!(err.to_string(), "validation error: empty content");
    }

    #[test]
    fn config_display_is_sensible() {
        let err = CoreError::Config("missing api key".to_string());
        assert_eq!(err.to_string(), "config error: missing api key");
    }

    #[test]
    fn provider_display_is_sensible() {
        let src = TestSourceError("boom".to_string());
        let err = CoreError::Provider {
            message: "llm failed".to_string(),
            source: Box::new(src),
        };
        assert_eq!(err.to_string(), "provider error: llm failed");
    }

    #[test]
    fn provider_source_returns_wrapped_error() {
        let src = TestSourceError("boom".to_string());
        let err = CoreError::Provider {
            message: "llm failed".to_string(),
            source: Box::new(src),
        };
        let source = std::error::Error::source(&err).expect("Provider must expose a source");
        assert_eq!(source.to_string(), "boom");
    }

    #[test]
    fn non_provider_variants_have_no_source() {
        assert!(std::error::Error::source(&CoreError::NotFound("x".to_string())).is_none());
        assert!(std::error::Error::source(&CoreError::Validation("x".to_string())).is_none());
        assert!(std::error::Error::source(&CoreError::Config("x".to_string())).is_none());
    }

    #[test]
    fn implements_std_error() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<CoreError>();
    }

    #[test]
    fn llm_timeout_routes_to_provider() {
        let err: CoreError = LlmError::Timeout.into();
        match err {
            CoreError::Provider { .. } => {}
            other => panic!("expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn embedding_timeout_routes_to_provider() {
        let err: CoreError = EmbeddingError::Timeout.into();
        match err {
            CoreError::Provider { .. } => {}
            other => panic!("expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn llm_malformed_routes_to_provider() {
        let err: CoreError = LlmError::Malformed("bad shape".to_string()).into();
        match err {
            CoreError::Provider { message, .. } => {
                assert_eq!(message, "malformed backend response: bad shape");
            }
            other => panic!("expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn llm_auth_failure_routes_to_provider() {
        let err: CoreError = LlmError::AuthFailure.into();
        match err {
            CoreError::Provider { message, .. } => {
                assert_eq!(message, "backend rejected credentials");
            }
            other => panic!("expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn rerank_backend_error_routes_to_provider() {
        let err: CoreError = RerankError::Backend("boom".to_string()).into();
        match err {
            CoreError::Provider { message, .. } => {
                assert_eq!(message, "backend error: boom");
            }
            other => panic!("expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn vector_store_dimension_mismatch_routes_to_provider() {
        let err: CoreError = VectorStoreError::DimensionMismatch { expected: 8, actual: 3 }.into();
        match err {
            CoreError::Provider { message, .. } => {
                assert!(message.contains('8'));
                assert!(message.contains('3'));
            }
            other => panic!("expected Provider, got {other:?}"),
        }
    }
}
