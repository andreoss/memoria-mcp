use crate::embedding::{EmbeddingError, EmbeddingProvider};
use crate::llm::{Completion, LlmError, LlmProvider, Message};

pub struct FakeLlmProvider {
    fail_with_backend: bool,
    fail_with_timeout: bool,
    fail_with_malformed: Option<String>,
    fail_with_auth: bool,
    raw_response: Option<String>,
}

impl FakeLlmProvider {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            fail_with_backend: false,
            fail_with_timeout: false,
            fail_with_malformed: None,
            fail_with_auth: false,
            raw_response: None,
        }
    }

    #[must_use]
    pub(crate) fn failing() -> Self {
        Self {
            fail_with_backend: true,
            fail_with_timeout: false,
            fail_with_malformed: None,
            fail_with_auth: false,
            raw_response: None,
        }
    }

    #[must_use]
    pub(crate) fn timing_out() -> Self {
        Self {
            fail_with_backend: false,
            fail_with_timeout: true,
            fail_with_malformed: None,
            fail_with_auth: false,
            raw_response: None,
        }
    }

    #[must_use]
    pub(crate) fn returning_malformed(reason: impl Into<String>) -> Self {
        Self {
            fail_with_backend: false,
            fail_with_timeout: false,
            fail_with_malformed: Some(reason.into()),
            fail_with_auth: false,
            raw_response: None,
        }
    }

    #[must_use]
    pub(crate) fn unauthorized() -> Self {
        Self {
            fail_with_backend: false,
            fail_with_timeout: false,
            fail_with_malformed: None,
            fail_with_auth: true,
            raw_response: None,
        }
    }

    #[must_use]
    pub(crate) fn with_response(content: impl Into<String>) -> Self {
        Self {
            fail_with_backend: false,
            fail_with_timeout: false,
            fail_with_malformed: None,
            fail_with_auth: false,
            raw_response: Some(content.into()),
        }
    }

    #[must_use]
    pub(crate) fn with_facts(content: impl Into<String>) -> Self {
        Self::with_response(content)
    }
}

impl LlmProvider for FakeLlmProvider {
    fn complete(&self, messages: &[Message]) -> Result<Completion, LlmError> {
        if self.fail_with_backend {
            return Err(LlmError::Backend("fake backend failure".to_string()));
        }
        if self.fail_with_timeout {
            return Err(LlmError::Timeout);
        }
        if let Some(reason) = &self.fail_with_malformed {
            return Err(LlmError::Malformed(reason.clone()));
        }
        if self.fail_with_auth {
            return Err(LlmError::AuthFailure);
        }
        if messages.is_empty() {
            return Err(LlmError::EmptyMessages);
        }
        let content = self
            .raw_response
            .clone()
            .unwrap_or_else(|| {
                messages
                    .iter()
                    .map(|m| m.content.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            });
        Ok(Completion::new(content))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FakeEmbeddingProvider {
    fail_with_backend: bool,
    fail_with_timeout: bool,
}

impl FakeEmbeddingProvider {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            fail_with_backend: false,
            fail_with_timeout: false,
        }
    }

    #[must_use]
    pub(crate) fn failing() -> Self {
        Self {
            fail_with_backend: true,
            fail_with_timeout: false,
        }
    }

    #[must_use]
    pub(crate) fn timing_out() -> Self {
        Self {
            fail_with_backend: false,
            fail_with_timeout: true,
        }
    }
}

impl EmbeddingProvider for FakeEmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        const DIM: u8 = 8;
        if self.fail_with_backend {
            return Err(EmbeddingError::Backend("fake backend failure".to_string()));
        }
        if self.fail_with_timeout {
            return Err(EmbeddingError::Timeout);
        }
        if text.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }
        let bytes = text.as_bytes();
        let mut vector = Vec::with_capacity(usize::from(DIM));
        for d in 0..DIM {
            let mut acc = f32::from(d);
            let mut idx: u8 = 0;
            for &b in bytes {
                let weight = f32::from(idx % DIM) + 1.0;
                acc = acc.mul_add(f32::from(b), weight);
                idx = idx.wrapping_add(1);
            }
            vector.push(acc);
        }
        Ok(vector)
    }
}
