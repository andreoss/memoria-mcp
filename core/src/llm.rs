use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    #[must_use]
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completion {
    pub content: String,
}

impl Completion {
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmError {
    EmptyMessages,
    Backend(String),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMessages => write!(f, "no messages provided"),
            Self::Backend(reason) => write!(f, "backend error: {reason}"),
        }
    }
}

impl std::error::Error for LlmError {}

pub trait LlmProvider {
    #[allow(clippy::missing_errors_doc)]
    fn complete(&self, messages: &[Message]) -> Result<Completion, LlmError>;
}

pub trait LlmContractTests: LlmProvider {
    fn contract_happy_path(&self) {
        let messages = [Message::new(Role::User, "hello")];
        let result = self.complete(&messages);
        assert!(result.is_ok(), "expected a successful completion");
    }

    fn contract_rejects_empty_messages(&self) {
        let result = self.complete(&[]);
        assert!(
            matches!(result, Err(LlmError::EmptyMessages)),
            "expected empty messages to be rejected"
        );
    }
}

impl<T: LlmProvider + ?Sized> LlmContractTests for T {}

#[cfg(test)]
mod tests {
    use super::{Completion, LlmContractTests, LlmError, LlmProvider, Message};

    struct FakeLlmProvider;

    impl LlmProvider for FakeLlmProvider {
        fn complete(&self, messages: &[Message]) -> Result<Completion, LlmError> {
            if messages.is_empty() {
                return Err(LlmError::EmptyMessages);
            }
            Ok(Completion::new("fake response"))
        }
    }

    #[test]
    fn fake_provider_passes_happy_path_contract() {
        FakeLlmProvider.contract_happy_path();
    }

    #[test]
    fn fake_provider_passes_error_contract() {
        FakeLlmProvider.contract_rejects_empty_messages();
    }
}
