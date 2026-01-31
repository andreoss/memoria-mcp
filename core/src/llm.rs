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

#[allow(clippy::missing_errors_doc)]
pub fn extract_facts(
    provider: &impl LlmProvider,
    conversation: &[Message],
) -> Result<Vec<String>, LlmError> {
    let system = Message::new(
        Role::System,
        "Extract discrete factual statements from the conversation. \
         Output one fact per line. Do not number them. Do not add blank lines.",
    );
    let user = Message::new(
        Role::User,
        format!(
            "Conversation:\n{}",
            conversation
                .iter()
                .map(|m| format!("{:?}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n")
        ),
    );
    let completion = provider.complete(&[system, user])?;
    Ok(parse_facts(&completion.content))
}

fn parse_facts(response: &str) -> Vec<String> {
    response
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
        .collect()
}

pub trait LlmContractTests: LlmProvider {
    fn contract_happy_path(&self) {
        let messages = [Message::new(Role::User, "hello")];
        let result = self.complete(&messages);
        let completion = result.expect("expected a successful completion");
        assert!(
            !completion.content.is_empty(),
            "completion content must not be empty"
        );
    }

    fn contract_rejects_empty_messages(&self) {
        let result = self.complete(&[]);
        assert!(
            matches!(result, Err(LlmError::EmptyMessages)),
            "expected empty messages to be rejected"
        );
    }

    fn contract_rejects_backend_error(&self) {
        let messages = [Message::new(Role::User, "hello")];
        let result = self.complete(&messages);
        assert!(
            matches!(result, Err(LlmError::Backend(_))),
            "expected a backend error"
        );
    }
}

impl<T: LlmProvider + ?Sized> LlmContractTests for T {}

#[cfg(test)]
mod tests {
    use super::{extract_facts, Completion, LlmContractTests, LlmError, LlmProvider, Message, Role};

    struct FakeLlmProvider {
        fail_with_backend: bool,
        raw_response: Option<String>,
    }

    impl FakeLlmProvider {
        #[must_use]
        fn new() -> Self {
            Self {
                fail_with_backend: false,
                raw_response: None,
            }
        }

        #[must_use]
        fn failing() -> Self {
            Self {
                fail_with_backend: true,
                raw_response: None,
            }
        }

        #[must_use]
        fn with_response(content: impl Into<String>) -> Self {
            Self {
                fail_with_backend: false,
                raw_response: Some(content.into()),
            }
        }
    }

    impl LlmProvider for FakeLlmProvider {
        fn complete(&self, messages: &[Message]) -> Result<Completion, LlmError> {
            if self.fail_with_backend {
                return Err(LlmError::Backend("fake backend failure".to_string()));
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

    #[test]
    fn empty_messages_display_is_sensible() {
        let err = LlmError::EmptyMessages;
        assert_eq!(err.to_string(), "no messages provided");
    }

    #[test]
    fn backend_display_is_sensible() {
        let err = LlmError::Backend("boom".to_string());
        assert_eq!(err.to_string(), "backend error: boom");
    }

    #[test]
    fn fake_provider_passes_happy_path_contract() {
        FakeLlmProvider::new().contract_happy_path();
    }

    #[test]
    fn fake_provider_passes_error_contract() {
        FakeLlmProvider::new().contract_rejects_empty_messages();
    }

    #[test]
    fn fake_provider_passes_backend_error_contract() {
        FakeLlmProvider::failing().contract_rejects_backend_error();
    }

    #[test]
    fn extract_facts_happy_path() {
        let response = "Alice is an engineer.\nBob lives in Berlin.\nThe project started in 2021.";
        let provider = FakeLlmProvider::with_response(response);
        let conversation = [Message::new(
            Role::User,
            "Alice is an engineer and Bob lives in Berlin. The project started in 2021.",
        )];
        let facts = extract_facts(&provider, &conversation).expect("expected facts");
        assert_eq!(facts.len(), 3);
        assert_eq!(facts[0], "Alice is an engineer.");
        assert_eq!(facts[1], "Bob lives in Berlin.");
        assert_eq!(facts[2], "The project started in 2021.");
    }

    #[test]
    fn extract_facts_empty_response() {
        let provider = FakeLlmProvider::with_response("");
        let conversation = [Message::new(Role::User, "anything")];
        let facts = extract_facts(&provider, &conversation)
            .expect("empty response should not be an error");
        assert!(facts.is_empty());
    }

    #[test]
    fn extract_facts_malformed_response_skips_blanks() {
        let response = "\n  \nFact one.\n\n   \nFact two.\n";
        let provider = FakeLlmProvider::with_response(response);
        let conversation = [Message::new(Role::User, "anything")];
        let facts = extract_facts(&provider, &conversation).expect("expected facts");
        assert_eq!(facts, vec!["Fact one.".to_string(), "Fact two.".to_string()]);
    }
}
