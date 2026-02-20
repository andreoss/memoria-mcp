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

#[derive(Clone, Debug, PartialEq)]
pub struct LlmConfig {
    pub model: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub temperature: Option<f32>,
}

impl LlmConfig {
    #[allow(clippy::missing_errors_doc)]
    pub fn validate(&self) -> Result<(), crate::CoreError> {
        if self.model.trim().is_empty() {
            return Err(crate::CoreError::Config("model must not be empty".to_string()));
        }
        if let Some(temperature) = self.temperature {
            if !(0.0..=2.0).contains(&temperature) {
                return Err(crate::CoreError::Config(format!(
                    "temperature must be between 0.0 and 2.0, got {temperature}"
                )));
            }
        }
        Ok(())
    }
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
    Timeout,
    Malformed(String),
    AuthFailure,
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMessages => write!(f, "no messages provided"),
            Self::Backend(reason) => write!(f, "backend error: {reason}"),
            Self::Timeout => write!(f, "backend timed out before responding"),
            Self::Malformed(reason) => write!(f, "malformed backend response: {reason}"),
            Self::AuthFailure => write!(f, "backend rejected credentials"),
        }
    }
}

impl std::error::Error for LlmError {}

pub trait LlmProvider {
    #[allow(clippy::missing_errors_doc)]
    fn complete(&self, messages: &[Message]) -> Result<Completion, LlmError>;
}

impl<T: LlmProvider + ?Sized> LlmProvider for Box<T> {
    fn complete(&self, messages: &[Message]) -> Result<Completion, LlmError> {
        self.as_ref().complete(messages)
    }
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

pub struct LocalSentenceLlmProvider;

impl LocalSentenceLlmProvider {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for LocalSentenceLlmProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmProvider for LocalSentenceLlmProvider {
    fn complete(&self, messages: &[Message]) -> Result<Completion, LlmError> {
        if messages.is_empty() {
            return Err(LlmError::EmptyMessages);
        }
        let conversation_text = messages.last().map_or("", |m| m.content.as_str());
        let content_lines: Vec<&str> = conversation_text
            .lines()
            .filter_map(|line| line.split_once(": ").map(|(_, content)| content))
            .collect();
        let combined = if content_lines.is_empty() {
            conversation_text.to_string()
        } else {
            content_lines.join(" ")
        };
        let sentences: Vec<String> = combined
            .split(['.', '!', '?'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| format!("{s}."))
            .collect();
        Ok(Completion::new(sentences.join("\n")))
    }
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

#[cfg(feature = "ollama")]
pub struct OllamaLlmProvider {
    client: reqwest::blocking::Client,
    base_url: String,
    model: String,
}

#[cfg(feature = "ollama")]
impl OllamaLlmProvider {
    #[allow(clippy::missing_errors_doc)]
    pub fn from_config(config: LlmConfig) -> Result<Self, crate::CoreError> {
        config.validate()?;
        Ok(Self {
            client: reqwest::blocking::Client::new(),
            base_url: config.base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
            model: config.model,
        })
    }
}

#[cfg(feature = "ollama")]
const fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

#[cfg(feature = "ollama")]
fn build_chat_request(model: &str, messages: &[Message]) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": messages.iter().map(|m| serde_json::json!({
            "role": role_str(m.role),
            "content": m.content,
        })).collect::<Vec<_>>(),
        "stream": false,
    })
}

#[cfg(feature = "ollama")]
fn parse_chat_response(json: &serde_json::Value) -> Result<Completion, LlmError> {
    json.get("message")
        .and_then(|message| message.get("content"))
        .and_then(serde_json::Value::as_str)
        .map(Completion::new)
        .ok_or_else(|| LlmError::Malformed("missing message.content field".to_string()))
}

#[cfg(feature = "ollama")]
impl LlmProvider for OllamaLlmProvider {
    fn complete(&self, messages: &[Message]) -> Result<Completion, LlmError> {
        if messages.is_empty() {
            return Err(LlmError::EmptyMessages);
        }
        let request = build_chat_request(&self.model, messages);
        let response = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&request)
            .send()
            .map_err(|err| {
                if err.is_timeout() {
                    LlmError::Timeout
                } else {
                    LlmError::Backend(err.to_string())
                }
            })?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(LlmError::AuthFailure);
        }
        if !response.status().is_success() {
            return Err(LlmError::Backend(format!("HTTP {}", response.status())));
        }

        let json: serde_json::Value = response.json().map_err(|err| LlmError::Malformed(err.to_string()))?;
        parse_chat_response(&json)
    }
}

#[cfg(test)]
mod tests {
    use super::{extract_facts, LlmConfig, LlmContractTests, LlmError, LlmProvider, LocalSentenceLlmProvider, Message, Role};
    use crate::test_support::{EchoLlmProvider, FakeLlmProvider};

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
    fn timeout_display_is_sensible() {
        let err = LlmError::Timeout;
        assert_eq!(err.to_string(), "backend timed out before responding");
    }

    #[test]
    fn malformed_display_is_sensible() {
        let err = LlmError::Malformed("unexpected json shape".to_string());
        assert_eq!(
            err.to_string(),
            "malformed backend response: unexpected json shape"
        );
    }

    #[test]
    fn auth_failure_display_is_sensible() {
        let err = LlmError::AuthFailure;
        assert_eq!(err.to_string(), "backend rejected credentials");
    }

    #[test]
    fn fake_provider_passes_happy_path_contract() {
        FakeLlmProvider::new().contract_happy_path();
    }

    #[test]
    fn echo_provider_passes_happy_path_contract() {
        EchoLlmProvider::new().contract_happy_path();
    }

    #[test]
    fn echo_provider_passes_rejects_empty_messages_contract() {
        EchoLlmProvider::new().contract_rejects_empty_messages();
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
    fn fake_provider_returns_timeout() {
        let messages = [Message::new(Role::User, "hello")];
        let result = FakeLlmProvider::timing_out().complete(&messages);
        assert!(
            matches!(result, Err(LlmError::Timeout)),
            "expected a timeout error"
        );
    }

    #[test]
    fn fake_provider_returns_malformed() {
        let messages = [Message::new(Role::User, "hello")];
        let result = FakeLlmProvider::returning_malformed("unexpected json shape").complete(&messages);
        assert!(
            matches!(result, Err(LlmError::Malformed(_))),
            "expected a malformed error"
        );
    }

    #[test]
    fn fake_provider_returns_auth_failure() {
        let messages = [Message::new(Role::User, "hello")];
        let result = FakeLlmProvider::unauthorized().complete(&messages);
        assert!(
            matches!(result, Err(LlmError::AuthFailure)),
            "expected an auth failure error"
        );
    }

    #[test]
    fn contract_complete_is_unmodified_passthrough() {
        let raw = "the sky is blue\nwater is wet";
        let result = FakeLlmProvider::with_response(raw)
            .complete(&[Message::new(Role::User, "ignored")]);
        let completion = result.expect("expected a completion");
        assert_eq!(
            completion.content, raw,
            "raw response must be returned unchanged without wrapping or reformatting"
        );
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

    #[test]
    fn config_with_valid_model_passes_validation() {
        let config = LlmConfig {
            model: "llama3".to_string(),
            base_url: None,
            api_key: None,
            temperature: None,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_with_empty_model_is_rejected() {
        let config = LlmConfig {
            model: String::new(),
            base_url: None,
            api_key: None,
            temperature: None,
        };
        assert!(matches!(config.validate(), Err(crate::CoreError::Config(_))));
    }

    #[test]
    fn config_with_whitespace_only_model_is_rejected() {
        let config = LlmConfig {
            model: "   ".to_string(),
            base_url: None,
            api_key: None,
            temperature: None,
        };
        assert!(matches!(config.validate(), Err(crate::CoreError::Config(_))));
    }

    #[test]
    fn config_with_temperature_in_range_passes_validation() {
        let config = LlmConfig {
            model: "llama3".to_string(),
            base_url: None,
            api_key: None,
            temperature: Some(0.7),
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_with_temperature_out_of_range_is_rejected() {
        let config = LlmConfig {
            model: "llama3".to_string(),
            base_url: None,
            api_key: None,
            temperature: Some(2.5),
        };
        assert!(matches!(config.validate(), Err(crate::CoreError::Config(_))));
    }

    #[test]
    fn config_with_negative_temperature_is_rejected() {
        let config = LlmConfig {
            model: "llama3".to_string(),
            base_url: None,
            api_key: None,
            temperature: Some(-0.1),
        };
        assert!(matches!(config.validate(), Err(crate::CoreError::Config(_))));
    }

    #[test]
    fn local_sentence_provider_splits_conversation_into_one_fact_per_sentence() {
        let provider = LocalSentenceLlmProvider::new();
        let wrapped = [
            Message::new(Role::System, "Extract discrete factual statements from the conversation."),
            Message::new(Role::User, "Conversation:\nUser: Alice is an engineer. Bob lives in Berlin."),
        ];
        let completion = provider.complete(&wrapped).expect("expected a successful completion");
        let facts: Vec<&str> = completion.content.lines().collect();
        assert_eq!(facts, vec!["Alice is an engineer.", "Bob lives in Berlin."]);
    }

    #[test]
    fn local_sentence_provider_rejects_empty_messages() {
        let provider = LocalSentenceLlmProvider::new();
        let result = provider.complete(&[]);
        assert!(matches!(result, Err(LlmError::EmptyMessages)));
    }

    #[test]
    fn local_sentence_provider_passes_happy_path_contract() {
        LocalSentenceLlmProvider::new().contract_happy_path();
    }

    #[test]
    fn local_sentence_provider_passes_rejects_empty_messages_contract() {
        LocalSentenceLlmProvider::new().contract_rejects_empty_messages();
    }

    #[test]
    fn local_sentence_provider_end_to_end_through_extract_facts() {
        let provider = LocalSentenceLlmProvider::new();
        let conversation = [Message::new(Role::User, "Alice is an engineer. Bob lives in Berlin.")];
        let facts = extract_facts(&provider, &conversation).expect("expected facts");
        assert_eq!(facts, vec!["Alice is an engineer.".to_string(), "Bob lives in Berlin.".to_string()]);
    }

    #[cfg(feature = "ollama")]
    mod ollama_tests {
        use super::super::{build_chat_request, parse_chat_response, OllamaLlmProvider};
        use super::*;

        #[test]
        fn build_chat_request_has_the_expected_shape() {
            let messages = [Message::new(Role::System, "be terse"), Message::new(Role::User, "hello")];
            let request = build_chat_request("qwen2.5:0.5b", &messages);
            assert_eq!(request["model"], "qwen2.5:0.5b");
            assert_eq!(request["stream"], false);
            assert_eq!(request["messages"][0]["role"], "system");
            assert_eq!(request["messages"][0]["content"], "be terse");
            assert_eq!(request["messages"][1]["role"], "user");
            assert_eq!(request["messages"][1]["content"], "hello");
        }

        #[test]
        fn parse_chat_response_extracts_message_content() {
            let json = serde_json::json!({"message": {"role": "assistant", "content": "hi there"}});
            let completion = parse_chat_response(&json).expect("expected a completion");
            assert_eq!(completion.content, "hi there");
        }

        #[test]
        fn parse_chat_response_rejects_a_missing_content_field() {
            let json = serde_json::json!({"message": {"role": "assistant"}});
            assert!(matches!(parse_chat_response(&json), Err(LlmError::Malformed(_))));
        }

        #[test]
        fn from_config_rejects_an_invalid_config_before_building_the_client() {
            let config = LlmConfig { model: String::new(), base_url: None, api_key: None, temperature: None };
            assert!(matches!(OllamaLlmProvider::from_config(config), Err(crate::CoreError::Config(_))));
        }

        #[test]
        fn contract_rejects_backend_error_against_an_unreachable_host() {
            let config = LlmConfig {
                model: "qwen2.5:0.5b".to_string(),
                base_url: Some("http://127.0.0.1:1".to_string()),
                api_key: None,
                temperature: None,
            };
            let provider = OllamaLlmProvider::from_config(config).expect("valid config should construct");
            provider.contract_rejects_backend_error();
        }

        #[test]
        fn rejects_empty_messages_without_a_network_call() {
            let config = LlmConfig {
                model: "qwen2.5:0.5b".to_string(),
                base_url: Some("http://127.0.0.1:1".to_string()),
                api_key: None,
                temperature: None,
            };
            let provider = OllamaLlmProvider::from_config(config).expect("valid config should construct");
            assert!(matches!(provider.complete(&[]), Err(LlmError::EmptyMessages)));
        }

        #[test]
        #[ignore = "requires a real Ollama instance reachable at MEMORIA_TEST_OLLAMA_URL"]
        fn real_ollama_extracts_a_sensible_completion() {
            let base_url = std::env::var("MEMORIA_TEST_OLLAMA_URL").unwrap_or_else(|_| "http://192.0.2.1:11434".to_string());
            let config = LlmConfig {
                model: "qwen2.5:0.5b".to_string(),
                base_url: Some(base_url),
                api_key: None,
                temperature: None,
            };
            let provider = OllamaLlmProvider::from_config(config).expect("valid config should construct");
            provider.contract_happy_path();
        }
    }
}
