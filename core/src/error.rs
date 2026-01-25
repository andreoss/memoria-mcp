use std::fmt;

#[derive(Debug)]
pub enum CoreError {
    NotFound(String),
    Validation(String),
    Provider {
        message: String,
        source: Box<dyn std::error::Error>,
    },
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(message) => write!(f, "not found: {message}"),
            Self::Validation(message) => write!(f, "validation error: {message}"),
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
