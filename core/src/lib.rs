#![forbid(unsafe_code)]

pub mod embedding;
pub mod error;
pub mod llm;
pub mod vector_store;

pub use error::CoreError;

#[must_use]
pub const fn placeholder() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_returns_true() {
        assert!(placeholder());
    }
}
