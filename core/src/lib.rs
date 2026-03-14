#![forbid(unsafe_code)]
#![allow(clippy::multiple_crate_versions)]

pub mod embedding;
pub mod entity;
pub mod error;
pub mod filter;
pub mod llm;
pub mod memory;
pub mod reranker;
pub mod vector_store;

pub use error::CoreError;

#[cfg(test)]
pub(crate) mod test_support;

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
