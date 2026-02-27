#![forbid(unsafe_code)]

pub mod embedding;
pub mod error;
pub mod filter;
pub mod llm;
pub mod memory;
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
