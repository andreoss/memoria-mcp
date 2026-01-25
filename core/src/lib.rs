#![forbid(unsafe_code)]

pub mod embedding;
pub mod llm;

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
