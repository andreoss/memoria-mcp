#[must_use]
pub fn placeholder() -> bool {
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
