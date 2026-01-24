fn crate_name() -> &'static str {
    "cli"
}

fn main() {
    println!("{}", crate_name());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_name_is_cli() {
        assert_eq!(crate_name(), "cli");
    }
}
