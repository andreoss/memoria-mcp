fn crate_name() -> &'static str {
    "server"
}

fn main() {
    println!("{}", crate_name());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_name_is_server() {
        assert_eq!(crate_name(), "server");
    }
}
