#![allow(dead_code)]

use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const HASH_LEN: usize = 32;
const SALT_LEN: usize = 16;
const PBKDF2_ITERATIONS: u32 = 600_000;

fn pbkdf2_block(password: &[u8], salt: &[u8], iterations: u32, block_index: u32) -> [u8; HASH_LEN] {
    let mut mac = HmacSha256::new_from_slice(password).expect("HMAC accepts any key length");
    mac.update(salt);
    mac.update(&block_index.to_be_bytes());
    let mut result: [u8; HASH_LEN] = mac.finalize().into_bytes().into();
    let mut previous = result;
    for _ in 1..iterations {
        let mut mac = HmacSha256::new_from_slice(password).expect("HMAC accepts any key length");
        mac.update(&previous);
        let next: [u8; HASH_LEN] = mac.finalize().into_bytes().into();
        for (r, b) in result.iter_mut().zip(next.iter()) {
            *r ^= b;
        }
        previous = next;
    }
    result
}

pub fn pbkdf2_hmac_sha256(password: &[u8], salt: &[u8], iterations: u32, output_len: usize) -> Vec<u8> {
    let mut output = Vec::with_capacity(output_len);
    let mut block_index: u32 = 1;
    while output.len() < output_len {
        let block = pbkdf2_block(password, salt, iterations, block_index);
        let take = (output_len - output.len()).min(HASH_LEN);
        output.extend_from_slice(&block[..take]);
        block_index += 1;
    }
    output
}

pub fn hash_password(password: &str) -> String {
    hash_password_with_iterations(password, PBKDF2_ITERATIONS)
}

fn hash_password_with_iterations(password: &str, iterations: u32) -> String {
    let mut salt = [0u8; SALT_LEN];
    getrandom::fill(&mut salt).expect("the OS random source is available");
    let hash = pbkdf2_hmac_sha256(password.as_bytes(), &salt, iterations, HASH_LEN);
    format!(
        "pbkdf2-sha256${iterations}${}${}",
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, salt),
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, hash),
    )
}

pub fn verify_password(password: &str, stored: &str) -> bool {
    let mut parts = stored.split('$');
    let (Some("pbkdf2-sha256"), Some(iterations_str), Some(salt_b64), Some(hash_b64), None) =
        (parts.next(), parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    let Ok(iterations) = iterations_str.parse::<u32>() else {
        return false;
    };
    let Ok(salt) = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, salt_b64) else {
        return false;
    };
    let Ok(expected_hash) = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, hash_b64) else {
        return false;
    };
    let actual_hash = pbkdf2_hmac_sha256(password.as_bytes(), &salt, iterations, expected_hash.len());
    crate::constant_time_eq(&actual_hash, &expected_hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pbkdf2_matches_a_real_vector_one_iteration() {
        let output = pbkdf2_hmac_sha256(b"password", b"salt", 1, 32);
        assert_eq!(
            hex_encode(&output),
            "120fb6cffcf8b32c43e7225256c4f837a86548c92ccc35480805987cb70be17b"
        );
    }

    #[test]
    fn pbkdf2_matches_a_real_vector_two_iterations() {
        let output = pbkdf2_hmac_sha256(b"password", b"salt", 2, 32);
        assert_eq!(
            hex_encode(&output),
            "ae4d0c95af6b46d32d0adff928f06dd02a303f8ef3c251dfd6e2d85a95474c43"
        );
    }

    #[test]
    fn pbkdf2_matches_a_real_vector_four_thousand_ninety_six_iterations() {
        let output = pbkdf2_hmac_sha256(b"password", b"salt", 4096, 32);
        assert_eq!(
            hex_encode(&output),
            "c5e478d59288c841aa530db6845c4c8d962893a001ce4e11a4963873aa98134a"
        );
    }

    #[test]
    fn pbkdf2_is_deterministic_for_the_same_inputs() {
        let a = pbkdf2_hmac_sha256(b"same", b"salt", 10, 32);
        let b = pbkdf2_hmac_sha256(b"same", b"salt", 10, 32);
        assert_eq!(a, b);
    }

    #[test]
    fn pbkdf2_differs_when_the_salt_differs() {
        let a = pbkdf2_hmac_sha256(b"same", b"salt-a", 10, 32);
        let b = pbkdf2_hmac_sha256(b"same", b"salt-b", 10, 32);
        assert_ne!(a, b);
    }

    #[test]
    fn pbkdf2_differs_when_the_iteration_count_differs() {
        let a = pbkdf2_hmac_sha256(b"same", b"salt", 10, 32);
        let b = pbkdf2_hmac_sha256(b"same", b"salt", 11, 32);
        assert_ne!(a, b);
    }

    #[test]
    fn pbkdf2_respects_an_output_length_longer_than_one_hash_block() {
        let output = pbkdf2_hmac_sha256(b"password", b"salt", 1, 64);
        assert_eq!(output.len(), 64);
    }

    const CHEAP_NON_PRODUCTION_ITERATIONS: u32 = 10;

    #[test]
    fn hash_password_uses_the_real_production_iteration_count() {
        let stored = hash_password("irrelevant");
        let iterations_field = stored.split('$').nth(1).expect("stored hash has an iterations field");
        assert_eq!(iterations_field, PBKDF2_ITERATIONS.to_string());
    }

    #[test]
    fn hash_password_verifies_against_the_correct_password() {
        let stored = hash_password_with_iterations("correct horse battery staple", CHEAP_NON_PRODUCTION_ITERATIONS);
        assert!(verify_password("correct horse battery staple", &stored));
    }

    #[test]
    fn hash_password_rejects_the_wrong_password() {
        let stored = hash_password_with_iterations("correct horse battery staple", CHEAP_NON_PRODUCTION_ITERATIONS);
        assert!(!verify_password("wrong password", &stored));
    }

    #[test]
    fn hash_password_produces_a_different_salt_each_time() {
        let a = hash_password_with_iterations("same password", CHEAP_NON_PRODUCTION_ITERATIONS);
        let b = hash_password_with_iterations("same password", CHEAP_NON_PRODUCTION_ITERATIONS);
        assert_ne!(a, b, "two hashes of the same password must not be identical (random salt)");
    }

    #[test]
    fn verify_password_rejects_a_single_bit_tampered_hash() {
        let mut stored = hash_password_with_iterations("tamper me", CHEAP_NON_PRODUCTION_ITERATIONS);
        let last_char = stored.pop().expect("stored hash is non-empty");
        let tampered_char = if last_char == 'A' { 'B' } else { 'A' };
        stored.push(tampered_char);
        assert!(!verify_password("tamper me", &stored));
    }

    #[test]
    fn verify_password_rejects_a_malformed_stored_value() {
        assert!(!verify_password("anything", "not-a-real-hash"));
        assert!(!verify_password("anything", "pbkdf2-sha256$not-a-number$salt$hash"));
        assert!(!verify_password("anything", "pbkdf2-sha256$1$salt"));
    }

    fn hex_encode(bytes: &[u8]) -> String {
        use std::fmt::Write;
        bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut out, b| {
            write!(out, "{b:02x}").expect("writing to a String never fails");
            out
        })
    }
}
