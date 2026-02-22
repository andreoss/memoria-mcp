#![allow(dead_code)]

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

pub fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    digest.iter().fold(String::with_capacity(digest.len() * 2), |mut out, b| {
        use std::fmt::Write;
        write!(out, "{b:02x}").expect("writing to a String never fails");
        out
    })
}

const HASH_LEN: usize = 32;
const SALT_LEN: usize = 16;
const PBKDF2_ITERATIONS: u32 = 600_000;

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; HASH_LEN] {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(message);
    mac.finalize().into_bytes().into()
}

fn pbkdf2_block(password: &[u8], salt: &[u8], iterations: u32, block_index: u32) -> [u8; HASH_LEN] {
    let mut first_message = salt.to_vec();
    first_message.extend_from_slice(&block_index.to_be_bytes());
    let mut result = hmac_sha256(password, &first_message);
    let mut previous = result;
    for _ in 1..iterations {
        let next = hmac_sha256(password, &previous);
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

pub fn hash_password_with_iterations(password: &str, iterations: u32) -> String {
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

#[derive(Debug, PartialEq, Eq)]
pub enum JwtError {
    Malformed,
    UnsupportedAlgorithm,
    BadSignature,
    Expired,
}

fn base64url_encode(bytes: &[u8]) -> String {
    base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, bytes)
}

fn base64url_decode(s: &str) -> Result<Vec<u8>, JwtError> {
    base64::Engine::decode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, s).map_err(|_| JwtError::Malformed)
}

pub fn encode_jwt(claims: &serde_json::Value, secret: &[u8]) -> String {
    let header = serde_json::json!({"alg": "HS256", "typ": "JWT"});
    let header_b64 = base64url_encode(&serde_json::to_vec(&header).expect("a static JSON object always serializes"));
    let payload_b64 = base64url_encode(&serde_json::to_vec(claims).expect("caller-provided claims must be valid JSON"));
    let signing_input = format!("{header_b64}.{payload_b64}");
    let signature = hmac_sha256(secret, signing_input.as_bytes());
    let signature_b64 = base64url_encode(&signature);
    format!("{signing_input}.{signature_b64}")
}

pub fn decode_jwt(token: &str, secret: &[u8], now_unix: u64) -> Result<serde_json::Value, JwtError> {
    let mut parts = token.split('.');
    let (Some(header_b64), Some(payload_b64), Some(signature_b64), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(JwtError::Malformed);
    };

    let header_bytes = base64url_decode(header_b64)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes).map_err(|_| JwtError::Malformed)?;
    if header.get("alg").and_then(serde_json::Value::as_str) != Some("HS256") {
        return Err(JwtError::UnsupportedAlgorithm);
    }

    let signing_input = format!("{header_b64}.{payload_b64}");
    let expected_signature = hmac_sha256(secret, signing_input.as_bytes());
    let actual_signature = base64url_decode(signature_b64)?;
    if !crate::constant_time_eq(&expected_signature, &actual_signature) {
        return Err(JwtError::BadSignature);
    }

    let payload_bytes = base64url_decode(payload_b64)?;
    let claims: serde_json::Value = serde_json::from_slice(&payload_bytes).map_err(|_| JwtError::Malformed)?;
    if let Some(exp) = claims.get("exp").and_then(serde_json::Value::as_u64) {
        if exp < now_unix {
            return Err(JwtError::Expired);
        }
    }
    Ok(claims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_hex_matches_the_real_empty_string_vector() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn sha256_hex_matches_a_real_known_vector() {
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn sha256_hex_is_deterministic() {
        assert_eq!(sha256_hex(b"same input"), sha256_hex(b"same input"));
    }

    #[test]
    fn sha256_hex_differs_for_different_input() {
        assert_ne!(sha256_hex(b"input a"), sha256_hex(b"input b"));
    }

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

    #[test]
    fn decode_jwt_accepts_a_real_token_from_an_independent_implementation() {
        let token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEiLCJleHAiOjIwMDAwMDAwMDAsImlhdCI6MTcwMDAwMDAwMH0.iH-RmEk064svZjjCuO1PCOL6k0OqmFSF-4EJx8nroBA";
        let claims = decode_jwt(token, b"test-secret-key", 1_800_000_000).expect("a real, untampered HS256 token must decode");
        assert_eq!(claims["sub"], "user-1");
        assert_eq!(claims["exp"], 2_000_000_000);
        assert_eq!(claims["iat"], 1_700_000_000);
    }

    #[test]
    fn decode_jwt_rejects_the_same_token_under_the_wrong_secret() {
        let token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEiLCJleHAiOjIwMDAwMDAwMDAsImlhdCI6MTcwMDAwMDAwMH0.iH-RmEk064svZjjCuO1PCOL6k0OqmFSF-4EJx8nroBA";
        assert_eq!(decode_jwt(token, b"wrong-secret", 1_800_000_000), Err(JwtError::BadSignature));
    }

    #[test]
    fn decode_jwt_rejects_the_same_token_once_expired() {
        let token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEiLCJleHAiOjIwMDAwMDAwMDAsImlhdCI6MTcwMDAwMDAwMH0.iH-RmEk064svZjjCuO1PCOL6k0OqmFSF-4EJx8nroBA";
        assert_eq!(decode_jwt(token, b"test-secret-key", 2_100_000_000), Err(JwtError::Expired));
    }

    #[test]
    fn decode_jwt_rejects_a_middle_character_of_the_signature_being_flipped() {
        let mut token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEiLCJleHAiOjIwMDAwMDAwMDAsImlhdCI6MTcwMDAwMDAwMH0.iH-RmEk064svZjjCuO1PCOL6k0OqmFSF-4EJx8nroBA".to_string();
        let tamper_index = token.len() / 2;
        let mut bytes: Vec<u8> = token.into_bytes();
        let tampered_char = if bytes[tamper_index] == b'A' { b'B' } else { b'A' };
        bytes[tamper_index] = tampered_char;
        token = String::from_utf8(bytes).expect("ASCII base64url stays valid UTF-8 after a single-byte swap");
        assert_eq!(decode_jwt(&token, b"test-secret-key", 1_800_000_000), Err(JwtError::BadSignature));
    }

    #[test]
    fn decode_jwt_rejects_the_last_character_of_the_signature_being_flipped() {
        let mut token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEiLCJleHAiOjIwMDAwMDAwMDAsImlhdCI6MTcwMDAwMDAwMH0.iH-RmEk064svZjjCuO1PCOL6k0OqmFSF-4EJx8nroBA".to_string();
        let last_char = token.pop().expect("token is non-empty");
        let tampered_char = if last_char == 'A' { 'B' } else { 'A' };
        token.push(tampered_char);
        assert!(decode_jwt(&token, b"test-secret-key", 1_800_000_000).is_err());
    }

    #[test]
    fn decode_jwt_rejects_a_malformed_token() {
        assert_eq!(decode_jwt("not-a-jwt", b"secret", 0), Err(JwtError::Malformed));
        assert_eq!(decode_jwt("a.b", b"secret", 0), Err(JwtError::Malformed));
        assert_eq!(decode_jwt("a.b.c.d", b"secret", 0), Err(JwtError::Malformed));
    }

    #[test]
    fn decode_jwt_rejects_an_unsupported_algorithm() {
        let alg_none_header = base64url_encode(br#"{"alg":"none","typ":"JWT"}"#);
        let payload = base64url_encode(br#"{"sub":"x"}"#);
        let token = format!("{alg_none_header}.{payload}.");
        assert_eq!(decode_jwt(&token, b"secret", 0), Err(JwtError::UnsupportedAlgorithm));
    }

    #[test]
    fn encode_jwt_then_decode_jwt_round_trips_real_claims() {
        let claims = serde_json::json!({"sub": "round-trip", "exp": 3_000_000_000u64});
        let token = encode_jwt(&claims, b"a-real-secret");
        let decoded = decode_jwt(&token, b"a-real-secret", 1_000_000_000).expect("a freshly encoded token must decode");
        assert_eq!(decoded, claims);
    }

    #[test]
    fn encode_jwt_output_is_rejected_under_a_different_secret() {
        let claims = serde_json::json!({"sub": "x"});
        let token = encode_jwt(&claims, b"secret-a");
        assert_eq!(decode_jwt(&token, b"secret-b", 0), Err(JwtError::BadSignature));
    }

    fn hex_encode(bytes: &[u8]) -> String {
        use std::fmt::Write;
        bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut out, b| {
            write!(out, "{b:02x}").expect("writing to a String never fails");
            out
        })
    }
}
