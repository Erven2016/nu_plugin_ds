//! Encryption for the stored sessions.
//!
//! A session file is JSON the model can be reminded of in full, so it is encrypted at rest
//! with ChaCha20-Poly1305 (from `ring`) under a 256-bit key kept in the operating system's
//! credential store. The format is
//!
//! ```text
//! "nupds-enc-v1\n" || nonce (12 bytes) || ciphertext+tag
//! ```
//!
//! A file that does not start with that marker is read as plaintext JSON, so sessions
//! written before encryption existed (or on a machine with no credential store) still load.
//!
//! Performance: the key is read once per process and cached, and ChaCha20-Poly1305 runs at
//! gigabytes per second, so encrypting a multi-megabyte session on every message costs about
//! a millisecond and never stalls the interface.

use std::sync::OnceLock;

use anyhow::{Result, anyhow, bail};
use ring::aead::{Aad, CHACHA20_POLY1305, LessSafeKey, Nonce, UnboundKey};
use ring::rand::{SecureRandom, SystemRandom};

use crate::credential::{Credential, SERVICE};

/// A 256-bit session key.
pub type Key = [u8; 32];

/// Marks an encrypted session file.
const MAGIC: &[u8] = b"nupds-enc-v1\n";
/// The nonce length ChaCha20-Poly1305 uses.
const NONCE_LEN: usize = 12;
/// Account suffix under which the session key is stored in the credential store.
const KEY_ACCOUNT_SUFFIX: &str = "-session-key";
/// Environment override holding the key as 64 hex characters.
///
/// Mainly for tests and for pinning a key in a scripted setup; it takes precedence over the
/// credential store.
pub const ENV_KEY: &str = "NU_PLUGIN_DS_SESSION_KEY";

/// Whether `blob` is one of our encrypted sessions.
pub fn is_encrypted(blob: &[u8]) -> bool {
    blob.starts_with(MAGIC)
}

/// Encrypt `plaintext`, returning `MAGIC || nonce || ciphertext`.
pub fn encrypt(key: &Key, plaintext: &[u8]) -> Result<Vec<u8>> {
    let mut nonce = [0u8; NONCE_LEN];
    SystemRandom::new()
        .fill(&mut nonce)
        .map_err(|_| anyhow!("could not gather randomness for the session nonce"))?;

    let mut in_out = plaintext.to_vec();
    less_safe_key(key)?
        .seal_in_place_append_tag(
            Nonce::assume_unique_for_key(nonce),
            Aad::empty(),
            &mut in_out,
        )
        .map_err(|_| anyhow!("could not encrypt the session"))?;

    let mut out = Vec::with_capacity(MAGIC.len() + NONCE_LEN + in_out.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&in_out);
    Ok(out)
}

/// Decrypt a blob produced by [`encrypt`].
pub fn decrypt(key: &Key, blob: &[u8]) -> Result<Vec<u8>> {
    let rest = blob
        .strip_prefix(MAGIC)
        .ok_or_else(|| anyhow!("not an encrypted session"))?;
    if rest.len() < NONCE_LEN {
        bail!("the encrypted session is truncated");
    }
    let (nonce, ciphertext) = rest.split_at(NONCE_LEN);
    let nonce: [u8; NONCE_LEN] = nonce.try_into().expect("split at NONCE_LEN");

    let mut in_out = ciphertext.to_vec();
    let plain = less_safe_key(key)?
        .open_in_place(
            Nonce::assume_unique_for_key(nonce),
            Aad::empty(),
            &mut in_out,
        )
        .map_err(|_| anyhow!("could not decrypt the session (wrong key?)"))?;
    Ok(plain.to_vec())
}

fn less_safe_key(key: &Key) -> Result<LessSafeKey> {
    let unbound = UnboundKey::new(&CHACHA20_POLY1305, key)
        .map_err(|_| anyhow!("the session key has the wrong length"))?;
    Ok(LessSafeKey::new(unbound))
}

static KEY: OnceLock<Option<Key>> = OnceLock::new();

/// The session key, resolved once per process and cached.
///
/// Returns `None` when sessions should stay plaintext: no key could be found or created.
pub fn load_key() -> Option<Key> {
    KEY.get_or_init(resolve_key).to_owned()
}

fn resolve_key() -> Option<Key> {
    if let Some(key) = key_from_env() {
        return Some(key);
    }
    // Unit tests must not touch the machine's credential store, so they stay plaintext
    // unless they set the environment override.
    if cfg!(test) {
        return None;
    }

    let credential = session_credential();
    match credential.load() {
        Ok(Some(stored)) => match parse_hex(&stored) {
            Some(key) => Some(key),
            None => {
                eprintln!(
                    "nu_plugin_ds: the stored session key is malformed; sessions will not be encrypted"
                );
                None
            }
        },
        Ok(None) => generate_key(&credential),
        Err(err) => {
            eprintln!(
                "nu_plugin_ds: could not read the session key ({err:#}); sessions will not be encrypted"
            );
            None
        }
    }
}

fn key_from_env() -> Option<Key> {
    parse_hex(std::env::var(ENV_KEY).ok()?.trim())
}

/// The credential entry the session key lives in. It hangs off the API key's account, so the
/// test override that isolates the API key isolates this too.
fn session_credential() -> Credential {
    let base = Credential::from_env();
    Credential::new(SERVICE, format!("{}{KEY_ACCOUNT_SUFFIX}", base.account()))
}

fn generate_key(credential: &Credential) -> Option<Key> {
    let mut key = [0u8; 32];
    if SystemRandom::new().fill(&mut key).is_err() {
        eprintln!("nu_plugin_ds: could not gather randomness; sessions will not be encrypted");
        return None;
    }
    if let Err(err) = credential.store(&to_hex(&key)) {
        eprintln!(
            "nu_plugin_ds: could not store the session key ({err:#}); sessions will not be encrypted"
        );
        return None;
    }
    Some(key)
}

fn parse_hex(raw: &str) -> Option<Key> {
    if raw.len() != 64 {
        return None;
    }
    let mut key = [0u8; 32];
    for (index, byte) in key.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&raw[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(key)
}

fn to_hex(key: &Key) -> String {
    let mut out = String::with_capacity(64);
    for byte in key {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: Key = [7u8; 32];

    #[test]
    fn a_round_trip_returns_the_original_bytes() {
        let plaintext = b"{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
        let blob = encrypt(&KEY, plaintext).unwrap();
        assert!(is_encrypted(&blob));
        assert_ne!(
            &blob[MAGIC.len()..],
            plaintext,
            "the payload must be hidden"
        );
        assert_eq!(decrypt(&KEY, &blob).unwrap(), plaintext);
    }

    #[test]
    fn each_encryption_uses_a_fresh_nonce() {
        let a = encrypt(&KEY, b"same").unwrap();
        let b = encrypt(&KEY, b"same").unwrap();
        assert_ne!(a, b, "the nonce must differ so the ciphertext differs");
    }

    #[test]
    fn a_wrong_key_cannot_decrypt() {
        let blob = encrypt(&KEY, b"secret").unwrap();
        let mut other = KEY;
        other[0] ^= 0xff;
        assert!(decrypt(&other, &blob).is_err());
    }

    #[test]
    fn a_tampered_ciphertext_is_rejected() {
        let mut blob = encrypt(&KEY, b"secret").unwrap();
        let last = blob.len() - 1;
        blob[last] ^= 0x01;
        assert!(
            decrypt(&KEY, &blob).is_err(),
            "the tag must reject tampering"
        );
    }

    #[test]
    fn plaintext_is_not_mistaken_for_ciphertext() {
        assert!(!is_encrypted(b"{\"id\":\"abc\"}"));
    }

    #[test]
    fn hex_round_trips() {
        let key = [0x00, 0x1f, 0xab, 0xff]
            .into_iter()
            .chain([0u8; 28])
            .collect::<Vec<_>>();
        let key: Key = key.try_into().unwrap();
        assert_eq!(parse_hex(&to_hex(&key)), Some(key));
        assert_eq!(parse_hex("not hex"), None);
        assert_eq!(parse_hex("abcd"), None, "too short must be rejected");
    }
}
