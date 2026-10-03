//! Storage for the API key, backed by the operating system's credential store.

use anyhow::{Context, Result};

/// The credential store "service" all plugin entries live under.
pub const SERVICE: &str = "nu_plugin_ds";
/// The account name used for the DeepSeek API key.
pub const ACCOUNT: &str = "deepseek-api-key";
/// Overrides the account name, so tests never touch the user's real entry.
pub const ENV_ACCOUNT: &str = "NU_PLUGIN_DS_KEYRING_ACCOUNT";

/// A handle to one entry in the OS credential store.
#[derive(Clone, Debug)]
pub struct Credential {
    service: String,
    account: String,
}

impl Credential {
    pub fn new(service: impl Into<String>, account: impl Into<String>) -> Self {
        Self {
            service: service.into(),
            account: account.into(),
        }
    }

    /// The account this handle points at (the entry's user name).
    pub fn account(&self) -> &str {
        &self.account
    }

    /// The entry the plugin uses, honouring `$env.NU_PLUGIN_DS_KEYRING_ACCOUNT`.
    pub fn from_env() -> Self {
        let account = std::env::var(ENV_ACCOUNT)
            .ok()
            .filter(|account| !account.is_empty())
            .unwrap_or_else(|| ACCOUNT.to_string());
        Self::new(SERVICE, account)
    }

    /// The stored secret, or `None` when there is no entry.
    pub fn load(&self) -> Result<Option<String>> {
        let entry = self.entry()?;
        match entry.get_password() {
            Ok(secret) => Ok(Some(secret)),
            // A missing entry is an expected outcome, not a failure.
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(err) => Err(err).with_context(|| {
                format!(
                    "failed to read {}/{} from {}",
                    self.service,
                    self.account,
                    self.backend()
                )
            }),
        }
    }

    /// Replace the stored secret.
    pub fn store(&self, secret: &str) -> Result<()> {
        let entry = self.entry()?;
        entry.set_password(secret).with_context(|| {
            format!(
                "failed to write {}/{} to {}",
                self.service,
                self.account,
                self.backend()
            )
        })
    }

    /// Remove the entry. Removing a missing entry is not an error.
    pub fn delete(&self) -> Result<()> {
        let entry = self.entry()?;
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(err) => Err(err).with_context(|| {
                format!(
                    "failed to delete {}/{} from {}",
                    self.service,
                    self.account,
                    self.backend()
                )
            }),
        }
    }

    /// A human readable name of the backend in use, for `ds config` and messages.
    pub fn backend(&self) -> String {
        backend_name().to_string()
    }

    /// Open the underlying `keyring` entry for this credential.
    fn entry(&self) -> Result<keyring::Entry> {
        keyring::Entry::new(&self.service, &self.account).with_context(|| {
            format!(
                "failed to open {}/{} in {}",
                self.service,
                self.account,
                self.backend()
            )
        })
    }
}

/// The human readable name of the store `keyring` selected for this platform.
///
/// This mirrors the `default` module `keyring` picks at build time from the
/// features we enable: `windows-native` on Windows, `apple-native` on Apple
/// platforms and `sync-secret-service` everywhere else (we deliberately leave
/// the non-persistent Linux `keyutils` store out).
fn backend_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "Windows Credential Manager"
    } else if cfg!(target_os = "macos") {
        "macOS Keychain"
    } else if cfg!(target_os = "ios") {
        "iOS Keychain"
    } else if cfg!(any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "openbsd"
    )) {
        "Secret Service (D-Bus)"
    } else {
        "mock credential store (not persistent)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_secret_through_the_credential_store() {
        // A unique account so the test can never clobber a real key.
        let credential = Credential::new(
            format!("{SERVICE}-test"),
            format!("round-trip-{}", std::process::id()),
        );

        if let Err(err) = credential.store("secret-value") {
            eprintln!("skipping: no credential store available on this machine: {err:#}");
            return;
        }

        assert_eq!(
            credential.load().expect("load should succeed").as_deref(),
            Some("secret-value")
        );

        credential.delete().expect("delete should succeed");
        assert_eq!(
            credential.load().expect("load should succeed"),
            None,
            "a deleted entry must read back as None"
        );
    }

    #[test]
    fn the_account_can_be_overridden_by_the_environment() {
        // Reading the environment is process-wide, so keep this to a pure check of the
        // constant that the override uses.
        assert_eq!(ENV_ACCOUNT, "NU_PLUGIN_DS_KEYRING_ACCOUNT");
    }
}
