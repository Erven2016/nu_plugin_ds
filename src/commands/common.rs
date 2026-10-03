//! Helpers shared by the plugin's commands.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use nu_plugin::{EngineInterface, EvaluatedCall};
use nu_protocol::{Record, Span, Value};

use crate::api::{DeepSeekClient, ModelInfo, Usage};
use crate::config::{ConfigPaths, ENV_API_KEY, ENV_BASE_URL, ENV_NU_BIN, Settings, ThinkingEffort};
use crate::credential::Credential;
use crate::session::Session;

/// Flags every command accepts, already merged into [`Settings`].
pub fn load_settings(call: &EvaluatedCall) -> Result<(ConfigPaths, Settings)> {
    let paths = ConfigPaths::discover()?;
    let mut settings = Settings::load(&paths)?;

    if let Some(base_url) = call.get_flag::<String>("base-url")? {
        settings.base_url = base_url;
    }
    if let Some(model) = call.get_flag::<String>("model")? {
        settings.model = model;
    }
    if let Some(context_limit) = call.get_flag::<i64>("context-limit")? {
        settings.context_limit = Some(
            u32::try_from(context_limit)
                .map_err(|_| anyhow!("`--context-limit` must be a positive number of tokens"))?,
        );
    }
    if let Some(temperature) = call.get_flag::<f64>("temperature")? {
        settings.temperature = Some(temperature as f32);
    }

    Ok((paths, settings))
}

/// The thinking level requested on the command line, if any.
pub fn thinking_from_call(call: &EvaluatedCall) -> Result<Option<ThinkingEffort>> {
    let Some(raw) = call.get_flag::<String>("think")? else {
        return Ok(None);
    };
    ThinkingEffort::parse(&raw)
        .map(Some)
        .ok_or_else(|| anyhow!("`--think` must be one of off, low, medium or high (got `{raw}`)"))
}

/// Where a resolved API key came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApiKeySource {
    Flag,
    NushellEnv,
    ProcessEnv,
    CredentialStore,
}

impl ApiKeySource {
    pub fn describe(self) -> &'static str {
        match self {
            ApiKeySource::Flag => "--api-key",
            ApiKeySource::NushellEnv => "the nushell environment",
            ApiKeySource::ProcessEnv => "the process environment",
            ApiKeySource::CredentialStore => "the OS credential store",
        }
    }
}

/// A key plus where it came from.
#[derive(Clone, Debug)]
pub struct ApiKey {
    pub key: String,
    pub source: ApiKeySource,
}

/// The result of looking for a key, without asking the user for one.
#[derive(Clone, Debug)]
pub enum ApiKeyLookup {
    Found(ApiKey),
    Missing,
}

/// Which of the four places a key comes from, in precedence order.
///
/// Blank candidates are ignored, so an empty `$env.DEEPSEEK_API_KEY` does not shadow a
/// stored key.
pub fn select_api_key(
    flag: Option<String>,
    nushell_env: Option<String>,
    process_env: Option<String>,
    stored: Option<String>,
) -> ApiKeyLookup {
    let candidates = [
        (flag, ApiKeySource::Flag),
        (nushell_env, ApiKeySource::NushellEnv),
        (process_env, ApiKeySource::ProcessEnv),
        (stored, ApiKeySource::CredentialStore),
    ];

    for (candidate, source) in candidates {
        if let Some(key) = candidate.filter(|key| !key.trim().is_empty()) {
            return ApiKeyLookup::Found(ApiKey {
                key: key.trim().to_owned(),
                source,
            });
        }
    }

    ApiKeyLookup::Missing
}

/// Look for a key without prompting. Reads the credential store, but never writes it.
pub fn lookup_api_key(engine: &EngineInterface, call: &EvaluatedCall) -> ApiKeyLookup {
    let credential = Credential::from_env();
    let stored = match credential.load() {
        Ok(stored) => stored,
        Err(err) => {
            // A broken credential store must not stop a key that comes from the
            // environment from working.
            eprintln!("nu_plugin_ds: could not read the credential store: {err:#}");
            None
        }
    };

    select_api_key(
        call.get_flag::<String>("api-key").ok().flatten(),
        env_string(engine, ENV_API_KEY),
        std::env::var(ENV_API_KEY).ok(),
        stored,
    )
}

/// The key to talk to the API with, asking for it once when it is not configured anywhere.
///
/// On first use the user is asked to paste the key; it is then stored in the OS credential
/// store so it is never written to a plaintext file.
pub fn resolve_api_key(engine: &EngineInterface, call: &EvaluatedCall) -> Result<ApiKey> {
    if let ApiKeyLookup::Found(key) = lookup_api_key(engine, call) {
        return Ok(key);
    }

    if !is_interactive(engine) {
        bail!(
            "no DeepSeek API key found, and there is no terminal to ask on; set \
             `$env.{ENV_API_KEY}`, pass `--api-key`, or run `ds change-api-key` from an \
             interactive session"
        );
    }

    eprintln!(
        "No DeepSeek API key is configured. Create one at \
         https://platform.deepseek.com/api_keys"
    );
    let key = prompt_secret("Paste your DeepSeek API key:")?;

    let credential = Credential::from_env();
    match credential.store(&key) {
        Ok(()) => eprintln!(
            "nu_plugin_ds: saved to {} (change it later with `ds change-api-key`)",
            credential.backend()
        ),
        // The key still works for this run; losing it is worth a warning, not a failure.
        Err(err) => eprintln!(
            "nu_plugin_ds: could not save the key to {}: {err:#}\n\
             nu_plugin_ds: using it for this session only",
            credential.backend()
        ),
    }

    Ok(ApiKey {
        key,
        source: ApiKeySource::CredentialStore,
    })
}

/// Build a client for an explicit key.
pub fn client_for(
    key: &str,
    engine: &EngineInterface,
    settings: &Settings,
) -> Result<DeepSeekClient> {
    let base_url = env_string(engine, ENV_BASE_URL).unwrap_or_else(|| settings.api_base());
    DeepSeekClient::new(key, base_url)
}

/// Build the API client from wherever the key happens to be.
pub fn build_client(
    engine: &EngineInterface,
    call: &EvaluatedCall,
    settings: &Settings,
) -> Result<DeepSeekClient> {
    let api_key = resolve_api_key(engine, call)?;
    client_for(&api_key.key, engine, settings)
}

/// Read a secret from the terminal without echoing it.
///
/// `Esc` and `Ctrl+C` cancel, `Ctrl+U` clears, `Backspace` deletes, and pasting is
/// accepted. The terminal is always put back, even when reading fails half way through.
pub fn prompt_secret(question: &str) -> Result<String> {
    use crossterm::cursor::MoveToColumn;
    use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
    use crossterm::terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode};

    /// Restores the terminal when it goes out of scope.
    struct Raw;
    impl Drop for Raw {
        fn drop(&mut self) {
            let _ = disable_raw_mode();
        }
    }

    let mut stderr = std::io::stderr();
    writeln!(stderr, "{question}")?;
    write!(stderr, "  (input is hidden, Esc to cancel) ")?;
    stderr.flush()?;

    enable_raw_mode().context("could not switch the terminal to raw mode")?;
    let guard = Raw;
    let mut secret = String::new();

    let result = loop {
        match crossterm::event::read() {
            Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => match key.code {
                KeyCode::Enter => break Ok(()),
                KeyCode::Esc => break Err(anyhow!("no API key was entered")),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    break Err(anyhow!("no API key was entered"));
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    secret.clear();
                }
                KeyCode::Backspace => {
                    secret.pop();
                }
                KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    secret.push(ch);
                }
                _ => {}
            },
            Ok(Event::Paste(text)) => secret.push_str(text.trim()),
            Ok(_) => {}
            Err(err) => break Err(anyhow::Error::new(err).context("could not read the key")),
        }

        // Rewrite the masked value on the current line.
        let masked = "*".repeat(secret.chars().count());
        crossterm::execute!(stderr, MoveToColumn(0), Clear(ClearType::UntilNewLine))?;
        write!(stderr, "  {masked}")?;
        stderr.flush()?;
    };

    drop(guard);
    eprintln!();

    let secret = secret.trim().to_owned();
    result.and_then(|()| {
        if secret.is_empty() {
            bail!("no API key was entered");
        }
        Ok(secret)
    })
}

/// Show just enough of a key to recognise it, never the whole thing.
pub fn mask_secret(key: &str) -> String {
    let key = key.trim();
    if key.len() <= 8 {
        return "****".to_owned();
    }
    format!("{}…{}", &key[..4], &key[key.len() - 4..])
}

/// Read a string from the nushell environment.
pub fn env_string(engine: &EngineInterface, name: &str) -> Option<String> {
    engine
        .get_env_var(name)
        .ok()
        .flatten()
        .and_then(|value| value.as_str().ok().map(str::to_owned))
}

/// Read a path-like environment variable, accepting both a string and a list of
/// strings (nushell exposes `$env.PATH` as a list on every platform).
pub fn env_paths(engine: &EngineInterface, name: &str) -> Vec<PathBuf> {
    let Ok(Some(value)) = engine.get_env_var(name) else {
        return Vec::new();
    };

    match value {
        Value::String { val, .. } => std::env::split_paths(&val).collect(),
        Value::List { vals, .. } => vals
            .iter()
            .filter_map(|value| value.as_str().ok())
            .map(PathBuf::from)
            .collect(),
        _ => Vec::new(),
    }
}

/// Choose the model to start with: an explicit request, then the configured model if
/// the API knows it, then the first model the API offered.
pub fn pick_model(settings: &Settings, models: &[ModelInfo], requested: Option<String>) -> String {
    if let Some(requested) = requested.filter(|model| !model.trim().is_empty()) {
        return requested;
    }
    if models.iter().any(|model| model.id == settings.model) {
        return settings.model.clone();
    }
    models
        .first()
        .map(|model| model.id.clone())
        .unwrap_or_else(|| settings.model.clone())
}

/// Best-effort model lookup used when we cannot reach the API.
pub fn offline_models(settings: &Settings) -> Vec<ModelInfo> {
    vec![ModelInfo {
        id: settings.model.clone(),
        owned_by: None,
        created: None,
    }]
}

/// Locate the `nu` executable used to run commands generated by `cc`.
pub fn nu_binary(
    engine: &EngineInterface,
    call: &EvaluatedCall,
    settings: &Settings,
) -> Result<PathBuf> {
    if let Some(path) = call.get_flag::<String>("nu")? {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = settings.nu_bin.clone() {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = env_string(engine, ENV_NU_BIN) {
        return Ok(PathBuf::from(path));
    }

    let direct: Vec<PathBuf> = env_paths(engine, "PATH");
    let fallback: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect())
        .unwrap_or_default();

    for dir in direct.iter().chain(fallback.iter()) {
        if let Some(found) = find_in_dir(dir, "nu") {
            return Ok(found);
        }
    }

    bail!(
        "could not find a `nu` executable on `$env.PATH`; pass `--nu <path>` or set \
         `nu_bin` in {}",
        ConfigPaths::discover()
            .map(|paths| paths.settings_file().display().to_string())
            .unwrap_or_else(|_| "the plugin settings".to_owned())
    )
}

fn find_in_dir(dir: &Path, name: &str) -> Option<PathBuf> {
    let extensions: &[&str] = if cfg!(windows) {
        &["exe", "cmd", "bat", "ps1"]
    } else {
        &[""]
    };

    for extension in extensions {
        let file_name = if extension.is_empty() {
            name.to_owned()
        } else {
            format!("{name}.{extension}")
        };
        let candidate = dir.join(file_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Sessions need a real terminal for the TUI, and the plugin protocol must not be on
/// stdin/stdout while we drive it.
pub fn terminal_hint(engine: &EngineInterface) -> Option<&'static str> {
    if engine.is_using_stdio() {
        Some(
            "this plugin is talking to nushell over stdin/stdout, so it cannot take over the \
             terminal; re-register it so it can use a local socket: plugin add --force <path to \
             nu_plugin_ds>",
        )
    } else {
        None
    }
}

/// True when the TUI can be started.
pub fn is_interactive(engine: &EngineInterface) -> bool {
    use std::io::IsTerminal;

    !engine.is_using_stdio() && std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

// ------------------------------------------------------------------ values

/// Render a session as the record returned by the commands.
pub fn session_value(session: &Session, span: Span) -> Value {
    let messages = session
        .messages
        .iter()
        .map(|message| {
            Value::record(
                Record::from_iter([
                    (
                        "role".to_owned(),
                        Value::string(message.role.as_str(), span),
                    ),
                    (
                        "content".to_owned(),
                        Value::string(message.content.clone(), span),
                    ),
                ]),
                span,
            )
        })
        .collect::<Vec<_>>();

    Value::record(
        Record::from_iter([
            ("id".to_owned(), Value::string(session.id.clone(), span)),
            ("title".to_owned(), Value::string(session.title(), span)),
            (
                "model".to_owned(),
                Value::string(session.model.clone(), span),
            ),
            (
                "thinking".to_owned(),
                Value::string(session.thinking.label(), span),
            ),
            (
                "created_at".to_owned(),
                Value::string(session.created_at.to_rfc3339(), span),
            ),
            (
                "updated_at".to_owned(),
                Value::string(session.updated_at.to_rfc3339(), span),
            ),
            (
                "turns".to_owned(),
                Value::int(session.user_turns() as i64, span),
            ),
            (
                "compactions".to_owned(),
                Value::int(session.compactions as i64, span),
            ),
            ("usage".to_owned(), usage_value(&session.usage, span)),
            ("messages".to_owned(), Value::list(messages, span)),
        ]),
        span,
    )
}

pub fn usage_value(usage: &Usage, span: Span) -> Value {
    Value::record(
        Record::from_iter([
            (
                "prompt_tokens".to_owned(),
                Value::int(usage.prompt_tokens as i64, span),
            ),
            (
                "completion_tokens".to_owned(),
                Value::int(usage.completion_tokens as i64, span),
            ),
            (
                "total_tokens".to_owned(),
                Value::int(usage.total_tokens as i64, span),
            ),
            (
                "prompt_cache_hit_tokens".to_owned(),
                Value::int(usage.prompt_cache_hit_tokens as i64, span),
            ),
            (
                "prompt_cache_miss_tokens".to_owned(),
                Value::int(usage.prompt_cache_miss_tokens as i64, span),
            ),
        ]),
        span,
    )
}

pub fn model_value(model: &ModelInfo, current: bool, span: Span) -> Value {
    Value::record(
        Record::from_iter([
            ("id".to_owned(), Value::string(model.id.clone(), span)),
            (
                "owned_by".to_owned(),
                model
                    .owned_by
                    .clone()
                    .map_or_else(|| Value::nothing(span), |owner| Value::string(owner, span)),
            ),
            ("active".to_owned(), Value::bool(current, span)),
        ]),
        span,
    )
}

/// Render a session as the record returned by the commands.
/// Wrap a blocking operation so it can be awaited from sync plugin code.
pub fn block_on<T>(future: impl std::future::Future<Output = Result<T>>) -> Result<T> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("could not start the async runtime")?;
    runtime.block_on(future)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored_key(lookup: ApiKeyLookup) -> Option<ApiKey> {
        match lookup {
            ApiKeyLookup::Found(key) => Some(key),
            ApiKeyLookup::Missing => None,
        }
    }

    #[test]
    fn the_first_configured_source_wins() {
        let all = select_api_key(
            Some("from-flag".to_owned()),
            Some("from-nushell".to_owned()),
            Some("from-process".to_owned()),
            Some("from-store".to_owned()),
        );
        let key = stored_key(all).unwrap();
        assert_eq!(key.key, "from-flag");
        assert_eq!(key.source, ApiKeySource::Flag);

        let without_flag = select_api_key(
            None,
            Some("from-nushell".to_owned()),
            Some("from-process".to_owned()),
            Some("from-store".to_owned()),
        );
        let key = stored_key(without_flag).unwrap();
        assert_eq!(key.key, "from-nushell");
        assert_eq!(key.source, ApiKeySource::NushellEnv);

        let only_env_and_store = select_api_key(
            None,
            None,
            Some("from-process".to_owned()),
            Some("from-store".to_owned()),
        );
        let key = stored_key(only_env_and_store).unwrap();
        assert_eq!(key.key, "from-process");
        assert_eq!(key.source, ApiKeySource::ProcessEnv);

        let only_store = select_api_key(None, None, None, Some("from-store".to_owned()));
        let key = stored_key(only_store).unwrap();
        assert_eq!(key.key, "from-store");
        assert_eq!(key.source, ApiKeySource::CredentialStore);
    }

    #[test]
    fn blank_candidates_are_skipped() {
        let lookup = select_api_key(
            Some("   ".to_owned()),
            Some(String::new()),
            Some("\t\n".to_owned()),
            Some("from-store".to_owned()),
        );
        let key = stored_key(lookup).unwrap();
        assert_eq!(key.key, "from-store");
        assert_eq!(key.source, ApiKeySource::CredentialStore);
    }

    #[test]
    fn no_candidates_at_all_is_missing() {
        assert!(stored_key(select_api_key(None, None, None, None)).is_none());
    }

    #[test]
    fn masks_all_but_the_ends() {
        assert_eq!(mask_secret("sk-1234567890abcdef"), "sk-1…cdef");
        assert_eq!(mask_secret("short"), "****");
    }
}
