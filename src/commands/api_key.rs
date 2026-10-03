//! `ds change-api-key` — store, replace or remove the DeepSeek API key.

use anyhow::{Context, Result, bail};
use nu_plugin::{EngineInterface, EvaluatedCall, SimplePluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, Record, Signature, Span, SyntaxShape, Type, Value,
};

use crate::DsPlugin;
use crate::credential::Credential;

use super::common;

/// Store, replace or remove the DeepSeek API key in the OS credential store.
pub struct ChangeApiKeyCommand;

impl SimplePluginCommand for ChangeApiKeyCommand {
    type Plugin = DsPlugin;

    fn name(&self) -> &str {
        "ds change-api-key"
    }

    fn description(&self) -> &str {
        "Store, replace or remove the DeepSeek API key"
    }

    fn extra_description(&self) -> &str {
        r#"The key is saved in the operating system's credential store (Windows Credential
Manager, the macOS Keychain, or the Linux Secret Service) and is never written to
`settings.json` or any other plaintext file. Commands read it back from there when both
`--api-key` and `$env.DEEPSEEK_API_KEY` are unset.

A newly stored key is checked against `GET /models` unless `--no-verify` is passed; a
failed check is reported but the key stays stored. On systems without a usable credential
store the environment variable `DEEPSEEK_API_KEY` is the fallback."#
    }

    fn signature(&self) -> Signature {
        Signature::build("ds change-api-key")
            .input_output_type(Type::Nothing, Type::Any)
            .named(
                "api-key",
                SyntaxShape::String,
                "Store this key instead of prompting for it (visible in the shell history)",
                None,
            )
            .switch("delete", "Remove the stored key", Some('d'))
            .switch(
                "no-verify",
                "Do not check the new key against the API",
                None,
            )
            .named(
                "base-url",
                SyntaxShape::String,
                "API base URL used for the check",
                None,
            )
            .category(Category::Experimental)
    }

    fn examples(&self) -> Vec<Example<'_>> {
        vec![
            Example {
                description: "Ask for the key and store it",
                example: "ds change-api-key",
                result: None,
            },
            Example {
                description: "Remove the stored key",
                example: "ds change-api-key --delete",
                result: None,
            },
        ]
    }

    fn run(
        &self,
        _plugin: &DsPlugin,
        engine: &EngineInterface,
        call: &EvaluatedCall,
        _input: &Value,
    ) -> Result<Value, LabeledError> {
        let span = call.head;
        run(engine, call).map_err(|err| crate::error::labeled(err, span))
    }
}

fn run(engine: &EngineInterface, call: &EvaluatedCall) -> Result<Value> {
    let span = call.head;
    let credential = Credential::from_env();

    if call.has_flag("delete")? {
        let had = credential.load().ok().flatten().is_some();
        // Removing a missing entry is not an error.
        credential.delete()?;

        let message = if had {
            format!("removed the stored key from {}", credential.backend())
        } else {
            format!("there was no stored key in {}", credential.backend())
        };
        return Ok(record_value(
            false,
            Value::nothing(span),
            message,
            &credential,
            span,
        ));
    }

    let key = match call.get_flag::<String>("api-key")? {
        Some(key) if !key.trim().is_empty() => key.trim().to_owned(),
        _ => {
            if !common::is_interactive(engine) {
                bail!("no terminal to prompt on; pass --api-key");
            }
            common::prompt_secret("Paste your DeepSeek API key:")?
        }
    };

    // Persisting the key is this command's whole job, so a failure here is fatal.
    credential
        .store(&key)
        .with_context(|| format!("could not save the key to {}", credential.backend()))?;

    // A failed check is worth reporting, but the key has already been stored successfully.
    let check = if call.has_flag("no-verify")? {
        None
    } else {
        Some(verify(&key, engine, call))
    };

    let masked = common::mask_secret(&key);
    let backend = credential.backend();
    let (verified, message) = match &check {
        None => (
            Value::nothing(span),
            format!("stored {masked} in {backend} (not checked against the API)"),
        ),
        Some(Ok(models)) => (
            Value::bool(true, span),
            format!("stored {masked} in {backend} and verified it against {models} models"),
        ),
        Some(Err(err)) => (
            Value::bool(false, span),
            format!("stored {masked} in {backend}, but the check failed: {err:#}"),
        ),
    };

    Ok(record_value(true, verified, message, &credential, span))
}

/// Check a key by listing the models it can see. Any failure is a check failure, not a
/// reason to reject the key.
fn verify(key: &str, engine: &EngineInterface, call: &EvaluatedCall) -> Result<usize> {
    let (_, settings) = common::load_settings(call)?;
    let client = common::client_for(key, engine, &settings)?;
    let models = common::block_on(client.list_models())?;
    Ok(models.len())
}

/// Render the record the command returns.
fn record_value(
    stored: bool,
    verified: Value,
    message: String,
    credential: &Credential,
    span: Span,
) -> Value {
    Value::record(
        Record::from_iter([
            ("stored".to_owned(), Value::bool(stored, span)),
            (
                "account".to_owned(),
                Value::string(credential.account().to_owned(), span),
            ),
            (
                "backend".to_owned(),
                Value::string(credential.backend(), span),
            ),
            ("verified".to_owned(), verified),
            ("message".to_owned(), Value::string(message, span)),
        ]),
        span,
    )
}
