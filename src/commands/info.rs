//! `ds config` — show the settings the plugin resolved.

use anyhow::Result;
use nu_plugin::{EngineInterface, EvaluatedCall, SimplePluginCommand};
use nu_protocol::{Category, Example, LabeledError, Record, Signature, SyntaxShape, Type, Value};

use crate::DsPlugin;
use crate::config::{ENV_API_KEY, Settings, model_context_limit};
use crate::credential::Credential;

use super::common;

/// Print the effective configuration, and where it comes from.
pub struct ConfigCommand;

impl SimplePluginCommand for ConfigCommand {
    type Plugin = DsPlugin;

    fn name(&self) -> &str {
        "ds config"
    }

    fn description(&self) -> &str {
        "Show the resolved plugin configuration and file locations"
    }

    fn signature(&self) -> Signature {
        Signature::build("ds config")
            .input_output_type(Type::Nothing, Type::Any)
            .named("api-key", SyntaxShape::String, "API key override", None)
            .named("base-url", SyntaxShape::String, "API base URL", None)
            .named("model", SyntaxShape::String, "Model override", Some('m'))
            .named(
                "context-limit",
                SyntaxShape::Int,
                "Context limit override",
                None,
            )
            .category(Category::Experimental)
    }

    fn examples(&self) -> Vec<Example<'_>> {
        vec![Example {
            description: "Show where settings and sessions are stored",
            example: "ds config",
            result: None,
        }]
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
    let (paths, settings) = common::load_settings(call)?;

    let key = common::lookup_api_key(engine, call);
    let (key_present, key_source) = match &key {
        common::ApiKeyLookup::Found(key) => (
            true,
            format!(
                "{} ({})",
                key.source.describe(),
                common::mask_secret(&key.key)
            ),
        ),
        common::ApiKeyLookup::Missing => (false, format!("not set; export $env.{ENV_API_KEY}")),
    };

    let credential = Credential::from_env();
    let api_key_store = credential.backend();
    let api_key_account = credential.account().to_owned();

    let context_window = model_context_limit(&settings.model);
    let compaction_threshold = settings.compact_at(&settings.model);

    let Settings {
        base_url,
        model,
        thinking,
        temperature,
        max_tokens,
        context_limit,
        compact_ratio,
        keep_recent_messages,
        system_prompt,
        nu_bin,
        markdown,
        tools,
        confirm_tool_reads,
        confirm_tool_writes,
        confirm_tool_commands,
        tool_command_timeout_secs,
        max_tool_rounds,
    } = settings;

    Ok(Value::record(
        Record::from_iter([
            ("base_url".to_owned(), Value::string(base_url, span)),
            ("model".to_owned(), Value::string(model, span)),
            ("thinking".to_owned(), Value::string(thinking.label(), span)),
            (
                "temperature".to_owned(),
                temperature.map_or_else(
                    || Value::nothing(span),
                    |value| Value::float(value as f64, span),
                ),
            ),
            (
                "max_tokens".to_owned(),
                max_tokens.map_or_else(
                    || Value::nothing(span),
                    |value| Value::int(value as i64, span),
                ),
            ),
            (
                "context_limit".to_owned(),
                context_limit.map_or_else(
                    || Value::nothing(span),
                    |value| Value::int(value as i64, span),
                ),
            ),
            (
                "context_window".to_owned(),
                Value::int(context_window as i64, span),
            ),
            (
                "compaction_threshold".to_owned(),
                Value::int(compaction_threshold as i64, span),
            ),
            (
                "compact_ratio".to_owned(),
                Value::float(compact_ratio as f64, span),
            ),
            (
                "keep_recent_messages".to_owned(),
                Value::int(keep_recent_messages as i64, span),
            ),
            (
                "system_prompt".to_owned(),
                system_prompt
                    .map_or_else(|| Value::nothing(span), |value| Value::string(value, span)),
            ),
            (
                "nu_bin".to_owned(),
                nu_bin.map_or_else(|| Value::nothing(span), |value| Value::string(value, span)),
            ),
            ("markdown".to_owned(), Value::bool(markdown, span)),
            ("tools".to_owned(), Value::bool(tools, span)),
            (
                "confirm_tool_reads".to_owned(),
                Value::bool(confirm_tool_reads, span),
            ),
            (
                "confirm_tool_writes".to_owned(),
                Value::bool(confirm_tool_writes, span),
            ),
            (
                "confirm_tool_commands".to_owned(),
                Value::bool(confirm_tool_commands, span),
            ),
            (
                "tool_command_timeout_secs".to_owned(),
                Value::int(tool_command_timeout_secs as i64, span),
            ),
            (
                "max_tool_rounds".to_owned(),
                Value::int(max_tool_rounds as i64, span),
            ),
            ("api_key_present".to_owned(), Value::bool(key_present, span)),
            ("api_key_source".to_owned(), Value::string(key_source, span)),
            (
                "api_key_store".to_owned(),
                Value::string(api_key_store, span),
            ),
            (
                "api_key_account".to_owned(),
                Value::string(api_key_account, span),
            ),
            (
                "config_dir".to_owned(),
                Value::string(paths.dir().display().to_string(), span),
            ),
            (
                "settings_file".to_owned(),
                Value::string(paths.settings_file().display().to_string(), span),
            ),
            (
                "sessions_dir".to_owned(),
                Value::string(paths.sessions_dir().display().to_string(), span),
            ),
            (
                "interactive".to_owned(),
                Value::bool(common::is_interactive(engine), span),
            ),
        ]),
        span,
    ))
}
