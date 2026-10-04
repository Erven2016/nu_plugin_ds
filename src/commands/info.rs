//! `ds config` — show the settings the plugin resolved, or read and change one of them.

use anyhow::{Result, anyhow, bail};
use nu_plugin::{EngineInterface, EvaluatedCall, SimplePluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, Record, Signature, Span, SyntaxShape, Type, Value,
};

use crate::DsPlugin;
use crate::config::{ENV_API_KEY, Settings, ThinkingEffort, model_context_limit};
use crate::credential::Credential;

use super::common;

/// Print the effective configuration, and where it comes from. A positional argument reads
/// or writes a single setting instead.
pub struct ConfigCommand;

impl SimplePluginCommand for ConfigCommand {
    type Plugin = DsPlugin;

    fn name(&self) -> &str {
        "ds config"
    }

    fn description(&self) -> &str {
        "Show the resolved configuration, or read and change one setting"
    }

    fn signature(&self) -> Signature {
        Signature::build("ds config")
            .input_output_type(Type::Nothing, Type::Any)
            .optional(
                "setting",
                SyntaxShape::String,
                "Show this setting instead of the whole configuration",
            )
            .optional(
                "value",
                SyntaxShape::String,
                "Set the setting to this value, e.g. `ds config thinking high`",
            )
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
        vec![
            Example {
                description: "Show where settings and sessions are stored",
                example: "ds config",
                result: None,
            },
            Example {
                description: "Read one setting, and see the values it accepts",
                example: "ds config thinking",
                result: None,
            },
            Example {
                description: "Change the reasoning level used for new sessions",
                example: "ds config thinking high",
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

/// A setting `ds config` can read and write: its name, the fixed values it accepts (empty
/// when the value is free-form) and a one-line description.
struct SettingSpec {
    name: &'static str,
    options: &'static [&'static str],
    help: &'static str,
}

/// Every writable setting, in the order they are shown. Anything not listed here is read
/// only (the derived values like `context_window`, or the environment it runs in).
const SETTING_SPECS: &[SettingSpec] = &[
    SettingSpec {
        name: "base_url",
        options: &[],
        help: "API base URL",
    },
    SettingSpec {
        name: "model",
        options: &[],
        help: "model used when a session does not pin one",
    },
    SettingSpec {
        name: "thinking",
        options: &["off", "low", "high", "max"],
        help: "reasoning level for new sessions",
    },
    SettingSpec {
        name: "temperature",
        options: &[],
        help: "sampling temperature, or `none`",
    },
    SettingSpec {
        name: "max_tokens",
        options: &[],
        help: "cap on generated tokens, or `none`",
    },
    SettingSpec {
        name: "context_limit",
        options: &[],
        help: "local token budget, or `none` (at least 1024)",
    },
    SettingSpec {
        name: "compact_ratio",
        options: &[],
        help: "fraction of the budget that triggers compaction (0-1)",
    },
    SettingSpec {
        name: "keep_recent_messages",
        options: &[],
        help: "recent messages kept verbatim when compacting",
    },
    SettingSpec {
        name: "system_prompt",
        options: &[],
        help: "system prompt for new sessions, or `none`",
    },
    SettingSpec {
        name: "nu_bin",
        options: &[],
        help: "path to the `nu` executable, or `none`",
    },
    SettingSpec {
        name: "markdown",
        options: &["true", "false"],
        help: "render answers as markdown",
    },
    SettingSpec {
        name: "tools",
        options: &["true", "false"],
        help: "offer the file tools to the model",
    },
    SettingSpec {
        name: "confirm_tool_reads",
        options: &["true", "false"],
        help: "confirm before reading a file",
    },
    SettingSpec {
        name: "confirm_tool_writes",
        options: &["true", "false"],
        help: "confirm before writing a file",
    },
    SettingSpec {
        name: "confirm_tool_commands",
        options: &["true", "false"],
        help: "confirm before running a command",
    },
    SettingSpec {
        name: "tool_command_timeout_secs",
        options: &[],
        help: "seconds a command may run",
    },
    SettingSpec {
        name: "max_tool_rounds",
        options: &[],
        help: "tool rounds allowed per turn",
    },
];

fn spec(name: &str) -> Option<&'static SettingSpec> {
    SETTING_SPECS.iter().find(|spec| spec.name == name)
}

fn run(engine: &EngineInterface, call: &EvaluatedCall) -> Result<Value> {
    let span = call.head;
    let (paths, mut settings) = common::load_settings(call)?;

    if let Some(name) = call.opt::<String>(0)? {
        let name = name.trim().to_owned();
        let Some(spec) = spec(&name) else {
            return Err(unknown_setting(&name));
        };
        return match call.opt::<String>(1)? {
            Some(value) => {
                apply(&mut settings, spec, &value)?;
                settings.save(&paths)?;
                Ok(setting_record(spec, &settings, span, true))
            }
            None => Ok(setting_record(spec, &settings, span, false)),
        };
    }

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

    let record = vec![
        setting_pair(&settings, "base_url", span),
        setting_pair(&settings, "model", span),
        setting_pair(&settings, "thinking", span),
        setting_pair(&settings, "temperature", span),
        setting_pair(&settings, "max_tokens", span),
        setting_pair(&settings, "context_limit", span),
        (
            "context_window".to_owned(),
            Value::int(model_context_limit(&settings.model) as i64, span),
        ),
        (
            "compaction_threshold".to_owned(),
            Value::int(settings.compact_at(&settings.model) as i64, span),
        ),
        setting_pair(&settings, "compact_ratio", span),
        setting_pair(&settings, "keep_recent_messages", span),
        setting_pair(&settings, "system_prompt", span),
        setting_pair(&settings, "nu_bin", span),
        setting_pair(&settings, "markdown", span),
        setting_pair(&settings, "tools", span),
        setting_pair(&settings, "confirm_tool_reads", span),
        setting_pair(&settings, "confirm_tool_writes", span),
        setting_pair(&settings, "confirm_tool_commands", span),
        setting_pair(&settings, "tool_command_timeout_secs", span),
        setting_pair(&settings, "max_tool_rounds", span),
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
    ];

    Ok(Value::record(Record::from_iter(record), span))
}

fn setting_pair(settings: &Settings, name: &str, span: Span) -> (String, Value) {
    (name.to_owned(), setting_value(settings, name, span))
}

/// The current value of one setting, typed as nushell would want it.
fn setting_value(settings: &Settings, name: &str, span: Span) -> Value {
    match name {
        "base_url" => Value::string(settings.base_url.clone(), span),
        "model" => Value::string(settings.model.clone(), span),
        "thinking" => Value::string(settings.thinking.label(), span),
        "temperature" => optional_value(settings.temperature, span, |value| {
            Value::float(value as f64, span)
        }),
        "max_tokens" => optional_value(settings.max_tokens, span, |value| {
            Value::int(value as i64, span)
        }),
        "context_limit" => optional_value(settings.context_limit, span, |value| {
            Value::int(value as i64, span)
        }),
        "compact_ratio" => Value::float(settings.compact_ratio as f64, span),
        "keep_recent_messages" => Value::int(settings.keep_recent_messages as i64, span),
        "system_prompt" => match &settings.system_prompt {
            Some(prompt) => Value::string(prompt.clone(), span),
            None => Value::nothing(span),
        },
        "nu_bin" => match &settings.nu_bin {
            Some(bin) => Value::string(bin.clone(), span),
            None => Value::nothing(span),
        },
        "markdown" => Value::bool(settings.markdown, span),
        "tools" => Value::bool(settings.tools, span),
        "confirm_tool_reads" => Value::bool(settings.confirm_tool_reads, span),
        "confirm_tool_writes" => Value::bool(settings.confirm_tool_writes, span),
        "confirm_tool_commands" => Value::bool(settings.confirm_tool_commands, span),
        "tool_command_timeout_secs" => Value::int(settings.tool_command_timeout_secs as i64, span),
        "max_tool_rounds" => Value::int(settings.max_tool_rounds as i64, span),
        _ => Value::nothing(span),
    }
}

fn optional_value<T>(value: Option<T>, span: Span, wrap: impl FnOnce(T) -> Value) -> Value {
    match value {
        Some(value) => wrap(value),
        None => Value::nothing(span),
    }
}

/// The record `ds config <key> [value]` returns: the value, and the accepted options for a
/// key that only takes a fixed set.
fn setting_record(spec: &SettingSpec, settings: &Settings, span: Span, changed: bool) -> Value {
    let mut record = vec![
        ("key".to_owned(), Value::string(spec.name, span)),
        ("value".to_owned(), setting_value(settings, spec.name, span)),
    ];
    if !spec.options.is_empty() {
        let options = spec
            .options
            .iter()
            .map(|option| Value::string(*option, span))
            .collect();
        record.push(("options".to_owned(), Value::list(options, span)));
    }
    record.push(("help".to_owned(), Value::string(spec.help, span)));
    if changed {
        record.push(("changed".to_owned(), Value::bool(true, span)));
    }
    Value::record(Record::from_iter(record), span)
}

/// Write `value` into `settings` under `spec`, parsing it according to the setting's type.
fn apply(settings: &mut Settings, spec: &SettingSpec, raw: &str) -> Result<()> {
    let value = raw.trim();
    match spec.name {
        "base_url" => settings.base_url = value.to_owned(),
        "model" => settings.model = value.to_owned(),
        "thinking" => {
            settings.thinking =
                ThinkingEffort::parse(value).ok_or_else(|| invalid_value(spec, value))?;
        }
        "temperature" => {
            settings.temperature = optional(value)
                .map(|text| parse_number(spec, text))
                .transpose()?;
        }
        "max_tokens" => {
            settings.max_tokens = optional(value)
                .map(|text| parse_uint(spec, text))
                .transpose()?;
        }
        "context_limit" => {
            settings.context_limit = optional(value)
                .map(|text| parse_uint(spec, text))
                .transpose()?;
        }
        "compact_ratio" => settings.compact_ratio = parse_number(spec, value)?,
        "keep_recent_messages" => settings.keep_recent_messages = parse_uint(spec, value)?,
        "system_prompt" => settings.system_prompt = optional(value).map(str::to_owned),
        "nu_bin" => settings.nu_bin = optional(value).map(str::to_owned),
        "markdown" => settings.markdown = parse_bool(spec, value)?,
        "tools" => settings.tools = parse_bool(spec, value)?,
        "confirm_tool_reads" => settings.confirm_tool_reads = parse_bool(spec, value)?,
        "confirm_tool_writes" => settings.confirm_tool_writes = parse_bool(spec, value)?,
        "confirm_tool_commands" => settings.confirm_tool_commands = parse_bool(spec, value)?,
        "tool_command_timeout_secs" => {
            settings.tool_command_timeout_secs = parse_uint(spec, value)?;
        }
        "max_tool_rounds" => settings.max_tool_rounds = parse_uint(spec, value)?,
        other => bail!("`{other}` is not a setting that `ds config` can change"),
    }
    Ok(())
}

/// `None` for the values that clear an optional setting.
fn optional(value: &str) -> Option<&str> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("none") || value.eq_ignore_ascii_case("null")
    {
        None
    } else {
        Some(value)
    }
}

fn parse_bool(spec: &SettingSpec, value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "yes" | "on" | "1" => Ok(true),
        "false" | "no" | "off" | "0" => Ok(false),
        _ => Err(invalid_value(spec, value)),
    }
}

fn parse_uint<T: std::str::FromStr>(spec: &SettingSpec, value: &str) -> Result<T> {
    value
        .parse()
        .map_err(|_| anyhow!("`{}` must be a whole number (got `{value}`)", spec.name))
}

fn parse_number(spec: &SettingSpec, value: &str) -> Result<f32> {
    value
        .parse()
        .map_err(|_| anyhow!("`{}` must be a number (got `{value}`)", spec.name))
}

/// The error for a value a fixed-option setting does not accept, naming every option.
fn invalid_value(spec: &SettingSpec, value: &str) -> anyhow::Error {
    if spec.options.is_empty() {
        anyhow!("`{}` is not a valid value (got `{value}`)", spec.name)
    } else {
        anyhow!(
            "`{}` must be one of {} (got `{value}`)",
            spec.name,
            spec.options.join(", ")
        )
    }
}

fn unknown_setting(name: &str) -> anyhow::Error {
    let names = SETTING_SPECS
        .iter()
        .map(|spec| spec.name)
        .collect::<Vec<_>>()
        .join(", ");
    anyhow!("`{name}` is not a setting; try one of: {names}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_of(name: &str) -> &'static SettingSpec {
        spec(name).expect("a known setting")
    }

    #[test]
    fn setting_thinking_accepts_the_known_levels_and_aliases() {
        let mut settings = Settings::default();
        apply(&mut settings, spec_of("thinking"), "high").unwrap();
        assert_eq!(settings.thinking, ThinkingEffort::High);
        apply(&mut settings, spec_of("thinking"), "ultra").unwrap();
        assert_eq!(settings.thinking, ThinkingEffort::Max);
    }

    #[test]
    fn an_invalid_value_names_the_options() {
        let mut settings = Settings::default();
        let message = format!(
            "{:#}",
            apply(&mut settings, spec_of("thinking"), "loud").unwrap_err()
        );
        for option in ["off", "low", "high", "max"] {
            assert!(message.contains(option), "missing `{option}`: {message}");
        }
    }

    #[test]
    fn an_invalid_boolean_names_true_and_false() {
        let mut settings = Settings::default();
        let message = format!(
            "{:#}",
            apply(&mut settings, spec_of("tools"), "maybe").unwrap_err()
        );
        assert!(message.contains("true"), "{message}");
        assert!(message.contains("false"), "{message}");
    }

    #[test]
    fn optional_values_can_be_cleared() {
        let mut settings = Settings::default();
        apply(&mut settings, spec_of("temperature"), "0.7").unwrap();
        assert_eq!(settings.temperature, Some(0.7));
        apply(&mut settings, spec_of("temperature"), "none").unwrap();
        assert_eq!(settings.temperature, None);

        apply(&mut settings, spec_of("system_prompt"), "be terse").unwrap();
        assert_eq!(settings.system_prompt.as_deref(), Some("be terse"));
        apply(&mut settings, spec_of("system_prompt"), "none").unwrap();
        assert_eq!(settings.system_prompt, None);
    }

    #[test]
    fn booleans_accept_words() {
        let mut settings = Settings::default();
        apply(&mut settings, spec_of("tools"), "off").unwrap();
        assert!(!settings.tools);
        apply(&mut settings, spec_of("tools"), "on").unwrap();
        assert!(settings.tools);
    }

    #[test]
    fn an_unknown_setting_is_rejected_with_the_list() {
        let message = format!("{:#}", unknown_setting("nope"));
        assert!(message.contains("thinking"), "{message}");
        assert!(message.contains("base_url"), "{message}");
    }

    #[test]
    fn an_out_of_range_value_fails_validation() {
        let mut settings = Settings::default();
        apply(&mut settings, spec_of("context_limit"), "10").unwrap();
        assert!(settings.validate().is_err());
    }

    #[test]
    fn a_limited_setting_exposes_its_options() {
        assert_eq!(spec_of("thinking").options, &["off", "low", "high", "max"]);
        assert_eq!(spec_of("markdown").options, &["true", "false"]);
        assert!(spec_of("base_url").options.is_empty());
    }
}
