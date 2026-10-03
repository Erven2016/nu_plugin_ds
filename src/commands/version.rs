//! `ds version` — print the plugin's version.

use anyhow::Result;
use nu_plugin::{EngineInterface, EvaluatedCall, SimplePluginCommand};
use nu_protocol::{Category, Example, LabeledError, Record, Signature, Type, Value};

use crate::DsPlugin;

/// The version compiled into the plugin, and the one handed to nushell when it registers
/// the plugin, so the two can never disagree.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The plugin's name, so the record says what the version belongs to.
pub const NAME: &str = "nu_plugin_ds";

/// Show the version of the plugin.
pub struct VersionCommand;

impl SimplePluginCommand for VersionCommand {
    type Plugin = DsPlugin;

    fn name(&self) -> &str {
        "ds version"
    }

    fn description(&self) -> &str {
        "Show the version of the nu_plugin_ds plugin"
    }

    fn signature(&self) -> Signature {
        Signature::build("ds version")
            .input_output_type(Type::Nothing, Type::Any)
            .category(Category::Experimental)
    }

    fn examples(&self) -> Vec<Example<'_>> {
        vec![Example {
            description: "Show the plugin version",
            example: "ds version",
            result: None,
        }]
    }

    fn run(
        &self,
        _plugin: &DsPlugin,
        _engine: &EngineInterface,
        call: &EvaluatedCall,
        _input: &Value,
    ) -> Result<Value, LabeledError> {
        let span = call.head;
        Ok(Value::record(
            Record::from_iter([
                ("name".to_owned(), Value::string(NAME, span)),
                ("version".to_owned(), Value::string(VERSION, span)),
            ]),
            span,
        ))
    }
}
