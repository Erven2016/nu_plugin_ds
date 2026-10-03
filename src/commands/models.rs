//! `ds models` — list the models the API key can use.

use anyhow::Result;
use nu_plugin::{EngineInterface, EvaluatedCall, SimplePluginCommand};
use nu_protocol::{Category, Example, LabeledError, Signature, SyntaxShape, Type, Value};

use crate::DsPlugin;

use super::common;

/// Show the models returned by `GET /models`.
pub struct ModelsCommand;

impl SimplePluginCommand for ModelsCommand {
    type Plugin = DsPlugin;

    fn name(&self) -> &str {
        "ds models"
    }

    fn description(&self) -> &str {
        "List the DeepSeek models available to the configured API key"
    }

    fn signature(&self) -> Signature {
        Signature::build("ds models")
            .input_output_type(Type::Nothing, Type::Any)
            .named(
                "model",
                SyntaxShape::String,
                "Model to mark as active in the result",
                Some('m'),
            )
            .named(
                "api-key",
                SyntaxShape::String,
                "API key to use instead of $env.DEEPSEEK_API_KEY",
                None,
            )
            .named("base-url", SyntaxShape::String, "API base URL", None)
            .category(Category::Experimental)
    }

    fn examples(&self) -> Vec<Example<'_>> {
        vec![Example {
            description: "List the available models",
            example: "ds models",
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
    let (_, settings) = common::load_settings(call)?;
    let client = common::build_client(engine, call, &settings)?;
    let models = common::block_on(client.list_models())?;

    let active = common::pick_model(
        &settings,
        &models,
        call.get_flag::<String>("model")?
            .filter(|model| !model.trim().is_empty()),
    );

    let values = models
        .iter()
        .map(|model| common::model_value(model, model.id == active, span))
        .collect();

    Ok(Value::list(values, span))
}
