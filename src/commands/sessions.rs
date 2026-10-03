//! `ds sessions` — inspect and manage stored conversations.

use anyhow::{Context, Result, bail};
use nu_plugin::{EngineInterface, EvaluatedCall, SimplePluginCommand};
use nu_protocol::{Category, Example, LabeledError, Record, Signature, SyntaxShape, Type, Value};

use crate::DsPlugin;
use crate::session::{Session, SessionStore};

use super::common;

/// List, open, create and delete chat sessions.
pub struct SessionsCommand;

impl SimplePluginCommand for SessionsCommand {
    type Plugin = DsPlugin;

    fn name(&self) -> &str {
        "ds sessions"
    }

    fn description(&self) -> &str {
        "List, show, create or delete stored chat sessions"
    }

    fn signature(&self) -> Signature {
        Signature::build("ds sessions")
            .input_output_type(Type::Nothing, Type::Any)
            .switch("dir", "Print the directory the sessions live in", Some('d'))
            .named(
                "show",
                SyntaxShape::String,
                "Full record of one session",
                Some('s'),
            )
            .named(
                "new",
                SyntaxShape::String,
                "Create a session with this name",
                Some('n'),
            )
            .named("delete", SyntaxShape::String, "Delete a session", None)
            .named(
                "model",
                SyntaxShape::String,
                "Model to use for --new",
                Some('m'),
            )
            .category(Category::Experimental)
    }

    fn examples(&self) -> Vec<Example<'_>> {
        vec![
            Example {
                description: "List past conversations",
                example: "ds sessions",
                result: None,
            },
            Example {
                description: "Create a conversation called planning",
                example: "ds sessions --new planning",
                result: None,
            },
            Example {
                description: "Show one conversation in full",
                example: "ds sessions --show planning | get messages | length",
                result: None,
            },
        ]
    }

    fn run(
        &self,
        _plugin: &DsPlugin,
        _engine: &EngineInterface,
        call: &EvaluatedCall,
        _input: &Value,
    ) -> Result<Value, LabeledError> {
        let span = call.head;
        run(call).map_err(|err| crate::error::labeled(err, span))
    }
}

fn run(call: &EvaluatedCall) -> Result<Value> {
    let span = call.head;
    let (paths, settings) = common::load_settings(call)?;
    let store = SessionStore::new(paths.sessions_dir())?;

    if call.has_flag("dir")? {
        return Ok(Value::string(store.dir().display().to_string(), span));
    }

    if let Some(id) = call.get_flag::<String>("delete")? {
        store
            .delete(&id)
            .with_context(|| format!("could not delete the session `{id}`"))?;
        return Ok(Value::string(format!("deleted {id}"), span));
    }

    if let Some(name) = call.get_flag::<String>("new")? {
        let model = common::pick_model(&settings, &[], call.get_flag::<String>("model")?);
        let session = Session::new(
            name,
            model,
            settings.thinking,
            settings.system_prompt.as_deref(),
        );
        store.save(&session)?;
        return Ok(common::session_value(&session, span));
    }

    if let Some(selector) = call.get_flag::<String>("show")? {
        if selector.trim().is_empty() {
            bail!("`--show` needs a session id or title");
        }
        let session = store.resolve(&selector)?;
        return Ok(common::session_value(&session, span));
    }

    let summaries = store.list()?;
    if summaries.is_empty() {
        return Ok(Value::list(Vec::new(), span));
    }

    let values = summaries
        .into_iter()
        .map(|summary| {
            Value::record(
                Record::from_iter([
                    ("id".to_owned(), Value::string(summary.id, span)),
                    ("title".to_owned(), Value::string(summary.title, span)),
                    ("model".to_owned(), Value::string(summary.model, span)),
                    (
                        "updated_at".to_owned(),
                        Value::string(summary.updated_at.to_rfc3339(), span),
                    ),
                    ("turns".to_owned(), Value::int(summary.turns as i64, span)),
                    (
                        "messages".to_owned(),
                        Value::int(summary.message_count as i64, span),
                    ),
                ]),
                span,
            )
        })
        .collect();

    Ok(Value::list(values, span))
}
