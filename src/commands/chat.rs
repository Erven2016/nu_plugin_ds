//! `chat` — an interactive DeepSeek client.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use nu_plugin::{EngineInterface, EvaluatedCall, SimplePluginCommand};
use nu_protocol::{Category, Example, LabeledError, Signature, SyntaxShape, Type, Value};

use crate::DsPlugin;
use crate::api::{ChatMessage, build_chat_request, with_tools};
use crate::chat::tools::Risk;
use crate::chat::{self, ChatSetup, tools};
use crate::config::ThinkingEffort;
use crate::session::{Session, SessionStore};

use super::common;

/// Start (or continue) an interactive conversation with DeepSeek.
pub struct ChatCommand;

impl SimplePluginCommand for ChatCommand {
    type Plugin = DsPlugin;

    fn name(&self) -> &str {
        "chat"
    }

    fn description(&self) -> &str {
        "Chat with DeepSeek in a full-screen client"
    }

    fn extra_description(&self) -> &str {
        r#"Opens a full-screen transcript with a status bar showing the model, the thinking
level, the session, the estimated context usage and the tokens spent so far.

The model list is fetched from the API when the chat starts (Ctrl+O to switch), and the
history is compacted automatically when it grows past `context_limit * compact_ratio`
tokens. Sessions are stored as JSON files under the plugin's config directory, and they
can be reopened with Ctrl+B or `--session`.

Keys: Enter sends, Shift+Enter inserts a newline, Ctrl+O switches model, Ctrl+T switches
the thinking level, Ctrl+B browses sessions, Ctrl+N starts a new session, Ctrl+R redoes
the last answer, Ctrl+C cancels or quits, Ctrl+X quits and Ctrl+/ shows the help card.
Type /help inside the chat for the full list.

Tools are classed by risk: reading a file is free, while writing one and running a command
are shown and confirmed first (Enter runs it, Esc skips it, and `a` allows every call of
that class for the session). /tools toggles them, and the settings `tools`,
`confirm_tool_reads`, `confirm_tool_writes`, `confirm_tool_commands` and
`tool_command_timeout_secs` control the defaults. Reading a file sends its contents to
DeepSeek.

`run_nu` runs each command in a fresh nushell process, so `cd` and `$env` changes do not
persist between calls and no shell state is changed. Its output is captured, truncated per
stream, and bounded by `tool_command_timeout_secs`. The command is shown before it runs,
and reading it is the only protection: there is no denylist.

`--allow-tools` runs calls without asking (a shorthand for `--allow-reads`, `--allow-writes`
and `--allow-commands`), `--no-tools` turns them off, and no single turn runs more than
`max_tool_rounds` rounds.

Closing the window returns nothing, so the shell stays clean; pass `--return-session` for
the session record, or use `ds sessions --show`. Without a terminal `chat` becomes a
single-turn request: it requires `--prompt` and returns the answer as a string."#
    }

    fn signature(&self) -> Signature {
        Signature::build("chat")
            .input_output_type(Type::Nothing, Type::Any)
            .named(
                "model",
                SyntaxShape::String,
                "Model to use for this session",
                Some('m'),
            )
            .named(
                "think",
                SyntaxShape::String,
                "Reasoning effort: off, low, high or max",
                Some('t'),
            )
            .named(
                "session",
                SyntaxShape::String,
                "Session id or title to open",
                Some('s'),
            )
            .switch(
                "new",
                "Start a new session instead of continuing the most recent one",
                Some('n'),
            )
            .named(
                "prompt",
                SyntaxShape::String,
                "Send this prompt straight away, and answer without a TUI when there is no terminal",
                Some('p'),
            )
            .switch(
                "return-session",
                "Return the session record instead of printing nothing",
                None,
            )
            .named(
                "api-key",
                SyntaxShape::String,
                "API key to use instead of $env.DEEPSEEK_API_KEY",
                None,
            )
            .named("base-url", SyntaxShape::String, "API base URL", None)
            .named(
                "context-limit",
                SyntaxShape::Int,
                "Token budget before the history is compacted",
                None,
            )
            .named(
                "temperature",
                SyntaxShape::Float,
                "Sampling temperature (ignored by thinking-only models)",
                None,
            )
            .switch(
                "no-tools",
                "Do not offer the file tools to the model",
                None,
            )
            .switch(
                "allow-tools",
                "Do not ask before any tool call (shorthand for the three below)",
                None,
            )
            .switch("allow-reads", "Do not ask before reading a file", None)
            .switch("allow-writes", "Do not ask before writing a file", None)
            .switch("allow-commands", "Do not ask before running a command", None)
            .category(Category::Experimental)
    }

    fn examples(&self) -> Vec<Example<'_>> {
        vec![
            Example {
                description: "Start or continue the most recent conversation",
                example: "chat",
                result: None,
            },
            Example {
                description: "Start a fresh conversation with the reasoning model",
                example: "chat --new --model deepseek-v4-pro --think high",
                result: None,
            },
            Example {
                description: "Reopen a named conversation",
                example: "chat --session planning",
                result: None,
            },
            Example {
                description: "Ask one question from a pipeline",
                example: "chat --new --prompt 'explain this error'",
                result: None,
            },
            Example {
                description: "Take the whole session record",
                example: "chat --new --prompt 'hi' --return-session | get messages | last | get content",
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
    let (paths, settings) = common::load_settings(call)?;
    let store = SessionStore::new(paths.sessions_dir())?;
    let client = common::build_client(engine, call, &settings)?;

    // The model list drives switching inside the TUI, but a network hiccup should not
    // stop the user from chatting.
    let models = match common::block_on(client.list_models()) {
        Ok(models) if !models.is_empty() => models,
        Ok(_) => common::offline_models(&settings),
        Err(err) => {
            eprintln!("nu_plugin_ds: could not fetch the model list: {err:#}");
            common::offline_models(&settings)
        }
    };

    let explicit_model = call
        .get_flag::<String>("model")?
        .filter(|model| !model.trim().is_empty());
    let thinking = common::thinking_from_call(call)?;

    let mut session = open_session(call, &store, &settings, &models, &explicit_model, thinking)?;

    if explicit_model.is_some() {
        session.model = explicit_model.clone().unwrap_or_default();
    }
    if let Some(thinking) = thinking {
        session.thinking = thinking;
    }

    // Say once, at startup, when the configured budget is larger than the model can hold.
    if let Some(warning) = settings.context_limit_warning(&session.model) {
        eprintln!("nu_plugin_ds: {warning}");
    }

    let return_session = call.has_flag("return-session")?;

    let cwd = PathBuf::from(
        engine
            .get_current_dir()
            .unwrap_or_else(|_| paths.dir().display().to_string()),
    );
    // `--no-tools` wins over the setting. The `--allow-*` flags pre-approve the risks they
    // name for this session: `--allow-tools` is the shorthand for all three. Both paths,
    // the TUI and the single-turn loop, need these.
    let tools_enabled = settings.tools && !call.has_flag("no-tools")?;
    let mut approved = BTreeSet::new();
    if call.has_flag("allow-tools")? {
        approved.extend([Risk::Read, Risk::Write, Risk::Command]);
    }
    if call.has_flag("allow-reads")? {
        approved.insert(Risk::Read);
    }
    if call.has_flag("allow-writes")? {
        approved.insert(Risk::Write);
    }
    if call.has_flag("allow-commands")? {
        approved.insert(Risk::Command);
    }

    // Locate `nu` for the command tool. When it cannot be found the tool is simply not
    // offered, and that is noted on stderr rather than stopping the chat.
    let nu_bin = if tools_enabled {
        match common::nu_binary(engine, call, &settings) {
            Ok(path) => Some(path),
            Err(err) => {
                eprintln!("nu_plugin_ds: the `run_nu` tool will not be offered: {err:#}");
                None
            }
        }
    } else {
        None
    };

    let context = tools::Context {
        cwd,
        nu_bin,
        command_timeout: std::time::Duration::from_secs(settings.tool_command_timeout_secs),
    };

    if !common::is_interactive(engine) {
        return single_turn(
            engine,
            call,
            &store,
            &settings,
            &client,
            session,
            return_session,
            context,
            tools_enabled,
            &approved,
        );
    }

    let initial_prompt = call
        .get_flag::<String>("prompt")?
        .filter(|prompt| !prompt.trim().is_empty());

    let setup = ChatSetup {
        client,
        settings,
        store,
        models,
        session,
        initial_prompt,
        context,
        tools_enabled,
        approved,
    };

    // Taking the foreground lets the chat own the terminal without nushell fighting it
    // for keystrokes. Failure is not fatal: input still arrives through the inherited
    // stdio in local socket mode.
    let guard = match engine.enter_foreground() {
        Ok(guard) => Some(guard),
        Err(err) => {
            eprintln!("nu_plugin_ds: could not take over the terminal ({err})");
            None
        }
    };

    let session = chat::run(setup).context("the chat session ended with an error")?;

    if let Some(guard) = guard {
        guard
            .leave()
            .context("could not return the terminal to nushell")?;
    }

    // Returning nothing keeps the shell clean after the window closes; the record is
    // available on request (and through `ds sessions --show`).
    if return_session {
        Ok(common::session_value(&session, span))
    } else {
        Ok(Value::nothing(span))
    }
}

fn open_session(
    call: &EvaluatedCall,
    store: &SessionStore,
    settings: &crate::config::Settings,
    models: &[crate::api::ModelInfo],
    explicit_model: &Option<String>,
    thinking: Option<ThinkingEffort>,
) -> Result<Session> {
    let model = common::pick_model(settings, models, explicit_model.clone());
    let thinking = thinking.unwrap_or(settings.thinking);

    if !call.has_flag("new")? {
        if let Some(selector) = call.get_flag::<String>("session")? {
            return store
                .resolve(&selector)
                .with_context(|| format!("could not open the session `{selector}`"));
        }
        if let Some(session) = store.latest()? {
            return Ok(session);
        }
    }

    Ok(Session::new(
        "",
        model,
        thinking,
        settings.system_prompt.as_deref(),
    ))
}

/// The non-interactive path: one prompt, and the tool loop that answers it.
#[allow(clippy::too_many_arguments)]
fn single_turn(
    engine: &EngineInterface,
    call: &EvaluatedCall,
    store: &SessionStore,
    settings: &crate::config::Settings,
    client: &crate::api::DeepSeekClient,
    mut session: Session,
    return_session: bool,
    context: tools::Context,
    tools_enabled: bool,
    approved: &BTreeSet<Risk>,
) -> Result<Value> {
    let span = call.head;

    let Some(prompt) = call
        .get_flag::<String>("prompt")?
        .filter(|prompt| !prompt.trim().is_empty())
    else {
        let hint = common::terminal_hint(engine)
            .unwrap_or("run `chat` from an interactive terminal to use the full-screen client");
        bail!(
            "there is no interactive terminal for the chat client, so only a single question \
             can be asked; pass `--prompt` to do that ({hint})"
        );
    };

    session.push(ChatMessage::user(prompt.trim()));

    // A legacy `*-reasoner` name marks a thinking-only model, which cannot call tools.
    let tools_active = tools_enabled && !session.model.contains("reasoner");
    // The command tool is only offered when a `nu` executable was found.
    let offer_commands = tools_active && context.nu_bin.is_some();
    let mut rounds = 0usize;
    let mut wrap_up = false;
    let mut hinted: BTreeSet<Risk> = BTreeSet::new();

    // A rejected-too-long request is compacted and retried the same way the TUI does.
    let keep = settings.keep_recent_messages.max(2);
    let mut context_retries = 0usize;

    let answer = loop {
        let request = build_chat_request(
            &session.model,
            session.thinking,
            settings.max_tokens,
            settings.temperature,
            &session.messages,
            false,
        );
        let request = if tools_active && !wrap_up {
            with_tools(request, tools::definitions(offer_commands))
        } else {
            request
        };

        let response = match common::block_on(client.complete(&request)) {
            Ok(response) => response,
            Err(err) => {
                let message = format!("{err:#}");
                if context_retries < chat::app::CONTEXT_RETRIES
                    && crate::api::is_context_overflow(&message)
                    && session.compaction_plan(keep).is_some()
                {
                    let to_summarise = session
                        .compaction_plan(keep)
                        .map(|(messages, _)| messages)
                        .unwrap_or_default();
                    let (summary, usage) = common::block_on(chat::app::summarise(
                        client,
                        &session.model,
                        &to_summarise,
                    ))?;
                    let removed = session.apply_summary(&summary, keep);
                    if let Some(usage) = usage {
                        session.usage.merge(&usage);
                    }
                    store.save(&session)?;
                    context_retries += 1;
                    eprintln!(
                        "nu_plugin_ds: the model rejected the history as too long; compacting \
                         {removed} messages and retrying ({context_retries}/{});",
                        chat::app::CONTEXT_RETRIES
                    );
                    continue;
                }
                return Err(err);
            }
        };
        if let Some(usage) = response.usage {
            session.usage.merge(&usage);
        }

        let calls = response
            .choices
            .first()
            .and_then(|choice| choice.message.tool_calls.clone())
            .unwrap_or_default();

        if calls.is_empty() {
            let answer = response.text();
            session.push(ChatMessage::assistant(answer.clone()));
            break answer;
        }

        let mut assistant = ChatMessage::assistant(response.text());
        assistant.tool_calls = Some(calls.clone());
        session.push(assistant);

        if rounds >= settings.max_tool_rounds {
            // Keep the history valid: every `tool_call` needs a matching result.
            for tool_call in &calls {
                session.push(ChatMessage::tool(
                    &tool_call.id,
                    format!(
                        "not run: this turn already used {} tool rounds",
                        settings.max_tool_rounds
                    ),
                ));
            }
            if wrap_up {
                // The wrap-up was requested and a tool was called anyway, which should not
                // be possible: stop rather than loop.
                eprintln!(
                    "nu_plugin_ds: stopping: the model used {} tool rounds in one turn",
                    settings.max_tool_rounds
                );
                break response.text();
            }
            // Ask once more without the tools, so the turn ends with the model telling the
            // user where things stand rather than a bare tool result.
            eprintln!(
                "nu_plugin_ds: the model used {} tool rounds in one turn; asking it to wrap up",
                settings.max_tool_rounds
            );
            wrap_up = true;
            continue;
        }
        rounds += 1;

        for tool_call in &calls {
            match tools::risk(&tool_call.function.name) {
                Some(risk) if !needs_approval(settings, approved, risk) => {
                    let outcome = tools::execute(tool_call, &context);
                    session.push(ChatMessage::tool(&tool_call.id, outcome.result));
                }
                Some(risk) => {
                    session.push(ChatMessage::tool(
                        &tool_call.id,
                        format!(
                            "not run: this session has not approved {} calls",
                            risk.label()
                        ),
                    ));
                    if hinted.insert(risk) {
                        eprintln!(
                            "nu_plugin_ds: the model asked to {}; pass `{}` to let this class \
                             run without asking (or `--allow-tools` for all of them)",
                            describe(risk),
                            allow_flag(risk),
                        );
                    }
                }
                // Nothing to approve for a tool that does not exist: record the error.
                None => {
                    let outcome = tools::execute(tool_call, &context);
                    session.push(ChatMessage::tool(&tool_call.id, outcome.result));
                }
            }
        }
    };

    store.save(&session)?;

    // A bare string prints as-is, so `chat -p ...` does not draw a table; ask for the
    // record explicitly when the whole session is wanted.
    if return_session {
        Ok(common::session_value(&session, span))
    } else {
        Ok(Value::string(answer, span))
    }
}

/// Whether a call of this risk has to be confirmed: it is confirmed when the settings ask
/// for it and the session's `--allow-*` flags did not pre-approve the class.
fn needs_approval(
    settings: &crate::config::Settings,
    approved: &BTreeSet<Risk>,
    risk: Risk,
) -> bool {
    let configured = match risk {
        Risk::Read => settings.confirm_tool_reads,
        Risk::Write => settings.confirm_tool_writes,
        Risk::Command => settings.confirm_tool_commands,
    };
    configured && !approved.contains(&risk)
}

/// The flag that pre-approves a risk class.
fn allow_flag(risk: Risk) -> &'static str {
    match risk {
        Risk::Read => "--allow-reads",
        Risk::Write => "--allow-writes",
        Risk::Command => "--allow-commands",
    }
}

/// How a risk reads in the hint printed when a call is refused.
fn describe(risk: Risk) -> &'static str {
    match risk {
        Risk::Read => "read a file",
        Risk::Write => "write a file",
        Risk::Command => "run a command",
    }
}
