//! `cc` — turn natural language into a nushell command, confirm it, then run it.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command as ProcessCommand;

use anyhow::{Context, Result, bail};
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use nu_plugin::{EngineInterface, EvaluatedCall, SimplePluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, Span, SyntaxShape, Type, Value,
};

use crate::DsPlugin;
use crate::api::{ChatRequest, WireMessage};

use super::common;

/// Ask the model for a nushell command and offer to run it.
pub struct CcCommand;

impl SimplePluginCommand for CcCommand {
    type Plugin = DsPlugin;

    fn name(&self) -> &str {
        "cc"
    }

    fn description(&self) -> &str {
        "Generate a nushell command from a natural language request"
    }

    fn extra_description(&self) -> &str {
        r#"Asks the model for a single nushell command that satisfies the request, prints it and
waits for confirmation before running it:

    > cc "列出当前目录所有文件"
    ls ./
    run this command? [Enter] yes  [Esc] no

Nothing is returned once the command has run: it writes its own output, and a value here
would only draw a table over it. Pass `--no-execute` to skip the confirmation and get the
command itself as a string instead.

The command is run by a fresh `nu --commands` process, so changes to the shell's own state
(`cd`, `$env` assignments) do not survive the call. Pass `--no-execute --shell` to evaluate
a single plain command in this session instead, so changes like `cd` do take effect; that
mode shows the command and asks before evaluating it, and `--yes` skips the question for
scripts. Without a terminal the command is never run."#
    }

    fn signature(&self) -> Signature {
        Signature::build("cc")
            .input_output_type(Type::Nothing, Type::Any)
            .required(
                "request",
                SyntaxShape::String,
                "What you want nushell to do, in plain language",
            )
            .switch(
                "no-execute",
                "Do not run the generated command in a new process; return it as text (see --shell)",
                Some('n'),
            )
            .switch(
                "shell",
                "Evaluate the generated command in this session instead of a new process (requires --no-execute)",
                Some('s'),
            )
            .switch(
                "yes",
                "Skip the confirmation that --shell asks for (for scripts)",
                Some('y'),
            )
            .named("model", SyntaxShape::String, "Model to ask", Some('m'))
            .named(
                "think",
                SyntaxShape::String,
                "Reasoning effort: off, low, medium or high",
                Some('t'),
            )
            .named(
                "nu",
                SyntaxShape::String,
                "Path to the `nu` executable used to run the command",
                None,
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
        vec![
            Example {
                description: "Generate a command, then confirm it",
                example: "cc '列出当前目录所有文件'",
                result: None,
            },
            Example {
                description: "Generate a command without running it",
                example: "cc --no-execute 'disk usage of the largest 5 folders'",
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
    let request_text = call.req::<String>(0)?;
    if request_text.trim().is_empty() {
        bail!("describe what you want nushell to do");
    }

    let (paths, mut settings) = common::load_settings(call)?;
    settings.thinking = common::thinking_from_call(call)?.unwrap_or(settings.thinking);

    let client = common::build_client(engine, call, &settings)?;
    let models = common::block_on(client.list_models()).unwrap_or_default();
    let model = common::pick_model(
        &settings,
        &models,
        call.get_flag::<String>("model")?
            .filter(|model| !model.trim().is_empty()),
    );

    let cwd = engine
        .get_current_dir()
        .unwrap_or_else(|_| paths.dir().display().to_string());

    let request = ChatRequest::new(
        &model,
        vec![
            WireMessage {
                role: "system".to_owned(),
                content: system_prompt(&cwd, &paths),
                tool_calls: None,
                tool_call_id: None,
            },
            WireMessage {
                role: "user".to_owned(),
                content: request_text.trim().to_owned(),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
    )
    .max_tokens(Some(1024))
    .temperature(Some(0.0))
    .reasoning_effort(settings.thinking.as_api());

    let response = common::block_on(client.complete(&request))?;
    let command = extract_command(&response.text());

    if command.is_empty() {
        bail!("the model did not produce a command");
    }

    let print_only = call.has_flag("no-execute")?;
    let in_session = call.has_flag("shell")?;

    if in_session && !print_only {
        bail!(
            "`--shell` evaluates the generated command in this session instead of a new \
             process, so it must be combined with `--no-execute` (which stops the plugin \
             from running the command itself)"
        );
    }

    if call.has_flag("yes")? && !in_session {
        bail!(
            "`--yes` only applies to `--shell`, which is the only mode it can skip a \
             confirmation for"
        );
    }

    // `--no-execute` is the inspection mode: the plugin never starts a process. On its
    // own it returns the command as text; with `--shell` the running session evaluates
    // it instead.
    if print_only {
        // On its own the returned string is the output, and printing it here too would
        // show it twice.
        if !in_session {
            return Ok(Value::string(command, span));
        }

        // The command is about to run in this session, so show it and get consent first.
        announce(&command)?;

        // `--yes` is consent given at the call site; otherwise the user gives it now.
        if !call.has_flag("yes")? {
            if !common::is_interactive(engine) {
                eprintln!(
                    "nu_plugin_ds: there is no terminal to confirm on, so the command was \
                     not evaluated in this session; returning it as text instead (pass \
                     --yes to skip the confirmation)"
                );
                return Ok(Value::string(command, span));
            }
            prompt("evaluate this command in this session?")?;
            if !confirm()? {
                eprintln!("nu_plugin_ds: cancelled");
                return Ok(Value::nothing(span));
            }
        }

        return evaluate_in_session(engine, &command, span);
    }

    show(&command, &model, settings.thinking.label())?;

    if !common::is_interactive(engine) {
        eprintln!(
            "nu_plugin_ds: no terminal available, so the command was not run; \
             use --no-execute to get the command itself"
        );
        return Ok(Value::nothing(span));
    }

    if !confirm()? {
        eprintln!("nu_plugin_ds: cancelled");
        return Ok(Value::nothing(span));
    }

    let nu = common::nu_binary(engine, call, &settings)?;
    let exit_code = run_command(&nu, &command, &cwd)?;
    if exit_code != 0 {
        eprintln!("nu_plugin_ds: the command exited with code {exit_code}");
    }

    // Nothing is returned: the command has already written its own output, and a value
    // here would only draw a table over it.
    Ok(Value::nothing(span))
}

/// A short description of the environment, so the model writes commands that fit it.
fn system_prompt(cwd: &str, paths: &crate::config::ConfigPaths) -> String {
    format!(
        r#"You translate a plain language request into exactly one nushell command.

Rules:
- Reply with the command only. No prose, no explanation, no markdown fences.
- Use nushell syntax, never bash or PowerShell. Examples: `ls`, `open`, `save`, `where`,
  `each`, `get`, `sort-by`, `group-by`, `str replace`, `to json`, `| `.
- Prefer a single line. Use `;` or newlines only when several steps are needed.
- Paths are nushell paths: use forward slashes, quote when they contain spaces.
- If the request is ambiguous, pick the most useful interpretation.

Environment:
- operating system: {os}
- shell: nushell {nu_version}
- current directory: {cwd}
- plugin config directory: {config}"#,
        os = std::env::consts::OS,
        nu_version = env!("CARGO_PKG_VERSION"),
        cwd = cwd,
        config = paths.dir().display(),
    )
}

/// Print the generated command on its own, without asking anything.
fn announce(command: &str) -> Result<()> {
    let mut stderr = std::io::stderr();
    writeln!(stderr)?;
    for line in command.lines() {
        writeln!(stderr, "  {line}")?;
    }
    stderr.flush()?;
    Ok(())
}

/// Print the confirmation question, without a trailing newline.
fn prompt(question: &str) -> Result<()> {
    let mut stderr = std::io::stderr();
    write!(stderr, "  {question} [Enter] yes  [Esc] no ")?;
    stderr.flush()?;
    Ok(())
}

/// Print the generated command and ask whether to run it.
fn show(command: &str, model: &str, thinking: &str) -> Result<()> {
    announce(command)?;
    let mut stderr = std::io::stderr();
    writeln!(stderr, "  ── {model} · think:{thinking}")?;
    prompt("run this command?")
}

/// Wait for a single key: `true` means the user pressed Enter.
fn confirm() -> Result<bool> {
    use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

    enable_raw_mode().context("could not switch the terminal to raw mode")?;
    let result = loop {
        match crossterm::event::read() {
            Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => match key.code {
                KeyCode::Enter => break Ok(true),
                KeyCode::Esc => break Ok(false),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    break Ok(false);
                }
                KeyCode::Char('q') => break Ok(false),
                _ => {}
            },
            Ok(_) => {}
            Err(err) => break Err(err).context("could not read a key"),
        }
    };

    let _ = disable_raw_mode();
    eprintln!();

    result
}

/// Run the generated command in a fresh nushell process, inheriting our terminal.
fn run_command(nu: &PathBuf, command: &str, cwd: &str) -> Result<i64> {
    let status = ProcessCommand::new(nu)
        .arg("--commands")
        .arg(command)
        .current_dir(cwd)
        .status()
        .with_context(|| format!("could not run {}", nu.display()))?;

    Ok(status.code().map_or(-1, i64::from))
}

/// Evaluate a generated command in the running session.
///
/// Only a single, plain command can be handed over: the plugin cannot parse new source,
/// it can only ask the engine to call one of its own declarations with arguments we build
/// ourselves. That call runs on the live stack, which is why `cd` here really does move
/// this session. Everything else falls back to returning the text.
fn evaluate_in_session(engine: &EngineInterface, command: &str, span: Span) -> Result<Value> {
    let Some((name, arguments)) = single_invocation(command) else {
        eprintln!(
            "nu_plugin_ds: `{command}` is not a single plain command, so it cannot be \
             evaluated in this session; returning it as text instead"
        );
        return Ok(Value::string(command, span));
    };

    let Some(decl_id) = engine.find_decl(&name)? else {
        eprintln!(
            "nu_plugin_ds: `{name}` is not a nushell declaration (external commands always \
             run in their own process), so the command cannot be evaluated in this session; \
             returning it as text instead"
        );
        return Ok(Value::string(command, span));
    };

    let mut evaluated = EvaluatedCall::new(span);
    for argument in arguments {
        evaluated.add_positional(Value::string(argument, span));
    }

    let output = engine.call_decl(decl_id, evaluated, PipelineData::empty(), false, false)?;

    // Most of these commands (`cd`, `mkdir`, `touch`) produce nothing; anything that does
    // produce a value is handed back so nushell prints it as usual.
    match output.into_value(span)? {
        Value::Nothing { .. } => Ok(Value::nothing(span)),
        value => Ok(value),
    }
}

/// Split `name arg arg` into its parts, when the command is simple enough to rebuild as a
/// declaration call: exactly one command, no pipelines, no redirections, no flags and no
/// expansions (a `$` or a glob would need the parser).
fn single_invocation(command: &str) -> Option<(String, Vec<String>)> {
    let trimmed = command.trim();
    if trimmed.is_empty()
        || trimmed.contains([
            '\n', '|', ';', '&', '>', '<', '`', '$', '(', ')', '{', '}', '[', ']', '*', '?',
        ])
    {
        return None;
    }

    let tokens = tokenize(trimmed)?;
    let (name, arguments) = tokens.split_first()?;
    if name.starts_with('-') || arguments.iter().any(|argument| argument.starts_with('-')) {
        return None;
    }

    Some((name.clone(), arguments.to_vec()))
}

/// Split on whitespace, keeping quoted runs together and dropping the quotes.
fn tokenize(command: &str) -> Option<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;

    for ch in command.chars() {
        match quote {
            Some(open) => {
                if ch == open {
                    quote = None;
                } else {
                    current.push(ch);
                }
            }
            None => {
                if ch == '"' || ch == '\'' {
                    quote = Some(ch);
                } else if ch.is_whitespace() {
                    if !current.is_empty() {
                        tokens.push(std::mem::take(&mut current));
                    }
                } else {
                    current.push(ch);
                }
            }
        }
    }

    // An unbalanced quote means we cannot know what the arguments were.
    if quote.is_some() {
        return None;
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    Some(tokens)
}

/// Pull the command out of whatever the model replied with.
fn extract_command(raw: &str) -> String {
    let text = raw.trim();

    // Prefer a fenced block when the model used one anyway.
    let body = if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        let body_start = after.find('\n').map_or(after.len(), |index| index + 1);
        let body = &after[body_start..];
        body.find("```").map_or(body, |end| &body[..end])
    } else {
        text
    };

    let mut lines: Vec<&str> = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim();
        // Drop common prompt markers some models like to add.
        let trimmed = trimmed
            .strip_prefix("nu> ")
            .or_else(|| trimmed.strip_prefix("> "))
            .or_else(|| trimmed.strip_prefix("$ "))
            .unwrap_or(trimmed);
        lines.push(trimmed);
    }

    while lines.first().is_some_and(|line| line.is_empty()) {
        lines.remove(0);
    }
    while lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }

    let joined = lines.join("\n");
    let trimmed = joined.trim();
    // Unwrap a single pair of backticks around the whole command.
    if trimmed.starts_with('`') && trimmed.ends_with('`') && trimmed.len() > 1 {
        return trimmed[1..trimmed.len() - 1].trim().to_owned();
    }
    trimmed.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_a_fenced_command() {
        let raw = "Here you go:\n```nu\nls ./ | where size > 1kb\n```\nHope that helps!";
        assert_eq!(extract_command(raw), "ls ./ | where size > 1kb");
    }

    #[test]
    fn strips_prompt_markers_and_backticks() {
        assert_eq!(extract_command("> ls ./"), "ls ./");
        assert_eq!(extract_command("`ls ./`"), "ls ./");
        assert_eq!(extract_command("nu> ls ./"), "ls ./");
    }

    #[test]
    fn keeps_multi_line_pipelines() {
        let raw = "ls ./\n| where type == file\n| length";
        assert_eq!(
            extract_command(raw),
            "ls ./\n| where type == file\n| length"
        );
    }

    #[test]
    fn does_not_unwrap_partial_backticks() {
        assert_eq!(extract_command("ls `./my dir`"), "ls `./my dir`");
    }

    #[test]
    fn accepts_a_single_plain_command() {
        assert_eq!(
            single_invocation("cd .."),
            Some(("cd".to_owned(), vec!["..".to_owned()]))
        );
        assert_eq!(
            single_invocation("cd \"my dir\""),
            Some(("cd".to_owned(), vec!["my dir".to_owned()]))
        );
        assert_eq!(
            single_invocation("mkdir foo/bar"),
            Some(("mkdir".to_owned(), vec!["foo/bar".to_owned()]))
        );
        assert_eq!(
            single_invocation("  cd   ..  ").map(|it| it.0),
            Some("cd".to_owned())
        );
    }

    #[test]
    fn rejects_anything_needing_the_parser() {
        for command in [
            "ls | where size > 1kb",
            "cd a; cd b",
            "mkdir -p nested",
            "cd $nu.home-path",
            "ls *.rs",
            "open \"unbalanced",
            "echo hi > out.txt",
            "",
        ] {
            assert_eq!(
                single_invocation(command),
                None,
                "should reject {command:?}"
            );
        }
    }

    #[test]
    fn a_quoted_path_survives_tokenising() {
        assert_eq!(
            single_invocation("cd \"C:/a b/c\""),
            Some(("cd".to_owned(), vec!["C:/a b/c".to_owned()]))
        );
    }
}
