//! The tools chat offers to the model.
//!
//! One to read a file, one to write one, and one to run a nushell command. They are plain
//! functions of their arguments plus a [`Context`], so they need no terminal and are easy to
//! test.

use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};

use crate::api::{ToolCall, ToolDefinition};

/// The largest file `write_file` will write.
const MAX_WRITE_BYTES: usize = 5 * 1024 * 1024;
/// How much of a file `read_file` hands back.
const MAX_READ_BYTES: usize = 32 * 1024;
/// How many lines the confirmation preview shows.
const PREVIEW_LINES: usize = 12;
/// The most output kept from a command, per stream.
const MAX_COMMAND_BYTES: usize = 32 * 1024;

pub const WRITE_FILE: &str = "write_file";
pub const READ_FILE: &str = "read_file";
pub const RUN_NU: &str = "run_nu";

/// How much a tool call could affect the user's machine, which decides whether it has to be
/// confirmed first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Risk {
    /// Reading a file. It cannot change anything, so it is free by default.
    Read,
    /// Writing a file.
    Write,
    /// Running a command, which can do anything.
    Command,
}

impl Risk {
    /// Used in messages and in the overlay title.
    pub fn label(self) -> &'static str {
        match self {
            Risk::Read => "read",
            Risk::Write => "write",
            Risk::Command => "command",
        }
    }
}

/// The risk of a tool by name, or `None` when there is no such tool.
pub fn risk(name: &str) -> Option<Risk> {
    match name {
        READ_FILE => Some(Risk::Read),
        WRITE_FILE => Some(Risk::Write),
        RUN_NU => Some(Risk::Command),
        _ => None,
    }
}

/// What a tool needs in order to run.
#[derive(Clone, Debug)]
pub struct Context {
    /// The session's working directory: relative paths resolve against it.
    pub cwd: PathBuf,
    /// The `nu` executable, when commands are allowed. `None` means the command tool is
    /// not offered at all.
    pub nu_bin: Option<PathBuf>,
    /// How long one command may run before it is killed.
    pub command_timeout: Duration,
}

/// What a call will do, shown to the user before it runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Preview {
    /// The tool being called.
    pub tool: String,
    /// The resolved, absolute path.
    pub path: PathBuf,
    /// One line about the effect, e.g. `creates a new file (42 bytes)`.
    pub summary: String,
    /// The first few lines of what will be written or read.
    pub lines: Vec<String>,
    /// How many further lines were not shown.
    pub hidden: usize,
    /// How much the call could change, used to decide and describe the approval.
    pub risk: Risk,
}

/// The result of running a call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    /// The text handed back to the model as the tool result.
    pub result: String,
    /// Whether the tool did what it was asked to do.
    pub ok: bool,
}

impl Outcome {
    pub fn ok(result: impl Into<String>) -> Self {
        Outcome {
            result: result.into(),
            ok: true,
        }
    }

    pub fn failed(result: impl Into<String>) -> Self {
        Outcome {
            result: result.into(),
            ok: false,
        }
    }
}

/// What the model is told it may call.
///
/// The command tool is only offered when `commands` is true, which the caller decides from
/// the settings and whether a `nu` executable was found.
pub fn definitions(commands: bool) -> Vec<ToolDefinition> {
    let mut tools = vec![
        ToolDefinition::function(
            WRITE_FILE,
            "Write a text file. Creates the file and any missing parent directories, and \
             replaces the file if it already exists. Use this whenever the user asks for a \
             file to be created or changed.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file to write, relative to the current directory or absolute."
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete contents of the file."
                    }
                },
                "required": ["path", "content"]
            }),
        ),
        ToolDefinition::function(
            READ_FILE,
            "Read a text file so you can see what is in it before changing it. Large files \
             are truncated.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file to read, relative to the current directory or absolute."
                    }
                },
                "required": ["path"]
            }),
        ),
    ];

    if commands {
        tools.push(ToolDefinition::function(
            RUN_NU,
            "Run a nushell command and read its output. Each call runs in a fresh nushell process, \
             so `cd` and `$env` changes do not persist between calls: chain the steps you need into \
             one command, or use absolute paths. Prefer this over guessing what a file contains.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The nushell command to run."
                    }
                },
                "required": ["command"]
            }),
        ));
    }

    tools
}

/// Describe a call so the user can decide about it. Fails when the arguments are unusable,
/// in which case there is nothing to approve.
pub fn preview(call: &ToolCall, context: &Context) -> Result<Preview> {
    let arguments = call.parse_arguments()?;
    match call.function.name.as_str() {
        WRITE_FILE => preview_write(&arguments, &context.cwd),
        READ_FILE => preview_read(&arguments, &context.cwd),
        RUN_NU => preview_run(&arguments, context),
        other => bail!("there is no tool called `{other}`"),
    }
}

/// Resolve a path from the model against the session's working directory.
///
/// `~` is expanded, and `.`/`..` are collapsed lexically so the confirmation shows a path a
/// human can check. No filesystem access, so it also works for files that do not exist yet.
pub fn resolve(cwd: &Path, path: &str) -> PathBuf {
    let path = path.trim();
    let expanded = if path == "~" {
        dirs::home_dir().unwrap_or_else(|| PathBuf::from(path))
    } else if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\")) {
        dirs::home_dir()
            .map(|home| home.join(rest))
            .unwrap_or_else(|| PathBuf::from(path))
    } else {
        PathBuf::from(path)
    };

    let joined = if expanded.is_absolute() {
        expanded
    } else {
        cwd.join(expanded)
    };
    normalize(&joined)
}

/// Run a call. Never fails: a failure becomes a tool result the model can read and act on.
pub fn execute(call: &ToolCall, context: &Context) -> Outcome {
    let arguments = match call.parse_arguments() {
        Ok(arguments) => arguments,
        Err(err) => return Outcome::failed(format!("{err:#}")),
    };

    match call.function.name.as_str() {
        WRITE_FILE => write_file(&arguments, &context.cwd),
        READ_FILE => read_file(&arguments, &context.cwd),
        RUN_NU => run_nu(&arguments, context),
        other => Outcome::failed(format!("there is no tool called `{other}`")),
    }
}

/// The result text used when the user refuses a call.
pub fn declined() -> Outcome {
    Outcome::failed("the user declined to run this call; do not repeat it, ask them how to proceed")
}

fn preview_write(arguments: &Value, cwd: &Path) -> Result<Preview> {
    let path = resolve(cwd, &required(arguments, "path")?);
    let content = required(arguments, "content")?;

    let summary = match std::fs::metadata(&path) {
        Ok(metadata) => format!("replaces the existing file (now {} bytes)", metadata.len()),
        Err(_) => format!("creates a new file ({} bytes)", content.len()),
    };

    let (lines, hidden) = head_lines(&content, PREVIEW_LINES);
    Ok(Preview {
        tool: WRITE_FILE.to_owned(),
        path,
        summary,
        lines,
        hidden,
        risk: Risk::Write,
    })
}

fn preview_read(arguments: &Value, cwd: &Path) -> Result<Preview> {
    let path = resolve(cwd, &required(arguments, "path")?);
    let size = std::fs::metadata(&path)
        .map(|metadata| format!("{} bytes", metadata.len()))
        .unwrap_or_else(|_| "missing".to_owned());

    Ok(Preview {
        tool: READ_FILE.to_owned(),
        path,
        // Reading a file sends it to DeepSeek, which the user deserves to know.
        summary: format!("reads the file ({size}) and sends its contents to DeepSeek"),
        lines: Vec::new(),
        hidden: 0,
        risk: Risk::Read,
    })
}

fn preview_run(arguments: &Value, context: &Context) -> Result<Preview> {
    let command = required(arguments, "command")?;

    Ok(Preview {
        tool: RUN_NU.to_owned(),
        path: context.cwd.clone(),
        summary: format!(
            "runs in {} (up to {}s)",
            context.cwd.display(),
            context.command_timeout.as_secs()
        ),
        // Show every line: the user is being asked to approve the whole command.
        lines: command.lines().map(str::to_owned).collect(),
        hidden: 0,
        risk: Risk::Command,
    })
}

fn write_file(arguments: &Value, cwd: &Path) -> Outcome {
    let path = resolve(
        cwd,
        &match required(arguments, "path") {
            Ok(path) => path,
            Err(err) => return Outcome::failed(format!("{err:#}")),
        },
    );
    let content = match required(arguments, "content") {
        Ok(content) => content,
        Err(err) => return Outcome::failed(format!("{err:#}")),
    };

    if content.len() > MAX_WRITE_BYTES {
        return Outcome::failed(format!(
            "refusing to write {} bytes to {}; the limit is {MAX_WRITE_BYTES} bytes",
            content.len(),
            path.display()
        ));
    }

    if let Some(parent) = path.parent()
        && let Err(err) = std::fs::create_dir_all(parent)
    {
        return Outcome::failed(format!("could not create {}: {err:#}", parent.display()));
    }

    match std::fs::write(&path, &content) {
        Ok(()) => Outcome::ok(format!(
            "wrote {} bytes to {}",
            content.len(),
            path.display()
        )),
        Err(err) => Outcome::failed(format!("could not write {}: {err:#}", path.display())),
    }
}

fn read_file(arguments: &Value, cwd: &Path) -> Outcome {
    let path = resolve(
        cwd,
        &match required(arguments, "path") {
            Ok(path) => path,
            Err(err) => return Outcome::failed(format!("{err:#}")),
        },
    );

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(err) => return Outcome::failed(format!("could not read {}: {err:#}", path.display())),
    };

    let text = match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(_) => {
            return Outcome::failed(format!("{} is not a text file", path.display()));
        }
    };

    if text.len() <= MAX_READ_BYTES {
        return Outcome::ok(text);
    }

    // Cut on a character boundary so the model never receives half a character.
    let mut end = MAX_READ_BYTES;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }

    Outcome::ok(format!(
        "{}\n… (truncated: showing the first {end} of {} bytes)",
        &text[..end],
        text.len()
    ))
}

/// Run a command in a fresh nushell process, capturing its output.
///
/// Both pipes are drained on their own threads so a command that chats away cannot fill a
/// pipe and deadlock, and the wait is bounded so a runaway command cannot freeze the chat.
fn run_nu(arguments: &Value, context: &Context) -> Outcome {
    let command = match required(arguments, "command") {
        Ok(command) => command,
        Err(err) => return Outcome::failed(format!("{err:#}")),
    };
    if command.trim().is_empty() {
        return Outcome::failed("the call is missing the `command` argument");
    }

    let Some(nu) = context.nu_bin.as_ref() else {
        return Outcome::failed("running commands is not available in this session");
    };

    let child = std::process::Command::new(nu)
        .arg("--commands")
        .arg(&command)
        .current_dir(&context.cwd)
        // Never inherit our stdin: it is the user's terminal.
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let mut child = match child {
        Ok(child) => child,
        Err(err) => {
            return Outcome::failed(format!("could not run {}: {err:#}", nu.display()));
        }
    };

    let stdout = drain(child.stdout.take());
    let stderr = drain(child.stderr.take());

    let started = std::time::Instant::now();
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if started.elapsed() >= context.command_timeout {
                    timed_out = true;
                    let _ = child.kill();
                    break child.wait().ok();
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(err) => {
                return Outcome::failed(format!("could not wait for the command: {err:#}"));
            }
        }
    };

    let stdout = stdout.join().unwrap_or_default();
    let stderr = stderr.join().unwrap_or_default();
    let elapsed = started.elapsed();

    if timed_out {
        return Outcome::failed(format!(
            "the command was stopped after {:.0}s (limit {}s):\n{}",
            elapsed.as_secs_f64(),
            context.command_timeout.as_secs(),
            head_of(&stdout),
        ));
    }

    let code = status.and_then(|status| status.code()).unwrap_or(-1);
    let mut result = format!(
        "exit {code} in {:.1}s\n{}",
        elapsed.as_secs_f64(),
        head_of(&stdout)
    );
    if !stderr.trim().is_empty() {
        result.push_str("\n[stderr]\n");
        result.push_str(&head_of(&stderr));
    }

    if code == 0 {
        Outcome::ok(result.trim_end().to_owned())
    } else {
        Outcome::failed(result.trim_end().to_owned())
    }
}

/// Read a pipe on its own thread, keeping up to `MAX_COMMAND_BYTES + 1` bytes and discarding
/// the rest so the child never blocks on a full pipe.
fn drain<R: std::io::Read + Send + 'static>(pipe: Option<R>) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let Some(mut pipe) = pipe else {
            return String::new();
        };

        let mut kept: Vec<u8> = Vec::new();
        let mut buffer = [0u8; 8192];
        while kept.len() <= MAX_COMMAND_BYTES {
            match pipe.read(&mut buffer) {
                Ok(0) | Err(_) => return String::from_utf8_lossy(&kept).into_owned(),
                Ok(read) => {
                    let room = MAX_COMMAND_BYTES + 1 - kept.len();
                    kept.extend_from_slice(&buffer[..read.min(room)]);
                }
            }
        }

        // Keep draining so the child can finish, but throw the rest away.
        loop {
            match pipe.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }

        String::from_utf8_lossy(&kept).into_owned()
    })
}

/// The first `MAX_COMMAND_BYTES` bytes of a command's output, on a character boundary, with a
/// note when it was cut.
fn head_of(text: &str) -> String {
    if text.len() <= MAX_COMMAND_BYTES {
        return text.to_owned();
    }

    let mut end = MAX_COMMAND_BYTES;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }

    format!(
        "{}\n… (truncated: showing the first {end} of {} bytes)",
        &text[..end],
        text.len()
    )
}

/// A required string argument. An empty string is a value; a missing key is an error.
fn required(arguments: &Value, key: &str) -> Result<String> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("the call is missing the `{key}` argument"))
}

/// The first `count` lines of `text`, and how many were left out.
fn head_lines(text: &str, count: usize) -> (Vec<String>, usize) {
    let lines: Vec<&str> = text.lines().collect();
    let shown = lines
        .iter()
        .take(count)
        .map(|line| (*line).to_owned())
        .collect();
    (shown, lines.len().saturating_sub(count))
}

/// Collapse `.` and `..` without touching the filesystem.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                // Only cancel a plain name; `..` after a root or another `..` has to stay.
                let cancellable =
                    matches!(out.components().next_back(), Some(Component::Normal(_)));
                if cancellable {
                    out.pop();
                } else {
                    out.push(component.as_os_str());
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::FunctionCall;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A unique scratch directory that cleans itself up.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "nu_plugin_ds_tools_{}_{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::SeqCst)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Scratch(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        /// A context rooted at this directory, without the command tool.
        fn context(&self) -> Context {
            Context {
                cwd: self.0.clone(),
                nu_bin: None,
                command_timeout: Duration::from_secs(5),
            }
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn call(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: "call_0".to_owned(),
            kind: "function".to_owned(),
            function: FunctionCall {
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            },
        }
    }

    #[test]
    fn resolve_handles_relative_absolute_and_tilde() {
        let cwd = Path::new("/work/project");
        assert_eq!(
            resolve(cwd, "notes.txt"),
            PathBuf::from("/work/project/notes.txt")
        );
        assert_eq!(resolve(cwd, "/etc/hosts"), PathBuf::from("/etc/hosts"));

        if let Some(home) = dirs::home_dir() {
            assert_eq!(resolve(cwd, "~/notes.txt"), home.join("notes.txt"));
            assert_eq!(resolve(cwd, "~"), home);
        }
    }

    #[test]
    fn resolve_collapses_dots_lexically() {
        let cwd = Path::new("/work/project");
        assert_eq!(
            resolve(cwd, "./a/../b.txt"),
            PathBuf::from("/work/project/b.txt")
        );
        assert_eq!(
            resolve(cwd, "../sibling/file.txt"),
            PathBuf::from("/work/sibling/file.txt")
        );
        // A `..` with nothing left to cancel has to stay.
        assert_eq!(resolve(cwd, "/../etc"), PathBuf::from("/../etc"));
    }

    #[test]
    fn write_file_creates_a_file_and_read_file_reads_it_back() {
        let scratch = Scratch::new();
        let arguments = json!({"path": "notes.txt", "content": "hello\nworld\n"}).to_string();
        let outcome = execute(&call(WRITE_FILE, &arguments), &scratch.context());
        assert!(outcome.ok, "{}", outcome.result);
        assert_eq!(
            std::fs::read_to_string(scratch.path().join("notes.txt")).unwrap(),
            "hello\nworld\n"
        );

        let read = execute(
            &call(READ_FILE, &json!({"path": "notes.txt"}).to_string()),
            &scratch.context(),
        );
        assert!(read.ok, "{}", read.result);
        assert_eq!(read.result, "hello\nworld\n");
    }

    #[test]
    fn write_file_creates_missing_parent_directories() {
        let scratch = Scratch::new();
        let arguments = json!({"path": "a/b/c.txt", "content": "nested"}).to_string();
        let outcome = execute(&call(WRITE_FILE, &arguments), &scratch.context());
        assert!(outcome.ok, "{}", outcome.result);
        assert_eq!(
            std::fs::read_to_string(scratch.path().join("a/b/c.txt")).unwrap(),
            "nested"
        );
    }

    #[test]
    fn read_file_truncates_a_long_file() {
        let scratch = Scratch::new();
        let content = "a".repeat(MAX_READ_BYTES + 50);
        std::fs::write(scratch.path().join("big.txt"), &content).unwrap();

        let outcome = execute(
            &call(READ_FILE, &json!({"path": "big.txt"}).to_string()),
            &scratch.context(),
        );
        assert!(outcome.ok, "{}", outcome.result);
        assert!(outcome.result.starts_with(&content[..MAX_READ_BYTES]));
        assert!(outcome.result.contains("truncated"), "{}", outcome.result);
        assert!(
            outcome.result.contains(&content.len().to_string()),
            "the truncation note should mention the full size: {}",
            outcome.result
        );
    }

    #[test]
    fn read_file_reports_a_missing_file() {
        let scratch = Scratch::new();
        let outcome = execute(
            &call(READ_FILE, &json!({"path": "nope.txt"}).to_string()),
            &scratch.context(),
        );
        assert!(!outcome.ok);
        assert!(
            outcome.result.contains("could not read"),
            "{}",
            outcome.result
        );
    }

    #[test]
    fn bad_calls_fail_with_a_useful_message() {
        let scratch = Scratch::new();

        let garbage = execute(&call(WRITE_FILE, "not json"), &scratch.context());
        assert!(!garbage.ok);
        assert!(
            garbage.result.contains("not valid JSON"),
            "{}",
            garbage.result
        );

        let missing = execute(&call(WRITE_FILE, r#"{"content":"hi"}"#), &scratch.context());
        assert!(!missing.ok);
        assert!(missing.result.contains("path"), "{}", missing.result);

        let unknown = execute(&call("delete_everything", "{}"), &scratch.context());
        assert!(!unknown.ok);
        assert!(
            unknown.result.contains("delete_everything"),
            "{}",
            unknown.result
        );
    }

    #[test]
    fn preview_write_reports_new_and_existing_files() {
        let scratch = Scratch::new();

        let fresh = preview(
            &call(
                WRITE_FILE,
                &json!({"path": "new.txt", "content": "hi\n"}).to_string(),
            ),
            &scratch.context(),
        )
        .unwrap();
        assert_eq!(fresh.tool, WRITE_FILE);
        assert!(
            fresh.summary.contains("creates a new file"),
            "{}",
            fresh.summary
        );
        assert_eq!(fresh.lines, vec!["hi".to_owned()]);
        assert_eq!(fresh.hidden, 0);

        std::fs::write(scratch.path().join("old.txt"), "old content").unwrap();
        let existing = preview(
            &call(
                WRITE_FILE,
                &json!({"path": "old.txt", "content": "new"}).to_string(),
            ),
            &scratch.context(),
        )
        .unwrap();
        assert!(
            existing.summary.contains("replaces the existing file"),
            "{}",
            existing.summary
        );
    }

    #[test]
    fn preview_shows_only_the_first_lines() {
        let scratch = Scratch::new();
        let content = (1..=20)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let preview = preview(
            &call(
                WRITE_FILE,
                &json!({"path": "many.txt", "content": content}).to_string(),
            ),
            &scratch.context(),
        )
        .unwrap();

        assert_eq!(preview.lines.len(), PREVIEW_LINES);
        assert_eq!(preview.hidden, 20 - PREVIEW_LINES);
        assert_eq!(preview.lines[0], "1");
    }

    /// A context whose command tool points at `nu`, or `None` when nushell is not on `PATH`.
    fn nu_context(timeout: Duration) -> Option<Context> {
        let available = std::process::Command::new("nu")
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success());
        if !available {
            eprintln!("skipping: no `nu` on PATH");
            return None;
        }
        Some(Context {
            cwd: std::env::temp_dir(),
            nu_bin: Some(PathBuf::from("nu")),
            command_timeout: timeout,
        })
    }

    #[test]
    fn risk_knows_the_command_tool() {
        assert_eq!(risk(RUN_NU), Some(Risk::Command));
        assert_eq!(risk(WRITE_FILE), Some(Risk::Write));
        assert_eq!(risk(READ_FILE), Some(Risk::Read));
        assert_eq!(risk("delete_everything"), None);
    }

    #[test]
    fn run_nu_round_trips_a_real_command() {
        let Some(context) = nu_context(Duration::from_secs(30)) else {
            return;
        };
        let outcome = execute(
            &call(RUN_NU, &json!({"command": "echo hello"}).to_string()),
            &context,
        );
        assert!(outcome.ok, "{}", outcome.result);
        assert!(outcome.result.contains("exit 0"), "{}", outcome.result);
        assert!(outcome.result.contains("hello"), "{}", outcome.result);
    }

    #[test]
    fn run_nu_reports_a_failing_command() {
        let Some(context) = nu_context(Duration::from_secs(30)) else {
            return;
        };
        let outcome = execute(
            &call(
                RUN_NU,
                &json!({"command": "print -e oops; exit 3"}).to_string(),
            ),
            &context,
        );
        assert!(!outcome.ok, "{}", outcome.result);
        assert!(outcome.result.contains("exit 3"), "{}", outcome.result);
        assert!(outcome.result.contains("[stderr]"), "{}", outcome.result);
        assert!(outcome.result.contains("oops"), "{}", outcome.result);
    }

    #[test]
    fn run_nu_stops_a_command_that_outlives_the_timeout() {
        let Some(context) = nu_context(Duration::from_millis(200)) else {
            return;
        };
        let outcome = execute(
            &call(RUN_NU, &json!({"command": "sleep 5sec"}).to_string()),
            &context,
        );
        assert!(!outcome.ok, "{}", outcome.result);
        assert!(outcome.result.contains("stopped"), "{}", outcome.result);
    }

    #[test]
    fn run_nu_without_a_command_fails_cleanly() {
        let Some(context) = nu_context(Duration::from_secs(5)) else {
            return;
        };
        let outcome = execute(&call(RUN_NU, "{}"), &context);
        assert!(!outcome.ok);
        assert!(outcome.result.contains("command"), "{}", outcome.result);
    }
}
