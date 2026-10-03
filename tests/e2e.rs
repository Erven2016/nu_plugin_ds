//! End-to-end tests that drive the real nushell binary.
//!
//! They are ignored by default because they need a `nu` on `PATH`. Run them with:
//!
//! ```text
//! cargo test --test e2e -- --ignored --nocapture
//! ```
//!
//! Each test starts a mock DeepSeek server, points the plugin at it through
//! `$env.DEEPSEEK_BASE_URL`, and isolates both nushell's plugin registry
//! (`--plugin-config`, `--plugins`) and the plugin's own config directory
//! (`$env.NU_PLUGIN_DS_CONFIG_DIR`) in a temporary directory.

mod support;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use nu_plugin_ds::api::ChatMessage;
use nu_plugin_ds::config::ThinkingEffort;
use nu_plugin_ds::session::{Session, SessionStore};

use support::{MockResponse, MockServer};

/// The plugin binary being tested.
const PLUGIN: &str = env!("CARGO_BIN_EXE_nu_plugin_ds");

/// A directory that is removed when the test finishes.
struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "nu_plugin_ds_e2e_{}_{}_{}",
            label,
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&path).expect("could not create the temp directory");
        TempDir(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A mock API server, the runtime that keeps it serving, and an isolated home.
struct Harness {
    /// Kept alive so the server's worker threads keep running for the whole test.
    _runtime: tokio::runtime::Runtime,
    server: MockServer,
    home: TempDir,
}

impl Harness {
    /// `None` when nushell is not installed, so the test can skip gracefully.
    fn new(label: &str, routes: Vec<(&'static str, MockResponse)>) -> Option<Self> {
        if !nu_available() {
            eprintln!("skipping: no `nu` on PATH");
            return None;
        }

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("could not start the runtime");
        let server = runtime.block_on(MockServer::routed(routes));

        Some(Harness {
            _runtime: runtime,
            server,
            home: TempDir::new(label),
        })
    }

    /// The base `nu` invocation, wired up to the mock server.
    fn command(&self, script: &str) -> Command {
        let mut command = Command::new("nu");
        command
            .args(["--no-config-file"])
            .arg("--plugin-config")
            .arg(self.home.path().join("plugins.msgpackz"))
            .arg("--plugins")
            .arg(PLUGIN)
            .arg("--commands")
            .arg(script)
            .env("DEEPSEEK_API_KEY", "test-key-123456")
            .env("DEEPSEEK_BASE_URL", self.server.base_url())
            .env("NU_PLUGIN_DS_CONFIG_DIR", self.home.path());
        command
    }

    /// Run a script through nushell, wired up to the mock server.
    fn nu(&self, script: &str) -> NuOutput {
        self.run(self.command(script))
    }

    /// Run a script with no `DEEPSEEK_API_KEY`, so the plugin has to fall back to the
    /// credential store. A test-only account keeps the user's real entry untouched.
    fn nu_without_api_key(&self, script: &str) -> NuOutput {
        let mut command = self.command(script);
        command.env_remove("DEEPSEEK_API_KEY").env(
            "NU_PLUGIN_DS_KEYRING_ACCOUNT",
            format!("nu_plugin_ds-e2e-{}", std::process::id()),
        );
        self.run(command)
    }

    fn run(&self, mut command: Command) -> NuOutput {
        let output = command.output().expect("could not run nu");

        NuOutput {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            success: output.status.success(),
        }
    }

    /// Run a script that must succeed, returning its standard output.
    fn ok(&self, script: &str) -> String {
        let output = self.nu(script);
        assert!(
            output.success,
            "nu failed\nscript: {script}\nstdout: {}\nstderr: {}",
            output.stdout, output.stderr
        );
        output.stdout
    }

    /// Like [`Harness::ok`], but without `DEEPSEEK_API_KEY`.
    fn ok_without_api_key(&self, script: &str) -> String {
        let output = self.nu_without_api_key(script);
        assert!(
            output.success,
            "nu failed\nscript: {script}\nstdout: {}\nstderr: {}",
            output.stdout, output.stderr
        );
        output.stdout
    }

    /// The JSON body sent to `path`.
    fn body_for(&self, path: &str) -> serde_json::Value {
        self.server.body_for(path)
    }
}

/// The result of running a script through nushell.
struct NuOutput {
    stdout: String,
    stderr: String,
    success: bool,
}

/// Skip the test when nushell is not installed.
fn nu_available() -> bool {
    Command::new("nu")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

const MODELS_PATH: &str = "/models";
const CHAT_PATH: &str = "/chat/completions";

fn models_response() -> MockResponse {
    MockResponse::json(
        r#"{"object":"list","data":[
            {"id":"deepseek-reasoner","object":"model","owned_by":"deepseek"},
            {"id":"deepseek-chat","object":"model","owned_by":"deepseek"}
        ]}"#,
    )
}

fn non_streaming_answer(text: &str) -> MockResponse {
    MockResponse::json(format!(
        r#"{{"model":"deepseek-chat","choices":[{{"index":0,"message":{{"role":"assistant","content":"{text}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":11,"completion_tokens":5,"total_tokens":16}}}}"#
    ))
}

/// A 400 rejection whose wording says the request was too long for the model.
fn context_overflow_error() -> MockResponse {
    MockResponse::status(
        400,
        r#"{"error":{"message":"This model's maximum context length is 131072 tokens. However, you requested 140000 tokens. Please reduce the length of the messages.","type":"invalid_request_error","code":"context_length_exceeded"}}"#,
    )
}

/// A blocking reply that asks for a tool call.
fn tool_call_answer(id: &str, name: &str, arguments: serde_json::Value) -> MockResponse {
    let body = serde_json::json!({
        "model": "deepseek-chat",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments.to_string()}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
    })
    .to_string();

    MockResponse::json(body)
}

fn models_and_escaped_answer(text: &str) -> Vec<(&'static str, MockResponse)> {
    let body = serde_json::json!({
        "model": "deepseek-chat",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
    })
    .to_string();

    vec![
        (MODELS_PATH, models_response()),
        (CHAT_PATH, MockResponse::json(body)),
    ]
}

fn models_route() -> Vec<(&'static str, MockResponse)> {
    vec![(MODELS_PATH, models_response())]
}

fn models_and_answer(text: &str) -> Vec<(&'static str, MockResponse)> {
    vec![
        (MODELS_PATH, models_response()),
        (CHAT_PATH, non_streaming_answer(text)),
    ]
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn lists_models_through_nushell() {
    let Some(harness) = Harness::new("models", models_route()) else {
        return;
    };

    let stdout = harness.ok(
        "let m = ds models; {count: ($m | length), active: ($m | first | get active)} | to json --raw",
    );
    assert!(
        stdout.contains("\"count\": 2") || stdout.contains("\"count\":2"),
        "got: {stdout}"
    );
    assert!(
        stdout.contains("true"),
        "the configured model should be active: {stdout}"
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn reports_the_plugin_version_through_nushell() {
    // `ds version` needs neither the API nor configuration, so the server has no routes.
    let Some(harness) = Harness::new("version", vec![]) else {
        return;
    };

    let stdout = harness.ok("ds version | to json --raw");
    assert!(
        stdout.contains(env!("CARGO_PKG_VERSION")),
        "`ds version` should report the crate version {}: {stdout}",
        env!("CARGO_PKG_VERSION")
    );
    assert!(stdout.contains("nu_plugin_ds"), "{stdout}");

    // A bare script can read just the version.
    let version = harness.ok("ds version | get version");
    assert_eq!(version.trim(), env!("CARGO_PKG_VERSION"));
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn asks_one_question_without_a_terminal() {
    let Some(harness) = Harness::new("chat", models_and_answer("hello from the mock")) else {
        return;
    };

    let stdout = harness.ok(
        "let a = chat --new --prompt 'say hi'; {type: ($a | describe), text: $a} | to json --raw",
    );
    assert!(stdout.contains("hello from the mock"), "{stdout}");
    assert!(
        stdout.contains("\"type\": \"string\"") || stdout.contains("\"type\":\"string\""),
        "a single answer should come back as a plain string: {stdout}"
    );
    assert!(
        !stdout.contains('╭'),
        "the answer must not be rendered as a table: {stdout}"
    );

    // The conversation must have been written to the isolated config directory.
    let sessions = harness.home.path().join("sessions");
    let count = std::fs::read_dir(&sessions)
        .map(|entries| entries.count())
        .unwrap_or(0);
    assert_eq!(count, 1, "the session should be stored under {sessions:?}");

    let body = harness.body_for(CHAT_PATH);
    assert_eq!(body["model"], serde_json::json!("deepseek-chat"));
    assert_eq!(body["messages"][0]["content"], serde_json::json!("say hi"));
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn generates_a_command_without_running_it() {
    let Some(harness) = Harness::new("cc", models_and_answer("ls ./")) else {
        return;
    };

    let stdout = harness.ok(
        "let c = cc --no-execute 'list every file here'; {type: ($c | describe), text: $c} | to json --raw",
    );
    assert!(stdout.contains("ls ./"), "unexpected command: {stdout}");
    assert!(
        stdout.contains("\"type\": \"string\"") || stdout.contains("\"type\":\"string\""),
        "the command should come back as a plain string: {stdout}"
    );
    assert!(
        !stdout.contains('╭'),
        "the generated command must not be rendered as a table: {stdout}"
    );

    let body = harness.body_for(CHAT_PATH);
    assert_eq!(
        body["messages"][1]["content"],
        serde_json::json!("list every file here")
    );
    assert_eq!(
        body["messages"][0]["role"],
        serde_json::json!("system"),
        "the request should carry the nushell system prompt"
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn running_a_command_returns_nothing() {
    let Some(harness) = Harness::new("cc-silent", models_and_answer("ls ./")) else {
        return;
    };

    // Without a terminal the confirmation cannot be shown, so `cc` only reports that on
    // stderr and must not return a value that nushell would draw as a table.
    let output = harness.nu("cc 'list every file here'");
    assert!(output.success, "stderr: {}", output.stderr);
    assert!(
        output.stdout.trim().is_empty(),
        "nothing should be written to stdout, got: {}",
        output.stdout
    );
    assert!(
        output.stderr.contains("not run"),
        "the reason should be explained on stderr: {}",
        output.stderr
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn manages_sessions_through_nushell() {
    // Creating and listing sessions never talks to the network.
    let Some(harness) = Harness::new("sessions", vec![]) else {
        return;
    };

    let stdout = harness.ok("ds sessions --new demo | get title");
    assert!(stdout.contains("demo"), "unexpected title: {stdout}");

    let stdout = harness.ok("ds sessions | length");
    assert!(stdout.trim().ends_with('1'), "unexpected list: {stdout}");

    let stdout = harness.ok("ds sessions | first | get model");
    assert!(
        stdout.contains("deepseek-chat"),
        "unexpected model: {stdout}"
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn explains_that_chat_needs_a_prompt_or_a_terminal() {
    let Some(harness) = Harness::new("no-prompt", vec![]) else {
        return;
    };

    let output = harness.nu("chat --new");
    assert!(!output.success, "chat without a prompt should fail");
    assert!(
        output.stderr.contains("--prompt"),
        "the error should suggest --prompt: {}",
        output.stderr
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn never_prints_the_api_key() {
    let Some(harness) = Harness::new("config", vec![]) else {
        return;
    };

    let stdout = harness.ok("ds config | select api_key_present api_key_source | to json --raw");
    assert!(
        stdout.contains("true"),
        "the key should be reported as present: {stdout}"
    );
    assert!(
        stdout.contains("nushell environment"),
        "the source should be the nushell environment: {stdout}"
    );
    assert!(
        !stdout.contains("test-key-123456"),
        "the key itself must never be printed: {stdout}"
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn evaluates_the_generated_command_in_this_session() {
    // The answer tells the session to change directory; `pwd` afterwards must report the
    // new one, which can only happen if the command ran on the live stack.
    let target = TempDir::new("cd-target");
    let answer = format!("cd \"{}\"", target.path().join("").display());

    let Some(harness) = Harness::new("cc-shell", models_and_escaped_answer(&answer)) else {
        return;
    };

    let stdout = harness.ok("cc --no-execute --shell --yes 'change to the target'; pwd");
    assert!(
        stdout.contains("nu_plugin_ds_e2e_cd-target"),
        "the session should have moved into the target directory, got: {stdout}"
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn asks_before_evaluating_in_this_session() {
    let Some(harness) = Harness::new("cc-shell-confirm", models_and_escaped_answer("cd \"..\""))
    else {
        return;
    };

    // `;` only prints the last statement's value, so `print` the returned text and keep
    // `pwd` last.
    let output = harness.nu("let c = cc --no-execute --shell 'go up'; print $c; pwd");
    assert!(output.success, "stderr: {}", output.stderr);

    // Without a terminal there is nobody to ask, so the command must not be evaluated:
    // the session has to still be in the project directory.
    assert!(
        output.stdout.contains("nu_plugin_ds"),
        "the session should not have moved up a level, got: {}",
        output.stdout
    );
    assert!(
        output.stderr.contains("--yes"),
        "the message should say how to skip the confirmation: {}",
        output.stderr
    );
    assert!(
        output.stdout.contains("cd \"..\""),
        "the command should come back as text: {}",
        output.stdout
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn stores_and_reuses_the_api_key_from_the_credential_store() {
    let Some(harness) = Harness::new("api-key", models_route()) else {
        return;
    };

    // No DEEPSEEK_API_KEY in the environment, so the plugin must use the credential store.
    let script = "let s = (ds change-api-key --api-key sk-e2e-stored | get stored); \
                  let c = (ds config | select api_key_source | to json --raw); \
                  {stored: $s, source: $c}";
    let stdout = harness.ok_without_api_key(script);

    // Remove the test entry before asserting, so a failing assertion can never leave it
    // behind in the credential store.
    let deleted = harness.ok_without_api_key("ds change-api-key --delete | get stored");

    assert!(
        stdout.contains("true"),
        "the key should have been stored: {stdout}"
    );
    assert!(
        stdout.contains("credential store"),
        "the key should be read back from the store: {stdout}"
    );

    // The verification call that `ds change-api-key` makes must have carried the new key.
    let saw_key = harness
        .server
        .raw_requests()
        .iter()
        .any(|request| request.to_lowercase().contains("bearer sk-e2e-stored"));
    assert!(saw_key, "the API client should have used the stored key");

    assert!(
        deleted.contains("false"),
        "the key should have been removed: {deleted}"
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn writes_a_file_through_the_tool_when_allowed() {
    let workspace = TempDir::new("tool-write");
    let target = workspace.path().join("from-tool.txt");

    let routes = vec![
        (MODELS_PATH, models_response()),
        (
            CHAT_PATH,
            tool_call_answer(
                "call_1",
                "write_file",
                serde_json::json!({
                    "path": target.display().to_string(),
                    "content": "hello from the tool\n"
                }),
            ),
        ),
        (CHAT_PATH, non_streaming_answer("done")),
    ];

    let Some(harness) = Harness::new("tool-write", routes) else {
        return;
    };

    let stdout = harness.ok("chat --new --prompt 'make the file' --allow-tools");
    assert!(
        stdout.contains("done"),
        "the final answer should be returned: {stdout}"
    );

    let written = std::fs::read_to_string(&target).expect("the tool should have written the file");
    assert_eq!(written, "hello from the tool\n");

    // The follow-up request must carry the assistant's call and the tool result.
    let bodies = harness.server.request_bodies();
    let follow_up = bodies
        .last()
        .expect("a second request should have been sent");
    assert_eq!(
        follow_up["messages"][1]["tool_calls"][0]["id"],
        serde_json::json!("call_1")
    );
    assert_eq!(follow_up["messages"][2]["role"], serde_json::json!("tool"));
    let content = follow_up["messages"][2]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(
        content.contains(&target.display().to_string()),
        "the tool result should mention the path: {content}"
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn does_not_write_without_approval() {
    let workspace = TempDir::new("tool-denied");
    let target = workspace.path().join("from-tool.txt");

    let routes = vec![
        (MODELS_PATH, models_response()),
        (
            CHAT_PATH,
            tool_call_answer(
                "call_1",
                "write_file",
                serde_json::json!({
                    "path": target.display().to_string(),
                    "content": "hello from the tool\n"
                }),
            ),
        ),
        (CHAT_PATH, non_streaming_answer("done")),
    ];

    let Some(harness) = Harness::new("tool-denied", routes) else {
        return;
    };

    let output = harness.nu("chat --new --prompt 'make the file'");
    assert!(output.success, "stderr: {}", output.stderr);
    assert!(
        !target.exists(),
        "the file must not be written without --allow-tools"
    );

    let bodies = harness.server.request_bodies();
    let follow_up = bodies
        .last()
        .expect("a second request should have been sent");
    assert_eq!(follow_up["messages"][2]["role"], serde_json::json!("tool"));
    let content = follow_up["messages"][2]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(
        content.contains("not run"),
        "the tool result should record that it was not run: {content}"
    );
    assert!(
        output.stderr.contains("--allow-tools"),
        "stderr should point at --allow-tools: {}",
        output.stderr
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn reads_a_file_without_approval() {
    // A read cannot change anything, so it is free by default: with no allow flags the
    // tool must have run and handed the file back to the model.
    let workspace = TempDir::new("tool-read");
    let target = workspace.path().join("input.txt");
    std::fs::write(&target, "the secret contents\n").expect("could not seed the file");

    let routes = vec![
        (MODELS_PATH, models_response()),
        (
            CHAT_PATH,
            tool_call_answer(
                "call_1",
                "read_file",
                serde_json::json!({"path": target.display().to_string()}),
            ),
        ),
        (CHAT_PATH, non_streaming_answer("done")),
    ];

    let Some(harness) = Harness::new("tool-read", routes) else {
        return;
    };

    let stdout = harness.ok("chat --new --prompt 'read the file'");
    assert!(
        stdout.contains("done"),
        "the final answer should be returned: {stdout}"
    );

    let bodies = harness.server.request_bodies();
    let follow_up = bodies
        .last()
        .expect("a second request should have been sent");
    assert_eq!(follow_up["messages"][2]["role"], serde_json::json!("tool"));
    let content = follow_up["messages"][2]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(
        content.contains("the secret contents"),
        "the read tool should have run and returned the file: {content}"
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn writes_a_file_with_allow_writes() {
    // `--allow-writes` on its own is enough for a write, so the flags really are split
    // per risk rather than only the `--allow-tools` shorthand working.
    let workspace = TempDir::new("tool-write-one");
    let target = workspace.path().join("from-tool.txt");

    let routes = vec![
        (MODELS_PATH, models_response()),
        (
            CHAT_PATH,
            tool_call_answer(
                "call_1",
                "write_file",
                serde_json::json!({
                    "path": target.display().to_string(),
                    "content": "hello from the tool\n"
                }),
            ),
        ),
        (CHAT_PATH, non_streaming_answer("done")),
    ];

    let Some(harness) = Harness::new("tool-write-one", routes) else {
        return;
    };

    let stdout = harness.ok("chat --new --prompt 'make the file' --allow-writes");
    assert!(
        stdout.contains("done"),
        "the final answer should be returned: {stdout}"
    );

    let written =
        std::fs::read_to_string(&target).expect("--allow-writes should let write_file run");
    assert_eq!(written, "hello from the tool\n");
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn runs_a_command_through_the_tool_when_allowed() {
    let routes = vec![
        (MODELS_PATH, models_response()),
        (
            CHAT_PATH,
            tool_call_answer(
                "call_1",
                "run_nu",
                serde_json::json!({"command": "echo from-the-tool"}),
            ),
        ),
        (CHAT_PATH, non_streaming_answer("ran it")),
    ];

    let Some(harness) = Harness::new("tool-run", routes) else {
        return;
    };

    let stdout = harness.ok("chat --new --prompt 'run it' --allow-commands");
    assert!(
        stdout.contains("ran it"),
        "the follow-up answer should be returned: {stdout}"
    );

    let bodies = harness.server.request_bodies();
    let follow_up = bodies
        .last()
        .expect("a second request should have been sent");
    assert_eq!(follow_up["messages"][2]["role"], serde_json::json!("tool"));
    let content = follow_up["messages"][2]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(
        content.contains("from-the-tool"),
        "the command should have run and returned its output: {content}"
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn does_not_run_a_command_without_approval() {
    let routes = vec![
        (MODELS_PATH, models_response()),
        (
            CHAT_PATH,
            tool_call_answer(
                "call_1",
                "run_nu",
                serde_json::json!({"command": "echo from-the-tool"}),
            ),
        ),
        (CHAT_PATH, non_streaming_answer("ran it")),
    ];

    let Some(harness) = Harness::new("tool-run-denied", routes) else {
        return;
    };

    let output = harness.nu("chat --new --prompt 'run it'");
    assert!(output.success, "stderr: {}", output.stderr);

    let bodies = harness.server.request_bodies();
    let follow_up = bodies
        .last()
        .expect("a second request should have been sent");
    assert_eq!(follow_up["messages"][2]["role"], serde_json::json!("tool"));
    let content = follow_up["messages"][2]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(
        content.contains("not run"),
        "the tool result should record that it was not run: {content}"
    );
    assert!(
        output.stderr.contains("--allow-commands"),
        "stderr should point at --allow-commands: {}",
        output.stderr
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn compacts_and_retries_when_the_model_rejects_the_history() {
    let routes = vec![
        (MODELS_PATH, models_response()),
        (CHAT_PATH, context_overflow_error()),
        (CHAT_PATH, non_streaming_answer("the summary")),
        (CHAT_PATH, non_streaming_answer("the final answer")),
    ];

    let Some(harness) = Harness::new("compact-retry", routes) else {
        return;
    };

    // A long history, written with the library's own types so the plugin reads it back.
    let store = SessionStore::new(harness.home.path().join("sessions"))
        .expect("could not open the session store");
    let mut session = Session::new("long", "deepseek-chat", ThinkingEffort::Off, None);
    for i in 0..6 {
        session.push(ChatMessage::user(format!("question {i}")));
        session.push(ChatMessage::assistant(format!("answer {i}")));
    }
    assert_eq!(session.messages.len(), 12);
    store.save(&session).expect("could not write the session");

    let output = harness.nu(&format!(
        "chat --session {} --prompt 'carry on'",
        session.id
    ));
    assert!(
        output.success,
        "nu failed\nstdout: {}\nstderr: {}",
        output.stdout, output.stderr
    );
    assert!(
        output.stdout.contains("the final answer"),
        "stdout should hold the answer from the retry: {}",
        output.stdout
    );
    assert!(
        output.stderr.contains("compacting"),
        "stderr should say it compacted and is retrying: {}",
        output.stderr
    );

    let chat: Vec<serde_json::Value> = harness
        .server
        .request_bodies()
        .into_iter()
        .filter(|body| body.get("messages").is_some())
        .collect();
    assert!(
        chat.len() >= 3,
        "expected the failed attempt, the summary and the retry, got {chat:#?}"
    );
    let first = chat.first().unwrap()["messages"].as_array().unwrap().len();
    let last = chat.last().unwrap()["messages"].as_array().unwrap().len();
    assert!(
        last < first,
        "the retry should send a shorter history ({last} vs {first})"
    );
}

#[test]
#[ignore = "needs a `nu` binary on PATH"]
fn asks_the_model_to_wrap_up_when_the_tool_limit_is_hit() {
    // Only one tool round is allowed this turn, so the second call is refused and the
    // model is asked for a final answer instead of the turn just stopping.
    let routes = vec![
        (MODELS_PATH, models_response()),
        (
            CHAT_PATH,
            tool_call_answer(
                "call_1",
                "run_nu",
                serde_json::json!({"command": "echo first"}),
            ),
        ),
        (
            CHAT_PATH,
            tool_call_answer(
                "call_2",
                "run_nu",
                serde_json::json!({"command": "echo second"}),
            ),
        ),
        (
            CHAT_PATH,
            non_streaming_answer("here is where things stand"),
        ),
    ];

    let Some(harness) = Harness::new("tool-limit", routes) else {
        return;
    };
    std::fs::write(
        harness.home.path().join("settings.json"),
        r#"{"max_tool_rounds": 1}"#,
    )
    .expect("could not write settings");

    let output = harness.nu("chat --new --prompt 'go' --allow-commands");
    assert!(
        output.success,
        "nu failed\nstdout: {}\nstderr: {}",
        output.stdout, output.stderr
    );
    assert!(
        output.stdout.contains("here is where things stand"),
        "the wrap-up answer should be returned: {}",
        output.stdout
    );
    assert!(
        output.stderr.contains("wrap up"),
        "stderr should say the model was asked to wrap up: {}",
        output.stderr
    );

    let chat: Vec<serde_json::Value> = harness
        .server
        .request_bodies()
        .into_iter()
        .filter(|body| body.get("messages").is_some())
        .collect();
    assert!(chat.len() >= 3, "expected three requests, got {chat:#?}");
    assert!(
        chat[0].get("tools").is_some(),
        "the first request should offer the tools: {:#?}",
        chat[0]
    );

    // The last request is the wrap-up: no tools, and the refused call is answered so the
    // history stays valid.
    let last = chat.last().unwrap();
    assert!(
        last.get("tools").is_none(),
        "the wrap-up must not offer the tools: {last:#?}"
    );
    let answered = last["messages"].as_array().unwrap().iter().any(|message| {
        message["role"] == "tool"
            && message["content"]
                .as_str()
                .unwrap_or_default()
                .contains("not run")
    });
    assert!(answered, "the refused call must be answered: {last:#?}");
}
