//! The interactive chat application: state, key handling and the streaming pipeline.

use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use futures_util::StreamExt;
use ratatui::DefaultTerminal;
use tokio::sync::Notify;

use crate::api::{
    BalanceInfo, BalanceProbe, ChatMessage, ChatRequest, DeepSeekClient, ModelInfo, StreamEvent,
    ToolCall, Usage, WireMessage,
};
use crate::config::{Settings, ThinkingEffort};
use crate::session::{Session, SessionStore, SessionSummary};
use crate::token::{estimate_messages, estimate_tokens};

use super::input::InputBuffer;
use super::tools;
use super::ui;

/// How long the event loop waits before redrawing when nothing happens.
const TICK: Duration = Duration::from_millis(60);

/// How many times a rejected-too-long request is retried after compacting.
pub const CONTEXT_RETRIES: usize = 2;

/// Everything the `chat` command prepares for the TUI.
pub struct ChatSetup {
    pub client: DeepSeekClient,
    pub settings: Settings,
    pub store: SessionStore,
    pub models: Vec<ModelInfo>,
    pub session: Session,
    /// Submitted automatically once the TUI is up.
    pub initial_prompt: Option<String>,
    /// Where the tools run: working directory, `nu` executable and command timeout.
    pub context: tools::Context,
    /// Whether the tools are offered at all.
    pub tools_enabled: bool,
    /// The tool risks the `--allow-*` flags pre-approved for this session.
    pub approved: BTreeSet<tools::Risk>,
}

/// Events funnelled into the TUI from the terminal, the network and the clock.
pub enum AppEvent {
    Term(Event),
    Stream {
        epoch: u64,
        event: StreamEvent,
    },
    StreamError {
        epoch: u64,
        message: String,
    },
    StreamDone {
        epoch: u64,
    },
    Compacted {
        epoch: u64,
        summary: String,
        removed: usize,
        usage: Option<Usage>,
    },
    CompactFailed {
        epoch: u64,
        message: String,
    },
    Models(Result<Vec<ModelInfo>, String>),
    /// A balance fetch finished. `report` is true when the user asked for it, so a
    /// failure is worth a status message.
    Balance {
        report: bool,
        probe: BalanceProbe,
    },
    /// A tool call finished on a worker thread.
    ToolDone {
        call: ToolCall,
        outcome: tools::Outcome,
    },
}

/// Which popup, if any, is on top of the transcript.
#[derive(Clone, Debug)]
pub enum Overlay {
    Models {
        index: usize,
    },
    Thinking {
        index: usize,
    },
    Sessions {
        index: usize,
    },
    Help,
    /// A confirmation before the window closes.
    Quit,
    /// A tool call waiting for approval.
    ToolCall {
        /// 1-based position of this call in the round.
        index: usize,
        total: usize,
        preview: tools::Preview,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusKind {
    Info,
    Warn,
    Error,
}

#[derive(Clone, Debug)]
pub struct Status {
    pub text: String,
    pub kind: StatusKind,
}

/// What the status bar knows about the account balance.
///
/// The balance endpoint is a DeepSeek extension, so a provider may not offer it at all.
/// Keeping the capability in the state lets the status bar show the balance exactly when
/// the provider has one, and stops the automatic refresh from polling one that does not.
#[derive(Clone, Debug, Default)]
pub enum Balance {
    /// Not fetched yet, or the last attempt was inconclusive.
    #[default]
    Unknown,
    /// The provider has no balance endpoint: show nothing and stop asking.
    Unsupported,
    /// The provider answered; show this figure and keep it refreshed.
    Known(BalanceInfo),
}

/// A tool call running on a worker thread.
#[derive(Clone, Debug)]
pub(super) struct RunningTool {
    /// The tool's name, e.g. `run_nu`.
    pub(super) name: String,
    /// When it started, so the status bar can animate and show how long it has run.
    pub(super) started: Instant,
}

/// A `Ctrl+F` search over the rendered transcript.
#[derive(Clone, Debug, Default)]
pub(super) struct Search {
    /// The query the matches were computed for, which is the prompt text while searching.
    pub(super) query: String,
    /// Transcript line indices that contain the query, in order.
    pub(super) matches: Vec<usize>,
    /// Which match is current, for the `n/total` counter and the stronger highlight.
    pub(super) current: usize,
    /// The transcript cache version the matches were computed against, so a re-render
    /// recomputes them instead of using stale line numbers.
    pub(super) version: u64,
}

/// The stage the current turn is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Summarising old history to make room.
    Compacting,
    /// Waiting for the first byte from the API.
    Waiting,
    /// Receiving tokens.
    Streaming,
    /// The stream ended; the answer is being written to the session.
    Finishing,
}

/// The in-flight turn.
pub struct StreamState {
    epoch: u64,
    pub phase: Phase,
    pub reasoning: String,
    pub content: String,
    pub usage: Option<Usage>,
    /// Tool calls the model asked for, assembled from streamed fragments.
    pub tool_calls: crate::api::ToolCallAccumulator,
    /// Set when the user cancelled the turn, so it does not carry on into a tool round.
    cancelled: bool,
    started: Instant,
}

impl StreamState {
    pub(super) fn new(epoch: u64, phase: Phase) -> Self {
        StreamState {
            epoch,
            phase,
            reasoning: String::new(),
            content: String::new(),
            usage: None,
            tool_calls: crate::api::ToolCallAccumulator::default(),
            cancelled: false,
            started: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

/// The interactive chat application.
pub struct ChatApp {
    client: DeepSeekClient,
    pub(super) settings: Settings,
    store: SessionStore,
    pub(super) models: Vec<ModelInfo>,
    /// The account balance last fetched, shown right-aligned in the status bar when the
    /// provider offers the endpoint.
    pub(super) balance: Balance,
    pub(super) session: Session,
    /// True after the open conversation was deleted from the picker: it is closed, so a
    /// later save must not recreate the file the user just removed.
    session_closed: bool,
    // Held here so the tool confirmation and the tool loop can use them without asking
    // the settings again.
    pub(super) context: tools::Context,
    pub(super) tools_enabled: bool,
    /// Tool calls from the round being streamed, waiting for a decision.
    pending: VecDeque<ToolCall>,
    /// How many calls the current round asked for, for the `1/2` in the overlay title.
    pending_total: usize,
    /// The risk classes the user allowed for the rest of the session, either up front with
    /// `--allow-*` or by pressing `a` on a call.
    pub(super) approved: BTreeSet<tools::Risk>,
    /// Tool rounds used by the turn in progress.
    tool_rounds: usize,
    /// The tool call running on a worker thread, with when it started so the status bar
    /// can animate it. The event loop keeps drawing while it runs, so a new turn must not
    /// be allowed to start in the meantime.
    pub(super) running_tool: Option<RunningTool>,
    /// Times the current turn has been compacted and retried after a "too long" refusal.
    context_retries: usize,
    /// Whether the request currently being built may carry the tools. Off for models that
    /// do not support them.
    tools_active: bool,
    /// Set when the tool-round limit is reached: the next request goes out without the
    /// tools so the model has to answer in words instead of calling another one.
    tools_withheld: bool,
    pub(super) sessions: Vec<SessionSummary>,
    pub(super) input: InputBuffer,
    pub(super) overlay: Option<Overlay>,
    pub(super) stream: Option<StreamState>,
    pub(super) status: Option<Status>,
    /// The active `Ctrl+F` search, if any.
    pub(super) search: Option<Search>,
    /// The prompt draft saved when the search opened, restored when it closes.
    search_draft: String,
    /// The rendered transcript, kept between frames so a long history is not re-rendered
    /// (and its markdown not re-parsed) on every frame.
    pub(super) transcript: ui::TranscriptCache,
    /// Lines scrolled up from the bottom of the transcript.
    pub(super) scroll: usize,
    /// The largest useful `scroll`, from the last frame. Paging up at the top must not add
    /// to `scroll` past this, or PageDown would have to burn the backlog before it moved.
    pub(super) scroll_max: usize,
    pub(super) stick_to_bottom: bool,
    /// Set when `d` was pressed once in the session picker.
    pub(super) pending_delete: bool,
    notices: Vec<String>,
    tx: Sender<AppEvent>,
    rx: Receiver<AppEvent>,
    runtime: Option<tokio::runtime::Runtime>,
    /// Bumped for every turn so stale network events can be discarded.
    epoch: u64,
    cancel: Option<Arc<Notify>>,
    should_quit: bool,
}

impl ChatApp {
    pub fn new(setup: ChatSetup) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .context("could not start the async runtime")?;

        let (tx, rx) = mpsc::channel();
        let sessions = setup.store.list().unwrap_or_default();

        let mut app = ChatApp {
            client: setup.client,
            settings: setup.settings,
            store: setup.store,
            models: setup.models,
            balance: Balance::Unknown,
            session: setup.session,
            session_closed: false,
            context: setup.context,
            tools_enabled: setup.tools_enabled,
            pending: VecDeque::new(),
            pending_total: 0,
            approved: setup.approved,
            tool_rounds: 0,
            running_tool: None,
            context_retries: 0,
            tools_active: setup.tools_enabled,
            tools_withheld: false,
            sessions,
            input: InputBuffer::new(),
            overlay: None,
            stream: None,
            status: None,
            search: None,
            search_draft: String::new(),
            transcript: ui::TranscriptCache::new(),
            scroll: 0,
            scroll_max: 0,
            stick_to_bottom: true,
            pending_delete: false,
            notices: Vec::new(),
            tx,
            rx,
            runtime: Some(runtime),
            epoch: 0,
            cancel: None,
            should_quit: false,
        };

        if let Some(prompt) = setup
            .initial_prompt
            .filter(|prompt| !prompt.trim().is_empty())
        {
            app.input.set_text(prompt);
        }

        Ok(app)
    }

    /// Run the TUI until the user quits, returning the (saved) session.
    pub fn run(mut self, terminal: &mut DefaultTerminal) -> Result<Session> {
        let stop = Arc::new(AtomicBool::new(false));
        let reader = {
            let tx = self.tx.clone();
            let stop = stop.clone();
            thread::Builder::new()
                .name("nu_plugin_ds-input".into())
                .spawn(move || read_terminal_events(tx, stop))
                .context("could not start the terminal reader")?
        };

        self.set_status(
            "Ctrl+O model · Ctrl+T thinking · Ctrl+B sessions · Ctrl+/ help · Ctrl+X quit",
            StatusKind::Info,
        );
        self.refresh_balance(false);

        if !self.input.text().trim().is_empty() {
            let prompt = self.input.take();
            self.submit(prompt);
        }

        // Draw when something changed, or while a spinner is animating, so an idle window
        // costs nothing even with a very long history.
        let mut redraw = true;
        while !self.should_quit {
            if redraw {
                terminal.draw(|frame| ui::draw(frame, &mut self))?;
            }

            redraw = match self.rx.recv_timeout(TICK) {
                Ok(event) => {
                    self.handle(event);
                    true
                }
                Err(mpsc::RecvTimeoutError::Timeout) => self.is_busy(),
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            };
        }

        // Stop the reader and wait for it so it cannot eat the user's next keystroke.
        stop.store(true, Ordering::SeqCst);
        let _ = reader.join();

        if let Some(cancel) = self.cancel.take() {
            cancel.notify_one();
        }
        self.save_session();

        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_timeout(Duration::from_millis(500));
        }

        Ok(self.session)
    }

    // ---------------------------------------------------------------- events

    fn handle(&mut self, event: AppEvent) {
        match event {
            AppEvent::Term(event) => self.handle_terminal(event),
            AppEvent::Stream { epoch, event } => {
                if self.current_epoch() != Some(epoch) {
                    return;
                }
                let Some(state) = self.stream.as_mut() else {
                    return;
                };
                match event {
                    StreamEvent::Reasoning(text) => {
                        state.reasoning.push_str(&text);
                        state.phase = Phase::Streaming;
                    }
                    StreamEvent::Content(text) => {
                        state.content.push_str(&text);
                        state.phase = Phase::Streaming;
                    }
                    StreamEvent::Usage(usage) => state.usage = Some(usage),
                    StreamEvent::ToolCall(fragment) => {
                        state.tool_calls.push(fragment);
                        state.phase = Phase::Streaming;
                    }
                    StreamEvent::Finished(_) => state.phase = Phase::Finishing,
                }
                self.follow_output();
            }
            AppEvent::StreamError { epoch, message } => {
                if self.current_epoch() != Some(epoch) {
                    return;
                }
                self.finish_stream();
                let keep = self.settings.keep_recent_messages.max(2);
                if crate::api::is_context_overflow(&message)
                    && self.context_retries < CONTEXT_RETRIES
                    && self.session.compaction_plan(keep).is_some()
                {
                    self.context_retries += 1;
                    self.set_status(
                        "the model rejected the history as too long; compacting and retrying",
                        StatusKind::Warn,
                    );
                    self.force_compaction();
                    return;
                }
                self.set_status(message, StatusKind::Error);
            }
            AppEvent::StreamDone { epoch } => {
                if self.current_epoch() != Some(epoch) {
                    return;
                }
                self.finish_stream();
            }
            AppEvent::Compacted {
                epoch,
                summary,
                removed,
                usage,
            } => {
                if self.current_epoch() != Some(epoch) {
                    return;
                }
                self.apply_compaction(summary, removed, usage);
                self.spawn_stream(epoch);
            }
            AppEvent::CompactFailed { epoch, message } => {
                if self.current_epoch() != Some(epoch) {
                    return;
                }
                self.set_status(
                    format!("compaction failed, sending the full history anyway: {message}"),
                    StatusKind::Warn,
                );
                self.spawn_stream(epoch);
            }
            AppEvent::Models(result) => match result {
                Ok(models) => {
                    if !models.iter().any(|model| model.id == self.session.model) {
                        self.notices.push(format!(
                            "`{}` is not in the model list returned by the API",
                            self.session.model
                        ));
                    }
                    self.models = models;
                    self.set_status("model list refreshed", StatusKind::Info);
                }
                Err(message) => self.set_status(message, StatusKind::Warn),
            },
            AppEvent::Balance { report, probe } => match probe {
                BalanceProbe::Available(info) => {
                    self.balance = Balance::Known(info);
                    if report {
                        self.set_status("balance updated", StatusKind::Info);
                    }
                }
                BalanceProbe::Unsupported => {
                    // The provider has no balance endpoint: stop showing a figure and stop
                    // polling it for the rest of the session.
                    self.balance = Balance::Unsupported;
                    if report {
                        self.set_status(
                            format!(
                                "{} does not offer a balance endpoint",
                                self.client.base_url()
                            ),
                            StatusKind::Warn,
                        );
                    }
                }
                // Inconclusive: keep whatever figure is on screen and try again later.
                BalanceProbe::Failed(message) => {
                    if report {
                        self.set_status(message, StatusKind::Error);
                    }
                }
            },
            AppEvent::ToolDone { call, outcome } => {
                self.running_tool = None;
                self.announce_tool(&call, &outcome);
                self.record_tool_result(&call, &outcome);
                self.follow_output();
                // Continue the round, or hand back to the model, now that it is answered.
                self.next_tool();
            }
        }
    }

    fn handle_terminal(&mut self, event: Event) {
        match event {
            Event::Key(key) if key.kind == KeyEventKind::Press => self.on_key(key),
            Event::Paste(text) => {
                self.input.insert_str(&text);
                self.follow_output();
            }
            _ => {}
        }
    }

    fn on_key(&mut self, key: KeyEvent) {
        if self.overlay.is_some() {
            self.on_overlay_key(key);
            return;
        }

        if is_help_shortcut(&key) {
            self.open_overlay(Overlay::Help);
            return;
        }

        if is_search_shortcut(&key) {
            if self.search.is_some() {
                self.close_search();
            } else {
                self.open_search();
            }
            return;
        }

        // While searching, a few keys drive the search; the rest fall through to the normal
        // input handling, which edits the query.
        if self.search.is_some() && self.on_search_key(key) {
            return;
        }

        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);

        // Keys with the control modifier first, so the plain-character arm below can
        // stay a simple catch-all.
        if ctrl {
            match key.code {
                KeyCode::Char('c') => {
                    if self.stream.is_some() {
                        self.cancel_stream();
                    } else if self.running_tool.is_some() {
                        // A detached worker cannot be stopped mid-call, so quitting here
                        // could cut a write short or orphan the command's process.
                        self.set_status(
                            "a tool call is still running; Ctrl+X quits anyway",
                            StatusKind::Warn,
                        );
                    } else {
                        self.request_quit();
                    }
                    return;
                }
                KeyCode::Char('x') => {
                    self.request_quit();
                    return;
                }
                KeyCode::Char('d') => {
                    if self.input.is_empty() {
                        self.request_quit();
                    } else {
                        self.input.delete();
                    }
                    return;
                }
                KeyCode::Char('o') => {
                    let index = self
                        .models
                        .iter()
                        .position(|model| model.id == self.session.model)
                        .unwrap_or(0);
                    self.open_overlay(Overlay::Models { index });
                    return;
                }
                KeyCode::Char('t') => {
                    let index = ThinkingEffort::ALL
                        .iter()
                        .position(|effort| *effort == self.session.thinking)
                        .unwrap_or(0);
                    self.open_overlay(Overlay::Thinking { index });
                    return;
                }
                KeyCode::Char('b') => {
                    self.refresh_sessions();
                    self.open_overlay(Overlay::Sessions { index: 0 });
                    return;
                }
                KeyCode::Char('n') => {
                    self.new_session(None);
                    return;
                }
                KeyCode::Char('r') => {
                    self.regenerate();
                    return;
                }
                KeyCode::Char('l') => {
                    self.scroll = 0;
                    self.stick_to_bottom = true;
                    self.set_status("jumped to the end of the transcript", StatusKind::Info);
                    return;
                }
                KeyCode::Char('w') => {
                    self.input.delete_word();
                    return;
                }
                KeyCode::Char('k') => {
                    self.input.kill_to_end();
                    return;
                }
                KeyCode::Char('u') => {
                    self.input.kill_line();
                    return;
                }
                KeyCode::Char('a') => {
                    self.input.move_home();
                    return;
                }
                KeyCode::Char('e') => {
                    self.input.move_end();
                    return;
                }
                KeyCode::Enter => {
                    self.input.insert_char('\n');
                    return;
                }
                KeyCode::Home => {
                    self.scroll_to_top();
                    return;
                }
                KeyCode::End => {
                    self.scroll = 0;
                    self.stick_to_bottom = true;
                    return;
                }
                _ => {}
            }
        }

        match key.code {
            KeyCode::Esc => {
                if self.stream.is_some() {
                    self.cancel_stream();
                } else if !self.input.is_empty() {
                    self.input.clear();
                }
            }
            // Shift+Enter inserts a newline instead of sending.
            KeyCode::Enter if shift => self.input.insert_char('\n'),
            // Alt+Enter is deliberately left unbound (some terminals use it for fullscreen),
            // and must not fall through to sending the prompt.
            KeyCode::Enter if alt => {}
            KeyCode::Enter => {
                let text = self.input.take();
                self.submit(text);
            }
            // Plain typing only: control chords that were not handled above are ignored
            // rather than inserting their letter.
            KeyCode::Char(ch) if !alt && !ctrl => {
                self.input.insert_char(ch);
                self.follow_output();
            }
            KeyCode::Backspace => {
                self.input.backspace();
                self.follow_output();
            }
            KeyCode::Delete => self.input.delete(),
            KeyCode::Left => self.input.move_left(),
            KeyCode::Right => self.input.move_right(),
            KeyCode::Up if alt => self.scroll_by(1),
            KeyCode::Down if alt => self.scroll_by(-1),
            KeyCode::Up => {
                if !self.input.move_up() && !self.input.history_prev() {
                    self.scroll_by(1);
                }
            }
            KeyCode::Down => {
                if !self.input.move_down() && !self.input.history_next() {
                    self.scroll_by(-1);
                }
            }
            KeyCode::PageUp => self.scroll_by(10),
            KeyCode::PageDown => self.scroll_by(-10),
            KeyCode::Home => self.input.move_home(),
            KeyCode::End => self.input.move_end(),
            _ => {}
        }
    }

    /// Handle a key while a search is open. Returns `true` when it is consumed; anything not
    /// handled here (typing, editing) falls through and edits the query.
    fn on_search_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        match key.code {
            KeyCode::Esc => self.close_search(),
            KeyCode::Enter if shift => self.step_match(-1),
            KeyCode::Enter if !alt && !ctrl => self.step_match(1),
            KeyCode::Up => self.step_match(-1),
            KeyCode::Down => self.step_match(1),
            _ => return false,
        }
        true
    }

    /// Open the search box, saving the prompt draft to restore when it closes.
    pub(super) fn open_search(&mut self) {
        if self.search.is_some() {
            return;
        }
        self.search_draft = self.input.take();
        self.search = Some(Search::default());
        self.set_status(
            "search: type to filter, Enter/Shift+Enter to step, Esc to close",
            StatusKind::Info,
        );
    }

    /// Close the search and put the prompt draft back.
    pub(super) fn close_search(&mut self) {
        if self.search.take().is_none() {
            return;
        }
        self.input.set_text(std::mem::take(&mut self.search_draft));
        self.set_status("search closed", StatusKind::Info);
    }

    fn on_overlay_key(&mut self, key: KeyEvent) {
        let Some(overlay) = self.overlay.clone() else {
            return;
        };

        match overlay {
            // Any key dismisses the help card.
            Overlay::Help => self.overlay = None,
            Overlay::Models { mut index } => {
                match key.code {
                    KeyCode::Esc => {
                        self.overlay = None;
                        return;
                    }
                    KeyCode::Up | KeyCode::Char('k') => index = index.saturating_sub(1),
                    KeyCode::Down | KeyCode::Char('j') => {
                        index = (index + 1).min(self.models.len().saturating_sub(1))
                    }
                    KeyCode::Char('r') => {
                        self.refresh_models();
                        return;
                    }
                    KeyCode::Enter => {
                        if let Some(model) = self.models.get(index) {
                            self.session.model = model.id.clone();
                            self.session.touch();
                            self.set_status(
                                format!("model switched to {}", model.id),
                                StatusKind::Info,
                            );
                        }
                        self.overlay = None;
                        return;
                    }
                    _ => {}
                }
                self.overlay = Some(Overlay::Models { index });
            }
            Overlay::Thinking { mut index } => {
                match key.code {
                    KeyCode::Esc => {
                        self.overlay = None;
                        return;
                    }
                    KeyCode::Up | KeyCode::Char('k') => index = index.saturating_sub(1),
                    KeyCode::Down | KeyCode::Char('j') => {
                        index = (index + 1).min(ThinkingEffort::ALL.len().saturating_sub(1))
                    }
                    KeyCode::Enter => {
                        if let Some(effort) = ThinkingEffort::ALL.get(index) {
                            self.session.thinking = *effort;
                            self.session.touch();
                            self.set_status(
                                format!("thinking level set to {}", effort.label()),
                                StatusKind::Info,
                            );
                        }
                        self.overlay = None;
                        return;
                    }
                    _ => {}
                }
                self.overlay = Some(Overlay::Thinking { index });
            }
            Overlay::Sessions { mut index } => {
                match key.code {
                    KeyCode::Esc => {
                        self.pending_delete = false;
                        self.overlay = None;
                        return;
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        self.pending_delete = false;
                        index = index.saturating_sub(1)
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        self.pending_delete = false;
                        index = (index + 1).min(self.sessions.len().saturating_sub(1))
                    }
                    KeyCode::Char('n') => {
                        self.new_session(None);
                        index = 0;
                    }
                    KeyCode::Char('d') => {
                        if self.pending_delete {
                            self.delete_session(index);
                            index = index.min(self.sessions.len().saturating_sub(1));
                        } else {
                            self.pending_delete = true;
                            self.set_status(
                                "press `d` again to delete this session",
                                StatusKind::Warn,
                            );
                        }
                    }
                    KeyCode::Enter => {
                        self.load_session(index);
                        self.overlay = None;
                        return;
                    }
                    _ => {}
                }
                self.overlay = Some(Overlay::Sessions { index });
            }
            Overlay::ToolCall { preview, .. } => match key.code {
                KeyCode::Enter => {
                    self.overlay = None;
                    self.answer_tool(true);
                }
                KeyCode::Esc => {
                    self.overlay = None;
                    self.answer_tool(false);
                }
                KeyCode::Char('a') => {
                    // Allow this class of call for the rest of the session, not everything.
                    self.approved.insert(preview.risk);
                    self.overlay = None;
                    self.answer_tool(true);
                }
                _ => {}
            },
            Overlay::Quit => match key.code {
                KeyCode::Enter | KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.should_quit = true;
                }
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Char('N') => {
                    self.overlay = None;
                }
                _ => {}
            },
        }
    }

    fn open_overlay(&mut self, overlay: Overlay) {
        self.pending_delete = false;
        self.overlay = Some(overlay);
    }

    /// Ask before quitting instead of closing on the first key. The session is saved on the
    /// way out either way.
    fn request_quit(&mut self) {
        self.open_overlay(Overlay::Quit);
    }

    // -------------------------------------------------------------- commands

    /// Either run a slash command or start a new turn.
    fn submit(&mut self, text: String) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }
        if trimmed.starts_with('/') && self.run_slash_command(trimmed) {
            return;
        }

        if self.stream.is_some() {
            self.set_status(
                "still waiting for the previous answer (Ctrl+C to cancel)",
                StatusKind::Warn,
            );
            self.input.set_text(text);
            return;
        }
        if self.running_tool.is_some() {
            self.set_status(
                "a tool call is still running; wait for it to finish",
                StatusKind::Warn,
            );
            self.input.set_text(text);
            return;
        }

        self.session_closed = false;
        self.session.push(ChatMessage::user(trimmed));
        self.scroll = 0;
        self.stick_to_bottom = true;
        self.start_turn();
    }

    /// Returns `true` when the line was a slash command.
    fn run_slash_command(&mut self, line: &str) -> bool {
        let mut parts = line.splitn(2, char::is_whitespace);
        let command = parts.next().unwrap_or_default();
        let argument = parts.next().unwrap_or_default().trim();

        match command {
            "/help" | "/?" => self.open_overlay(Overlay::Help),
            "/quit" | "/exit" | "/q" => self.request_quit(),
            "/clear" => {
                self.session.clear_transcript();
                // The cleared transcript has to reach the file here: switching away and back
                // reloads the session from disk, which would otherwise restore it.
                self.save_session_now();
                self.refresh_sessions();
                self.set_status("transcript cleared", StatusKind::Info);
            }
            "/new" => self.new_session((!argument.is_empty()).then_some(argument)),
            "/save" => {
                self.save_session();
                self.set_status("session saved", StatusKind::Info);
            }
            "/system" => self.set_system_prompt(argument),
            "/markdown" => self.toggle_markdown(),
            "/tools" => self.toggle_tools(),
            "/model" => {
                if argument.is_empty() {
                    let index = self
                        .models
                        .iter()
                        .position(|model| model.id == self.session.model)
                        .unwrap_or(0);
                    self.open_overlay(Overlay::Models { index });
                } else {
                    self.session.model = argument.to_owned();
                    self.session.touch();
                    self.set_status(format!("model switched to {argument}"), StatusKind::Info);
                }
            }
            "/think" => {
                if argument.is_empty() {
                    let index = ThinkingEffort::ALL
                        .iter()
                        .position(|effort| *effort == self.session.thinking)
                        .unwrap_or(0);
                    self.open_overlay(Overlay::Thinking { index });
                } else if let Some(effort) = ThinkingEffort::parse(argument) {
                    self.session.thinking = effort;
                    self.session.touch();
                    self.set_status(
                        format!("thinking level set to {}", effort.label()),
                        StatusKind::Info,
                    );
                } else {
                    self.set_status(
                        format!("`{argument}` is not a thinking level (off/low/medium/high)"),
                        StatusKind::Warn,
                    );
                }
            }
            "/sessions" => {
                self.refresh_sessions();
                self.open_overlay(Overlay::Sessions { index: 0 });
            }
            "/models" => self.refresh_models(),
            "/balance" => self.refresh_balance(true),
            "/compact" => self.force_compaction(),
            "/regenerate" => self.regenerate(),
            other => self.set_status(
                format!("unknown command `{other}`; try /help"),
                StatusKind::Warn,
            ),
        }
        true
    }

    fn force_compaction(&mut self) {
        if self.is_busy() {
            self.set_status(
                "cannot compact while the answer or a tool call is still running",
                StatusKind::Warn,
            );
            return;
        }
        let epoch = self.begin_epoch();
        self.stream = Some(StreamState::new(epoch, Phase::Compacting));
        self.spawn_compaction(epoch, true);
    }

    fn set_system_prompt(&mut self, prompt: &str) {
        if self.session.system_prompt().is_some() {
            self.session.messages.remove(0);
        }
        if prompt.trim().is_empty() {
            self.set_status("system prompt removed", StatusKind::Info);
        } else {
            self.session.messages.insert(0, ChatMessage::system(prompt));
            self.set_status("system prompt updated", StatusKind::Info);
        }
        self.session.touch();
    }

    fn toggle_markdown(&mut self) {
        self.settings.markdown = !self.settings.markdown;
        let state = if self.settings.markdown { "on" } else { "off" };
        self.set_status(format!("markdown formatting {state}"), StatusKind::Info);
    }

    fn toggle_tools(&mut self) {
        self.tools_enabled = !self.tools_enabled;
        self.tools_active = self.tools_enabled;
        self.approved.clear();
        let state = if self.tools_enabled { "on" } else { "off" };
        self.set_status(format!("file tools {state}"), StatusKind::Info);
    }

    fn new_session(&mut self, name: Option<&str>) {
        self.save_session();
        self.session = Session::new(
            name.unwrap_or(""),
            self.session.model.clone(),
            self.session.thinking,
            self.settings.system_prompt.as_deref(),
        );
        self.session_closed = false;
        self.refresh_sessions();
        self.scroll = 0;
        self.stick_to_bottom = true;
        self.input.clear();
        self.set_status(format!("new session {}", self.session.id), StatusKind::Info);
    }

    fn load_session(&mut self, index: usize) {
        let Some(summary) = self.sessions.get(index).cloned() else {
            return;
        };
        self.save_session();
        match self.store.load(&summary.id) {
            Ok(session) => {
                self.session = session;
                self.session_closed = false;
                self.scroll = 0;
                self.stick_to_bottom = true;
                self.set_status(
                    format!("switched to session {}", summary.title),
                    StatusKind::Info,
                );
            }
            Err(err) => self.set_status(format!("{err:#}"), StatusKind::Error),
        }
    }

    fn delete_session(&mut self, index: usize) {
        let Some(summary) = self.sessions.get(index).cloned() else {
            return;
        };
        let was_current = summary.id == self.session.id;
        match self.store.delete(&summary.id) {
            Ok(()) => {
                self.pending_delete = false;
                self.sessions.retain(|entry| entry.id != summary.id);
                if was_current {
                    // Close the conversation that was deleted: replace it with a blank one
                    // that is never saved, so the file cannot come back, and leave the picker
                    // open so another conversation can be chosen.
                    self.close_current_session();
                    self.set_status(
                        format!(
                            "deleted and closed {}; choose another conversation",
                            summary.title
                        ),
                        StatusKind::Info,
                    );
                } else {
                    self.set_status(format!("deleted {}", summary.title), StatusKind::Info);
                }
            }
            Err(err) => self.set_status(format!("{err:#}"), StatusKind::Error),
        }
    }

    /// Replace the open conversation with a blank one that is not written to disk until the
    /// next turn starts.
    fn close_current_session(&mut self) {
        self.session = Session::new(
            "",
            self.session.model.clone(),
            self.session.thinking,
            self.settings.system_prompt.as_deref(),
        );
        self.session_closed = true;
        self.scroll = 0;
        self.stick_to_bottom = true;
        self.input.clear();
    }

    fn refresh_sessions(&mut self) {
        self.sessions = self.store.list().unwrap_or_default();
    }

    fn regenerate(&mut self) {
        if self.is_busy() {
            self.set_status(
                "still waiting for the previous answer or tool call",
                StatusKind::Warn,
            );
            return;
        }
        if self.session.drop_last_answer() {
            self.start_turn();
        } else {
            self.set_status("there is no answer to regenerate", StatusKind::Warn);
        }
    }

    fn save_session(&mut self) {
        if self.session.messages.is_empty() {
            return;
        }
        self.save_session_now();
    }

    /// Write the session to disk even when it holds no messages.
    ///
    /// [`Self::save_session`] deliberately skips a brand new, still empty session so it does
    /// not litter the store, but `/clear` leaves a transcript empty on purpose and that empty
    /// state has to reach the file for the clear to stick.
    fn save_session_now(&mut self) {
        // A conversation the user deleted must stay deleted: its file is gone and the
        // in-memory copy is closed, so saving would only bring it back.
        if self.session_closed {
            return;
        }
        if let Err(err) = self.store.save(&self.session) {
            self.set_status(
                format!("could not save the session: {err:#}"),
                StatusKind::Error,
            );
        }
    }

    // -------------------------------------------------------------- pipeline

    fn current_epoch(&self) -> Option<u64> {
        self.stream.as_ref().map(|state| state.epoch)
    }

    fn begin_epoch(&mut self) -> u64 {
        self.epoch += 1;
        self.epoch
    }

    fn spawn<F>(&self, future: F)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        if let Some(runtime) = &self.runtime {
            runtime.spawn(future);
        }
    }

    /// Build the request for the current session.
    fn build_request(&self, stream: bool, messages: &[ChatMessage]) -> ChatRequest {
        let request = crate::api::build_chat_request(
            &self.session.model,
            self.session.thinking,
            self.settings.max_tokens,
            self.settings.temperature,
            messages,
            stream,
        );

        if self.tools_active && !self.tools_withheld {
            let commands = self.tools_enabled && self.context.nu_bin.is_some();
            crate::api::with_tools(request, tools::definitions(commands))
        } else {
            request
        }
    }

    fn start_turn(&mut self) {
        self.tool_rounds = 0;
        self.context_retries = 0;
        self.tools_withheld = false;

        // `deepseek-reasoner` does not support tool calling, so the tools are left out
        // rather than making every request fail.
        self.tools_active = self.tools_enabled && !self.session.model.contains("reasoner");
        if self.tools_enabled && !self.tools_active {
            let notice = format!(
                "`{}` does not support tool calls, so the file tools are off",
                self.session.model
            );
            if !self.notices.contains(&notice) {
                self.notices.push(notice);
            }
        }

        let epoch = self.begin_epoch();
        self.stream = Some(StreamState::new(epoch, Phase::Waiting));

        if self.compaction_plan().is_some() {
            if let Some(state) = self.stream.as_mut() {
                state.phase = Phase::Compacting;
            }
            self.spawn_compaction(epoch, false);
        } else {
            self.spawn_stream(epoch);
        }
    }

    /// The token count at which the history is summarised.
    fn compact_at(&self) -> usize {
        self.settings.compact_at(&self.session.model)
    }

    pub(super) fn context_usage(&self) -> (usize, usize) {
        // The committed history is estimated once per change (the transcript cache keeps
        // it), not once per frame, so a long conversation does not cost anything here.
        let mut used = self.transcript.tokens();
        if let Some(state) = &self.stream {
            used += estimate_tokens(&state.reasoning);
            used += estimate_tokens(&state.content);
        }
        (
            used,
            crate::config::model_context_limit(&self.session.model) as usize,
        )
    }

    /// The messages to summarise and how many of them would be dropped.
    ///
    /// The token estimate is checked here, not in [`Session::compaction_plan`]: the session
    /// owns *which* messages go, the app decides *when*.
    fn compaction_plan(&self) -> Option<(Vec<ChatMessage>, usize)> {
        self.plan_keeping(self.settings.keep_recent_messages.max(2))
    }

    /// Like [`ChatApp::compaction_plan`], but always yields something when there is
    /// anything at all to compress (used by `/compact`).
    fn forced_compaction_plan(&self) -> Option<(Vec<ChatMessage>, usize)> {
        self.session.compaction_plan(2)
    }

    fn plan_keeping(&self, keep: usize) -> Option<(Vec<ChatMessage>, usize)> {
        if estimate_messages(&self.session.messages) < self.compact_at() {
            return None;
        }
        self.session.compaction_plan(keep)
    }

    fn spawn_compaction(&mut self, epoch: u64, forced: bool) {
        let plan = if forced {
            self.forced_compaction_plan()
        } else {
            self.compaction_plan()
        };

        let Some((to_summarise, removed)) = plan else {
            if forced {
                self.stream = None;
                self.set_status("there is not enough history to compact", StatusKind::Warn);
            } else {
                self.spawn_stream(epoch);
            }
            return;
        };

        let client = self.client.clone();
        let model = self.session.model.clone();
        let tx = self.tx.clone();

        self.spawn(async move {
            match summarise(&client, &model, &to_summarise).await {
                Ok((summary, usage)) => {
                    let _ = tx.send(AppEvent::Compacted {
                        epoch,
                        summary,
                        removed,
                        usage,
                    });
                }
                Err(err) => {
                    let _ = tx.send(AppEvent::CompactFailed {
                        epoch,
                        message: format!("{err:#}"),
                    });
                }
            }
        });
    }

    fn apply_compaction(&mut self, summary: String, removed: usize, usage: Option<Usage>) {
        let keep = self.settings.keep_recent_messages.max(2);
        self.session.apply_summary(&summary, keep);
        if let Some(usage) = usage {
            self.session.usage.merge(&usage);
        }
        self.save_session();
        self.scroll = 0;
        self.stick_to_bottom = true;

        self.set_status(
            format!(
                "compacted {removed} messages into a summary ({} tokens left in the estimate)",
                estimate_messages(&self.session.messages)
            ),
            StatusKind::Info,
        );
    }

    fn spawn_stream(&mut self, epoch: u64) {
        self.repair_history();
        let request = self.build_request(true, &self.session.messages);
        let client = self.client.clone();
        let tx = self.tx.clone();
        let cancel = Arc::new(Notify::new());
        self.cancel = Some(cancel.clone());

        if let Some(state) = self.stream.as_mut() {
            state.phase = Phase::Waiting;
            state.started = Instant::now();
        }

        self.spawn(async move {
            let mut stream = match client.stream(&request).await {
                Ok(stream) => stream,
                Err(err) => {
                    let _ = tx.send(AppEvent::StreamError {
                        epoch,
                        message: format!("{err:#}"),
                    });
                    return;
                }
            };

            loop {
                tokio::select! {
                    _ = cancel.notified() => break,
                    item = stream.next() => {
                        match item {
                            None => break,
                            Some(Err(err)) => {
                                let _ = tx.send(AppEvent::StreamError {
                                    epoch,
                                    message: format!("{err:#}"),
                                });
                                return;
                            }
                            Some(Ok(event)) => {
                                let finished = matches!(event, StreamEvent::Finished(_));
                                if tx.send(AppEvent::Stream { epoch, event }).is_err() {
                                    return;
                                }
                                if finished {
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            let _ = tx.send(AppEvent::StreamDone { epoch });
        });
    }

    /// Turn the finished stream into messages in the session, and start a tool round when
    /// the model asked for one.
    fn finish_stream(&mut self) {
        let Some(state) = self.stream.take() else {
            return;
        };
        self.cancel = None;

        if let Some(usage) = state.usage {
            self.session.usage.merge(&usage);
        }

        let calls = state.tool_calls.finish();
        let cancelled = state.cancelled;

        if !state.content.trim().is_empty() || !calls.is_empty() {
            let mut message = ChatMessage::assistant(state.content.clone())
                .with_reasoning(state.reasoning.clone());
            if !calls.is_empty() {
                message.tool_calls = Some(calls.clone());
            }
            self.session.push(message);
            self.save_session();
        }

        self.follow_output();

        if calls.is_empty() {
            self.refresh_balance(false);
            return;
        }

        if cancelled {
            // Answer the calls so the history stays valid, but do not keep going.
            for call in &calls {
                self.record_tool_result(call, &tools::declined());
            }
            self.refresh_balance(false);
            return;
        }

        self.begin_tool_round(calls);
    }

    /// Start a round of tool calls, or refuse to start another one.
    fn begin_tool_round(&mut self, calls: Vec<ToolCall>) {
        if self.tool_rounds < self.settings.max_tool_rounds {
            self.tool_rounds += 1;
            self.pending = calls.into_iter().collect();
            self.pending_total = self.pending.len();
            self.next_tool();
            return;
        }

        // The budget is spent. Answer the outstanding calls anyway, or the API would
        // reject the history for a `tool_call` with no result.
        for call in &calls {
            let outcome = tools::Outcome::failed(format!(
                "not run: this turn already used {} tool rounds",
                self.settings.max_tool_rounds
            ));
            self.record_tool_result(call, &outcome);
        }

        if self.tools_withheld {
            // The wrap-up was already asked for and the model still called a tool, which
            // it should not be able to do: stop rather than loop.
            self.set_status(
                "stopped: too many tool rounds in one turn",
                StatusKind::Warn,
            );
            return;
        }

        // One last request without the tools, so the turn ends with the model explaining
        // where things stand instead of leaving the user with a bare tool result.
        self.tools_withheld = true;
        self.set_status(
            "tool limit reached; asking the model to wrap up",
            StatusKind::Warn,
        );
        let epoch = self.epoch;
        self.stream = Some(StreamState::new(epoch, Phase::Waiting));
        self.spawn_stream(epoch);
    }

    /// Show, run or skip the next pending call.
    fn next_tool(&mut self) {
        let Some(call) = self.pending.front().cloned() else {
            // Everything is answered; let the model carry on with the same turn.
            let epoch = self.epoch;
            self.stream = Some(StreamState::new(epoch, Phase::Waiting));
            self.spawn_stream(epoch);
            return;
        };

        // `front` was cloned, so the position is the one about to be consumed.
        let position = self.pending_total - self.pending.len() + 1;

        // An unknown tool has no risk to approve: run it so the call fails with the
        // `there is no tool called` result instead of asking.
        let risk = tools::risk(&call.function.name);
        if risk.is_none_or(|risk| !self.needs_approval(risk)) {
            self.pending.pop_front();
            // The tool runs on a worker thread; `ToolDone` carries on with the round.
            self.run_tool(call);
            return;
        }

        match tools::preview(&call, &self.context) {
            Ok(preview) => {
                self.overlay = Some(Overlay::ToolCall {
                    index: position,
                    total: self.pending_total,
                    preview,
                });
            }
            // Nothing to approve when the arguments are unusable: tell the model instead.
            Err(err) => {
                self.pending.pop_front();
                let outcome =
                    tools::Outcome::failed(format!("the arguments could not be used: {err:#}"));
                self.record_tool_result(&call, &outcome);
                self.next_tool();
            }
        }
    }

    /// Whether a turn, a compaction or a tool call is in flight.
    pub(super) fn is_busy(&self) -> bool {
        self.stream.is_some() || self.running_tool.is_some()
    }

    /// Whether a call of this risk has to be confirmed.
    fn needs_approval(&self, risk: tools::Risk) -> bool {
        let configured = match risk {
            tools::Risk::Read => self.settings.confirm_tool_reads,
            tools::Risk::Write => self.settings.confirm_tool_writes,
            tools::Risk::Command => self.settings.confirm_tool_commands,
        };
        configured && !self.approved.contains(&risk)
    }

    /// Whether no outstanding class of call needs confirmation any more, so the status bar
    /// can say the tools run automatically.
    pub(super) fn tools_auto(&self) -> bool {
        [tools::Risk::Read, tools::Risk::Write, tools::Risk::Command]
            .iter()
            .all(|risk| !self.needs_approval(*risk))
    }

    /// Apply the user's decision to the call the overlay is showing.
    fn answer_tool(&mut self, run: bool) {
        let Some(call) = self.pending.pop_front() else {
            return;
        };
        if run {
            // The tool runs on a worker thread; `ToolDone` carries on with the round.
            self.run_tool(call);
        } else {
            self.record_tool_result(&call, &tools::declined());
            self.set_status(format!("skipped {}", call.function.name), StatusKind::Warn);
            self.next_tool();
        }
    }

    /// Run one call on a worker thread, reporting the result as [`AppEvent::ToolDone`].
    ///
    /// This is the UI thread, so a slow command must not be executed here: doing so would
    /// freeze the whole interface until it finished. The work is handed to a plain thread
    /// and the outcome comes back through the event channel, which keeps the window drawing
    /// and responsive while the command runs.
    fn run_tool(&mut self, call: ToolCall) {
        self.running_tool = Some(RunningTool {
            name: call.function.name.clone(),
            started: Instant::now(),
        });

        let context = self.context.clone();
        let tx = self.tx.clone();
        let spawned = {
            let call = call.clone();
            std::thread::Builder::new()
                .name("nu_plugin_ds-tool".into())
                .spawn(move || {
                    let outcome = tools::execute(&call, &context);
                    let _ = tx.send(AppEvent::ToolDone { call, outcome });
                })
        };

        if let Err(err) = spawned {
            // No worker thread: answer the call here so the turn can still finish.
            self.running_tool = None;
            let outcome = tools::Outcome::failed(format!("could not start a worker thread: {err}"));
            self.announce_tool(&call, &outcome);
            self.record_tool_result(&call, &outcome);
            self.next_tool();
        }
    }

    /// Put a finished call's outcome in the status line.
    fn announce_tool(&mut self, call: &ToolCall, outcome: &tools::Outcome) {
        let summary = first_line(&outcome.result);
        if outcome.ok {
            self.set_status(
                format!("{}: {summary}", call.function.name),
                StatusKind::Info,
            );
        } else {
            self.set_status(
                format!("{} failed: {summary}", call.function.name),
                StatusKind::Warn,
            );
        }
    }

    fn record_tool_result(&mut self, call: &ToolCall, outcome: &tools::Outcome) {
        self.session
            .push(ChatMessage::tool(&call.id, outcome.result.clone()));
        self.save_session();
    }

    /// Make sure every tool call in the history has an answer before it is sent: the API
    /// rejects a conversation where one is missing, and a turn can end early.
    fn repair_history(&mut self) {
        for call in self.session.unanswered_tool_calls() {
            let outcome = tools::Outcome::failed("not run: this call was never executed");
            self.record_tool_result(&call, &outcome);
        }
    }

    fn cancel_stream(&mut self) {
        if let Some(state) = self.stream.as_mut() {
            state.cancelled = true;
        }
        if let Some(cancel) = self.cancel.take() {
            cancel.notify_one();
            self.set_status("cancelling…", StatusKind::Warn);
        }
    }

    fn refresh_models(&mut self) {
        let client = self.client.clone();
        let tx = self.tx.clone();
        self.spawn(async move {
            let result = client
                .list_models()
                .await
                .map_err(|err| format!("could not refresh the model list: {err:#}"));
            let _ = tx.send(AppEvent::Models(result));
        });
    }

    /// Fetch the account balance. `report` makes a failure visible; the automatic refresh
    /// after a turn keeps quiet so a base URL without the endpoint does not interrupt.
    ///
    /// Once the provider is known to lack the endpoint the automatic refresh stops asking;
    /// an explicit `/balance` always re-probes.
    fn refresh_balance(&mut self, report: bool) {
        if !report && matches!(self.balance, Balance::Unsupported) {
            return;
        }
        let client = self.client.clone();
        let tx = self.tx.clone();
        self.spawn(async move {
            let probe = client.probe_balance().await;
            let _ = tx.send(AppEvent::Balance { report, probe });
        });
    }

    // --------------------------------------------------------------- display

    fn set_status(&mut self, text: impl Into<String>, kind: StatusKind) {
        self.status = Some(Status {
            text: text.into(),
            kind,
        });
    }

    fn follow_output(&mut self) {
        if self.stick_to_bottom {
            self.scroll = 0;
        }
    }

    fn scroll_by(&mut self, delta: isize) {
        let next = if delta >= 0 {
            self.scroll.saturating_add(delta as usize)
        } else {
            self.scroll.saturating_sub(delta.unsigned_abs())
        };
        self.scroll = next.min(self.scroll_max);
        self.stick_to_bottom = self.scroll == 0;
    }

    fn scroll_to_top(&mut self) {
        self.scroll = self.scroll_max;
        self.stick_to_bottom = self.scroll == 0;
    }

    /// Move to the next (`delta > 0`) or previous match, wrapping around, and scroll it into
    /// view.
    fn step_match(&mut self, delta: isize) {
        let line = {
            let Some(search) = self.search.as_mut() else {
                return;
            };
            if search.matches.is_empty() {
                return;
            }
            let len = search.matches.len();
            let step = delta.unsigned_abs() % len;
            search.current = if delta >= 0 {
                (search.current + step) % len
            } else {
                (search.current + len - step) % len
            };
            search.matches[search.current]
        };
        self.scroll_to_line(line);
    }

    /// Scroll so `line`, a transcript line index from the top, sits at the top of the view.
    pub(super) fn scroll_to_line(&mut self, line: usize) {
        let offset = line.min(self.scroll_max);
        self.scroll = self.scroll_max - offset;
        self.stick_to_bottom = false;
    }

    pub(super) fn notices(&self) -> &[String] {
        &self.notices
    }
}

/// Only this much of a transcript is sent to the summariser: a history long enough to
/// overflow the model would overflow the summarising request too.
const SUMMARISE_MAX_BYTES: usize = 120_000;

/// Keep the tail of a transcript, which is the part that matters, and say what was cut.
pub fn cap_transcript(text: &str, max_bytes: usize) -> (String, bool) {
    if text.len() <= max_bytes {
        return (text.to_owned(), false);
    }
    let mut start = text.len() - max_bytes;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    (
        format!("[earlier messages omitted]\n{}", &text[start..]),
        true,
    )
}

/// Summarise a slice of the conversation, returning the summary and the tokens spent.
pub(crate) async fn summarise(
    client: &DeepSeekClient,
    model: &str,
    messages: &[ChatMessage],
) -> Result<(String, Option<Usage>)> {
    let transcript = messages
        .iter()
        .map(|message| format!("{}: {}", message.role.as_str(), message.content))
        .collect::<Vec<_>>()
        .join("\n\n");
    let (transcript, truncated) = cap_transcript(&transcript, SUMMARISE_MAX_BYTES);

    let mut system = "You compress chat transcripts. Write a dense summary that keeps every \
                      fact, decision, file name, command and open question needed to carry \
                      on the conversation. Respond with the summary only."
        .to_owned();
    if truncated {
        system.push_str(
            " The excerpt was truncated to its most recent part, so the oldest messages \
             may be missing.",
        );
    }

    let request = ChatRequest::new(
        model,
        vec![
            WireMessage {
                role: "system".to_owned(),
                content: system,
                tool_calls: None,
                tool_call_id: None,
            },
            WireMessage {
                role: "user".to_owned(),
                content: format!(
                    "Summarise this conversation excerpt so that it can replace the original \
                     messages as context:\n\n{transcript}"
                ),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
    )
    .max_tokens(Some(1024))
    .temperature(Some(0.2));

    let response = client.complete(&request).await?;
    let summary = response.text();
    if summary.trim().is_empty() {
        anyhow::bail!("the model returned an empty summary");
    }
    Ok((summary, response.usage))
}

/// The first line of a tool result, for the one-line status message.
fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or(text);
    crate::api::types::truncate(line, 60)
}

/// Whether a key event is `Ctrl+/`, which opens the help card.
///
/// Terminals disagree about how to spell it and crossterm passes through whatever it
/// receives: Windows sends `/` with the control modifier, while a terminal sending the
/// US control character (0x1F) is reported as `7` with the control modifier, because
/// that byte is also what Ctrl+7 produces. Accept every spelling, so that none of them
/// can end up in the prompt as a literal character. Ctrl+7 doubling as Ctrl+/ is the
/// price of the terminal's ambiguity.
fn is_help_shortcut(key: &KeyEvent) -> bool {
    let control = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Char('/') | KeyCode::Char('7') => control,
        // Some terminals hand us the raw control character instead.
        KeyCode::Char('\u{1f}') => true,
        _ => false,
    }
}

/// Whether a key event is `Ctrl+F`, which opens or closes the transcript search.
fn is_search_shortcut(key: &KeyEvent) -> bool {
    key.modifiers.contains(KeyModifiers::CONTROL)
        && matches!(key.code, KeyCode::Char('f') | KeyCode::Char('F'))
}

/// Forward terminal events into the app until told to stop.
fn read_terminal_events(tx: Sender<AppEvent>, stop: Arc<AtomicBool>) {
    while !stop.load(Ordering::SeqCst) {
        match crossterm::event::poll(TICK) {
            Ok(false) => continue,
            Ok(true) => match crossterm::event::read() {
                Ok(event) => {
                    if tx.send(AppEvent::Term(event)).is_err() {
                        return;
                    }
                }
                Err(_) => return,
            },
            Err(_) => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_transcript_is_left_alone() {
        let (text, truncated) = cap_transcript("hello", 100);
        assert_eq!(text, "hello");
        assert!(!truncated);
    }

    #[test]
    fn a_long_transcript_keeps_its_tail() {
        let input = "a".repeat(50) + "TAIL";
        let (text, truncated) = cap_transcript(&input, 6);
        assert!(truncated);
        assert!(text.ends_with("TAIL"));
        assert!(text.starts_with("[earlier messages omitted]"));
    }

    #[test]
    fn truncation_never_splits_a_character() {
        // A multi-byte character straddling the cut must be skipped, not sliced in half.
        let input = format!("{}ééé", "x".repeat(10));
        let (text, truncated) = cap_transcript(&input, 5);
        assert!(truncated);
        // The retained tail is still valid UTF-8 and inside the budget.
        let tail = text.strip_prefix("[earlier messages omitted]\n").unwrap();
        assert!(tail.len() <= 5);
        assert!(tail.chars().all(|ch| ch == 'é'));
    }

    /// A throwaway directory for the store-backed tests below.
    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "nu_plugin_ds_app_test_{}_{}",
            std::process::id(),
            tag
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn test_app(dir: &std::path::Path, session: Session) -> ChatApp {
        let store = SessionStore::new(dir.join("sessions")).unwrap();
        store.save(&session).unwrap();
        let client = DeepSeekClient::new("test-key", "http://127.0.0.1:9").unwrap();
        ChatApp::new(ChatSetup {
            client,
            settings: Settings::default(),
            store,
            models: Vec::new(),
            session,
            initial_prompt: None,
            context: tools::Context {
                cwd: dir.to_path_buf(),
                nu_bin: None,
                command_timeout: Duration::from_secs(5),
            },
            tools_enabled: false,
            approved: BTreeSet::new(),
        })
        .unwrap()
    }

    #[test]
    fn clearing_persists_the_empty_transcript() {
        let dir = temp_dir("clear");
        let mut session = Session::new("demo", "deepseek-chat", ThinkingEffort::Off, None);
        session.push(ChatMessage::user("hello"));
        session.push(ChatMessage::assistant("hi"));
        let id = session.id.clone();
        let mut app = test_app(&dir, session);

        assert!(app.run_slash_command("/clear"));

        // The in-memory transcript is gone...
        assert!(app.session.messages.is_empty());
        // ...and so is the copy on disk, so switching away and back cannot restore it.
        let reloaded = app.store.load(&id).unwrap();
        assert!(
            reloaded.messages.is_empty(),
            "the cleared transcript came back: {:?}",
            reloaded.messages
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_new_turn_is_refused_while_a_tool_runs() {
        let dir = temp_dir("busy");
        let session = Session::new("demo", "deepseek-chat", ThinkingEffort::Off, None);
        let mut app = test_app(&dir, session);
        app.running_tool = Some(RunningTool {
            name: "run_nu".to_owned(),
            started: Instant::now(),
        });

        let before = app.session.messages.len();
        app.submit("hello".to_owned());

        assert_eq!(app.session.messages.len(), before, "no turn should start");
        let status = app.status.as_ref().expect("a status message").text.clone();
        assert!(status.contains("tool"), "unexpected status: {status}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn running_a_tool_reports_the_outcome_as_an_event() {
        let dir = temp_dir("tool-event");
        let session = Session::new("demo", "deepseek-chat", ThinkingEffort::Off, None);
        let mut app = test_app(&dir, session);
        // No `nu` binary: the call fails, but only on the worker thread.
        app.context.nu_bin = None;

        let call = ToolCall {
            id: "call_1".to_owned(),
            kind: "function".to_owned(),
            function: crate::api::FunctionCall {
                name: tools::RUN_NU.to_owned(),
                arguments: r#"{"command":"echo hi"}"#.to_owned(),
            },
        };

        app.run_tool(call);
        assert!(
            app.running_tool.is_some(),
            "the call must be marked as running"
        );

        // The outcome must arrive as an event rather than by blocking the caller.
        let event = app
            .rx
            .recv_timeout(Duration::from_secs(10))
            .expect("a tool event should arrive");
        match event {
            AppEvent::ToolDone { outcome, .. } => {
                assert!(!outcome.ok, "run_nu without a `nu` binary should fail");
                assert!(
                    outcome.result.contains("not available"),
                    "{}",
                    outcome.result
                );
            }
            _ => panic!("expected a ToolDone event"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ctrl_c_does_not_quit_while_a_tool_runs() {
        let dir = temp_dir("ctrl-c");
        let session = Session::new("demo", "deepseek-chat", ThinkingEffort::Off, None);
        let mut app = test_app(&dir, session);
        app.running_tool = Some(RunningTool {
            name: "run_nu".to_owned(),
            started: Instant::now(),
        });

        app.on_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(!app.should_quit, "Ctrl+C must not quit mid-tool");

        app.running_tool = None;
        app.on_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(
            matches!(app.overlay, Some(Overlay::Quit)),
            "Ctrl+C should ask to quit once nothing is running"
        );
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(app.should_quit, "Enter should confirm the quit");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_wrap_up_round_withholds_the_tools() {
        let dir = temp_dir("wrap-up");
        let session = Session::new("demo", "deepseek-chat", ThinkingEffort::Off, None);
        let mut app = test_app(&dir, session);
        app.tools_enabled = true;
        app.tools_active = true;

        let offering = app.build_request(false, &app.session.messages);
        assert!(
            serde_json::to_value(&offering)
                .unwrap()
                .get("tools")
                .is_some(),
            "an ordinary request should offer the tools"
        );

        app.tools_withheld = true;
        let wrapping = app.build_request(false, &app.session.messages);
        assert!(
            serde_json::to_value(&wrapping)
                .unwrap()
                .get("tools")
                .is_none(),
            "the wrap-up request must not offer the tools"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hitting_the_tool_limit_asks_the_model_to_wrap_up() {
        let dir = temp_dir("tool-limit");
        let session = Session::new("demo", "deepseek-chat", ThinkingEffort::Off, None);
        let mut app = test_app(&dir, session);
        app.settings.max_tool_rounds = 1;
        app.tool_rounds = 1; // the budget is already spent

        let call = ToolCall {
            id: "call_1".to_owned(),
            kind: "function".to_owned(),
            function: crate::api::FunctionCall {
                name: tools::READ_FILE.to_owned(),
                arguments: r#"{"path":"x"}"#.to_owned(),
            },
        };
        app.begin_tool_round(vec![call]);

        // The call has to be answered, or the API would reject the history...
        let last = app.session.messages.last().expect("a tool result");
        assert_eq!(last.role, crate::api::Role::Tool);
        assert!(last.content.contains("not run"), "{}", last.content);
        // ...and a tool-less wrap-up request must be under way.
        assert!(app.tools_withheld, "the tools should be withheld now");
        assert!(
            app.stream.is_some(),
            "a wrap-up request should be streaming"
        );
        let status = app.status.as_ref().expect("a status message").text.clone();
        assert!(status.contains("wrap up"), "unexpected status: {status}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stepping_through_search_matches_scrolls_to_each() {
        let dir = temp_dir("search-step");
        let session = Session::new("demo", "deepseek-chat", ThinkingEffort::Off, None);
        let mut app = test_app(&dir, session);
        app.scroll_max = 100;
        app.scroll = 0;
        app.search = Some(Search {
            query: "x".to_owned(),
            matches: vec![5, 40, 80],
            current: 0,
            version: 0,
        });

        // Enter steps forward and scrolls each match to the top of the view.
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.scroll, 100 - 40);
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.scroll, 100 - 80);

        // Stepping past the last match wraps around to the first.
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.scroll, 100 - 5);

        // Shift+Enter steps back.
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
        assert_eq!(app.scroll, 100 - 80);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn paging_back_down_works_after_reaching_the_top() {
        let dir = temp_dir("scroll");
        let session = Session::new("demo", "deepseek-chat", ThinkingEffort::Off, None);
        let mut app = test_app(&dir, session);
        // As if the transcript were long enough to scroll 100 lines.
        app.scroll_max = 100;
        app.stick_to_bottom = true;

        // Paging up past the top must clamp there instead of piling up a backlog.
        for _ in 0..20 {
            app.on_key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
        }
        assert_eq!(app.scroll, 100, "the scroll must stop at the top");
        assert!(!app.stick_to_bottom);

        app.on_key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));
        assert_eq!(app.scroll, 90, "PageDown must scroll down right away");

        // Jumping to the top must leave PageDown working too.
        app.scroll_to_top();
        assert_eq!(app.scroll, 100, "Home should reach the top");
        app.on_key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));
        assert_eq!(app.scroll, 90, "PageDown must work after Home");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deleting_the_open_session_closes_it_for_good() {
        let dir = temp_dir("delete-current");
        let mut session = Session::new("open", "deepseek-chat", ThinkingEffort::Off, None);
        session.push(ChatMessage::user("hello"));
        let deleted_id = session.id.clone();
        let mut app = test_app(&dir, session);

        // Open the picker on the session that is currently open and delete it (`d` arms,
        // a second `d` deletes).
        app.refresh_sessions();
        app.overlay = Some(Overlay::Sessions { index: 0 });
        app.on_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE));
        app.on_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE));

        // The file is gone, the picker stayed open, and the open session is a fresh one.
        assert!(
            app.store.load(&deleted_id).is_err(),
            "the deleted session is still on disk"
        );
        assert!(matches!(app.overlay, Some(Overlay::Sessions { .. })));
        assert_ne!(
            app.session.id, deleted_id,
            "the closed session should be replaced"
        );
        assert!(app.session_closed, "the session should be closed");

        // A later save, as switching or quitting would do, must not recreate it.
        app.save_session_now();
        assert!(
            app.store.load(&deleted_id).is_err(),
            "a later save brought the deleted session back"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn alt_enter_neither_sends_nor_edits() {
        let dir = temp_dir("alt-enter");
        let session = Session::new("demo", "deepseek-chat", ThinkingEffort::Off, None);
        let mut app = test_app(&dir, session);
        app.input.set_text("hello");

        let before = app.session.messages.len();
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT));

        assert_eq!(
            app.session.messages.len(),
            before,
            "Alt+Enter must not send the prompt"
        );
        assert_eq!(
            app.input.text(),
            "hello",
            "Alt+Enter must not edit the prompt"
        );

        // Shift+Enter still inserts a newline.
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
        assert!(
            app.input.text().contains('\n'),
            "Shift+Enter should insert a newline"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quitting_asks_for_confirmation_first() {
        let dir = temp_dir("quit-confirm");
        let session = Session::new("demo", "deepseek-chat", ThinkingEffort::Off, None);
        let mut app = test_app(&dir, session);

        // Ctrl+X no longer closes the window; it asks.
        app.on_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL));
        assert!(!app.should_quit, "Ctrl+X must not quit immediately");
        assert!(matches!(app.overlay, Some(Overlay::Quit)));

        // Esc stays open.
        app.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!app.should_quit, "Esc must cancel the quit");
        assert!(app.overlay.is_none(), "the confirmation should close");

        // Ctrl+X then Enter quits.
        app.on_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL));
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(app.should_quit, "Enter should confirm the quit");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_quit_slash_command_asks_too() {
        let dir = temp_dir("quit-slash");
        let session = Session::new("demo", "deepseek-chat", ThinkingEffort::Off, None);
        let mut app = test_app(&dir, session);

        assert!(app.run_slash_command("/quit"));
        assert!(!app.should_quit, "/quit must ask first");
        assert!(matches!(app.overlay, Some(Overlay::Quit)));

        // `y` confirms.
        app.on_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE));
        assert!(app.should_quit, "y should confirm the quit");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
