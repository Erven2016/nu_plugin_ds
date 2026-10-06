//! The interactive chat application: state, key handling and the streaming pipeline.

use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use futures_util::StreamExt;
use ratatui::DefaultTerminal;
use tokio::sync::Notify;

use crate::api::{
    BalanceInfo, BalanceProbe, ChatMessage, ChatRequest, DeepSeekClient, ModelInfo, StreamEvent,
    ToolCall, Usage,
};
use crate::config::{Settings, ThinkingEffort};
use crate::session::{Session, SessionStore, SessionSummary};
use crate::token::{estimate_messages, token_cost};

use super::input::InputBuffer;
use super::tools;
use super::ui;

/// How long the event loop waits before redrawing when nothing happens.
const TICK: Duration = Duration::from_millis(60);

/// The shortest gap between two frames while a turn is running: a fast token stream is
/// coalesced into ~30 frames per second instead of one frame per token.
const FRAME: Duration = Duration::from_millis(33);

/// How many transcript lines one notch of the mouse wheel scrolls.
const WHEEL_LINES: isize = 3;

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
        /// Whether the turn should carry on after compacting. A manual `/compact` only
        /// compacts; an automatic compaction is mid-turn and must continue it.
        continue_turn: bool,
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
    /// A context menu opened with the right mouse button, near where it was clicked.
    Menu {
        /// Requested top-left corner in screen cells; the renderer clamps it on screen.
        x: u16,
        y: u16,
        /// The highlighted item.
        index: usize,
    },
}

/// The entries of the right-click context menu, in order.
pub(super) const MENU_ITEMS: [&str; 2] = ["Copy", "Deselect"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusKind {
    /// A low-priority hint (the startup key reminders). The status bar drops it when the
    /// window is too narrow to show it in full, so it never crowds the context gauge.
    Hint,
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

/// A run of the transcript the user selected with the mouse.
///
/// Positions are `(transcript line, char index)`, not screen cells, so a selection stays
/// valid while the view scrolls and the streamed tail grows underneath it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Selection {
    pub(super) anchor: (usize, usize),
    pub(super) cursor: (usize, usize),
    /// True between the button going down and coming back up, so only a drag extends it.
    pub(super) dragging: bool,
}

impl Selection {
    /// The two ends, in reading order.
    pub(super) fn normalized(&self) -> ((usize, usize), (usize, usize)) {
        if self.anchor <= self.cursor {
            (self.anchor, self.cursor)
        } else {
            (self.cursor, self.anchor)
        }
    }
}

/// Where the transcript was drawn last frame, so a mouse position can be mapped back to a
/// line and a column. Published by the renderer, like [`ChatApp::scroll_max`].
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TranscriptView {
    /// The transcript's inner rectangle (inside the border).
    pub(super) x: u16,
    pub(super) y: u16,
    pub(super) width: u16,
    pub(super) height: u16,
    /// The transcript line drawn on the first inner row.
    pub(super) first_line: usize,
    /// How many committed transcript lines exist; rows past this show the streaming tail.
    pub(super) committed: usize,
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
    /// The running (fractional) token cost of `reasoning` + `content`, accumulated as the
    /// deltas arrive so the status bar never has to re-walk the whole answer each frame.
    cost: f32,
    /// The same, for `content` alone, so the streaming label can show the answer's size.
    content_cost: f32,
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
            cost: 0.0,
            content_cost: 0.0,
            cancelled: false,
            started: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// The estimated tokens generated for the answer text so far.
    pub(super) fn content_tokens(&self) -> usize {
        self.content_cost.ceil() as usize
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
    /// The mouse selection over the transcript, if any.
    pub(super) selection: Option<Selection>,
    /// Where the transcript was drawn last frame, used to map mouse events back to lines.
    pub(super) transcript_view: TranscriptView,
    /// Where the context menu was drawn last frame, as `(x, y, width, height)`, so a click
    /// can be matched to a menu item. `None` when no menu is showing.
    pub(super) menu_area: Option<(u16, u16, u16, u16)>,
    /// The session file's modification time as we last read or wrote it, so a change made by
    /// another window can be told apart from our own save.
    last_modified: Option<SystemTime>,
    /// Set once the "no stored reasoning" message has been shown for this session, so it is
    /// not repeated every turn.
    reasoning_warned: bool,
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
            selection: None,
            transcript_view: TranscriptView::default(),
            menu_area: None,
            last_modified: None,
            reasoning_warned: false,
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

        // Remember the file's mtime so another window's later write is detectable.
        app.note_modified();

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
            StatusKind::Hint,
        );
        self.refresh_balance(false);

        if !self.input.text().trim().is_empty() {
            let prompt = self.input.take();
            self.submit(prompt);
        }

        // Draw when something changed, or while a spinner is animating, so an idle window
        // costs nothing even with a very long history. While busy the frame rate is capped,
        // and every event already queued is handled before the next draw, so a fast token
        // stream is coalesced into one frame instead of one frame per token.
        let mut redraw = true;
        let mut last_draw: Option<Instant> = None;
        while !self.should_quit {
            // Notice a change another window made to the same conversation. Only while idle,
            // so a turn in flight is never yanked out from under the stream.
            if !self.is_busy() && self.pull_remote(true) {
                redraw = true;
            }

            if redraw {
                let now = Instant::now();
                let due = last_draw.is_none_or(|last| now.duration_since(last) >= FRAME);
                // An idle window redraws on any change immediately; a busy one redraws at
                // most once per FRAME, which is what keeps a long answer off the CPU.
                if due || !self.is_busy() {
                    terminal.draw(|frame| ui::draw(frame, &mut self))?;
                    last_draw = Some(now);
                }
            }

            let timeout = if self.is_busy() { FRAME } else { TICK };
            redraw = match self.rx.recv_timeout(timeout) {
                Ok(event) => {
                    self.handle(event);
                    // Coalesce everything that arrived while this frame was being drawn, so
                    // a burst of deltas becomes a single redraw.
                    while let Ok(more) = self.rx.try_recv() {
                        self.handle(more);
                    }
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
                        state.cost += token_cost(&text);
                        state.reasoning.push_str(&text);
                        state.phase = Phase::Streaming;
                    }
                    StreamEvent::Content(text) => {
                        state.cost += token_cost(&text);
                        state.content_cost += token_cost(&text);
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
                    self.force_compaction(true);
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
                continue_turn,
            } => {
                if self.current_epoch() != Some(epoch) {
                    return;
                }
                self.apply_compaction(summary, removed, usage);
                if continue_turn {
                    self.spawn_stream(epoch);
                } else {
                    // `/compact` compacts the history; it does not ask another question.
                    self.stream = None;
                }
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
            Event::Mouse(mouse) => self.on_mouse(mouse),
            Event::Paste(text) => {
                self.input.insert_str(&text);
                self.follow_output();
            }
            _ => {}
        }
    }

    /// The mouse drives three things: the wheel scrolls the transcript a few lines per notch,
    /// a left-button drag selects a run of text, and the right button opens a context menu
    /// over the selection (copy / deselect). While any other overlay is open the mouse is
    /// ignored, since the overlay owns the input.
    fn on_mouse(&mut self, mouse: MouseEvent) {
        if let Some(Overlay::Menu { index, .. }) = self.overlay.clone() {
            self.on_menu_mouse(mouse, index);
            return;
        }
        if self.overlay.is_some() {
            return;
        }
        match mouse.kind {
            MouseEventKind::ScrollUp => self.scroll_by(WHEEL_LINES),
            MouseEventKind::ScrollDown => self.scroll_by(-WHEEL_LINES),
            MouseEventKind::Down(MouseButton::Left) => {
                self.begin_selection(mouse.column, mouse.row)
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                self.extend_selection(mouse.column, mouse.row)
            }
            MouseEventKind::Up(MouseButton::Left) => self.end_selection(),
            MouseEventKind::Down(MouseButton::Right) => self.open_menu(mouse.column, mouse.row),
            _ => {}
        }
    }

    /// Drive the open context menu with the mouse: a left click activates the item under the
    /// cursor (or closes the menu when it lands outside), any other button closes it.
    fn on_menu_mouse(&mut self, mouse: MouseEvent, index: usize) {
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                match self.menu_item_at(mouse.column, mouse.row) {
                    Some(item) => self.run_menu_item(item),
                    None => self.overlay = None,
                }
            }
            MouseEventKind::Down(MouseButton::Right) => {
                // A second right click moves the menu rather than stacking another one.
                let (x, y) = (mouse.column, mouse.row);
                self.overlay = Some(Overlay::Menu { x, y, index });
            }
            _ => {}
        }
    }

    /// Open the context menu at a click, when there is a selection to act on.
    fn open_menu(&mut self, column: u16, row: u16) {
        if self.selection.is_none() {
            return;
        }
        self.overlay = Some(Overlay::Menu {
            x: column,
            y: row,
            index: 0,
        });
    }

    /// The menu item under a screen cell, using the rectangle the renderer published.
    fn menu_item_at(&self, column: u16, row: u16) -> Option<usize> {
        let (x, y, width, height) = self.menu_area?;
        // The border occupies the first and last column/row of the panel.
        if column <= x || column >= x + width.saturating_sub(1) {
            return None;
        }
        if row <= y || row >= y + height.saturating_sub(1) {
            return None;
        }
        let item = (row - y - 1) as usize;
        (item < MENU_ITEMS.len()).then_some(item)
    }

    /// Act on a context-menu entry and close the menu.
    fn run_menu_item(&mut self, index: usize) {
        self.overlay = None;
        match MENU_ITEMS.get(index) {
            Some(&"Copy") => {
                if !self.copy_selection() {
                    self.set_status("nothing is selected", StatusKind::Warn);
                }
            }
            Some(&"Deselect") => {
                self.selection = None;
                self.set_status("selection cleared", StatusKind::Info);
            }
            _ => {}
        }
    }

    /// Copy the current selection to the clipboard, reporting whether there was one.
    fn copy_selection(&mut self) -> bool {
        let Some(selection) = self.selection else {
            return false;
        };
        match self
            .selected_text(&selection)
            .filter(|text| !text.is_empty())
        {
            Some(text) => {
                let chars = text.chars().count();
                copy_to_clipboard(&text);
                self.set_status(
                    format!("copied {chars} characters to the clipboard"),
                    StatusKind::Info,
                );
                true
            }
            None => false,
        }
    }

    /// Start a selection where the button went down, or clear any existing one when the
    /// press did not land on a committed transcript line.
    fn begin_selection(&mut self, column: u16, row: u16) {
        self.selection = self
            .transcript_position(column, row)
            .map(|position| Selection {
                anchor: position,
                cursor: position,
                dragging: true,
            });
    }

    fn extend_selection(&mut self, column: u16, row: u16) {
        let dragging = self
            .selection
            .as_ref()
            .is_some_and(|selection| selection.dragging);
        if !dragging {
            return;
        }
        if let Some(position) = self.clamped_transcript_position(column, row)
            && let Some(selection) = self.selection.as_mut()
        {
            selection.cursor = position;
        }
    }

    /// Finish the drag. The selection is kept (and hinted at) rather than copied: copying is
    /// a deliberate action, taken from the right-click menu.
    fn end_selection(&mut self) {
        let Some(selection) = self.selection.as_mut() else {
            return;
        };
        selection.dragging = false;
        let selection = *selection;
        if selection.normalized().0 == selection.normalized().1 {
            self.selection = None;
        } else {
            self.set_status("text selected — right-click to copy", StatusKind::Info);
        }
    }

    /// The transcript position under a screen cell, when it lands on a committed line.
    fn transcript_position(&self, column: u16, row: u16) -> Option<(usize, usize)> {
        let view = &self.transcript_view;
        if column < view.x
            || column >= view.x + view.width
            || row < view.y
            || row >= view.y + view.height
        {
            return None;
        }
        let line = view.first_line + (row - view.y) as usize;
        if line >= view.committed {
            return None;
        }
        let ch = self.transcript.char_index_at(line, column - view.x)?;
        Some((line, ch))
    }

    /// Like [`Self::transcript_position`], but clamps a drag that ran off the transcript
    /// back onto the nearest committed cell, so a selection can be dragged past the edges.
    fn clamped_transcript_position(&self, column: u16, row: u16) -> Option<(usize, usize)> {
        let view = &self.transcript_view;
        if view.committed == 0 || view.width == 0 || view.height == 0 {
            return None;
        }
        let col = column.saturating_sub(view.x).min(view.width - 1);
        let row = row.clamp(view.y, view.y + view.height - 1);
        let line = (view.first_line + (row - view.y) as usize).min(view.committed - 1);
        self.transcript
            .char_index_at(line, col)
            .map(|ch| (line, ch))
    }

    /// The plain text the selection covers, joined with newlines between lines.
    pub(super) fn selected_text(&self, selection: &Selection) -> Option<String> {
        let (start, end) = selection.normalized();
        let mut out = String::new();
        for line in start.0..=end.0 {
            let chars: Vec<char> = self.transcript.text_of(line)?.chars().collect();
            let from = if line == start.0 {
                start.1.min(chars.len())
            } else {
                0
            };
            let to = if line == end.0 {
                end.1.min(chars.len())
            } else {
                chars.len()
            };
            if from < to {
                out.extend(&chars[from..to]);
            }
            if line != end.0 {
                out.push('\n');
            }
        }
        Some(out)
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
                        // Clone the id out first so the borrow of `self.models` ends before
                        // the mutable `save_session` below.
                        let id = self.models.get(index).map(|model| model.id.clone());
                        if let Some(id) = id {
                            self.session.model = id.clone();
                            self.session.touch();
                            // Persist immediately so the other windows follow.
                            self.save_session();
                            self.set_status(format!("model switched to {id}"), StatusKind::Info);
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
                            // Persist immediately so the other windows follow.
                            self.save_session();
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
            Overlay::Menu { x, y, mut index } => {
                match key.code {
                    KeyCode::Esc => {
                        self.overlay = None;
                        return;
                    }
                    KeyCode::Up | KeyCode::Char('k') => index = index.saturating_sub(1),
                    KeyCode::Down | KeyCode::Char('j') => {
                        index = (index + 1).min(MENU_ITEMS.len().saturating_sub(1))
                    }
                    KeyCode::Char('c') => {
                        self.run_menu_item(0);
                        return;
                    }
                    KeyCode::Enter => {
                        self.run_menu_item(index);
                        return;
                    }
                    _ => {}
                }
                self.overlay = Some(Overlay::Menu { x, y, index });
            }
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
            "/rename" => self.rename_session(argument),
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
                    // Persist immediately so the other windows on this conversation follow.
                    self.save_session();
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
                    // Persist immediately so the other windows on this conversation follow.
                    self.save_session();
                    self.set_status(
                        format!("thinking level set to {}", effort.label()),
                        StatusKind::Info,
                    );
                } else {
                    self.set_status(
                        format!("`{argument}` is not a thinking level (off/low/high/max)"),
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
            "/compact" => self.force_compaction(false),
            "/regenerate" => self.regenerate(),
            other => self.set_status(
                format!("unknown command `{other}`; try /help"),
                StatusKind::Warn,
            ),
        }
        true
    }

    fn force_compaction(&mut self, continue_turn: bool) {
        if self.is_busy() {
            self.set_status(
                "cannot compact while the answer or a tool call is still running",
                StatusKind::Warn,
            );
            return;
        }
        let epoch = self.begin_epoch();
        self.stream = Some(StreamState::new(epoch, Phase::Compacting));
        self.spawn_compaction(epoch, true, continue_turn);
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
        // The prompt is part of the conversation, so persist it for the other windows.
        self.save_session();
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

    fn rename_session(&mut self, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            self.set_status("usage: /rename <new name>", StatusKind::Warn);
            return;
        }
        if self.session_closed {
            self.set_status("there is no open conversation to rename", StatusKind::Warn);
            return;
        }
        self.session.name = name.to_owned();
        self.session.touch();
        // A rename is a deliberate change, so persist it even before the first message.
        self.save_session_now();
        self.refresh_sessions();
        self.set_status(
            format!("renamed the conversation to {name}"),
            StatusKind::Info,
        );
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
        self.last_modified = None;
        self.reasoning_warned = false;
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
                self.reasoning_warned = false;
                // Track this file from now on, so a change by another window is noticed.
                self.note_modified();
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
        self.last_modified = None;
        self.reasoning_warned = false;
        self.scroll = 0;
        self.stick_to_bottom = true;
        self.input.clear();
    }

    /// Another window deleted the conversation we have open: close ours too (so nothing
    /// writes it back) and drop it from the picker's list.
    fn close_deleted_session(&mut self) {
        self.close_current_session();
        self.refresh_sessions();
        self.set_status(
            "the open conversation was deleted in another window",
            StatusKind::Warn,
        );
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
        // Fold in anything another window wrote while we were not looking, so this save
        // does not drop their turns — or recreates a session they deleted.
        self.pull_remote(false);
        if self.session_closed {
            return;
        }
        if let Err(err) = self.store.save(&self.session) {
            self.set_status(
                format!("could not save the session: {err:#}"),
                StatusKind::Error,
            );
        }
        self.note_modified();
    }

    /// Remember the session file's current mtime as ours, so our own writes are not mistaken
    /// for another window's.
    fn note_modified(&mut self) {
        self.last_modified = self.store.modified(&self.session.id);
    }

    /// Pull in a change another `chat` window made to the same conversation.
    ///
    /// Returns whether the session changed. `allow_structural` controls what happens when the
    /// other window compacted or cleared the history: a `true` (a quiet poll) lets the file
    /// win, a `false` (called just before saving) leaves our copy alone so the turn in flight
    /// is not thrown away.
    fn pull_remote(&mut self, allow_structural: bool) -> bool {
        if self.session_closed {
            return false;
        }
        let Some(modified) = self.store.modified(&self.session.id) else {
            // The file is gone. If we had seen it before, another window deleted the
            // conversation; close ours too so nothing writes it back.
            if self.last_modified.is_some() {
                self.close_deleted_session();
                return true;
            }
            return false;
        };
        if Some(modified) == self.last_modified {
            return false;
        }
        let remote = match self.store.load(&self.session.id) {
            Ok(session) => session,
            // Unreadable right now (e.g. mid-replace): try again on the next poll.
            Err(_) => return false,
        };
        self.last_modified = Some(modified);

        if remote.compactions != self.session.compactions {
            if !allow_structural {
                return false;
            }
            self.session = remote;
            return true;
        }

        let mut params_changed = false;
        // The conversation's parameters (model, thinking level, name) changed in the other
        // window. Only adopted on a quiet poll, so a param this window just changed is not
        // undone by the save that follows it.
        if allow_structural {
            if remote.model != self.session.model {
                self.session.model = remote.model.clone();
                params_changed = true;
            }
            if remote.thinking != self.session.thinking {
                self.session.thinking = remote.thinking;
                params_changed = true;
            }
            if remote.name != self.session.name {
                self.session.name = remote.name.clone();
                params_changed = true;
            }
        }

        let merged = merge_messages(&self.session.messages, &remote.messages);
        let messages_changed = merged != self.session.messages;
        if messages_changed {
            self.session.messages = merged;
            self.session.updated_at = self.session.updated_at.max(remote.updated_at);
        }
        if params_changed {
            // The picker lists titles/models, so keep it in step.
            self.refresh_sessions();
        }
        params_changed || messages_changed
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
        // Tools require the thinking from every earlier assistant turn to be echoed back. When
        // some is missing (an older session, or thinking toggled mid-conversation), thinking
        // is turned off for this request so the tools can still be offered.
        let thinking = self.effective_thinking();
        let request = crate::api::build_chat_request(
            &self.session.model,
            thinking,
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

    /// The thinking level to send: the session's, unless the history is missing reasoning
    /// that the API would demand whenever `tools` are on the wire (then thinking is skipped,
    /// so the tools still work).
    fn effective_thinking(&self) -> ThinkingEffort {
        if self.session.thinking != ThinkingEffort::Off && !self.session.reasoning_complete() {
            ThinkingEffort::Off
        } else {
            self.session.thinking
        }
    }

    fn start_turn(&mut self) {
        self.tool_rounds = 0;
        self.context_retries = 0;
        self.tools_withheld = false;

        // A legacy `*-reasoner` name marks a thinking-only model, which cannot call tools,
        // so the tools are left out for it rather than making every request fail.
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
            self.spawn_compaction(epoch, false, true);
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
        // it), not once per frame. The in-flight answer's cost is accumulated as its
        // deltas arrive, so a long stream does not cost anything here either.
        let mut used = self.transcript.tokens();
        if let Some(state) = &self.stream {
            used += state.cost.ceil() as usize;
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
    ///
    /// It has to keep the same number of recent messages as [`ChatApp::apply_compaction`],
    /// otherwise the summary is built from a wider range than the one that is actually
    /// replaced: the surplus messages are then both summarised and kept verbatim (so the
    /// history can even grow), and the effect looks like the command did nothing.
    fn forced_compaction_plan(&self) -> Option<(Vec<ChatMessage>, usize)> {
        self.session
            .compaction_plan(self.settings.keep_recent_messages.max(2))
    }

    fn plan_keeping(&self, keep: usize) -> Option<(Vec<ChatMessage>, usize)> {
        if self.transcript.estimate(&self.session) < self.compact_at() {
            return None;
        }
        self.session.compaction_plan(keep)
    }

    fn spawn_compaction(&mut self, epoch: u64, forced: bool, continue_turn: bool) {
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
                        continue_turn,
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
        let before = estimate_messages(&self.session.messages);
        self.session.apply_summary(&summary, keep);
        if let Some(usage) = usage {
            self.session.usage.merge(&usage);
        }
        self.save_session();
        self.scroll = 0;
        self.stick_to_bottom = true;

        let after = estimate_messages(&self.session.messages);
        self.set_status(
            format!(
                "compacted {removed} messages into a summary (estimate ~{before} → ~{after} tokens)"
            ),
            StatusKind::Info,
        );
    }

    fn spawn_stream(&mut self, epoch: u64) {
        self.repair_history();
        if self.session.thinking != ThinkingEffort::Off
            && !self.session.reasoning_complete()
            && !self.reasoning_warned
        {
            self.reasoning_warned = true;
            self.set_status(
                "thinking is off for this session: an earlier answer has no stored reasoning, \
                 which the API requires when tools are sent",
                StatusKind::Warn,
            );
        }
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
    /// rejects a conversation where one is missing, and a turn can end early. It also drops a
    /// `tool` result left without its call (a compaction boundary or a cross-window merge can
    /// leave one behind), which the API rejects the same way.
    fn repair_history(&mut self) {
        for call in self.session.unanswered_tool_calls() {
            let outcome = tools::Outcome::failed("not run: this call was never executed");
            self.record_tool_result(&call, &outcome);
        }
        self.session.drop_orphan_tool_results();
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

    let request = summarise_request(model, system, &transcript);

    let response = client.complete(&request).await?;
    let summary = response.text();
    if summary.trim().is_empty() {
        anyhow::bail!("the model returned an empty summary");
    }
    Ok((summary, response.usage))
}

/// The request that compresses a transcript into a summary.
///
/// Thinking is on by default, but this request must not leave it that way: with a small
/// token cap the model can spend the whole budget on hidden reasoning and emit no content,
/// which failed compaction with "the model returned an empty summary". Summarising needs no
/// reasoning, so it is disabled explicitly and the cap leaves room for a dense summary.
fn summarise_request(model: &str, system: String, transcript: &str) -> ChatRequest {
    crate::api::build_chat_request(
        model,
        ThinkingEffort::Off,
        Some(4096),
        Some(0.2),
        &[
            ChatMessage::system(system),
            ChatMessage::user(format!(
                "Summarise this conversation excerpt so that it can replace the original \
                 messages as context:\n\n{transcript}"
            )),
        ],
        false,
    )
}

/// The first line of a tool result, for the one-line status message.
fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or(text);
    crate::api::types::truncate(line, 60)
}

/// Merge two views of the same conversation so neither window loses a turn.
///
/// Appends are the common case: when one list is a prefix of the other, the longer wins.
/// When both windows appended since they last synced, the shared prefix is kept and the two
/// tails are concatenated (the other window's turns first).
fn merge_messages(local: &[ChatMessage], remote: &[ChatMessage]) -> Vec<ChatMessage> {
    let prefix = local
        .iter()
        .zip(remote)
        .take_while(|(left, right)| left == right)
        .count();
    if prefix == local.len() {
        return remote.to_vec();
    }
    if prefix == remote.len() {
        return local.to_vec();
    }
    let mut merged = local[..prefix].to_vec();
    merged.extend_from_slice(&remote[prefix..]);
    merged.extend_from_slice(&local[prefix..]);
    merged
}

/// Put `text` on the system clipboard using the `OSC 52` escape sequence.
///
/// The plugin captures the mouse, so the terminal's own select-and-copy no longer works;
/// this writes the selection out the way a terminal clipboard integration does, without
/// pulling in a platform clipboard crate. Terminals that do not implement OSC 52 ignore it.
fn copy_to_clipboard(text: &str) {
    use std::io::Write;

    let payload = base64(text.as_bytes());
    let mut stdout = std::io::stdout();
    let _ = write!(stdout, "\x1b]52;c;{payload}\x07");
    let _ = stdout.flush();
}

/// Standard base64, so an arbitrary selection can ride inside the OSC 52 payload.
fn base64(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
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
        let mut session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
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
    fn renaming_updates_the_title_and_persists_it() {
        let dir = temp_dir("rename");
        let mut session = Session::new("", "deepseek-flash", ThinkingEffort::Off, None);
        session.push(ChatMessage::user("hello"));
        session.push(ChatMessage::assistant("hi"));
        let id = session.id.clone();
        let mut app = test_app(&dir, session);

        assert!(app.run_slash_command("/rename planning"));
        assert_eq!(app.session.title(), "planning");

        // The new name must survive a reload and show up in the picker's list.
        let reloaded = app.store.load(&id).unwrap();
        assert_eq!(reloaded.title(), "planning");
        assert!(app.sessions.iter().any(|entry| entry.title == "planning"));

        // An empty name is refused with a hint instead of blanking the title.
        assert!(app.run_slash_command("/rename   "));
        assert_eq!(app.session.title(), "planning");
        let status = app.status.as_ref().expect("a status").text.clone();
        assert!(status.contains("usage"), "unexpected status: {status}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compacting_replaces_exactly_what_the_plan_summarised() {
        let dir = temp_dir("compact");
        let mut session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
        for i in 0..12 {
            session.push(ChatMessage::user(format!("u{i}")));
            session.push(ChatMessage::assistant(format!("a{i}")));
        }
        let mut app = test_app(&dir, session);

        // The summariser must be handed everything except the tail that will be kept.
        let (to_summarise, removed) = app.forced_compaction_plan().expect("a plan");
        let keep = app.settings.keep_recent_messages.max(2);
        assert_eq!(
            app.session.messages.len() - to_summarise.len(),
            keep,
            "the plan and the compaction must agree on how much to keep"
        );
        assert_eq!(removed, to_summarise.len());

        app.apply_compaction("the gist".to_owned(), removed, None);

        // One summary plus the kept tail: nothing that was summarised is also kept verbatim.
        assert_eq!(app.session.messages.len(), keep + 1);
        assert_eq!(app.session.compactions, 1);
        let status = app.status.as_ref().expect("a status").text.clone();
        assert!(status.contains("compacted"), "unexpected status: {status}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_summariser_turns_thinking_off_and_leaves_room() {
        let request = summarise_request("deepseek-v4-pro", "be dense".to_owned(), "u: hi");
        let json = serde_json::to_value(&request).unwrap();

        // Thinking on by default can swallow the whole token budget and return no content.
        assert_eq!(json["thinking"]["type"], serde_json::json!("disabled"));
        assert_eq!(json["stream"], serde_json::json!(false));
        assert!(
            json["max_tokens"].as_u64().is_some_and(|cap| cap >= 2048),
            "the cap must leave room for a summary: {json}"
        );
        let body = request
            .messages
            .iter()
            .map(|message| message.content.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            body.contains("u: hi"),
            "the transcript must reach the model: {body}"
        );
    }

    #[test]
    fn a_new_turn_is_refused_while_a_tool_runs() {
        let dir = temp_dir("busy");
        let session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
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
        let session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
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
        let session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
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
        let session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
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
    fn an_incomplete_history_skips_thinking_but_keeps_the_tools() {
        let dir = temp_dir("reasoning-tools");
        let mut session = Session::new("demo", "deepseek-v4-pro", ThinkingEffort::High, None);
        session.push(ChatMessage::user("hi"));
        session.push(ChatMessage::assistant("answer")); // no reasoning stored
        let mut app = test_app(&dir, session);
        app.tools_enabled = true;
        app.tools_active = true;

        // Thinking is on but an assistant turn has no reasoning: thinking is dropped for the
        // request (the API requires it back when tools are sent), but the tools remain.
        let request = app.build_request(false, &app.session.messages);
        let json = serde_json::to_value(&request).unwrap();
        assert!(
            json.get("tools").is_some(),
            "the tools should still be offered"
        );
        assert_eq!(json["thinking"]["type"], serde_json::json!("disabled"));

        // With the reasoning restored, thinking is sent again.
        app.session.messages[1].reasoning_content = Some("thoughts".to_owned());
        let request = app.build_request(false, &app.session.messages);
        let json = serde_json::to_value(&request).unwrap();
        assert!(json.get("tools").is_some());
        assert!(json.get("thinking").is_none());
        assert_eq!(json["reasoning_effort"], serde_json::json!("high"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_manual_compaction_does_not_start_a_new_turn() {
        let dir = temp_dir("compact-manual");
        let session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
        let mut app = test_app(&dir, session);
        let epoch = app.begin_epoch();
        app.stream = Some(StreamState::new(epoch, Phase::Compacting));

        app.handle(AppEvent::Compacted {
            epoch,
            summary: "SUM".to_owned(),
            removed: 0,
            usage: None,
            continue_turn: false,
        });

        assert!(
            app.stream.is_none(),
            "/compact should compact without asking another question"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hitting_the_tool_limit_asks_the_model_to_wrap_up() {
        let dir = temp_dir("tool-limit");
        let session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
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
        // The assistant asked for the call; the API pairs a `tool` result with this message.
        let mut assistant = ChatMessage::assistant("");
        assistant.tool_calls = Some(vec![call.clone()]);
        app.session.push(assistant);
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
        let session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
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
        let session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
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
        let mut session = Session::new("open", "deepseek-flash", ThinkingEffort::Off, None);
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
    fn the_mouse_wheel_scrolls_the_transcript() {
        let dir = temp_dir("mouse");
        let session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
        let mut app = test_app(&dir, session);
        // Normally published by the last frame; a draw would set it.
        app.scroll_max = 100;
        app.stick_to_bottom = true;

        let wheel = |kind| {
            AppEvent::Term(Event::Mouse(MouseEvent {
                kind,
                column: 0,
                row: 0,
                modifiers: KeyModifiers::NONE,
            }))
        };

        app.handle(wheel(MouseEventKind::ScrollUp));
        assert_eq!(app.scroll, WHEEL_LINES as usize);
        assert!(!app.stick_to_bottom, "scrolling up leaves the bottom");

        app.handle(wheel(MouseEventKind::ScrollUp));
        assert_eq!(app.scroll, (WHEEL_LINES * 2) as usize);

        app.handle(wheel(MouseEventKind::ScrollDown));
        app.handle(wheel(MouseEventKind::ScrollDown));
        assert_eq!(app.scroll, 0);
        assert!(
            app.stick_to_bottom,
            "back at the bottom, output follows again"
        );

        // An open overlay owns the input, so the wheel must not scroll behind it.
        app.overlay = Some(Overlay::Help);
        app.handle(wheel(MouseEventKind::ScrollUp));
        assert_eq!(app.scroll, 0, "the wheel must not scroll behind an overlay");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dragging_the_mouse_selects_and_copies_transcript_text() {
        let dir = temp_dir("selection");
        let mut session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
        session.push(ChatMessage::user("hello world"));
        let mut app = test_app(&dir, session);

        // Render a frame so the transcript is built and its geometry published.
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(60, 20)).unwrap();
        terminal.draw(|frame| ui::draw(frame, &mut app)).unwrap();

        let view = app.transcript_view;
        let line = (view.first_line..view.first_line + view.height as usize)
            .find(|index| {
                app.transcript
                    .text_of(*index)
                    .is_some_and(|text| text.contains("hello world"))
            })
            .expect("the user's text should be on screen");
        let start = app.transcript.text_of(line).unwrap().find("hello").unwrap() as u16;
        let row = view.y + (line - view.first_line) as u16;

        let event = |kind, column| {
            AppEvent::Term(Event::Mouse(MouseEvent {
                kind,
                column,
                row,
                modifiers: KeyModifiers::NONE,
            }))
        };
        app.handle(event(
            MouseEventKind::Down(MouseButton::Left),
            view.x + start,
        ));
        app.handle(event(
            MouseEventKind::Drag(MouseButton::Left),
            view.x + start + 5,
        ));
        assert!(app.selection.is_some(), "a drag starts a selection");

        app.handle(event(
            MouseEventKind::Up(MouseButton::Left),
            view.x + start + 5,
        ));
        app.handle(event(
            MouseEventKind::Up(MouseButton::Left),
            view.x + start + 5,
        ));
        let selection = app.selection.expect("the selection survives the release");
        assert_eq!(app.selected_text(&selection).as_deref(), Some("hello"));

        // Releasing the button does not copy by itself: it only points at the menu.
        let status = app.status.as_ref().expect("a hint").text.clone();
        assert!(
            status.contains("right-click"),
            "unexpected status: {status}"
        );

        // The right button opens the context menu, and Enter runs the highlighted item.
        app.handle(event(
            MouseEventKind::Down(MouseButton::Right),
            view.x + start,
        ));
        assert!(
            matches!(app.overlay, Some(Overlay::Menu { .. })),
            "a right click should open the context menu"
        );
        app.handle(AppEvent::Term(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        ))));
        assert!(app.overlay.is_none(), "the menu closes after the action");
        let status = app.status.as_ref().expect("a status").text.clone();
        assert!(status.contains("copied"), "unexpected status: {status}");
        assert_eq!(
            app.selected_text(app.selection.as_ref().expect("kept"))
                .as_deref(),
            Some("hello"),
            "copying keeps the selection"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_change_from_another_window_is_pulled_in() {
        let dir = temp_dir("sync");
        let mut session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
        session.push(ChatMessage::user("from window A"));
        let mut app = test_app(&dir, session);
        app.note_modified();

        // Another `chat` window appends to the same file and saves it.
        std::thread::sleep(Duration::from_millis(10));
        let mut remote = app.store.load(&app.session.id).unwrap();
        remote.push(ChatMessage::assistant("from window B"));
        app.store.save(&remote).unwrap();

        assert!(
            app.pull_remote(true),
            "the external write should be noticed"
        );
        assert!(
            app.session
                .messages
                .iter()
                .any(|message| message.content == "from window B"),
            "the other window's turn should appear: {:?}",
            app.session.messages
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_turns_from_two_windows_are_merged() {
        let dir = temp_dir("sync-merge");
        let mut session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
        session.push(ChatMessage::user("base"));
        let mut app = test_app(&dir, session);
        app.note_modified();

        // We append a turn locally, not saved yet.
        app.session.push(ChatMessage::user("mine"));

        // The other window appended a different turn and saved it.
        std::thread::sleep(Duration::from_millis(10));
        let mut remote = app.store.load(&app.session.id).unwrap();
        remote.push(ChatMessage::user("theirs"));
        app.store.save(&remote).unwrap();

        // Saving now must keep both turns, on disk as well as in memory.
        app.save_session_now();
        let in_memory: Vec<&str> = app
            .session
            .messages
            .iter()
            .map(|message| message.content.as_str())
            .collect();
        assert!(in_memory.contains(&"mine"), "{in_memory:?}");
        assert!(in_memory.contains(&"theirs"), "{in_memory:?}");

        let reloaded = app.store.load(&app.session.id).unwrap();
        let on_disk: Vec<&str> = reloaded
            .messages
            .iter()
            .map(|message| message.content.as_str())
            .collect();
        assert!(on_disk.contains(&"mine"), "{on_disk:?}");
        assert!(on_disk.contains(&"theirs"), "{on_disk:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deleting_a_session_in_another_window_closes_it_here() {
        let dir = temp_dir("sync-delete");
        let mut session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
        session.push(ChatMessage::user("hello"));
        let mut app = test_app(&dir, session);
        app.note_modified();
        let deleted = app.session.id.clone();

        // Another window removes the conversation.
        app.store.delete(&deleted).unwrap();

        assert!(app.pull_remote(true), "the deletion should be noticed");
        assert!(app.session_closed, "the conversation should be closed here");
        assert_ne!(
            app.session.id, deleted,
            "a blank conversation takes its place"
        );

        // A later save must not write the deleted conversation back.
        app.save_session_now();
        assert!(
            app.store.load(&deleted).is_err(),
            "the deleted conversation must stay deleted"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_parameter_change_from_another_window_is_pulled_in() {
        let dir = temp_dir("sync-params");
        let mut session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
        session.push(ChatMessage::user("hello"));
        let mut app = test_app(&dir, session);
        app.note_modified();

        // Another window switches the model and thinking level, then saves.
        std::thread::sleep(Duration::from_millis(10));
        let mut remote = app.store.load(&app.session.id).unwrap();
        remote.model = "deepseek-v4-pro".to_owned();
        remote.thinking = ThinkingEffort::Max;
        app.store.save(&remote).unwrap();

        assert!(
            app.pull_remote(true),
            "the parameter change should be noticed"
        );
        assert_eq!(app.session.model, "deepseek-v4-pro");
        assert_eq!(app.session.thinking, ThinkingEffort::Max);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_local_parameter_change_survives_its_own_save() {
        let dir = temp_dir("sync-params-local");
        let mut session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
        session.push(ChatMessage::user("hello"));
        let mut app = test_app(&dir, session);
        app.note_modified();

        // The other window left a different model on disk.
        std::thread::sleep(Duration::from_millis(10));
        let mut remote = app.store.load(&app.session.id).unwrap();
        remote.model = "deepseek-v4-pro".to_owned();
        app.store.save(&remote).unwrap();

        // This window keeps its own model; saving must not adopt the remote's value.
        app.session.model = "deepseek-flash".to_owned();
        app.save_session_now();
        assert_eq!(app.session.model, "deepseek-flash");
        let on_disk = app.store.load(&app.session.id).unwrap();
        assert_eq!(on_disk.model, "deepseek-flash");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn alt_enter_neither_sends_nor_edits() {
        let dir = temp_dir("alt-enter");
        let session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
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
        let session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
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
        let session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
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
