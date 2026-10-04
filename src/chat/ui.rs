//! Rendering for the chat TUI.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Clear, Paragraph};

use crate::api::{ChatMessage, Role, ToolCall, Usage};
use crate::token::{estimate_messages, estimate_tokens, format_tokens};

use super::app::{Balance, ChatApp, Overlay, Phase, StatusKind};
use super::markdown;
use super::tools;

/// Number of spaces content is indented by under its role header.
const INDENT: usize = 2;

/// How much room the status bar must have left after the context gauge before a diagnostic
/// status message is worth drawing.
const MIN_STATUS_WIDTH: u16 = 24;

/// Draw one frame of the chat UI.
pub fn draw(frame: &mut Frame, app: &mut ChatApp) {
    let area = frame.area();

    if area.height < 4 || area.width < 20 {
        frame.render_widget(
            Paragraph::new("the chat needs a terminal of at least 20x4"),
            area,
        );
        return;
    }

    // Shrink the fixed-height rows before handing them to the layout, so a narrow
    // terminal does not push a section outside the frame.
    let input_height = (app.input.line_count() as u16 + 2)
        .clamp(3, 9)
        .min(area.height.saturating_sub(3));
    let status_height = 2u16.min(area.height.saturating_sub(1));

    let [transcript_area, input_area, status_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(input_height),
        Constraint::Length(status_height),
    ])
    .areas(area);

    draw_transcript(frame, app, clamp(transcript_area, area));
    draw_input(frame, app, clamp(input_area, area));
    draw_status(frame, app, clamp(status_area, area));

    if let Some(overlay) = app.overlay.clone() {
        draw_overlay(frame, app, overlay, area);
    }
}

// --------------------------------------------------------------- transcript

/// The rendered transcript, reused across frames.
///
/// Rendering markdown for the whole history on every frame is what makes a long
/// conversation lag. The lines are kept until something they depend on changes; the key
/// below is cheap to compute and covers every way the transcript can change.
///
/// The estimate of the history's tokens is kept here too, so the status bar does not have
/// to walk every message on every frame either.
pub(super) struct TranscriptCache {
    key: Option<TranscriptKey>,
    lines: Vec<Line<'static>>,
    /// The lower-cased plain text of each line, for fast search matching.
    texts: Vec<String>,
    tokens: usize,
    /// Bumped on every rebuild, so the search can tell when its line numbers went stale.
    version: u64,
}

#[derive(Clone, PartialEq)]
struct TranscriptKey {
    /// The session, so switching history always re-renders even at the same length.
    session: String,
    messages: usize,
    compactions: u32,
    /// The first message is the only one that can be edited in place, via `/system`.
    first_message: u64,
    width: usize,
    markdown: bool,
}

impl TranscriptCache {
    pub(super) fn new() -> Self {
        TranscriptCache {
            key: None,
            lines: Vec::new(),
            texts: Vec::new(),
            tokens: 0,
            version: 0,
        }
    }

    /// The estimated tokens in the committed history, kept in step with the lines.
    pub(super) fn tokens(&self) -> usize {
        self.tokens
    }

    /// The lower-cased plain text of each line, used to find search matches.
    pub(super) fn texts(&self) -> &[String] {
        &self.texts
    }

    /// A counter bumped whenever the lines are rebuilt.
    pub(super) fn version(&self) -> u64 {
        self.version
    }
}

impl TranscriptKey {
    fn of(app: &ChatApp, width: usize) -> Self {
        let first_message = app.session.messages.first().map_or(0, |message| {
            let mut hasher = DefaultHasher::new();
            message.content.hash(&mut hasher);
            hasher.finish()
        });
        TranscriptKey {
            session: app.session.id.clone(),
            messages: app.session.messages.len(),
            compactions: app.session.compactions,
            first_message,
            width,
            markdown: app.settings.markdown,
        }
    }
}

fn draw_transcript(frame: &mut Frame, app: &mut ChatApp, area: Rect) {
    let inner_width = area.width.saturating_sub(2) as usize;
    let inner_height = area.height.saturating_sub(2) as usize;

    // Re-render the committed history only when something it depends on has changed.
    let key = TranscriptKey::of(app, inner_width);
    if app.transcript.key.as_ref() != Some(&key) {
        let mut lines = Vec::new();
        for (index, message) in app.session.messages.iter().enumerate() {
            if index > 0 {
                lines.push(Line::default());
            }
            push_message(&mut lines, message, inner_width, app.settings.markdown);
        }
        let texts: Vec<String> = lines
            .iter()
            .map(|line| line_text(line).to_lowercase())
            .collect();
        app.transcript.lines = lines;
        app.transcript.texts = texts;
        app.transcript.tokens = estimate_messages(&app.session.messages);
        app.transcript.version += 1;
        app.transcript.key = Some(key);
    }

    // The turn still being streamed changes with every token, so it is never cached.
    let mut tail: Vec<Line<'static>> = Vec::new();
    if let Some(state) = &app.stream {
        if !state.reasoning.is_empty() {
            if app.transcript.lines.len() + tail.len() > 0 {
                tail.push(Line::default());
            }
            push_reasoning(&mut tail, &state.reasoning, inner_width);
        }
        if !state.content.is_empty() {
            if app.transcript.lines.len() + tail.len() > 0 {
                tail.push(Line::default());
            }
            push_message(
                &mut tail,
                &ChatMessage::assistant(state.content.clone()),
                inner_width,
                app.settings.markdown,
            );
        }
    }

    let block = Block::bordered()
        .title(Line::from(vec![
            Span::styled(" conversation ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("[{}] ", app.session.id),
                Style::default().fg(Color::DarkGray),
            ),
        ]))
        .border_style(Style::default().fg(Color::DarkGray));

    let cached = app.transcript.lines.len();
    let total = cached + tail.len();
    let max_offset = total.saturating_sub(inner_height);
    // Publish the limit and keep the scroll inside it, so a shrinking history cannot leave
    // the view stuck past the top.
    app.scroll_max = max_offset;
    app.scroll = app.scroll.min(max_offset);

    // Recompute the search against the freshly rendered lines when the query or the lines
    // changed; this may move the view onto the first hit.
    sync_search(app);

    let offset = if app.stick_to_bottom {
        max_offset
    } else {
        max_offset - app.scroll.min(max_offset)
    };

    // Collect the visible rows together with their transcript line numbers, so search hits
    // can be highlighted while the streamed tail (which has no line number) is left alone.
    let mut visible: Vec<(Option<usize>, Line<'static>)> = Vec::with_capacity(inner_height);
    if offset < cached {
        let end = cached.min(offset + inner_height);
        for index in offset..end {
            visible.push((Some(index), app.transcript.lines[index].clone()));
        }
    }
    let stream_start = offset.saturating_sub(cached);
    if visible.len() < inner_height && stream_start < tail.len() {
        let take = (inner_height - visible.len()).min(tail.len() - stream_start);
        for line in &tail[stream_start..stream_start + take] {
            visible.push((None, line.clone()));
        }
    }

    let query = app
        .search
        .as_ref()
        .map(|search| search.query.clone())
        .filter(|query| !query.is_empty());
    let current = app
        .search
        .as_ref()
        .and_then(|search| search.matches.get(search.current).copied());

    let window: Vec<Line<'static>> = visible
        .into_iter()
        .map(|(index, line)| match (&query, index) {
            (Some(query), Some(index)) if is_matched(app, index) => {
                highlight_line(line, query, current == Some(index))
            }
            _ => line,
        })
        .collect();

    frame.render_widget(Paragraph::new(Text::from(window)).block(block), area);
}

/// Recompute the search matches when the query or the rendered lines changed, and jump to the
/// first hit so typing has a visible effect.
fn sync_search(app: &mut ChatApp) {
    let Some(search) = app.search.as_ref() else {
        return;
    };
    let query = app.input.text().to_owned();
    let version = app.transcript.version();
    if search.query == query && search.version == version {
        return;
    }

    let matches = if query.is_empty() {
        Vec::new()
    } else {
        let needle = query.to_lowercase();
        app.transcript
            .texts()
            .iter()
            .enumerate()
            .filter(|(_, text)| text.contains(&needle))
            .map(|(index, _)| index)
            .collect()
    };

    let first = {
        let search = app.search.as_mut().expect("checked above");
        search.query = query;
        search.matches = matches;
        search.current = 0;
        search.version = version;
        search.matches.first().copied()
    };
    if let Some(line) = first {
        app.scroll_to_line(line);
    }
}

/// Whether transcript line `index` is one of the current search's matches.
fn is_matched(app: &ChatApp, index: usize) -> bool {
    app.search
        .as_ref()
        .is_some_and(|search| search.matches.binary_search(&index).is_ok())
}

/// The plain text of a line, with the spans joined back together.
fn line_text(line: &Line<'_>) -> String {
    line.spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
}

/// The character ranges of `needle` in `haystack`, compared case-insensitively.
fn find_all_ignore_case(haystack: &str, needle: &str) -> Vec<(usize, usize)> {
    let hay: Vec<char> = haystack.chars().collect();
    let ned: Vec<char> = needle.chars().collect();
    if ned.is_empty() || ned.len() > hay.len() {
        return Vec::new();
    }

    let mut found = Vec::new();
    let mut start = 0;
    while start + ned.len() <= hay.len() {
        if (0..ned.len()).all(|k| chars_eq_ignore_case(hay[start + k], ned[k])) {
            found.push((start, start + ned.len()));
            start += ned.len();
        } else {
            start += 1;
        }
    }
    found
}

fn chars_eq_ignore_case(a: char, b: char) -> bool {
    if a.is_ascii() && b.is_ascii() {
        a.eq_ignore_ascii_case(&b)
    } else {
        a == b || a.to_lowercase().eq(b.to_lowercase())
    }
}

/// Rebuild `line` with every occurrence of `query` highlighted. The current match is given a
/// stronger colour than the rest.
fn highlight_line(line: Line<'static>, query: &str, current: bool) -> Line<'static> {
    let ranges = find_all_ignore_case(&line_text(&line), query);
    if ranges.is_empty() {
        return line;
    }
    let style = if current {
        Style::default().fg(Color::Black).bg(Color::LightGreen)
    } else {
        Style::default().fg(Color::Black).bg(Color::Yellow)
    };

    let mut spans = Vec::new();
    let mut offset = 0usize;
    for span in line.spans {
        let chars: Vec<char> = span.content.chars().collect();
        let mut i = 0usize;
        while i < chars.len() {
            let position = offset + i;
            if let Some((_, end)) = ranges
                .iter()
                .find(|(start, end)| *start <= position && position < *end)
            {
                let take = (end - position).min(chars.len() - i);
                let text: String = chars[i..i + take].iter().collect();
                spans.push(Span::styled(text, style));
                i += take;
            } else {
                let next = ranges
                    .iter()
                    .map(|(start, _)| *start)
                    .filter(|start| *start > position)
                    .min()
                    .unwrap_or(offset + chars.len());
                let take = (next - position).min(chars.len() - i);
                let text: String = chars[i..i + take].iter().collect();
                spans.push(Span::styled(text, span.style));
                i += take;
            }
        }
        offset += chars.len();
    }
    Line::from(spans)
}

fn push_message(
    lines: &mut Vec<Line<'static>>,
    message: &ChatMessage,
    width: usize,
    markdown_enabled: bool,
) {
    let (label, color) = match message.role {
        Role::User => ("you", Color::Cyan),
        Role::Assistant => ("deepseek", Color::Green),
        Role::System => ("system", Color::Yellow),
        Role::Tool => ("tool", Color::DarkGray),
    };

    lines.push(Line::from(Span::styled(
        label.to_owned(),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )));

    // Answers are markdown. What the user typed and the system prompt are shown exactly
    // as they were written, so the transcript matches the prompt line.
    match message.role {
        Role::Assistant if markdown_enabled => {
            lines.extend(markdown::lines(&message.content, width, INDENT));
        }
        Role::System => {
            lines.extend(markdown::plain_lines(
                &message.content,
                width,
                INDENT,
                Style::default().fg(Color::Gray),
            ));
        }
        // What a tool did is read here, so it stays visible but quiet.
        Role::Tool => {
            lines.extend(markdown::plain_lines(
                &message.content,
                width,
                INDENT,
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::DIM),
            ));
        }
        _ => {
            lines.extend(markdown::plain_lines(
                &message.content,
                width,
                INDENT,
                Style::default(),
            ));
        }
    }

    // An assistant turn that asked for tools shows the request too, so the transcript reads
    // as "asked for, then result" rather than only showing the result.
    if let Some(calls) = &message.tool_calls {
        push_tool_calls(lines, calls, width);
    }
}

/// One dim line per call the assistant asked for: the tool name, then the first line of the
/// arguments. Kept visually quiet: the tool has not run yet.
fn push_tool_calls(lines: &mut Vec<Line<'static>>, calls: &[ToolCall], width: usize) {
    let style = Style::default()
        .fg(Color::DarkGray)
        .add_modifier(Modifier::DIM);

    for call in calls {
        let first = call.function.arguments.lines().next().unwrap_or_default();
        let text = if first.trim().is_empty() {
            format!("{}→ {}", " ".repeat(INDENT), call.function.name)
        } else {
            format!("{}→ {}  {}", " ".repeat(INDENT), call.function.name, first)
        };
        lines.push(Line::from(Span::styled(truncate(&text, width), style)));
    }
}

fn push_reasoning(lines: &mut Vec<Line<'static>>, reasoning: &str, width: usize) {
    let dim = Style::default()
        .fg(Color::DarkGray)
        .add_modifier(Modifier::ITALIC);

    lines.push(Line::from(Span::styled(
        "reasoning".to_owned(),
        Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD),
    )));

    // Only the tail is interesting; the whole trace can be enormous.
    let tail = tail_lines(reasoning, 6);
    let was_truncated = tail.len() < reasoning.lines().count();

    if was_truncated {
        lines.push(Line::from(Span::styled(
            format!("{}… (earlier reasoning hidden)", " ".repeat(INDENT)),
            dim,
        )));
    }
    lines.extend(markdown::plain_lines(&tail, width, INDENT, dim));
}

/// The last `count` lines of `text`, joined back together.
fn tail_lines(text: &str, count: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= count {
        return text.to_owned();
    }
    lines[lines.len() - count..].join("\n")
}

// -------------------------------------------------------------------- input

fn draw_input(frame: &mut Frame, app: &ChatApp, area: Rect) {
    let inner = inner(area);
    let height = inner.height as usize;
    let (cursor_row, cursor_col) = app.input.display_position();

    // Keep the cursor's line inside the visible window.
    let start = (cursor_row + 1).saturating_sub(height);

    let mut lines = Vec::new();
    for (index, text) in app.input.lines().enumerate().skip(start) {
        let prefix = if index == start && index == 0 {
            "> "
        } else {
            "  "
        };
        lines.push(Line::from(vec![
            Span::styled(prefix, Style::default().fg(Color::Cyan)),
            Span::raw(text.to_owned()),
        ]));
    }

    let (title, accent) = if let Some(search) = &app.search {
        let title = if search.query.is_empty() {
            " search ".to_owned()
        } else if search.matches.is_empty() {
            " search · no matches ".to_owned()
        } else {
            format!(" search {}/{} ", search.current + 1, search.matches.len())
        };
        (title, Color::Cyan)
    } else if app.is_busy() {
        (" prompt ".to_owned(), Color::Yellow)
    } else {
        (" prompt ".to_owned(), Color::Gray)
    };
    let border = if app.search.is_some() {
        Color::Cyan
    } else if app.is_busy() {
        Color::Yellow
    } else {
        Color::DarkGray
    };

    let block = Block::bordered()
        .title(Line::from(Span::styled(title, Style::default().fg(accent))))
        .border_style(Style::default().fg(border));

    frame.render_widget(Paragraph::new(Text::from(lines)).block(block), area);

    // Place the terminal cursor.
    let visible_row = cursor_row.saturating_sub(start) as u16;
    let x = inner.x + 2 + cursor_col as u16;
    let y = inner.y + visible_row;
    if y < inner.y + inner.height {
        frame.set_cursor_position((x.min(inner.x + inner.width.saturating_sub(1)), y));
    }
}

// ------------------------------------------------------------------- status

fn draw_status(frame: &mut Frame, app: &ChatApp, area: Rect) {
    let (used, limit) = app.context_usage();
    let ratio = if limit == 0 {
        0.0
    } else {
        used as f32 / limit as f32
    };

    let mut left = Vec::new();
    if let Some(state) = &app.stream {
        left.push(Span::styled(
            format!("{} ", spinner(state.elapsed())),
            Style::default().fg(Color::Yellow),
        ));
        left.push(Span::styled(
            match state.phase {
                Phase::Compacting => "compacting context…".to_owned(),
                Phase::Waiting => "waiting for the first token…".to_owned(),
                Phase::Streaming => format!(
                    "streaming ({:.1}s, ~{} tokens)",
                    state.elapsed().as_secs_f32(),
                    estimate_tokens(&state.content)
                ),
                Phase::Finishing => "finishing…".to_owned(),
            },
            Style::default().fg(Color::Yellow),
        ));
        left.push(Span::styled("  ·  ", Style::default().fg(Color::DarkGray)));
    } else if let Some(tool) = &app.running_tool {
        // A tool call runs on a worker thread; animate it exactly like a streaming answer
        // so a long command never looks like the interface has frozen.
        let elapsed = tool.started.elapsed();
        left.push(Span::styled(
            format!("{} ", spinner(elapsed)),
            Style::default().fg(Color::Yellow),
        ));
        left.push(Span::styled(
            format!("running {} ({:.1}s)", tool.name, elapsed.as_secs_f32()),
            Style::default().fg(Color::Yellow),
        ));
        left.push(Span::styled("  ·  ", Style::default().fg(Color::DarkGray)));
    }

    left.push(Span::styled(
        app.session.model.clone(),
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD),
    ));
    left.push(Span::styled(
        format!("  think:{}", app.session.thinking.label()),
        Style::default().fg(Color::Magenta),
    ));
    if !app.tools_enabled {
        left.push(Span::styled(
            "  tools:off".to_owned(),
            Style::default().fg(Color::DarkGray),
        ));
    } else if app.tools_auto() {
        left.push(Span::styled(
            "  tools:auto".to_owned(),
            Style::default().fg(Color::Yellow),
        ));
    }
    left.push(Span::styled(
        format!("  {}", truncate(&app.session.title(), 28)),
        Style::default().fg(Color::Cyan),
    ));
    if app.session.compactions > 0 {
        left.push(Span::styled(
            format!("  compacted x{}", app.session.compactions),
            Style::default().fg(Color::Yellow),
        ));
    }

    let gauge = format!(
        "ctx {} {:>3.0}% ({}/{})  ↑{} ↓{} Σ{}  compact:{:.0}%  cache:{}",
        gauge_bar(ratio, 10),
        (ratio * 100.0).min(999.0),
        format_tokens(used as u32),
        format_tokens(limit as u32),
        format_tokens(app.session.usage.prompt_tokens),
        format_tokens(app.session.usage.completion_tokens),
        format_tokens(app.session.usage.total_tokens),
        app.settings.compact_ratio * 100.0,
        cache_hit_rate(&app.session.usage),
    );

    let balance = match &app.balance {
        Balance::Known(info) => Some(format_balance(info)),
        Balance::Unknown | Balance::Unsupported => None,
    };
    let row1 = clamp(Rect { height: 1, ..area }, area);
    match balance {
        Some(text) => {
            let width = text.chars().count() as u16;
            let [left_area, right_area] =
                Layout::horizontal([Constraint::Min(0), Constraint::Length(width)]).areas(row1);
            frame.render_widget(Paragraph::new(Line::from(left)), left_area);
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    text,
                    Style::default().fg(Color::LightCyan),
                )))
                .alignment(Alignment::Right),
                right_area,
            );
        }
        None => {
            frame.render_widget(Paragraph::new(Line::from(left)), row1);
        }
    }
    let row2 = clamp(
        Rect {
            y: area.y + 1,
            height: 1,
            ..area
        },
        area,
    );
    let gauge_width = gauge.chars().count() as u16;
    let gauge_line = Line::from(Span::styled(gauge, Style::default().fg(gauge_color(ratio))));
    // Whatever is left of the row after the gauge is what a status message can use. If there
    // is too little of it, the gauge — the important information — keeps the whole row.
    let available = row2.width.saturating_sub(gauge_width);
    match &app.status {
        None => frame.render_widget(Paragraph::new(gauge_line), row2),
        Some(status) => {
            // Status messages (a compaction, a switched model, a failed tool) would otherwise
            // be written but never drawn, making commands like `/compact` look inert. Draw
            // them right-aligned. A hint is only worth showing in full and is dropped when
            // the window is too narrow; diagnostics may be truncated, so they just need a
            // minimum amount of room.
            let color = match status.kind {
                StatusKind::Hint => Color::DarkGray,
                StatusKind::Info => Color::Gray,
                StatusKind::Warn => Color::Yellow,
                StatusKind::Error => Color::Red,
            };
            let (needed, budget) = match status.kind {
                StatusKind::Hint => {
                    let full = status.text.chars().count() as u16;
                    (full, full as usize)
                }
                _ => (
                    MIN_STATUS_WIDTH,
                    available.saturating_sub(1).min(row2.width / 2) as usize,
                ),
            };
            if available < needed {
                frame.render_widget(Paragraph::new(gauge_line), row2);
                return;
            }
            let text = truncate(&status.text, budget.max(1));
            let width = text.chars().count() as u16;
            let [gauge_area, status_area] =
                Layout::horizontal([Constraint::Min(0), Constraint::Length(width)]).areas(row2);
            frame.render_widget(Paragraph::new(gauge_line), gauge_area);
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(text, Style::default().fg(color))))
                    .alignment(Alignment::Right),
                status_area,
            );
        }
    }
}

fn gauge_color(ratio: f32) -> Color {
    if ratio >= 0.9 {
        Color::Red
    } else if ratio >= 0.7 {
        Color::Yellow
    } else {
        Color::DarkGray
    }
}

fn gauge_bar(ratio: f32, width: usize) -> String {
    let filled = ((ratio.clamp(0.0, 1.0)) * width as f32).round() as usize;
    let filled = filled.min(width);
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

/// The balance line: total credit, with the granted part called out when there is one, so
/// "including gifted credit" is visible rather than hidden inside the total.
fn format_balance(info: &crate::api::BalanceInfo) -> String {
    let mut text = if info.currency.trim().is_empty() {
        format!("balance {}", info.total_balance)
    } else {
        format!("balance {} {}", info.currency, info.total_balance)
    };
    if info
        .granted_balance
        .parse::<f64>()
        .is_ok_and(|granted| granted > 0.0)
    {
        text.push_str(&format!(" (gift {})", info.granted_balance));
    }
    text
}

/// The share of prompt tokens served from the context cache, as a percentage.
fn cache_hit_rate(usage: &Usage) -> String {
    let served = usage
        .prompt_cache_hit_tokens
        .saturating_add(usage.prompt_cache_miss_tokens);
    if served == 0 {
        return "--".to_owned();
    }
    format!(
        "{:.0}%",
        usage.prompt_cache_hit_tokens as f32 / served as f32 * 100.0
    )
}

fn spinner(elapsed: std::time::Duration) -> char {
    const FRAMES: [char; 4] = ['|', '/', '-', '\\'];
    FRAMES[(elapsed.as_millis() / 120) as usize % FRAMES.len()]
}

// ------------------------------------------------------------------ overlays

fn draw_overlay(frame: &mut Frame, app: &ChatApp, overlay: Overlay, area: Rect) {
    let (title, lines, highlight): (String, Vec<Line<'static>>, Option<usize>) = match overlay {
        Overlay::Models { index } => (
            " models (Enter select · r refresh · Esc close) ".to_owned(),
            app.models
                .iter()
                .enumerate()
                .map(|(position, model)| {
                    let selected = position == index;
                    let current = model.id == app.session.model;
                    let text = format!(
                        "{}{}{}",
                        if selected { "> " } else { "  " },
                        model.id,
                        if current { "  (current)" } else { "" }
                    );
                    Line::from(Span::styled(
                        text,
                        if selected {
                            Style::default()
                                .fg(Color::Black)
                                .bg(Color::Cyan)
                                .add_modifier(Modifier::BOLD)
                        } else if current {
                            Style::default().fg(Color::Green)
                        } else {
                            Style::default()
                        },
                    ))
                })
                .collect(),
            Some(index),
        ),
        Overlay::Thinking { index } => (
            " thinking level (Enter select · Esc close) ".to_owned(),
            crate::config::ThinkingEffort::ALL
                .iter()
                .enumerate()
                .map(|(position, effort)| {
                    let selected = position == index;
                    let text = format!(
                        "{}{}{}",
                        if selected { "> " } else { "  " },
                        effort.label(),
                        match effort.as_api() {
                            None => "  (no reasoning tokens)",
                            Some(_) => "  (reasoning_effort)",
                        }
                    );
                    Line::from(Span::styled(
                        text,
                        if selected {
                            Style::default()
                                .fg(Color::Black)
                                .bg(Color::Magenta)
                                .add_modifier(Modifier::BOLD)
                        } else {
                            Style::default()
                        },
                    ))
                })
                .collect(),
            Some(index),
        ),
        Overlay::Sessions { index } => (
            " sessions (Enter open · n new · d delete · Esc close) ".to_owned(),
            app.sessions
                .iter()
                .enumerate()
                .map(|(position, session)| {
                    let selected = position == index;
                    let current = session.id == app.session.id;
                    let text = format!(
                        "{}{}  {}  ({} turns, {} msg){}",
                        if selected { "> " } else { "  " },
                        session.title,
                        session.updated_at.format("%m-%d %H:%M"),
                        session.turns,
                        session.message_count,
                        if current { "  (open)" } else { "" }
                    );
                    Line::from(Span::styled(
                        text,
                        if selected {
                            Style::default()
                                .fg(Color::Black)
                                .bg(Color::Yellow)
                                .add_modifier(Modifier::BOLD)
                        } else if current {
                            Style::default().fg(Color::Green)
                        } else {
                            Style::default()
                        },
                    ))
                })
                .collect(),
            Some(index),
        ),
        Overlay::Help => (" help (any key closes) ".to_owned(), help_lines(app), None),
        Overlay::ToolCall {
            index,
            total,
            preview,
        } => (
            format!(
                " tool call {index}/{total} (Enter run · Esc skip · a allow all {} calls) ",
                preview.risk.label()
            ),
            tool_call_lines(&preview),
            None,
        ),
        Overlay::Quit => (
            " quit (Enter/y quit · Esc/n cancel) ".to_owned(),
            vec![Line::from(Span::styled(
                "Quit the chat? The session is saved either way.",
                Style::default().fg(Color::Yellow),
            ))],
            None,
        ),
    };

    let mut lines = lines;
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "nothing to show",
            Style::default().fg(Color::DarkGray),
        )));
    }

    draw_panel(frame, area, &title, lines, highlight);
}

/// The body of the tool confirmation card: what it will do, where, and a preview of the
/// content. Kept readable in an 80-column terminal.
fn tool_call_lines(preview: &tools::Preview) -> Vec<Line<'static>> {
    let name = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);

    // Lead with how much the call could change, so the risk is impossible to miss before
    // the key is pressed.
    let (warning, warning_style) = match preview.risk {
        tools::Risk::Read => (
            "Reads a file — its contents are sent to DeepSeek.",
            Style::default().fg(Color::Gray),
        ),
        tools::Risk::Write => (
            "Writes to your disk — an existing file is replaced.",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        tools::Risk::Command => (
            "Runs a command with your privileges — there is no denylist. Read it carefully.",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
    };

    let mut lines = vec![
        Line::from(Span::styled(warning, warning_style)),
        Line::default(),
        Line::from(vec![
            Span::styled(preview.tool.clone(), name),
            Span::styled(
                format!("  {}", preview.summary),
                Style::default().fg(Color::Gray),
            ),
        ]),
        // The full path, so the user can see exactly which file is touched.
        Line::from(Span::styled(
            preview.path.display().to_string(),
            Style::default().fg(Color::Yellow),
        )),
    ];

    if !preview.lines.is_empty() {
        lines.push(Line::default());
        for line in &preview.lines {
            lines.push(Line::from(Span::styled(
                format!("  {line}"),
                Style::default(),
            )));
        }
        if preview.hidden > 0 {
            lines.push(Line::from(Span::styled(
                format!("  … {} more line(s) not shown", preview.hidden),
                Style::default().fg(Color::DarkGray),
            )));
        }
    }

    lines
}

fn draw_panel(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    lines: Vec<Line<'static>>,
    highlight: Option<usize>,
) {
    let width = (area.width.saturating_sub(4)).clamp(20, 78);
    let height = (lines.len() as u16 + 2)
        .min(area.height.saturating_sub(2))
        .max(3);

    let panel = clamp(
        Rect {
            x: area.x + (area.width.saturating_sub(width)) / 2,
            y: area.y + (area.height.saturating_sub(height)) / 2,
            width,
            height,
        },
        area,
    );

    // Scroll so the highlighted entry stays visible.
    let text_height = panel.height.saturating_sub(2) as usize;
    let offset = match highlight {
        Some(index) => index.saturating_add(1).saturating_sub(text_height),
        None => 0,
    }
    .min(lines.len().saturating_sub(text_height));

    frame.render_widget(Clear, panel);
    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .block(
                Block::bordered()
                    .title(Line::from(Span::styled(
                        title.to_owned(),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )))
                    .border_style(Style::default().fg(Color::Cyan)),
            )
            .scroll((offset.min(u16::MAX as usize) as u16, 0)),
        panel,
    );
}

fn help_lines(app: &ChatApp) -> Vec<Line<'static>> {
    let key = Style::default().fg(Color::Cyan);
    let description = Style::default().fg(Color::Gray);

    let mut lines = vec![
        entry("Enter", "send the prompt", key, description),
        entry("Shift+Enter", "insert a newline", key, description),
        entry("Ctrl+O", "switch model", key, description),
        entry("Ctrl+T", "switch thinking level", key, description),
        entry("Ctrl+B", "switch session / history", key, description),
        entry("Ctrl+N", "start a new session", key, description),
        entry("Ctrl+R", "regenerate the last answer", key, description),
        entry("Ctrl+C", "cancel the answer, or quit", key, description),
        entry("Ctrl+X", "quit", key, description),
        entry("Ctrl+/", "this help card", key, description),
        entry(
            "Esc",
            "cancel the answer / clear the prompt",
            key,
            description,
        ),
        entry("PgUp/PgDn", "scroll the transcript", key, description),
        entry("Ctrl+F", "search the transcript", key, description),
        entry(
            "Up/Down",
            "move in the prompt, or browse history",
            key,
            description,
        ),
        Line::default(),
        Line::from(Span::styled(
            "slash commands",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )),
        entry("/help", "this card", key, description),
        entry("/model [name]", "show or set the model", key, description),
        entry(
            "/think [level]",
            "show or set off/low/high/max",
            key,
            description,
        ),
        entry("/sessions", "browse past conversations", key, description),
        entry(
            "/new [name]",
            "start a fresh conversation",
            key,
            description,
        ),
        entry(
            "/rename <name>",
            "rename the current conversation",
            key,
            description,
        ),
        entry("/compact", "summarise the history now", key, description),
        entry("/clear", "forget the current transcript", key, description),
        entry("/balance", "refresh the account balance", key, description),
        entry(
            "/system [text]",
            "show, set or clear the system prompt",
            key,
            description,
        ),
        entry("/markdown", "toggle formatted answers", key, description),
        entry("/tools", "toggle the file tools", key, description),
        entry("/save", "write the session to disk", key, description),
        entry("/quit", "leave the chat", key, description),
        Line::default(),
        Line::from(Span::styled(
            "tool calls (while a call is waiting)",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )),
        entry("Enter", "run the tool call", key, description),
        entry("Esc", "skip the tool call", key, description),
        entry(
            "a",
            "allow every call of this class for the session",
            key,
            description,
        ),
    ];

    if !app.notices().is_empty() {
        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            "notices",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )));
        for notice in app.notices() {
            lines.push(Line::from(Span::styled(
                format!("  {notice}"),
                Style::default().fg(Color::Yellow),
            )));
        }
    }

    lines
}

fn entry(
    key: &str,
    description: &str,
    key_style: Style,
    description_style: Style,
) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {key:<14}"), key_style),
        Span::styled(description.to_owned(), description_style),
    ])
}

// ------------------------------------------------------------------- helpers

fn inner(area: Rect) -> Rect {
    Rect {
        x: area.x + 1,
        y: area.y + 1,
        width: area.width.saturating_sub(2),
        height: area.height.saturating_sub(2),
    }
}

/// Clip `rect` so it can never point outside `bounds`.
///
/// Ratatui panics when a widget is asked to draw outside the buffer, and the layout can
/// produce exactly that on very small terminals, so every sub-rect goes through here
/// before it is rendered.
fn clamp(rect: Rect, bounds: Rect) -> Rect {
    let x = rect.x.max(bounds.x).min(bounds.right());
    let y = rect.y.max(bounds.y).min(bounds.bottom());
    Rect {
        x,
        y,
        width: rect.width.min(bounds.right().saturating_sub(x)),
        height: rect.height.min(bounds.bottom().saturating_sub(y)),
    }
}

fn truncate(text: &str, max: usize) -> String {
    crate::api::types::truncate(text, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ChatMessage, DeepSeekClient, ModelInfo};
    use crate::chat::app::{ChatApp, ChatSetup, Phase, Status, StatusKind};
    use crate::config::Settings;
    use crate::session::{Session, SessionStore};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A throwaway directory for the session store.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "nu_plugin_ds_ui_{}_{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::SeqCst)
            ));
            std::fs::create_dir_all(&path).unwrap();
            TempDir(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// An app with a prepared transcript. The client is never used: the UI tests only
    /// render state.
    fn app(session: Session) -> (TempDir, ChatApp) {
        let dir = TempDir::new();
        let store = SessionStore::new(dir.0.clone()).unwrap();
        let setup = ChatSetup {
            client: DeepSeekClient::new("test-key", "http://127.0.0.1:1").unwrap(),
            settings: Settings::default(),
            store,
            models: vec![ModelInfo {
                id: "deepseek-flash".to_owned(),
                owned_by: None,
                created: None,
            }],
            session,
            initial_prompt: None,
            context: tools::Context {
                cwd: std::env::temp_dir(),
                nu_bin: None,
                command_timeout: std::time::Duration::from_secs(120),
            },
            tools_enabled: true,
            approved: Default::default(),
        };
        (dir, ChatApp::new(setup).unwrap())
    }

    fn transcript() -> Session {
        let mut session = Session::new(
            "demo",
            "deepseek-flash",
            crate::config::ThinkingEffort::High,
            Some("be concise"),
        );
        session.push(ChatMessage::user("list the files"));
        session.push(ChatMessage::assistant("Here you go:\n```\nls ./\n```"));
        session
    }

    /// Render one frame and return the visible characters, one string per row.
    fn snapshot(app: &mut ChatApp, width: u16, height: u16) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();

        let buffer = terminal.backend().buffer();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect()
    }

    fn text(rows: &[String]) -> String {
        rows.join("\n")
    }

    #[test]
    fn shows_the_transcript_and_status_bar() {
        let (_dir, mut app) = app(transcript());
        let rows = snapshot(&mut app, 80, 24);
        let screen = text(&rows);

        assert!(
            screen.contains("you"),
            "the user turn should be labelled:\n{screen}"
        );
        assert!(
            screen.contains("deepseek"),
            "the answer should be labelled:\n{screen}"
        );
        assert!(screen.contains("list the files"), "{screen}");
        assert!(screen.contains("ls ./"), "code should be shown:\n{screen}");
        assert!(
            screen.contains("deepseek-flash"),
            "the model belongs in the status bar:\n{screen}"
        );
        assert!(screen.contains("think:high"), "{screen}");
        assert!(
            screen.contains("ctx"),
            "the context gauge belongs in the status bar:\n{screen}"
        );
        assert!(
            screen.contains("prompt"),
            "the input box should be titled:\n{screen}"
        );
    }

    #[test]
    fn draws_status_messages_and_keeps_the_gauge() {
        let (_dir, mut app) = app(transcript());
        app.status = Some(Status {
            text: "compacted 6 messages into a summary".to_owned(),
            kind: StatusKind::Info,
        });

        let screen = text(&snapshot(&mut app, 160, 24));
        assert!(
            screen.contains("compacted 6 messages"),
            "the status message must be drawn:\n{screen}"
        );
        assert!(
            screen.contains("ctx"),
            "the context gauge must still share the row:\n{screen}"
        );
    }

    #[test]
    fn hides_the_key_hint_when_the_window_is_narrow() {
        let (_dir, mut app) = app(transcript());
        app.status = Some(Status {
            text: "Ctrl+O model · Ctrl+T thinking · Ctrl+B sessions · Ctrl+/ help · Ctrl+X quit"
                .to_owned(),
            kind: StatusKind::Hint,
        });

        // Too narrow to show the hint without squeezing the gauge: the hint is dropped.
        let narrow = text(&snapshot(&mut app, 80, 24));
        assert!(
            !narrow.contains("Ctrl+O"),
            "the hint must be hidden when the window is narrow:\n{narrow}"
        );
        assert!(narrow.contains("ctx"), "the gauge must stay:\n{narrow}");

        // Wide enough for both: the hint is shown in full.
        let wide = text(&snapshot(&mut app, 200, 24));
        assert!(
            wide.contains("Ctrl+O"),
            "the hint should show when there is room:\n{wide}"
        );
        assert!(
            wide.contains("Ctrl+X quit"),
            "the hint must be complete:\n{wide}"
        );
    }

    #[test]
    fn survives_tiny_terminals() {
        let (_dir, mut app) = app(transcript());
        // Anything below the guard renders a single hint line without panicking.
        for (width, height) in [(10u16, 3u16), (24, 6), (30, 7), (40, 8)] {
            let rows = snapshot(&mut app, width, height);
            assert_eq!(rows.len(), height as usize);
        }
    }

    #[test]
    fn keeps_the_newest_output_in_view() {
        let mut session = Session::new("demo", "deepseek-flash", Default::default(), None);
        for index in 0..40 {
            session.push(ChatMessage::user(format!("question {index}")));
            session.push(ChatMessage::assistant(format!("answer {index}")));
        }
        let (_dir, mut app) = app(session);
        app.stick_to_bottom = true;

        let screen = text(&snapshot(&mut app, 60, 20));
        assert!(
            screen.contains("answer 39"),
            "the newest answer should be visible:\n{screen}"
        );
        assert!(
            !screen.contains("question 0"),
            "the oldest turn should have scrolled away:\n{screen}"
        );
    }

    #[test]
    fn scrolls_back_through_the_transcript() {
        let (_dir, mut app) = app(transcript());
        app.stick_to_bottom = true;
        let bottom = text(&snapshot(&mut app, 60, 12));
        assert!(bottom.contains("list the files"), "{bottom}");

        // Scrolling further than the transcript is long must not panic, and once we
        // return to the end the newest content is back.
        app.scroll = usize::MAX;
        app.stick_to_bottom = false;
        let top = text(&snapshot(&mut app, 60, 12));
        assert!(
            top.contains("be concise") || top.contains("system"),
            "\n{top}"
        );

        app.scroll = 0;
        app.stick_to_bottom = true;
        let bottom_again = text(&snapshot(&mut app, 60, 12));
        assert!(bottom_again.contains("ls ./"), "{bottom_again}");
    }

    #[test]
    fn renders_the_help_overlay() {
        let (_dir, mut app) = app(transcript());
        app.overlay = Some(Overlay::Help);
        let screen = text(&snapshot(&mut app, 90, 30));

        assert!(screen.contains("help"), "{screen}");
        assert!(screen.contains("Ctrl+O"), "{screen}");
        assert!(screen.contains("/compact"), "{screen}");
    }

    #[test]
    fn renders_the_session_picker() {
        let (_dir, mut app) = app(transcript());
        app.sessions = vec![crate::session::SessionSummary {
            id: "abc".to_owned(),
            title: "planning".to_owned(),
            model: "deepseek-flash".to_owned(),
            updated_at: chrono::Local::now(),
            message_count: 4,
            turns: 2,
        }];
        app.overlay = Some(Overlay::Sessions { index: 0 });
        let screen = text(&snapshot(&mut app, 90, 24));

        assert!(screen.contains("sessions"), "{screen}");
        assert!(screen.contains("planning"), "{screen}");
    }

    #[test]
    fn shows_a_live_answer_while_streaming() {
        let (_dir, mut app) = app(transcript());
        let mut state = crate::chat::app::StreamState::new(1, Phase::Streaming);
        state.reasoning = "thinking about it".to_owned();
        state.content = "partial answer".to_owned();
        app.stream = Some(state);

        let screen = text(&snapshot(&mut app, 80, 24));
        assert!(screen.contains("reasoning"), "{screen}");
        assert!(screen.contains("thinking about it"), "{screen}");
        assert!(screen.contains("partial answer"), "{screen}");
        assert!(
            screen.contains("streaming"),
            "the status bar should show progress:\n{screen}"
        );
    }

    #[test]
    fn wraps_long_lines_instead_of_clipping_them() {
        let mut session = Session::new("demo", "deepseek-flash", Default::default(), None);
        let long: String = (0..40).map(|index| format!("word{index} ")).collect();
        session.push(ChatMessage::user(long.clone()));
        let (_dir, mut app) = app(session);
        app.stick_to_bottom = true;

        let screen = text(&snapshot(&mut app, 40, 30));
        assert!(screen.contains("word0 "), "{screen}");
        assert!(screen.contains("word39"), "{screen}");

        let short_screen = text(&snapshot(&mut app, 40, 8));
        assert!(short_screen.contains("word39"), "{short_screen}");
    }

    #[test]
    fn formats_answers_as_markdown() {
        let mut session = Session::new("demo", "deepseek-flash", Default::default(), None);
        session.push(ChatMessage::assistant(
            "# Heading\n\nsome **strong** words\n\n- a bullet",
        ));
        let (_dir, mut app) = app(session);
        let screen = text(&snapshot(&mut app, 60, 20));

        assert!(screen.contains("Heading"), "{screen}");
        assert!(
            !screen.contains('#'),
            "heading markers should be gone: {screen}"
        );
        assert!(
            !screen.contains("**"),
            "emphasis markers should be gone: {screen}"
        );
        assert!(screen.contains("strong"), "{screen}");
        assert!(screen.contains("• a bullet"), "{screen}");
    }

    #[test]
    fn markdown_can_be_turned_off() {
        let mut session = Session::new("demo", "deepseek-flash", Default::default(), None);
        session.push(ChatMessage::assistant("# Heading\n\nsome **strong** words"));
        let (_dir, mut app) = app(session);
        app.settings.markdown = false;

        let screen = text(&snapshot(&mut app, 60, 20));
        assert!(screen.contains("# Heading"), "{screen}");
        assert!(screen.contains("**strong**"), "{screen}");
    }

    #[test]
    fn shows_the_balance_cache_rate_and_compact_ratio() {
        let (_dir, mut app) = app(transcript());
        app.settings.compact_ratio = 0.8;
        app.session.usage = Usage {
            prompt_tokens: 2_000,
            completion_tokens: 100,
            total_tokens: 2_100,
            prompt_cache_hit_tokens: 1_500,
            prompt_cache_miss_tokens: 500,
        };
        app.balance = Balance::Known(crate::api::BalanceInfo {
            currency: "CNY".to_owned(),
            total_balance: "110.00".to_owned(),
            granted_balance: "10.00".to_owned(),
            topped_up_balance: "100.00".to_owned(),
        });

        let rows = snapshot(&mut app, 100, 24);
        let screen = text(&rows);

        assert!(screen.contains("compact:80%"), "{screen}");
        assert!(screen.contains("cache:75%"), "{screen}");

        // The balance must sit flush against the right edge of its row.
        let row = rows
            .iter()
            .find(|row| row.contains("balance"))
            .unwrap_or_else(|| panic!("no balance row in:\n{screen}"));
        assert!(
            row.ends_with("balance CNY 110.00 (gift 10.00)"),
            "the balance should be right-aligned: {row:?}"
        );
    }

    #[test]
    fn shows_a_running_tool_with_a_spinner() {
        let (_dir, mut app) = app(transcript());
        app.running_tool = Some(crate::chat::app::RunningTool {
            name: "run_nu".to_owned(),
            started: std::time::Instant::now(),
        });

        let screen = text(&snapshot(&mut app, 80, 24));
        assert!(screen.contains("running run_nu ("), "{screen}");
        assert!(screen.contains("deepseek-flash"), "{screen}");
    }

    #[test]
    fn the_spinner_animates_over_time() {
        use std::time::Duration;
        // The frame must change as time passes, or nothing would ever animate.
        assert_ne!(
            spinner(Duration::from_millis(0)),
            spinner(Duration::from_millis(120))
        );
        assert_ne!(
            spinner(Duration::from_millis(120)),
            spinner(Duration::from_millis(240))
        );
    }

    #[test]
    fn hides_the_balance_when_the_provider_has_no_endpoint() {
        let (_dir, mut app) = app(transcript());
        app.balance = Balance::Unsupported;

        let screen = text(&snapshot(&mut app, 100, 24));
        assert!(
            !screen.contains("balance"),
            "no balance should be drawn for a provider without the endpoint:\n{screen}"
        );
        assert!(screen.contains("deepseek-flash"), "{screen}");
    }

    #[test]
    fn a_new_message_shows_up_after_the_cache_is_warm() {
        let (_dir, mut app) = app(transcript());
        // Draw once to warm the cache, then append: the next frame must show the new turn.
        let first = text(&snapshot(&mut app, 60, 20));
        assert!(first.contains("list the files"), "{first}");

        app.session.push(ChatMessage::user("second question"));
        let second = text(&snapshot(&mut app, 60, 20));
        assert!(second.contains("second question"), "{second}");
    }

    #[test]
    fn editing_the_system_prompt_rerenders_without_an_append() {
        let (_dir, mut app) = app(transcript());
        let first = text(&snapshot(&mut app, 60, 20));
        assert!(first.contains("be concise"), "{first}");

        // Same message count, different content: `/system` does exactly this.
        app.session.messages[0] = ChatMessage::system("a brand new prompt");
        let second = text(&snapshot(&mut app, 60, 20));
        assert!(second.contains("a brand new prompt"), "{second}");
        assert!(!second.contains("be concise"), "{second}");
    }

    #[test]
    fn clearing_the_transcript_empties_the_render() {
        let (_dir, mut app) = app(transcript());
        let first = text(&snapshot(&mut app, 60, 20));
        assert!(first.contains("ls ./"), "{first}");

        app.session.clear_transcript();
        let second = text(&snapshot(&mut app, 60, 20));
        assert!(!second.contains("ls ./"), "{second}");
        assert!(!second.contains("list the files"), "{second}");
    }

    #[test]
    fn switching_to_another_session_rerenders() {
        let (_dir, mut app) = app(transcript());
        let first = text(&snapshot(&mut app, 60, 20));
        assert!(first.contains("list the files"), "{first}");

        // Same number of messages, different session: the cache must not be reused.
        let mut other = Session::new("demo", "deepseek-flash", Default::default(), None);
        other.push(ChatMessage::user("totally"));
        other.push(ChatMessage::assistant("different"));
        other.push(ChatMessage::user("history"));
        app.session = other;

        let second = text(&snapshot(&mut app, 60, 20));
        assert!(second.contains("totally"), "{second}");
        assert!(!second.contains("list the files"), "{second}");
    }

    #[test]
    fn finds_matches_case_insensitively() {
        assert_eq!(
            find_all_ignore_case("The Quick brown", "quick"),
            vec![(4, 9)]
        );
        assert_eq!(find_all_ignore_case("aaaa", "aa"), vec![(0, 2), (2, 4)]);
        assert!(find_all_ignore_case("hello", "z").is_empty());
        assert!(find_all_ignore_case("hi", "hello").is_empty());
    }

    #[test]
    fn ctrl_f_search_highlights_and_counts_matches() {
        let mut session = Session::new("demo", "deepseek-flash", Default::default(), None);
        session.push(ChatMessage::user("the quick brown fox"));
        session.push(ChatMessage::assistant("a lazy dog sleeps"));
        session.push(ChatMessage::user("the quick red fox"));
        let (_dir, mut app) = app(session);

        app.open_search();
        app.input.set_text("quick");

        let mut terminal = Terminal::new(TestBackend::new(60, 20)).unwrap();
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let buffer = terminal.backend().buffer();

        let screen: String = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            screen.contains("search 1/2"),
            "the counter should show both matches:\n{screen}"
        );

        let mut highlighted = false;
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                if buffer[(x, y)].bg == Color::Yellow {
                    highlighted = true;
                }
            }
        }
        assert!(highlighted, "a match should be highlighted:\n{screen}");
    }

    #[test]
    fn renders_the_quit_confirmation() {
        let (_dir, mut app) = app(transcript());
        app.overlay = Some(Overlay::Quit);
        let screen = text(&snapshot(&mut app, 80, 24));

        assert!(screen.contains("Quit the chat?"), "{screen}");
        assert!(screen.contains("Esc/n cancel"), "{screen}");
    }

    #[test]
    fn the_tool_confirmation_leads_with_the_risk() {
        let (_dir, mut app) = app(transcript());
        app.overlay = Some(Overlay::ToolCall {
            index: 1,
            total: 1,
            preview: tools::Preview {
                tool: "run_nu".to_owned(),
                path: std::path::PathBuf::from("."),
                summary: "runs in . (up to 120s)".to_owned(),
                lines: vec!["echo hi".to_owned()],
                hidden: 0,
                risk: tools::Risk::Command,
            },
        });
        let screen = text(&snapshot(&mut app, 90, 30));

        assert!(screen.contains("no denylist"), "{screen}");
        assert!(screen.contains("run_nu"), "{screen}");
    }
}
