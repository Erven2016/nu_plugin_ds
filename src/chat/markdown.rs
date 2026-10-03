//! Markdown rendering for the chat transcript.
//!
//! The renderer turns a markdown document into the exact visual lines ratatui has to
//! draw. It wraps text itself rather than letting `ratatui::widgets::Paragraph` do it,
//! because the transcript is scrolled by line and the line count has to be known before
//! rendering.
//!
//! It is deliberately forgiving: answers arrive one token at a time, so an unterminated
//! code fence or a lone `**` mid-stream must render as literal text rather than break or
//! swallow the rest of the message.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthChar;

/// A run of characters without line breaks, carrying a style per character.
///
/// Styles live per character so that emphasis, inline code and plain text can be wrapped
/// and split across lines without losing their formatting.
pub type Styled = Vec<(char, Style)>;

/// Render markdown into transcript lines.
pub fn lines(text: &str, width: usize, indent: usize) -> Vec<Line<'static>> {
    let mut renderer = Renderer {
        width: width.max(12),
        indent,
        out: Vec::new(),
    };
    renderer.run(text);
    renderer.out
}

/// Render unstyled text, still wrapped and indented like the rest of the transcript.
pub fn plain_lines(text: &str, width: usize, indent: usize, style: Style) -> Vec<Line<'static>> {
    let width = width.max(12).saturating_sub(indent).max(8);
    let prefix = vec![Span::styled(" ".repeat(indent), style)];
    let mut out = Vec::new();

    for source in text.split('\n') {
        if source.trim().is_empty() {
            out.push(Line::default());
            continue;
        }
        let styled: Styled = source.chars().map(|ch| (ch, style)).collect();
        for chunk in wrap_words(&styled, width) {
            out.push(to_line(&prefix, &chunk));
        }
    }

    out
}

// ---------------------------------------------------------------- renderer

struct Renderer {
    width: usize,
    indent: usize,
    out: Vec<Line<'static>>,
}

impl Renderer {
    fn run(&mut self, text: &str) {
        let source: Vec<&str> = text.split('\n').collect();
        let mut index = 0;
        let mut in_code = false;

        while index < source.len() {
            let line = source[index];

            if in_code {
                if fence_language(line).is_some() {
                    in_code = false;
                } else {
                    self.push_code(line);
                }
                index += 1;
                continue;
            }

            if fence_language(line).is_some() {
                in_code = true;
                index += 1;
                continue;
            }

            if let Some((level, title)) = heading(line) {
                self.push_heading(level, &title);
                index += 1;
                continue;
            }

            if is_rule(line) {
                self.push_rule();
                index += 1;
                continue;
            }

            if let Some(item) = list_item(line) {
                self.push_item(item);
                index += 1;
                continue;
            }

            if line.trim_start().starts_with('>') {
                self.push_quote(line);
                index += 1;
                continue;
            }

            if let Some((header, rows, consumed)) = table_at(&source, index) {
                self.push_table(header, rows);
                index += consumed;
                continue;
            }

            self.push_paragraph(line);
            index += 1;
        }
    }

    /// Width available for content once the indent and any gutter are removed.
    fn content_width(&self, gutter: usize) -> usize {
        self.width.saturating_sub(self.indent + gutter).max(8)
    }

    fn indent_prefix(&self) -> Vec<Span<'static>> {
        if self.indent == 0 {
            Vec::new()
        } else {
            vec![Span::raw(" ".repeat(self.indent))]
        }
    }

    fn push_paragraph(&mut self, line: &str) {
        if line.trim().is_empty() {
            self.out.push(Line::default());
            return;
        }
        let width = self.content_width(0);
        let prefix = self.indent_prefix();
        for chunk in wrap_words(&inline(line), width) {
            self.out.push(to_line(&prefix, &chunk));
        }
    }

    fn push_heading(&mut self, level: usize, title: &str) {
        let style = heading_style(level);
        let width = self.content_width(0);
        let prefix = self.indent_prefix();

        let mut styled = inline(title);
        for entry in styled.iter_mut() {
            entry.1 = entry.1.patch(style);
        }

        for chunk in wrap_words(&styled, width) {
            self.out.push(to_line(&prefix, &chunk));
        }
    }

    fn push_rule(&mut self) {
        let width = self.content_width(0);
        self.out.push(Line::from(vec![
            Span::raw(" ".repeat(self.indent)),
            Span::styled("─".repeat(width), Style::default().fg(Color::DarkGray)),
        ]));
    }

    fn push_item(&mut self, item: ListItem) {
        let marker = item.marker();
        let gutter = item.level + marker.chars().count();
        let width = self.content_width(gutter);

        let first = vec![
            Span::raw(" ".repeat(self.indent + item.level)),
            Span::styled(
                marker,
                Style::default().fg(if item.task.is_some() {
                    Color::Yellow
                } else {
                    Color::Cyan
                }),
            ),
        ];
        let rest = vec![Span::raw(" ".repeat(self.indent + gutter))];

        let mut chunks = wrap_words(&inline(&item.content), width);
        if chunks.is_empty() {
            chunks.push(Vec::new());
        }

        for (position, chunk) in chunks.iter().enumerate() {
            let prefix = if position == 0 { &first } else { &rest };
            self.out.push(to_line(prefix, chunk));
        }
    }

    fn push_quote(&mut self, line: &str) {
        let text = line.trim_start().trim_start_matches('>').trim_start();
        let style = Style::default()
            .fg(Color::Gray)
            .add_modifier(Modifier::ITALIC);

        let mut styled = inline(text);
        for entry in styled.iter_mut() {
            entry.1 = entry.1.patch(style);
        }

        let width = self.content_width(2);
        let prefix = vec![
            Span::raw(" ".repeat(self.indent)),
            Span::styled("▏ ", Style::default().fg(Color::DarkGray)),
        ];

        let chunks = wrap_words(&styled, width);
        if chunks.is_empty() {
            self.out.push(to_line(&prefix, &Vec::new()));
            return;
        }
        for chunk in &chunks {
            self.out.push(to_line(&prefix, chunk));
        }
    }

    fn push_code(&mut self, line: &str) {
        // Tabs would be measured as zero width, so make them concrete.
        let expanded = line.replace('\t', "    ");
        let style = Style::default().fg(Color::LightBlue);
        let width = self.content_width(2);
        let prefix = vec![
            Span::raw(" ".repeat(self.indent)),
            Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        ];

        let styled: Styled = expanded.chars().map(|ch| (ch, style)).collect();
        for chunk in hard_split(&styled, width) {
            self.out.push(to_line(&prefix, &chunk));
        }
    }

    fn push_table(&mut self, header: Vec<String>, rows: Vec<Vec<String>>) {
        let columns = rows
            .iter()
            .map(Vec::len)
            .chain(std::iter::once(header.len()))
            .max()
            .unwrap_or(0);
        if columns == 0 {
            return;
        }

        // Lay out the cells, then shrink the widest columns until the row fits.
        let mut cells: Vec<Vec<Styled>> = Vec::with_capacity(rows.len() + 1);
        cells.push(header.iter().map(|cell| inline(cell)).collect());
        cells.extend(
            rows.iter()
                .map(|row| row.iter().map(|cell| inline(cell)).collect()),
        );

        let mut widths: Vec<usize> = vec![0; columns];
        for row in &cells {
            for (index, cell) in row.iter().enumerate() {
                widths[index] = widths[index].max(measure(cell));
            }
        }

        // A rendered row is `│` plus one ` cell ` per column, so the border characters
        // cost three columns per cell on top of the leading bar.
        let budget = self
            .width
            .saturating_sub(self.indent + 1 + 3 * columns)
            .max(columns * 3);
        shrink_to_fit(&mut widths, budget);

        let border = Style::default().fg(Color::DarkGray);

        for (position, row) in cells.iter().enumerate() {
            let mut spans = vec![
                Span::raw(" ".repeat(self.indent)),
                Span::styled("│", border),
            ];
            for (index, width) in widths.iter().enumerate() {
                let empty = Vec::new();
                let cell = row.get(index).unwrap_or(&empty);
                let mut content = clip(cell, *width);
                let used = measure(&content);
                let style = if position == 0 {
                    Style::default().add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                for entry in content.iter_mut() {
                    entry.1 = entry.1.patch(style);
                }

                spans.push(Span::styled(" ", border));
                spans.extend(to_line(&[], &content).spans);
                if used < *width {
                    spans.push(Span::raw(" ".repeat(width - used)));
                }
                spans.push(Span::styled(" ", border));
                spans.push(Span::styled("│", border));
            }
            self.out.push(Line::from(spans));

            if position == 0 {
                let mut divider = vec![
                    Span::raw(" ".repeat(self.indent)),
                    Span::styled("├", border),
                ];
                for width in &widths {
                    divider.push(Span::styled("─".repeat(width + 2), border));
                    divider.push(Span::styled("┼", border));
                }
                divider.pop();
                divider.push(Span::styled("┤", border));
                self.out.push(Line::from(divider));
            }
        }
    }
}

// -------------------------------------------------------------------- lists

struct ListItem {
    /// Spaces of extra indentation from nested lists.
    level: usize,
    /// Either a bullet, or the original number.
    bullet: String,
    content: String,
    /// `Some(done)` for `- [ ]` / `- [x]` items.
    task: Option<bool>,
}

impl ListItem {
    fn marker(&self) -> String {
        match self.task {
            Some(done) => format!("[{}] ", if done { 'x' } else { ' ' }),
            None => self.bullet.clone(),
        }
    }
}

fn list_item(line: &str) -> Option<ListItem> {
    let indent = line.len() - line.trim_start().len();
    let trimmed = line.trim_start();

    let (bullet, rest) = if let Some(rest) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))
    {
        ("• ".to_owned(), rest)
    } else {
        let digits: String = trimmed.chars().take_while(char::is_ascii_digit).collect();
        if digits.is_empty() {
            return None;
        }
        let after = &trimmed[digits.len()..];
        let rest = after
            .strip_prefix('.')
            .or_else(|| after.strip_prefix(')'))?
            .strip_prefix(' ')?;
        (format!("{digits}. "), rest)
    };

    let (content, task) = if let Some(rest) = rest.strip_prefix("[ ] ") {
        (rest, Some(false))
    } else if let Some(rest) = rest
        .strip_prefix("[x] ")
        .or_else(|| rest.strip_prefix("[X] "))
    {
        (rest, Some(true))
    } else {
        (rest, None)
    };

    Some(ListItem {
        // Half the indent, so both two and four space nesting look right.
        level: (indent / 2).min(6),
        bullet,
        content: content.to_owned(),
        task,
    })
}

// ------------------------------------------------------------------- tables

/// If a table starts at `index`, return its header, body rows and line count.
fn table_at(source: &[&str], index: usize) -> Option<(Vec<String>, Vec<Vec<String>>, usize)> {
    let header = source.get(index)?;
    let separator = source.get(index + 1)?;

    if !header.contains('|') || !is_table_separator(separator) {
        return None;
    }

    let header = parse_cells(header);
    let mut rows = Vec::new();
    let mut consumed = 2;

    for line in source.iter().skip(index + 2) {
        if !line.contains('|') || line.trim().is_empty() {
            break;
        }
        rows.push(parse_cells(line));
        consumed += 1;
    }

    Some((header, rows, consumed))
}

fn is_table_separator(line: &str) -> bool {
    if !line.contains('-') {
        return false;
    }
    let cells = parse_cells(line);
    !cells.is_empty()
        && cells.iter().all(|cell| {
            let trimmed = cell.trim_matches(':');
            !trimmed.is_empty() && trimmed.chars().all(|ch| ch == '-')
        })
}

fn parse_cells(line: &str) -> Vec<String> {
    let trimmed = line.trim().trim_start_matches('|').trim_end_matches('|');
    trimmed
        .split('|')
        .map(|cell| cell.trim().to_owned())
        .collect()
}

fn shrink_to_fit(widths: &mut [usize], budget: usize) {
    while widths.iter().sum::<usize>() > budget {
        let Some(widest) = widths
            .iter()
            .enumerate()
            .filter(|(_, width)| **width > 3)
            .max_by_key(|(_, width)| **width)
            .map(|(index, _)| index)
        else {
            return;
        };
        widths[widest] -= 1;
    }
}

// ------------------------------------------------------------------- inline

/// Parse inline markdown, stopping at anything that cannot be closed.
fn inline(text: &str) -> Styled {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Styled = Vec::new();
    let mut index = 0;

    while index < chars.len() {
        // Backslash escapes keep their character and drop the backslash.
        if chars[index] == '\\' && index + 1 < chars.len() && is_marker(chars[index + 1]) {
            out.push((chars[index + 1], Style::default()));
            index += 2;
            continue;
        }

        // Inline code.
        if let Some((body, next)) = span_between(&chars, index, '`', '`') {
            for ch in body {
                out.push((ch, Style::default().fg(Color::LightBlue)));
            }
            index = next;
            continue;
        }

        let mut matched = false;

        // Triple emphasis, then bold, then strike.
        for (open, close, style) in [
            (
                "***",
                "***",
                Style::default().add_modifier(Modifier::BOLD | Modifier::ITALIC),
            ),
            ("**", "**", Style::default().add_modifier(Modifier::BOLD)),
            ("__", "__", Style::default().add_modifier(Modifier::BOLD)),
            (
                "~~",
                "~~",
                Style::default().add_modifier(Modifier::CROSSED_OUT),
            ),
        ] {
            if let Some((body, next)) = span_between_str(&chars, index, open, close) {
                for ch in body {
                    out.push((ch, style));
                }
                index = next;
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }

        // Single `*` emphasis. A doubled marker belongs to the cases above, `_` also
        // needs a word boundary so snake_case survives, and the emphasis must not start
        // or end with whitespace (`3 * 4 * 5` is arithmetic, not emphasis).
        for (marker, style) in [
            ('*', Style::default().add_modifier(Modifier::ITALIC)),
            ('_', Style::default().add_modifier(Modifier::ITALIC)),
        ] {
            if chars[index] != marker || chars.get(index + 1) == Some(&marker) {
                continue;
            }
            if marker == '_' && index > 0 && chars[index - 1].is_alphanumeric() {
                continue;
            }
            let Some((body, next)) = span_between(&chars, index, marker, marker) else {
                continue;
            };
            let flanking = body.first().is_some_and(|ch| !ch.is_whitespace())
                && body.last().is_some_and(|ch| !ch.is_whitespace());
            if !flanking {
                continue;
            }
            for ch in body {
                out.push((ch, style));
            }
            index = next;
            matched = true;
            break;
        }
        if matched {
            continue;
        }

        // Images render like links, using the alt text.
        if chars[index] == '!'
            && chars.get(index + 1) == Some(&'[')
            && let Some((label, target, next)) = link_at(&chars, index + 1)
        {
            push_link(&mut out, &label, &target);
            index = next;
            continue;
        }

        if chars[index] == '['
            && let Some((label, target, next)) = link_at(&chars, index)
        {
            push_link(&mut out, &label, &target);
            index = next;
            continue;
        }

        out.push((chars[index], Style::default()));
        index += 1;
    }

    out
}

fn push_link(out: &mut Styled, label: &[char], target: &str) {
    let style = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::UNDERLINED);
    for ch in label {
        out.push((*ch, style));
    }
    if !target.is_empty() {
        for ch in format!(" ({target})").chars() {
            out.push((ch, Style::default().fg(Color::DarkGray)));
        }
    }
}

fn is_marker(ch: char) -> bool {
    matches!(ch, '*' | '_' | '`' | '~' | '[' | ']' | '(' | ')' | '\\')
}

/// Find `[label](target)` at `index`, returning the label, the target and the next index.
fn link_at(chars: &[char], index: usize) -> Option<(Vec<char>, String, usize)> {
    if chars.get(index) != Some(&'[') {
        return None;
    }
    let close = (index + 1..chars.len()).find(|position| chars[*position] == ']')?;
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let end = (close + 2..chars.len()).find(|position| chars[*position] == ')')?;
    let label: Vec<char> = chars[index + 1..close].to_vec();
    let target: String = chars[close + 2..end].iter().collect();
    Some((label, target, end + 1))
}

/// Content between a single-character delimiter pair.
fn span_between(
    chars: &[char],
    index: usize,
    open: char,
    close: char,
) -> Option<(Vec<char>, usize)> {
    if chars.get(index) != Some(&open) {
        return None;
    }
    let end = (index + 1..chars.len()).find(|position| chars[*position] == close)?;
    Some((chars[index + 1..end].to_vec(), end + 1))
}

/// Content between a multi-character delimiter pair, which must not be empty.
fn span_between_str(
    chars: &[char],
    index: usize,
    open: &str,
    close: &str,
) -> Option<(Vec<char>, usize)> {
    let open: Vec<char> = open.chars().collect();
    let close: Vec<char> = close.chars().collect();
    if !chars[index..].starts_with(&open) {
        return None;
    }
    let from = index + open.len();
    let end = (from..chars.len().saturating_sub(close.len()) + 1)
        .find(|position| chars[*position..].starts_with(&close))?;
    if end == from {
        // `****` is not emphasis.
        return None;
    }
    Some((chars[from..end].to_vec(), end + close.len()))
}

// ------------------------------------------------------------------ wrapping

fn char_width(ch: char) -> usize {
    UnicodeWidthChar::width(ch).unwrap_or(0)
}

fn measure(text: &[(char, Style)]) -> usize {
    text.iter().map(|(ch, _)| char_width(*ch)).sum()
}

/// Greedy word wrapping, collapsing runs of whitespace as markdown does.
fn wrap_words(text: &[(char, Style)], width: usize) -> Vec<Styled> {
    let mut lines: Vec<Styled> = Vec::new();
    let mut current: Styled = Vec::new();
    let mut current_width = 0usize;
    let mut space: Option<Style> = None;
    let mut index = 0;

    while index < text.len() {
        if text[index].0.is_whitespace() {
            let style = text[index].1;
            while index < text.len() && text[index].0.is_whitespace() {
                index += 1;
            }
            if !current.is_empty() {
                space = Some(style);
            }
            continue;
        }

        let start = index;
        while index < text.len() && !text[index].0.is_whitespace() {
            index += 1;
        }
        let word = &text[start..index];
        let word_width = measure(word);
        let separator = usize::from(space.is_some());
        space = None;

        if !current.is_empty() && current_width + separator + word_width > width {
            lines.push(std::mem::take(&mut current));
            current_width = 0;
        } else if !current.is_empty() {
            current.push((' ', text[start].1));
            current_width += 1;
        }

        if current_width + word_width <= width {
            current.extend_from_slice(word);
            current_width += word_width;
        } else {
            let mut pieces = hard_split(word, width);
            let tail = pieces.pop().unwrap_or_default();
            lines.append(&mut pieces);
            current_width = measure(&tail);
            current = tail;
        }
    }

    if !current.is_empty() || lines.is_empty() {
        lines.push(current);
    }
    lines
}

/// Split on exactly `width` display cells, keeping every character.
fn hard_split(text: &[(char, Style)], width: usize) -> Vec<Styled> {
    let mut out: Vec<Styled> = Vec::new();
    let mut current: Styled = Vec::new();
    let mut current_width = 0;

    for &(ch, style) in text {
        let cell = char_width(ch);
        if current_width + cell > width && !current.is_empty() {
            out.push(std::mem::take(&mut current));
            current_width = 0;
        }
        current.push((ch, style));
        current_width += cell;
    }

    if !current.is_empty() {
        out.push(current);
    }
    if out.is_empty() {
        out.push(Vec::new());
    }
    out
}

/// Merge styled characters back into as few spans as possible.
fn to_line(prefix: &[Span<'static>], text: &[(char, Style)]) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = prefix.to_vec();
    let mut buffer = String::new();
    let mut style: Option<Style> = None;

    for &(ch, current) in text {
        if style != Some(current) {
            if let Some(previous) = style {
                spans.push(Span::styled(std::mem::take(&mut buffer), previous));
            }
            style = Some(current);
        }
        buffer.push(ch);
    }
    if let Some(previous) = style {
        spans.push(Span::styled(buffer, previous));
    }

    Line::from(spans)
}

/// Cut text down to `width` display cells, marking the cut with an ellipsis.
fn clip(text: &[(char, Style)], width: usize) -> Styled {
    if measure(text) <= width {
        return text.to_vec();
    }
    let mut out = Vec::new();
    let mut used = 0;
    for &(ch, style) in text {
        let cell = char_width(ch);
        if used + cell + 1 > width {
            break;
        }
        out.push((ch, style));
        used += cell;
    }
    out.push(('…', Style::default().fg(Color::DarkGray)));
    out
}

// ------------------------------------------------------------------ markers

fn fence_language(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    for marker in ["```", "~~~"] {
        if let Some(rest) = trimmed.strip_prefix(marker) {
            return Some(rest.trim().to_owned());
        }
    }
    None
}

fn heading(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim_start();
    let level = trimmed.chars().take_while(|ch| *ch == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &trimmed[level..];
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    let title = rest.trim().trim_end_matches('#').trim_end();
    Some((level, title.to_owned()))
}

fn is_rule(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.len() < 3 {
        return false;
    }
    let marker = trimmed.chars().next().unwrap_or(' ');
    matches!(marker, '-' | '*' | '_')
        && trimmed.chars().all(|ch| ch == marker || ch == ' ')
        && trimmed.chars().filter(|ch| *ch == marker).count() >= 3
}

fn heading_style(level: usize) -> Style {
    match level {
        1 => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
        2 => Style::default()
            .fg(Color::LightBlue)
            .add_modifier(Modifier::BOLD),
        _ => Style::default()
            .fg(Color::Gray)
            .add_modifier(Modifier::BOLD),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plain text of a line, with markers removed.
    fn text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>()
    }

    fn all(lines: &[Line<'_>]) -> String {
        lines.iter().map(text).collect::<Vec<_>>().join("\n")
    }

    /// Every non-blank span with its style, trimmed: which side of a span a space lands
    /// on is an implementation detail, so the assertions ignore it.
    fn styles(lines: &[Line<'_>]) -> Vec<(String, Style)> {
        lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .filter(|span| !span.content.trim().is_empty())
            .map(|span| (span.content.trim().to_owned(), span.style))
            .collect()
    }

    fn has(lines: &[Line<'_>], needle: &str, check: impl Fn(&Style) -> bool) -> bool {
        styles(lines)
            .iter()
            .any(|(text, style)| text == needle && check(style))
    }

    /// Convert a rendered line back to styled characters, for width assertions.
    fn to_styled(line: &Line<'_>) -> Styled {
        line.spans
            .iter()
            .flat_map(|span| {
                let style = span.style;
                span.content.chars().map(move |ch| (ch, style))
            })
            .collect()
    }

    /// Nothing may be rendered wider than the width it was given.
    fn assert_fits(lines: &[Line<'_>], width: usize) {
        for line in lines {
            assert!(
                measure(&to_styled(line)) <= width,
                "{:?} is wider than {width}",
                text(line)
            );
        }
    }

    #[test]
    fn strips_emphasis_markers_and_styles_the_text() {
        let out = lines("some **bold** and *italic* and `code` text", 60, 0);
        assert_eq!(text(&out[0]), "some bold and italic and code text");

        assert!(has(&out, "bold", |style| style
            .add_modifier
            .contains(Modifier::BOLD)));
        assert!(has(&out, "italic", |style| style
            .add_modifier
            .contains(Modifier::ITALIC)));
        assert!(has(&out, "code", |style| style.fg == Some(Color::LightBlue)));
    }

    #[test]
    fn leaves_unclosed_markers_alone_while_streaming() {
        let out = lines("an **unfinished sentence", 40, 0);
        assert_eq!(text(&out[0]), "an **unfinished sentence");

        let out = lines("code `unclosed", 40, 0);
        assert_eq!(text(&out[0]), "code `unclosed");

        let out = lines("3 * 4 * 5", 40, 0);
        assert_eq!(text(&out[0]), "3 * 4 * 5");
    }

    #[test]
    fn renders_headings_without_the_hashes() {
        let out = lines("## A title\n\ntext", 40, 0);
        assert_eq!(text(&out[0]), "A title");
        assert!(
            out[0]
                .spans
                .iter()
                .any(|span| span.style.add_modifier.contains(Modifier::BOLD))
        );
        assert_eq!(text(&out[2]), "text");
    }

    #[test]
    fn keeps_code_blocks_verbatim() {
        let source = "before\n```rust\nlet x = *ptr;  // keep\n```\nafter";
        let out = lines(source, 40, 0);
        let rendered = all(&out);
        assert!(rendered.contains("│ let x = *ptr;  // keep"), "{rendered}");
        assert!(
            !rendered.contains("let x = ptr"),
            "the asterisk must survive: {rendered}"
        );
        assert!(rendered.contains("after"));
    }

    #[test]
    fn wraps_code_that_is_wider_than_the_view() {
        let out = lines("```\n0123456789abcdefghij\n```", 14, 0);
        // The gutter takes two columns, leaving twelve characters per line.
        let body: Vec<String> = out
            .iter()
            .map(text)
            .filter(|line| line.starts_with('│'))
            .collect();
        assert_eq!(body.len(), 2, "{body:?}");
        assert!(body[0].ends_with("0123456789ab"), "{body:?}");
        assert_fits(&out, 14);
    }

    #[test]
    fn an_unterminated_fence_still_renders_as_code() {
        let out = lines("```\nstill streaming", 40, 0);
        assert!(
            text(&out[0]).starts_with("│ still streaming"),
            "{:?}",
            text(&out[0])
        );
    }

    #[test]
    fn bullets_and_numbers_get_a_hanging_indent() {
        let out = lines(
            "- first item is long enough to wrap around\n- second",
            24,
            2,
        );
        assert!(
            text(&out[0]).starts_with("  • first item"),
            "{:?}",
            text(&out[0])
        );
        let last = out
            .iter()
            .position(|line| text(line).contains("second"))
            .unwrap();
        assert!(text(&out[last]).starts_with("  • second"));
        assert_fits(&out, 24);

        let out = lines("1. ordered\n", 30, 2);
        assert!(
            text(&out[0]).starts_with("  1. ordered"),
            "{:?}",
            text(&out[0])
        );
    }

    #[test]
    fn wrapped_list_continuations_line_up_under_the_text() {
        let out = lines("- alpha beta gamma delta epsilon", 20, 0);
        assert!(text(&out[0]).starts_with("• alpha"));
        assert!(
            text(&out[1]).starts_with("  "),
            "the continuation should be indented: {:?}",
            text(&out[1])
        );
        assert!(!text(&out[1]).trim_start().starts_with('•'));
        assert_fits(&out, 20);
    }

    #[test]
    fn task_items_show_their_state() {
        let out = lines("- [x] done\n- [ ] todo", 40, 0);
        assert!(text(&out[0]).starts_with("[x] done"), "{:?}", text(&out[0]));
        assert!(text(&out[1]).starts_with("[ ] todo"), "{:?}", text(&out[1]));
    }

    #[test]
    fn quotes_and_rules_are_marked() {
        let out = lines("> quoted words\n\n---", 40, 0);
        assert_eq!(text(&out[0]), "▏ quoted words");
        assert!(out[1].spans.is_empty(), "{:?}", text(&out[1]));
        assert!(text(&out[2]).contains('─'), "{:?}", text(&out[2]));
        assert_fits(&out, 40);
    }

    #[test]
    fn links_show_the_label_and_the_target() {
        let out = lines("see [the docs](https://example.com) now", 60, 0);
        assert_eq!(text(&out[0]), "see the docs (https://example.com) now");
        assert!(has(&out, "the docs", |style| style
            .add_modifier
            .contains(Modifier::UNDERLINED)));
        assert!(has(&out, "(https://example.com)", |style| style.fg
            == Some(Color::DarkGray)));
    }

    #[test]
    fn snake_case_is_not_italicised() {
        let out = lines("call into_value first", 40, 0);
        assert_eq!(text(&out[0]), "call into_value first");
        assert!(
            !out[0]
                .spans
                .iter()
                .any(|span| span.style.add_modifier.contains(Modifier::ITALIC))
        );
    }

    #[test]
    fn tables_align_their_columns() {
        let source = "| name | size |\n| --- | ---: |\n| main.rs | 12kb |\n| lib.rs | 3kb |";
        let out = lines(source, 40, 0);
        let rendered: Vec<String> = out.iter().map(text).collect();

        assert!(rendered[0].starts_with("│ name "), "{rendered:?}");
        assert!(rendered[0].contains("size"), "{rendered:?}");
        assert!(rendered[1].contains('├'), "{rendered:?}");
        assert!(rendered[2].contains("main.rs"), "{rendered:?}");
        assert!(rendered[3].contains("lib.rs"), "{rendered:?}");
        let widths: Vec<usize> = rendered.iter().map(|line| line.chars().count()).collect();
        assert!(
            widths.windows(2).all(|pair| pair[0] == pair[1]),
            "the columns should align: {rendered:?}"
        );
        assert_fits(&out, 40);
    }

    #[test]
    fn a_table_wider_than_the_view_is_clipped_not_broken() {
        let source =
            "| column | description |\n| --- | --- |\n| a | a very long description here |";
        let out = lines(source, 30, 0);
        assert_fits(&out, 30);
        assert!(all(&out).contains('…'), "{}", all(&out));
    }

    #[test]
    fn a_header_without_a_separator_is_just_text() {
        let out = lines("| not really | a table |", 40, 0);
        assert!(
            text(&out[0]).contains("| not really | a table |"),
            "{:?}",
            text(&out[0])
        );
    }

    #[test]
    fn wide_characters_are_measured_by_display_width() {
        let source = "中文中文中文中文中文中文";
        let out = lines(source, 14, 0);
        assert!(out.len() > 1, "the text should wrap");
        assert_eq!(out.iter().map(text).collect::<String>(), source);
        assert_fits(&out, 14);
    }

    #[test]
    fn long_unbreakable_words_are_split() {
        let source = "aaaa bbbbbbbbbbbbbbbbbbbbbbbb cccc";
        let out = lines(source, 12, 0);
        assert!(
            out.len() >= 3,
            "{:?}",
            out.iter().map(text).collect::<Vec<_>>()
        );
        // A hard split is indistinguishable from a word boundary once the lines are
        // rejoined, so compare the characters with all whitespace removed.
        let strip = |text: &str| {
            text.chars()
                .filter(|ch| !ch.is_whitespace())
                .collect::<String>()
        };
        assert_eq!(strip(&all(&out)), strip(source));
        assert_fits(&out, 12);
    }

    #[test]
    fn plain_text_wraps_and_keeps_its_style() {
        let style = Style::default().fg(Color::Cyan);
        let out = plain_lines("one two three four five six", 12, 2, style);
        assert!(out.len() > 1);
        for line in &out {
            assert_eq!(line.spans[0].content, "  ");
            assert_eq!(line.spans[0].style.fg, Some(Color::Cyan));
        }
        assert_fits(&out, 12);
    }
}
