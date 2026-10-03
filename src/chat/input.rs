//! The multi-line prompt editor used by the chat TUI.
//!
//! The editor works on a single `String` with a byte cursor that is always kept on a
//! character boundary. Only the helpers needed by the TUI are implemented, but they
//! cover the usual readline motions so the input feels familiar.

use unicode_width::UnicodeWidthStr;

#[derive(Default)]
pub struct InputBuffer {
    text: String,
    cursor: usize,
    history: Vec<String>,
    /// `Some(n)` while browsing history, where `n` counts back from the newest entry.
    history_offset: Option<usize>,
    /// The text that was being edited before history browsing started.
    history_stash: Option<String>,
}

impl InputBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.history_offset = None;
        self.history_stash = None;
    }

    /// The lines of the prompt, used for rendering.
    pub fn lines(&self) -> impl Iterator<Item = &str> {
        self.text.split('\n')
    }

    pub fn line_count(&self) -> usize {
        self.text.matches('\n').count() + 1
    }

    pub fn history(&self) -> &[String] {
        &self.history
    }

    /// Take the current text, recording it in the history.
    pub fn take(&mut self) -> String {
        let submitted = std::mem::take(&mut self.text);
        self.cursor = 0;
        self.history_offset = None;
        self.history_stash = None;

        let trimmed = submitted.trim_end().to_owned();
        if !trimmed.trim().is_empty() && self.history.last() != Some(&trimmed) {
            self.history.push(trimmed);
        }
        submitted
    }

    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.cursor = self.text.len();
        self.history_offset = None;
        self.history_stash = None;
    }

    pub fn insert_str(&mut self, value: &str) {
        let value = value.replace("\r\n", "\n").replace('\r', "\n");
        self.text.insert_str(self.cursor, &value);
        self.cursor += value.len();
        self.history_offset = None;
    }

    pub fn insert_char(&mut self, ch: char) {
        self.text.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
        self.history_offset = None;
    }

    pub fn backspace(&mut self) {
        if let Some(previous) = self.prev_boundary() {
            self.text.replace_range(previous..self.cursor, "");
            self.cursor = previous;
        }
        self.history_offset = None;
    }

    pub fn delete(&mut self) {
        if let Some(next) = self.next_boundary() {
            self.text.replace_range(self.cursor..next, "");
        }
        self.history_offset = None;
    }

    /// Delete from the cursor back to the start of the word.
    pub fn delete_word(&mut self) {
        let before = &self.text[..self.cursor];
        let without_whitespace = before.trim_end();
        let boundary = if without_whitespace.len() == before.len() {
            // The cursor sits directly after a word: cut the word itself.
            word_start(without_whitespace)
        } else {
            // There is whitespace between the cursor and the word: cut that instead.
            without_whitespace.len()
        };
        self.text.replace_range(boundary..self.cursor, "");
        self.cursor = boundary;
        self.history_offset = None;
    }

    /// Delete from the cursor to the end of the line.
    pub fn kill_to_end(&mut self) {
        let end = self.text[self.cursor..]
            .find('\n')
            .map_or(self.text.len(), |offset| self.cursor + offset);
        self.text.replace_range(self.cursor..end, "");
        self.history_offset = None;
    }

    /// Delete the whole current line (readline's `Ctrl+U` behaviour).
    pub fn kill_line(&mut self) {
        let start = self.text[..self.cursor]
            .rfind('\n')
            .map_or(0, |index| index + 1);
        let end = self.text[self.cursor..]
            .find('\n')
            .map_or(self.text.len(), |offset| self.cursor + offset);
        self.text.replace_range(start..end, "");
        self.cursor = start;
        self.history_offset = None;
    }

    pub fn move_left(&mut self) {
        if let Some(previous) = self.prev_boundary() {
            self.cursor = previous;
        }
    }

    pub fn move_right(&mut self) {
        if let Some(next) = self.next_boundary() {
            self.cursor = next;
        }
    }

    pub fn move_home(&mut self) {
        self.cursor = self.text[..self.cursor]
            .rfind('\n')
            .map_or(0, |index| index + 1);
    }

    pub fn move_end(&mut self) {
        self.cursor = self.text[self.cursor..]
            .find('\n')
            .map_or(self.text.len(), |offset| self.cursor + offset);
    }

    pub fn move_document_start(&mut self) {
        self.cursor = 0;
    }

    pub fn move_document_end(&mut self) {
        self.cursor = self.text.len();
    }

    /// Move up one logical line. Returns `false` when the cursor is already on the
    /// first line, which lets the caller fall back to history navigation.
    pub fn move_up(&mut self) -> bool {
        let (row, column) = self.row_col();
        if row == 0 {
            return false;
        }
        self.cursor = self.offset_for(row - 1, column);
        true
    }

    /// Move down one logical line, reporting whether the move was possible.
    pub fn move_down(&mut self) -> bool {
        let (row, column) = self.row_col();
        if row + 1 >= self.line_count() {
            return false;
        }
        self.cursor = self.offset_for(row + 1, column);
        true
    }

    /// Browse to an older entry. Returns `true` when the history was consumed.
    pub fn history_prev(&mut self) -> bool {
        if self.history.is_empty() {
            return false;
        }
        let oldest = self.history.len() - 1;
        let next = match self.history_offset {
            None => {
                self.history_stash = Some(std::mem::take(&mut self.text));
                0
            }
            Some(offset) => (offset + 1).min(oldest),
        };
        self.history_offset = Some(next);
        let entry = self.history[self.history.len() - 1 - next].clone();
        self.set_entry(entry);
        true
    }

    /// Browse to a newer entry, restoring the stashed draft at the end.
    pub fn history_next(&mut self) -> bool {
        let Some(offset) = self.history_offset else {
            return false;
        };
        if offset == 0 {
            self.history_offset = None;
            let stash = self.history_stash.take().unwrap_or_default();
            self.set_entry(stash);
        } else {
            self.history_offset = Some(offset - 1);
            let entry = self.history[self.history.len() - offset].clone();
            self.set_entry(entry);
        }
        true
    }

    fn set_entry(&mut self, text: String) {
        self.text = text;
        self.cursor = self.text.len();
    }

    /// Cursor position as `(row, column)` in *characters*, used for vertical motion.
    fn row_col(&self) -> (usize, usize) {
        let before = &self.text[..self.cursor];
        (
            before.matches('\n').count(),
            before.chars().rev().take_while(|ch| *ch != '\n').count(),
        )
    }

    /// Cursor position as `(row, display column)`, used to place the terminal cursor.
    pub fn display_position(&self) -> (usize, usize) {
        let before = &self.text[..self.cursor];
        let row = before.matches('\n').count();
        let column = before.rsplit('\n').next().unwrap_or("").width();
        (row, column)
    }

    /// Byte offset of `column` characters into `row`.
    fn offset_for(&self, row: usize, column: usize) -> usize {
        let mut offset = 0;
        for (index, line) in self.text.split('\n').enumerate() {
            if index == row {
                let advance: usize = line.chars().take(column).map(|ch| ch.len_utf8()).sum();
                return offset + advance;
            }
            offset += line.len() + 1;
        }
        self.text.len()
    }

    fn prev_boundary(&self) -> Option<usize> {
        self.text[..self.cursor]
            .char_indices()
            .next_back()
            .map(|(index, _)| index)
    }

    fn next_boundary(&self) -> Option<usize> {
        self.text[self.cursor..]
            .chars()
            .next()
            .map(|ch| self.cursor + ch.len_utf8())
    }
}

/// Byte offset just past the last whitespace character in `text`.
fn word_start(text: &str) -> usize {
    text.char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map_or(0, |(index, ch)| index + ch.len_utf8())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typing_and_deleting() {
        let mut input = InputBuffer::new();
        input.insert_str("hello");
        assert_eq!(input.text(), "hello");
        input.backspace();
        assert_eq!(input.text(), "hell");
        input.move_left();
        input.insert_char('X');
        assert_eq!(input.text(), "helXl");
        input.move_left();
        input.move_left();
        input.delete();
        assert_eq!(input.text(), "heXl");
    }

    #[test]
    fn vertical_motion_preserves_the_column() {
        let mut input = InputBuffer::new();
        input.set_text("long line\nhi\nanother");
        input.move_document_start();
        input.move_down();
        assert_eq!(input.cursor(), "long line\n".len());
        input.move_down();
        assert_eq!(input.cursor(), "long line\nhi\n".len());
        assert!(!input.move_down());
    }

    #[test]
    fn vertical_motion_stops_at_the_first_line() {
        let mut input = InputBuffer::new();
        input.set_text("first\nsecond");
        input.move_document_start();
        assert!(!input.move_up());
    }

    #[test]
    fn history_round_trip_restores_the_draft() {
        let mut input = InputBuffer::new();
        input.insert_str("first");
        input.take();
        input.insert_str("second");
        input.take();

        input.insert_str("draft");
        assert!(input.history_prev());
        assert_eq!(input.text(), "second");
        assert!(input.history_prev());
        assert_eq!(input.text(), "first");
        assert!(input.history_next());
        assert_eq!(input.text(), "second");
        assert!(input.history_next());
        assert_eq!(input.text(), "draft");
        assert!(!input.history_next());
    }

    #[test]
    fn kill_line_only_removes_the_current_line() {
        let mut input = InputBuffer::new();
        input.set_text("one\ntwo\nthree");
        input.move_document_start();
        input.kill_to_end();
        assert_eq!(input.text(), "\ntwo\nthree");
    }

    #[test]
    fn multibyte_motion_keeps_char_boundaries() {
        let mut input = InputBuffer::new();
        input.insert_str("你好");
        assert_eq!(input.cursor(), 6);
        input.move_left();
        assert_eq!(input.cursor(), 3);
        input.backspace();
        assert_eq!(input.text(), "好");
    }
}
