//! The interactive `chat` experience.

pub mod app;
pub mod input;
pub mod markdown;
pub mod tools;
pub mod ui;

pub use app::{ChatApp, ChatSetup};

use anyhow::Result;
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
};
use crossterm::execute;

use crate::session::Session;

/// Run the chat TUI on the current terminal and return the resulting session.
///
/// The caller is responsible for having taken over the terminal (`enter_foreground`)
/// and for not using stdin/stdout for the plugin protocol.
pub fn run(setup: ChatSetup) -> Result<Session> {
    let mut terminal = ratatui::init();
    // Bracketed paste keeps a multi-line paste in one event; mouse capture is what makes the
    // wheel report scroll events at all.
    let _ = execute!(std::io::stdout(), EnableBracketedPaste, EnableMouseCapture);

    let result = ChatApp::new(setup).and_then(|app| app.run(&mut terminal));

    let _ = execute!(
        std::io::stdout(),
        DisableMouseCapture,
        DisableBracketedPaste
    );
    ratatui::restore();

    result
}
