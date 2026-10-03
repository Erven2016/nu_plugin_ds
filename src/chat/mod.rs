//! The interactive `chat` experience.

pub mod app;
pub mod input;
pub mod markdown;
pub mod tools;
pub mod ui;

pub use app::{ChatApp, ChatSetup};

use anyhow::Result;
use crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use crossterm::execute;

use crate::session::Session;

/// Run the chat TUI on the current terminal and return the resulting session.
///
/// The caller is responsible for having taken over the terminal (`enter_foreground`)
/// and for not using stdin/stdout for the plugin protocol.
pub fn run(setup: ChatSetup) -> Result<Session> {
    let mut terminal = ratatui::init();
    let _ = execute!(std::io::stdout(), EnableBracketedPaste);

    let result = ChatApp::new(setup).and_then(|app| app.run(&mut terminal));

    let _ = execute!(std::io::stdout(), DisableBracketedPaste);
    ratatui::restore();

    result
}
