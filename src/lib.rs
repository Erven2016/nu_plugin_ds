//! A [nushell](https://www.nushell.sh) plugin for talking to the DeepSeek API.
//!
//! The plugin is registered under the `ds` namespace by nushell and provides:
//!
//! * [`chat`](commands::chat::ChatCommand) — an interactive chat client with a status
//!   bar, live token accounting, model/thinking/session switching and automatic
//!   context compaction.
//! * [`cc`](commands::cc::CcCommand) — turns a natural language request into a
//!   nushell command and runs it after the user confirms it.
//! * `ds models`, `ds sessions`, `ds config` and `ds version` — small helpers for
//!   inspecting the pieces `chat` uses.

pub mod api;
pub mod chat;
pub mod commands;
pub mod config;
pub mod credential;
pub mod error;
pub mod session;
pub mod token;

use nu_plugin::{Plugin, PluginCommand};

/// The plugin entry point.
pub struct DsPlugin;

impl Plugin for DsPlugin {
    fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").into()
    }

    fn commands(&self) -> Vec<Box<dyn PluginCommand<Plugin = Self>>> {
        vec![
            Box::new(commands::chat::ChatCommand),
            Box::new(commands::cc::CcCommand),
            Box::new(commands::models::ModelsCommand),
            Box::new(commands::sessions::SessionsCommand),
            Box::new(commands::info::ConfigCommand),
            Box::new(commands::api_key::ChangeApiKeyCommand),
            Box::new(commands::version::VersionCommand),
        ]
    }
}
