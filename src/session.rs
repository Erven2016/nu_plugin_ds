//! Conversation sessions and their on-disk store.
//!
//! A session is a single JSON document under `<config>/sessions/<id>.json`. Keeping one
//! file per session means switching history is a plain file read, and a corrupted
//! session only ever affects itself.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};

use crate::api::{ChatMessage, Role, Usage};
use crate::config::ThinkingEffort;

/// A conversation, as persisted between runs.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Session {
    pub id: String,
    pub name: String,
    pub created_at: DateTime<Local>,
    pub updated_at: DateTime<Local>,
    pub model: String,
    pub thinking: ThinkingEffort,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    /// Token usage accumulated over the whole session.
    #[serde(default)]
    pub usage: Usage,
    /// How many times the history has been compacted.
    #[serde(default)]
    pub compactions: u32,
}

impl Session {
    /// Start a new session. `system_prompt` is stored as the first message so that it is
    /// visible in the transcript and survives compaction.
    pub fn new(
        name: impl Into<String>,
        model: impl Into<String>,
        thinking: ThinkingEffort,
        system_prompt: Option<&str>,
    ) -> Self {
        let now = Local::now();
        let mut session = Session {
            id: new_id(),
            name: name.into(),
            created_at: now,
            updated_at: now,
            model: model.into(),
            thinking,
            messages: Vec::new(),
            usage: Usage::default(),
            compactions: 0,
        };
        if let Some(prompt) = system_prompt.filter(|prompt| !prompt.trim().is_empty()) {
            session.messages.push(ChatMessage::system(prompt));
        }
        session
    }

    /// The system prompt, if this session has one.
    pub fn system_prompt(&self) -> Option<&str> {
        match self.messages.first() {
            Some(message) if message.role == Role::System => Some(&message.content),
            _ => None,
        }
    }

    /// Append a message and refresh the modification time.
    pub fn push(&mut self, message: ChatMessage) {
        self.messages.push(message);
        self.touch();
    }

    pub fn touch(&mut self) {
        self.updated_at = Local::now();
    }

    /// Forget the transcript, keeping this session's current system prompt (if any).
    ///
    /// The id, name, model and accumulated usage are left alone, so clearing is a fresh
    /// start rather than a new session. The caller has to persist the result: an empty
    /// session is normally not written to disk, but a deliberate clear must be.
    pub fn clear_transcript(&mut self) {
        let system_prompt = self.system_prompt().map(str::to_owned);
        self.messages.clear();
        if let Some(prompt) = system_prompt {
            self.messages.push(ChatMessage::system(prompt));
        }
        self.touch();
    }

    /// A human readable label for pickers and listings.
    pub fn title(&self) -> String {
        if !self.name.trim().is_empty() {
            return self.name.clone();
        }
        self.auto_title()
    }

    fn auto_title(&self) -> String {
        self.messages
            .iter()
            .find(|message| message.role == Role::User)
            .map(|message| message.preview(48))
            .filter(|title| !title.is_empty())
            .unwrap_or_else(|| "(empty session)".to_owned())
    }

    /// The number of turns the user has taken.
    pub fn user_turns(&self) -> usize {
        self.messages
            .iter()
            .filter(|message| message.role == Role::User)
            .count()
    }

    /// Index of the last user message, used by "regenerate".
    pub fn last_user_index(&self) -> Option<usize> {
        self.messages
            .iter()
            .rposition(|message| message.role == Role::User)
    }

    /// Drop the trailing assistant answer so it can be regenerated.
    pub fn drop_last_answer(&mut self) -> bool {
        let Some(index) = self.last_user_index() else {
            return false;
        };
        let removed = self.messages.len() > index + 1;
        self.messages.truncate(index + 1);
        self.touch();
        removed
    }

    /// Assistant tool calls that no `tool` message has answered yet.
    ///
    /// Every call has to be answered before the conversation is sent again.
    pub fn unanswered_tool_calls(&self) -> Vec<crate::api::ToolCall> {
        let mut open: Vec<crate::api::ToolCall> = Vec::new();
        for message in &self.messages {
            match message.role {
                Role::Assistant => {
                    if let Some(calls) = &message.tool_calls {
                        open.extend(calls.iter().cloned());
                    }
                }
                Role::Tool => {
                    if let Some(id) = &message.tool_call_id {
                        open.retain(|call| &call.id != id);
                    }
                }
                _ => {}
            }
        }
        open
    }

    /// Everything that should be sent to the API for the next turn.
    pub fn wire_messages(&self) -> Vec<crate::api::WireMessage> {
        self.messages
            .iter()
            .filter(|message| !message.is_empty())
            .map(Into::into)
            .collect()
    }

    /// The messages a compaction should replace, and how many that is.
    ///
    /// The system prompt and the most recent `keep` messages are always left alone.
    pub fn compaction_plan(&self, keep: usize) -> Option<(Vec<ChatMessage>, usize)> {
        let keep = keep.max(2);
        let start = usize::from(self.system_prompt().is_some());
        if self.messages.len() <= start + keep + 1 {
            return None;
        }
        let split = self.messages.len() - keep;
        Some((self.messages[start..split].to_vec(), split - start))
    }

    /// Replace the planned range with a summary message, keeping the system prompt and the
    /// most recent `keep` messages. Returns how many messages were dropped.
    pub fn apply_summary(&mut self, summary: &str, keep: usize) -> usize {
        let keep = keep.max(2);
        let start = usize::from(self.system_prompt().is_some());
        let split = self.messages.len().saturating_sub(keep).max(start);

        let system_prompt = self.system_prompt().map(str::to_owned);
        let kept: Vec<ChatMessage> = self.messages.split_off(split);

        self.messages.clear();
        if let Some(prompt) = system_prompt {
            self.messages.push(ChatMessage::system(prompt));
        }
        self.messages.push(ChatMessage::system(format!(
            "The earlier part of this conversation was compressed to save context. \
             Summary of everything before the remaining messages:\n{summary}"
        )));
        self.messages.extend(kept);
        self.compactions += 1;
        self.touch();
        split - start
    }
}

/// A cheap description of a session, used by listings and pickers.
#[derive(Clone, Debug)]
pub struct SessionSummary {
    pub id: String,
    pub title: String,
    pub model: String,
    pub updated_at: DateTime<Local>,
    pub message_count: usize,
    pub turns: usize,
}

/// Reads and writes sessions inside a directory.
#[derive(Clone, Debug)]
pub struct SessionStore {
    dir: PathBuf,
}

impl SessionStore {
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)
            .with_context(|| format!("could not create the session directory {}", dir.display()))?;
        Ok(SessionStore { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path_for(&self, id: &str) -> Result<PathBuf> {
        if id.is_empty()
            || !id
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
        {
            bail!("`{id}` is not a valid session id");
        }
        Ok(self.dir.join(format!("{id}.json")))
    }

    /// All known sessions, most recently used first.
    pub fn list(&self) -> Result<Vec<SessionSummary>> {
        let mut summaries = Vec::new();

        for entry in fs::read_dir(&self.dir)
            .with_context(|| format!("could not read {}", self.dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }

            match read_session(&path) {
                Ok(session) => summaries.push(SessionSummary {
                    id: session.id.clone(),
                    title: session.title(),
                    model: session.model.clone(),
                    updated_at: session.updated_at,
                    message_count: session.messages.len(),
                    turns: session.user_turns(),
                }),
                // A single unreadable session should not hide the rest.
                Err(err) => eprintln!("nu_plugin_ds: skipping {path:?}: {err:#}"),
            }
        }

        summaries.sort_by_key(|summary| std::cmp::Reverse(summary.updated_at));
        Ok(summaries)
    }

    pub fn load(&self, id: &str) -> Result<Session> {
        let path = self.path_for(id)?;
        if !path.exists() {
            bail!("there is no session with id `{id}`");
        }
        read_session(&path)
    }

    pub fn save(&self, session: &Session) -> Result<()> {
        let path = self.path_for(&session.id)?;
        let raw = serde_json::to_string_pretty(session)?;
        // Write to a sibling file first so an interrupted write cannot lose the session.
        let temporary = path.with_extension("json.tmp");
        fs::write(&temporary, raw)
            .with_context(|| format!("could not write {}", temporary.display()))?;
        fs::rename(&temporary, &path)
            .with_context(|| format!("could not replace {}", path.display()))?;
        Ok(())
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        let path = self.path_for(id)?;
        if !path.exists() {
            bail!("there is no session with id `{id}`");
        }
        fs::remove_file(&path).with_context(|| format!("could not remove {}", path.display()))
    }

    /// The most recently updated session, if there is one.
    pub fn latest(&self) -> Result<Option<Session>> {
        let Some(summary) = self.list()?.into_iter().next() else {
            return Ok(None);
        };
        Ok(Some(self.load(&summary.id)?))
    }

    /// Find a session by id or by a unique title/prefix match.
    pub fn resolve(&self, needle: &str) -> Result<Session> {
        let summaries = self.list()?;

        if let Some(found) = summaries.iter().find(|summary| summary.id == needle) {
            return self.load(&found.id);
        }

        let matches: Vec<_> = summaries
            .iter()
            .filter(|summary| {
                summary.title == needle
                    || summary.title.starts_with(needle)
                    || summary.id.starts_with(needle)
            })
            .collect();

        match matches.as_slice() {
            [] => Err(anyhow!("no session matches `{needle}`")),
            [found] => self.load(&found.id),
            _ => {
                let ids = matches
                    .iter()
                    .map(|summary| summary.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                Err(anyhow!("`{needle}` matches several sessions: {ids}"))
            }
        }
    }
}

fn read_session(path: &Path) -> Result<Session> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("could not read the session {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("{} is not a valid session file", path.display()))
}

/// Generate a session id that is unique enough without pulling in a uuid crate.
fn new_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or_default();
    format!("{nanos:016x}{:04x}", std::process::id() & 0xffff)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> (tempdir::TempDir, SessionStore) {
        let dir = tempdir::TempDir::new();
        let store = SessionStore::new(dir.path()).unwrap();
        (dir, store)
    }

    /// A minimal temp directory helper so the tests stay dependency free.
    mod tempdir {
        use std::path::{Path, PathBuf};
        use std::sync::atomic::{AtomicUsize, Ordering};

        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        pub struct TempDir(PathBuf);

        impl TempDir {
            pub fn new() -> Self {
                let path = std::env::temp_dir().join(format!(
                    "nu_plugin_ds_test_{}_{}",
                    std::process::id(),
                    COUNTER.fetch_add(1, Ordering::SeqCst)
                ));
                std::fs::create_dir_all(&path).unwrap();
                TempDir(path)
            }

            pub fn path(&self) -> &Path {
                &self.0
            }
        }

        impl Drop for TempDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    #[test]
    fn clearing_forgets_the_messages_but_keeps_the_system_prompt() {
        let mut session = Session::new(
            "demo",
            "deepseek-flash",
            ThinkingEffort::Off,
            Some("be nice"),
        );
        session.push(ChatMessage::user("hello"));
        session.push(ChatMessage::assistant("hi"));

        session.clear_transcript();

        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.system_prompt(), Some("be nice"));
        assert_eq!(session.user_turns(), 0);
    }

    #[test]
    fn clearing_keeps_the_prompt_the_session_actually_has() {
        // `/system` rewrites the first message without touching the settings, so a clear has
        // to preserve the session's own prompt rather than whatever the settings hold.
        let mut session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
        session
            .messages
            .insert(0, ChatMessage::system("edited prompt"));
        session.push(ChatMessage::user("hello"));

        session.clear_transcript();

        assert_eq!(session.system_prompt(), Some("edited prompt"));
        assert_eq!(session.messages.len(), 1);
    }

    #[test]
    fn clearing_without_a_system_prompt_leaves_nothing() {
        let mut session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
        session.push(ChatMessage::user("hello"));
        session.push(ChatMessage::assistant("hi"));

        session.clear_transcript();

        assert!(session.messages.is_empty());
    }

    #[test]
    fn an_emptied_session_round_trips_through_the_store() {
        // The cleared transcript has to survive a reload, which is the whole point of the
        // clear: writing the now-empty session must replace the old file on disk.
        let (_dir, store) = temp_store();
        let mut session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
        session.push(ChatMessage::user("hello"));
        session.push(ChatMessage::assistant("hi"));
        store.save(&session).unwrap();

        session.clear_transcript();
        store.save(&session).unwrap();

        let loaded = store.load(&session.id).unwrap();
        assert!(loaded.messages.is_empty(), "{:?}", loaded.messages);
    }

    #[test]
    fn round_trips_a_session() {
        let (_dir, store) = temp_store();
        let mut session = Session::new(
            "demo",
            "deepseek-flash",
            ThinkingEffort::Off,
            Some("be nice"),
        );
        session.push(ChatMessage::user("hello"));
        session.push(ChatMessage::assistant("hi"));

        store.save(&session).unwrap();
        let loaded = store.load(&session.id).unwrap();

        assert_eq!(loaded.messages.len(), 3);
        assert_eq!(loaded.system_prompt(), Some("be nice"));
        assert_eq!(loaded.user_turns(), 1);
        assert_eq!(loaded.title(), "demo");
        assert_eq!(store.list().unwrap().len(), 1);
    }

    #[test]
    fn titles_fall_back_to_the_first_user_message() {
        let mut session = Session::new("", "deepseek-flash", ThinkingEffort::Off, None);
        session.push(ChatMessage::user("列出当前目录所有文件"));
        assert_eq!(session.title(), "列出当前目录所有文件");
    }

    #[test]
    fn regeneration_drops_only_the_answer() {
        let mut session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, None);
        session.push(ChatMessage::user("one"));
        session.push(ChatMessage::assistant("first"));
        session.push(ChatMessage::assistant("second"));

        assert!(session.drop_last_answer());
        assert_eq!(session.messages.len(), 1);
        assert!(!session.drop_last_answer());
    }

    #[test]
    fn rejects_path_traversal_in_ids() {
        let (_dir, store) = temp_store();
        assert!(store.load("../escape").is_err());
        assert!(store.delete("nested/id").is_err());
    }

    fn tool_call(id: &str) -> crate::api::ToolCall {
        crate::api::ToolCall {
            id: id.to_owned(),
            kind: "function".to_owned(),
            function: crate::api::FunctionCall {
                name: "read_file".to_owned(),
                arguments: "{}".to_owned(),
            },
        }
    }

    fn session_with_calls(ids: &[&str]) -> Session {
        let mut session = Session::new("", "deepseek-flash", ThinkingEffort::Off, None);
        let mut assistant = ChatMessage::assistant("");
        assistant.tool_calls = Some(ids.iter().map(|id| tool_call(id)).collect());
        session.push(assistant);
        session
    }

    #[test]
    fn answered_tool_calls_leave_nothing_open() {
        let mut session = session_with_calls(&["call_0"]);
        session.push(ChatMessage::tool("call_0", "ok"));
        assert!(session.unanswered_tool_calls().is_empty());
    }

    #[test]
    fn an_unanswered_tool_call_is_reported() {
        let session = session_with_calls(&["call_0"]);
        let open = session.unanswered_tool_calls();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].id, "call_0");
    }

    #[test]
    fn tool_results_are_matched_by_id_not_order() {
        let mut session = session_with_calls(&["call_0", "call_1"]);
        session.push(ChatMessage::tool("call_1", "second"));
        session.push(ChatMessage::tool("call_0", "first"));
        assert!(session.unanswered_tool_calls().is_empty());
    }

    #[test]
    fn a_partial_answer_leaves_the_rest_open() {
        let mut session = session_with_calls(&["call_0", "call_1"]);
        session.push(ChatMessage::tool("call_1", "second"));
        let open = session.unanswered_tool_calls();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].id, "call_0");
    }

    fn long_session() -> Session {
        let mut session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, Some("sys"));
        for i in 0..5 {
            session.push(ChatMessage::user(format!("question {i}")));
            session.push(ChatMessage::assistant(format!("answer {i}")));
        }
        session
    }

    #[test]
    fn a_short_session_plans_nothing() {
        let mut session = Session::new("demo", "deepseek-flash", ThinkingEffort::Off, Some("sys"));
        session.push(ChatMessage::user("hello"));
        session.push(ChatMessage::assistant("hi"));
        assert!(session.compaction_plan(2).is_none());
    }

    #[test]
    fn a_long_session_plans_all_but_the_prompt_and_the_recent_tail() {
        let session = long_session();
        let (planned, removed) = session.compaction_plan(2).unwrap();

        assert_eq!(removed, 8);
        assert_eq!(planned.len(), 8);
        assert_eq!(planned[0].content, "question 0");
        assert_eq!(planned.last().unwrap().content, "answer 3");
    }

    #[test]
    fn applying_a_summary_keeps_the_prompt_the_tail_and_the_summary() {
        let mut session = long_session();
        let tail = session.messages[9..].to_vec();

        let dropped = session.apply_summary("SUM", 2);

        assert_eq!(dropped, 8);
        assert_eq!(session.compactions, 1);
        assert_eq!(session.messages[0].role, Role::System);
        assert_eq!(session.messages[0].content, "sys");
        assert_eq!(session.messages[1].role, Role::System);
        assert!(session.messages[1].content.contains("SUM"));
        assert_eq!(&session.messages[2..], tail.as_slice());
    }

    #[test]
    fn resolves_by_title_prefix() {
        let (_dir, store) = temp_store();
        let session = Session::new("planning", "deepseek-flash", ThinkingEffort::Off, None);
        store.save(&session).unwrap();

        assert_eq!(store.resolve("plan").unwrap().id, session.id);
        assert!(store.resolve("nope").is_err());
    }
}
