//! Wire types for the DeepSeek (OpenAI compatible) chat completions API.

use serde::{Deserialize, Serialize};

/// A message as stored in a session and sent back to the API.
///
/// `reasoning_content` is persisted so the TUI can redraw old answers. It is also echoed
/// back on the wire: once a request carries `tools`, DeepSeek requires the thinking from
/// earlier turns to be returned, or it rejects the call with a 400.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Set when the assistant asked for tools instead of (or as well as) answering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Set on a `tool` role message: which call it answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// A `tool` role message: the result of a call the assistant asked for.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: Role::Tool,
            content: content.into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    pub fn new(role: Role, content: impl Into<String>) -> Self {
        ChatMessage {
            role,
            content: content.into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        let reasoning = reasoning.into();
        if !reasoning.is_empty() {
            self.reasoning_content = Some(reasoning);
        }
        self
    }

    /// Messages with nothing to say are dropped before being sent. An assistant message
    /// that only asks for tools still has something to say.
    pub fn is_empty(&self) -> bool {
        self.content.trim().is_empty() && self.tool_calls.as_ref().is_none_or(Vec::is_empty)
    }

    /// One-line preview used in session listings.
    pub fn preview(&self, max: usize) -> String {
        let flattened = self
            .content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        truncate(&flattened, max)
    }
}

/// Conversation role.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "you",
            Role::Assistant => "deepseek",
            Role::Tool => "tool",
        }
    }
}

/// The reduced message shape actually accepted by the API.
#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
pub struct WireMessage {
    pub role: String,
    pub content: String,
    /// The assistant's thinking. Once tools are in play the API requires this back, or it
    /// rejects the request with a 400; without tools it is ignored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl From<&ChatMessage> for WireMessage {
    fn from(message: &ChatMessage) -> Self {
        WireMessage {
            role: message.role.as_str().to_owned(),
            content: message.content.clone(),
            reasoning_content: message.reasoning_content.clone(),
            tool_calls: message.tool_calls.clone(),
            tool_call_id: message.tool_call_id.clone(),
        }
    }
}

/// A tool the model may call.
#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDefinition,
}

impl ToolDefinition {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        ToolDefinition {
            kind: "function".to_owned(),
            function: FunctionDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// A call the model asked for.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", default = "function_kind")]
    pub kind: String,
    pub function: FunctionCall,
}

fn function_kind() -> String {
    "function".to_owned()
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct FunctionCall {
    pub name: String,
    /// The arguments exactly as the model sent them: a JSON object encoded as a string.
    pub arguments: String,
}

impl ToolCall {
    /// The arguments decoded into JSON.
    pub fn parse_arguments(&self) -> anyhow::Result<serde_json::Value> {
        serde_json::from_str(&self.function.arguments).map_err(|err| {
            anyhow::Error::new(err).context(format!(
                "`{}` was called with arguments that are not valid JSON: {}",
                self.function.name, self.function.arguments
            ))
        })
    }
}

/// One streamed fragment of a tool call.
///
/// The model streams a call in pieces: the first fragment for an index carries the id and
/// the name, and every later fragment for the same index carries more of the arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCallFragment {
    /// The model's index for this call within the response.
    pub index: u32,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: String,
}

/// Assembles streamed fragments back into complete calls.
#[derive(Debug, Default)]
pub struct ToolCallAccumulator {
    partial: std::collections::BTreeMap<u32, ToolCall>,
}

impl ToolCallAccumulator {
    pub fn is_empty(&self) -> bool {
        self.partial.is_empty()
    }

    /// Fold in one fragment. Fragments may arrive in any order and interleaved.
    pub fn push(&mut self, fragment: ToolCallFragment) {
        let call = self
            .partial
            .entry(fragment.index)
            .or_insert_with(|| ToolCall {
                id: String::new(),
                kind: "function".to_owned(),
                function: FunctionCall {
                    name: String::new(),
                    arguments: String::new(),
                },
            });
        if let Some(id) = fragment.id {
            call.id = id;
        }
        if let Some(name) = fragment.name {
            call.function.name = name;
        }
        call.function.arguments.push_str(&fragment.arguments);
    }

    /// The complete calls, ordered by the model's index. A fragment set that never
    /// received an id and a name is incomplete and is dropped rather than sent back to the
    /// API, which would reject it.
    pub fn finish(self) -> Vec<ToolCall> {
        self.partial
            .into_values()
            .filter(|call| !call.id.is_empty() && !call.function.name.is_empty())
            .collect()
    }
}

/// Token accounting as reported by the API.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
    /// Tokens of the prompt served from DeepSeek's context cache.
    #[serde(default)]
    pub prompt_cache_hit_tokens: u32,
    /// Tokens of the prompt that had to be billed in full.
    #[serde(default)]
    pub prompt_cache_miss_tokens: u32,
}

impl Usage {
    /// Fold another usage report into this one, used to accumulate a session total.
    pub fn merge(&mut self, other: &Usage) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(other.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(other.completion_tokens);
        self.total_tokens = self.total_tokens.saturating_add(other.total_tokens);
        self.prompt_cache_hit_tokens = self
            .prompt_cache_hit_tokens
            .saturating_add(other.prompt_cache_hit_tokens);
        self.prompt_cache_miss_tokens = self
            .prompt_cache_miss_tokens
            .saturating_add(other.prompt_cache_miss_tokens);
    }
}

/// A model as returned by `GET /models`.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub owned_by: Option<String>,
    #[serde(default)]
    pub created: Option<u64>,
}

#[derive(Deserialize, Debug)]
pub struct ModelsResponse {
    #[serde(default)]
    pub data: Vec<ModelInfo>,
}

/// The account balance, as returned by `GET /user/balance`.
#[derive(Deserialize, Serialize, Clone, Debug, Default, PartialEq)]
pub struct BalanceResponse {
    /// Whether the account can still be used.
    #[serde(default)]
    pub is_available: bool,
    /// One entry per currency; DeepSeek currently returns a single one.
    #[serde(default)]
    pub balance_infos: Vec<BalanceInfo>,
}

/// The credit left in one currency.
#[derive(Deserialize, Serialize, Clone, Debug, Default, PartialEq)]
pub struct BalanceInfo {
    #[serde(default)]
    pub currency: String,
    /// Remaining credit in total: granted plus topped up.
    #[serde(default)]
    pub total_balance: String,
    /// The part of the total that came from granted credit.
    #[serde(default)]
    pub granted_balance: String,
    /// The part of the total the user paid for.
    #[serde(default)]
    pub topped_up_balance: String,
}

/// Streaming options; asking for usage makes the final chunk carry token counts.
#[derive(Serialize, Clone, Copy, Debug)]
pub struct StreamOptions {
    pub include_usage: bool,
}

/// DeepSeek's switch for the thinking mode, which is on by default.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThinkingToggle {
    #[serde(rename = "type")]
    pub kind: ThinkingKind,
}

#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingKind {
    Enabled,
    Disabled,
}

/// A chat completions request.
#[derive(Serialize, Clone, Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Turns the thinking mode on or off. It is on by default, so this is only sent to turn
    /// it off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingToggle>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<WireMessage>) -> Self {
        ChatRequest {
            model: model.into(),
            messages,
            stream: false,
            stream_options: None,
            max_tokens: None,
            temperature: None,
            reasoning_effort: None,
            thinking: None,
            tools: None,
            tool_choice: None,
        }
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self.stream_options = stream.then_some(StreamOptions {
            include_usage: true,
        });
        self
    }

    pub fn max_tokens(mut self, max_tokens: Option<u32>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn temperature(mut self, temperature: Option<f32>) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn reasoning_effort(mut self, effort: Option<impl Into<String>>) -> Self {
        self.reasoning_effort = effort.map(Into::into);
        self
    }

    pub fn thinking(mut self, thinking: Option<ThinkingToggle>) -> Self {
        self.thinking = thinking;
        self
    }

    pub fn tools(mut self, tools: Option<Vec<ToolDefinition>>) -> Self {
        self.tools = tools;
        self
    }

    pub fn tool_choice(mut self, tool_choice: Option<impl Into<String>>) -> Self {
        self.tool_choice = tool_choice.map(Into::into);
        self
    }
}

/// A non-streaming completion.
#[derive(Deserialize, Debug)]
pub struct ChatResponse {
    #[serde(default)]
    pub choices: Vec<ResponseChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub model: Option<String>,
}

impl ChatResponse {
    /// The text of the first choice.
    pub fn text(&self) -> String {
        self.choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_default()
    }

    /// The first choice's thinking, when the model returned any. It has to be stored and
    /// echoed back: once a request carries `tools`, the API requires it for every assistant
    /// turn, and rejects the request otherwise.
    pub fn reasoning(&self) -> Option<String> {
        self.choices
            .first()
            .and_then(|choice| choice.message.reasoning_content.clone())
            .filter(|reasoning| !reasoning.is_empty())
    }
}

#[derive(Deserialize, Debug)]
pub struct ResponseChoice {
    #[serde(default)]
    pub message: ResponseMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
pub struct ResponseMessage {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// A single server-sent event of a streaming completion.
#[derive(Deserialize, Debug)]
pub struct ChatChunk {
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
pub struct ChunkChoice {
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub delta: Delta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
pub struct Delta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallDelta>,
}

#[derive(Deserialize, Debug, Default)]
pub struct ToolCallDelta {
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Deserialize, Debug, Default)]
pub struct FunctionCallDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

/// A decoded, UI friendly streaming event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamEvent {
    /// A chunk of the model's private reasoning.
    Reasoning(String),
    /// A chunk of the visible answer.
    Content(String),
    /// A piece of a tool call the model is asking for.
    ToolCall(ToolCallFragment),
    /// Token accounting, sent as the last meaningful event.
    Usage(Usage),
    /// The stream ended; carries the API's `finish_reason` when known.
    Finished(Option<String>),
}

/// An error reported by the API itself (as opposed to a transport failure).
#[derive(Debug)]
pub struct ApiError {
    pub status: Option<u16>,
    pub message: String,
    pub hint: Option<String>,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.status {
            Some(status) => write!(f, "DeepSeek API error (HTTP {status}): {}", self.message),
            None => write!(f, "DeepSeek API error: {}", self.message),
        }?;
        if let Some(hint) = &self.hint {
            write!(f, "\n{hint}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ApiError {}

/// The error envelope DeepSeek returns for failed requests.
#[derive(Deserialize, Debug)]
pub struct ApiErrorBody {
    #[serde(default)]
    pub error: Option<ApiErrorDetail>,
}

#[derive(Deserialize, Debug)]
pub struct ApiErrorDetail {
    #[serde(default)]
    pub message: String,
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
}

/// Truncate a string on a character boundary, appending an ellipsis when cut.
pub fn truncate(value: &str, max: usize) -> String {
    let mut out = String::with_capacity(max + 1);
    for (index, ch) in value.chars().enumerate() {
        if index == max {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_response_exposes_its_reasoning() {
        let raw = r#"{"model":"deepseek-v4-pro","choices":[{"message":{"role":"assistant","content":"hi","reasoning_content":"thought"},"finish_reason":"stop"}]}"#;
        let response: ChatResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(response.text(), "hi");
        assert_eq!(response.reasoning().as_deref(), Some("thought"));

        // A response with no thinking, or an empty one, reports nothing to store.
        let raw = r#"{"choices":[{"message":{"role":"assistant","content":"hi","reasoning_content":""}}]}"#;
        let response: ChatResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(response.reasoning(), None);
    }

    #[test]
    fn usage_accumulates() {
        let mut total = Usage {
            prompt_tokens: 10,
            completion_tokens: 2,
            total_tokens: 12,
            ..Usage::default()
        };
        total.merge(&Usage {
            prompt_tokens: 5,
            completion_tokens: 1,
            total_tokens: 6,
            prompt_cache_hit_tokens: 4,
            ..Usage::default()
        });
        assert_eq!(total.prompt_tokens, 15);
        assert_eq!(total.total_tokens, 18);
        assert_eq!(total.prompt_cache_hit_tokens, 4);
    }

    #[test]
    fn streaming_request_asks_for_usage() {
        let request = ChatRequest::new("deepseek-flash", vec![]).stream(true);
        assert!(request.stream_options.is_some_and(|o| o.include_usage));

        let request = ChatRequest::new("deepseek-flash", vec![]).stream(false);
        assert!(request.stream_options.is_none());
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("你好世界", 2), "你好…");
    }

    fn fragment(
        index: u32,
        id: Option<&str>,
        name: Option<&str>,
        arguments: &str,
    ) -> ToolCallFragment {
        ToolCallFragment {
            index,
            id: id.map(str::to_owned),
            name: name.map(str::to_owned),
            arguments: arguments.to_owned(),
        }
    }

    #[test]
    fn accumulator_assembles_interleaved_out_of_order_fragments() {
        let mut accumulator = ToolCallAccumulator::default();
        assert!(accumulator.is_empty());

        // A later-shaped call arrives first, its arguments split in two; the second call
        // then starts (with an empty arguments fragment that must not be dropped) and its
        // arguments interleave with the tail of the first.
        accumulator.push(fragment(1, Some("call_1"), Some("read_file"), "{\"path\":"));
        accumulator.push(fragment(0, Some("call_0"), Some("write_file"), ""));
        accumulator.push(fragment(1, None, None, "\"a.txt\"}"));
        accumulator.push(fragment(0, None, None, "{\"path\":\"b.txt\","));
        accumulator.push(fragment(0, None, None, "\"content\":\"hi\"}"));

        let calls = accumulator.finish();
        assert_eq!(
            calls,
            vec![
                ToolCall {
                    id: "call_0".to_owned(),
                    kind: "function".to_owned(),
                    function: FunctionCall {
                        name: "write_file".to_owned(),
                        arguments: "{\"path\":\"b.txt\",\"content\":\"hi\"}".to_owned(),
                    },
                },
                ToolCall {
                    id: "call_1".to_owned(),
                    kind: "function".to_owned(),
                    function: FunctionCall {
                        name: "read_file".to_owned(),
                        arguments: "{\"path\":\"a.txt\"}".to_owned(),
                    },
                },
            ],
            "calls should be complete and ordered by the model's index"
        );
    }

    #[test]
    fn accumulator_drops_incomplete_calls() {
        let mut accumulator = ToolCallAccumulator::default();
        // Arguments but no id or name: the API would reject this if sent back.
        accumulator.push(fragment(0, None, None, "{\"path\":\"a.txt\"}"));
        assert!(accumulator.finish().is_empty());

        // An id without a name is still incomplete.
        let mut accumulator = ToolCallAccumulator::default();
        accumulator.push(fragment(0, Some("call_0"), None, "{}"));
        assert!(accumulator.finish().is_empty());
    }

    #[test]
    fn parse_arguments_decodes_json_and_reports_garbage() {
        let call = ToolCall {
            id: "call_0".to_owned(),
            kind: "function".to_owned(),
            function: FunctionCall {
                name: "write_file".to_owned(),
                arguments: "{\"path\":\"a.txt\"}".to_owned(),
            },
        };
        assert_eq!(
            call.parse_arguments().unwrap(),
            serde_json::json!({"path": "a.txt"})
        );

        let broken = ToolCall {
            function: FunctionCall {
                name: "write_file".to_owned(),
                arguments: "not json".to_owned(),
            },
            ..call.clone()
        };
        let error = broken.parse_arguments().expect_err("garbage should fail");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("write_file"), "got: {rendered}");
        assert!(rendered.contains("not json"), "got: {rendered}");
    }

    #[test]
    fn tool_call_message_is_not_empty() {
        let mut message = ChatMessage::assistant("");
        message.tool_calls = Some(vec![ToolCall {
            id: "call_0".to_owned(),
            kind: "function".to_owned(),
            function: FunctionCall {
                name: "write_file".to_owned(),
                arguments: "{}".to_owned(),
            },
        }]);
        assert!(!message.is_empty());

        // An empty list of calls is still nothing to say.
        let mut empty = ChatMessage::assistant("  ");
        empty.tool_calls = Some(Vec::new());
        assert!(empty.is_empty());
    }

    #[test]
    fn wire_messages_keep_tools_and_reasoning() {
        let mut assistant = ChatMessage::assistant("").with_reasoning("private thoughts");
        assistant.tool_calls = Some(vec![ToolCall {
            id: "call_0".to_owned(),
            kind: "function".to_owned(),
            function: FunctionCall {
                name: "write_file".to_owned(),
                arguments: "{}".to_owned(),
            },
        }]);
        let tool = ChatMessage {
            role: Role::Tool,
            content: "ok".to_owned(),
            tool_call_id: Some("call_0".to_owned()),
            ..ChatMessage::new(Role::Tool, "ok")
        };

        let request = ChatRequest::new("deepseek-flash", vec![(&assistant).into(), (&tool).into()]);
        let json = serde_json::to_value(&request).unwrap();

        assert_eq!(
            json["messages"][0]["tool_calls"][0]["function"]["name"],
            "write_file"
        );
        assert_eq!(
            json["messages"][0]["reasoning_content"],
            serde_json::json!("private thoughts")
        );
        assert_eq!(json["messages"][1]["role"], "tool");
        assert_eq!(json["messages"][1]["tool_call_id"], "call_0");
    }

    #[test]
    fn thinking_mode_is_switched_off_explicitly() {
        let disabled = ChatRequest::new("deepseek-flash", vec![]).thinking(Some(ThinkingToggle {
            kind: ThinkingKind::Disabled,
        }));
        let json = serde_json::to_value(&disabled).unwrap();
        assert_eq!(json["thinking"]["type"], serde_json::json!("disabled"));

        let enabled = ChatRequest::new("deepseek-flash", vec![]).thinking(Some(ThinkingToggle {
            kind: ThinkingKind::Enabled,
        }));
        let json = serde_json::to_value(&enabled).unwrap();
        assert_eq!(json["thinking"]["type"], serde_json::json!("enabled"));
    }
}
