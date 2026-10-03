//! A small async client for the DeepSeek chat completions API.
//!
//! Only the parts the plugin needs are implemented: listing models, a blocking
//! completion (used for summarising and for `cc`) and a streaming completion.

pub mod stream;
pub mod types;

pub use stream::EventStream;
pub use types::*;

use std::time::Duration;

use anyhow::{Context, Result, bail};

/// An API key plus the settings needed to talk to DeepSeek.
///
/// Cloning is cheap: the underlying [`reqwest::Client`] is reference counted.
#[derive(Clone)]
pub struct DeepSeekClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl DeepSeekClient {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Result<Self> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            bail!("no API key configured");
        }

        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(20))
            .user_agent(concat!("nu_plugin_ds/", env!("CARGO_PKG_VERSION")))
            .build()
            .context("could not create the HTTP client")?;

        Ok(DeepSeekClient {
            http,
            api_key,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }

    /// List the models the API key has access to.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let response = self
            .http
            .get(self.endpoint("models"))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .context("could not reach the DeepSeek API")?;
        let response = fail_on_error(response).await?;
        let models: ModelsResponse = response
            .json()
            .await
            .context("could not decode the model list")?;
        let mut models = models.data;
        models.sort_by_key(|model| model.id.clone());
        Ok(models)
    }

    /// The account balance, which includes any granted credit.
    pub async fn balance(&self) -> Result<BalanceResponse> {
        let response = self
            .http
            .get(self.endpoint("user/balance"))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .context("could not reach the DeepSeek API")?;
        let response = fail_on_error(response).await?;
        response
            .json()
            .await
            .context("could not decode the account balance")
    }

    /// Ask the provider for the account balance, reporting whether it has the endpoint.
    ///
    /// A 404 or 405 means the provider does not implement `GET /user/balance`, which is
    /// permanent; any other failure is inconclusive and worth retrying.
    pub async fn probe_balance(&self) -> BalanceProbe {
        match self.balance().await {
            Ok(response) => match response.balance_infos.into_iter().next() {
                Some(info) => BalanceProbe::Available(info),
                None => BalanceProbe::Failed("the account balance came back empty".to_owned()),
            },
            Err(err) => match err
                .downcast_ref::<ApiError>()
                .and_then(|error| error.status)
            {
                Some(404 | 405) => BalanceProbe::Unsupported,
                _ => BalanceProbe::Failed(format!("could not fetch the balance: {err:#}")),
            },
        }
    }

    /// Run a request to completion without streaming.
    pub async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let response = self
            .http
            .post(self.endpoint("chat/completions"))
            .bearer_auth(&self.api_key)
            .json(request)
            .send()
            .await
            .context("could not reach the DeepSeek API")?;
        let response = fail_on_error(response).await?;
        response
            .json()
            .await
            .context("could not decode the completion")
    }

    /// Run a request and return the decoded event stream.
    pub async fn stream(&self, request: &ChatRequest) -> Result<EventStream> {
        let response = self
            .http
            .post(self.endpoint("chat/completions"))
            .bearer_auth(&self.api_key)
            .json(request)
            .send()
            .await
            .context("could not reach the DeepSeek API")?;
        let response = fail_on_error(response).await?;
        Ok(EventStream::new(stream::boxed_byte_stream(response)))
    }
}

/// What asking the provider for the account balance told us.
///
/// `GET /user/balance` is a DeepSeek extension rather than part of the OpenAI compatible
/// API, so a gateway may or may not implement it. This separates "no endpoint" from
/// "the request failed", which are handled very differently.
#[derive(Clone, Debug, PartialEq)]
pub enum BalanceProbe {
    /// The provider has a balance endpoint, and this is what it returned.
    Available(BalanceInfo),
    /// The provider has no balance endpoint.
    Unsupported,
    /// The request failed in a way that says nothing about whether the endpoint exists.
    Failed(String),
}

/// Build the request for one turn of a conversation.
///
/// Two DeepSeek quirks are handled here: `reasoning_content` is never sent back, and
/// thinking-only models do not get a custom temperature.
pub fn build_chat_request(
    model: &str,
    thinking: crate::config::ThinkingEffort,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    messages: &[ChatMessage],
    stream: bool,
) -> ChatRequest {
    let wire = messages
        .iter()
        .filter(|message| !message.is_empty())
        .map(Into::into)
        .collect();

    let temperature = if model.contains("reasoner") {
        None
    } else {
        temperature
    };

    ChatRequest::new(model, wire)
        .stream(stream)
        .max_tokens(max_tokens)
        .temperature(temperature)
        .reasoning_effort(thinking.as_api())
}

/// Attach the tools the model may call to a request that was already built.
///
/// Kept separate from [`build_chat_request`] so that callers without tools need no
/// change; the interactive chat builds its request and then offers the tools here.
pub fn with_tools(request: ChatRequest, tools: Vec<ToolDefinition>) -> ChatRequest {
    request.tools(Some(tools))
}

/// Map a non-success response into a descriptive [`ApiError`].
async fn fail_on_error(response: reqwest::Response) -> Result<reqwest::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let body = response.text().await.unwrap_or_default();
    let detail = serde_json::from_str::<ApiErrorBody>(&body)
        .ok()
        .and_then(|body| body.error);

    let message = match detail {
        Some(detail) if !detail.message.is_empty() => match detail.code.filter(|c| !c.is_empty()) {
            Some(code) => format!("{} ({code})", detail.message),
            None => detail.message,
        },
        _ => {
            let trimmed = body.trim();
            if trimmed.is_empty() {
                status
                    .canonical_reason()
                    .unwrap_or("request failed")
                    .to_owned()
            } else {
                types::truncate(trimmed, 500)
            }
        }
    };

    Err(anyhow::Error::new(ApiError {
        status: Some(status.as_u16()),
        message,
        hint: hint_for_status(status.as_u16()),
    }))
}

/// Whether an API error means the conversation was too long for the model.
///
/// Matched on the message because the providers differ in status code and wording.
pub fn is_context_overflow(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    [
        "context length",
        "context_length_exceeded",
        "maximum context",
        "too many tokens",
        "exceeds the maximum",
        "reduce the length",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

fn hint_for_status(status: u16) -> Option<String> {
    let hint = match status {
        401 => concat!(
            "check that `$env.DEEPSEEK_API_KEY` holds a valid key ",
            "(create one at https://platform.deepseek.com/api_keys)"
        ),
        402 => "the account has run out of credit; top it up on platform.deepseek.com",
        422 => "the request was rejected; try lowering the thinking level with `/think off`",
        429 => "rate limited; wait a moment and retry",
        500 | 502 | 503 | 504 => "the API is having trouble; retrying usually helps",
        _ => return None,
    };
    Some(hint.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::{ChatMessage, ChatRequest, WireMessage};

    #[test]
    fn empty_api_key_is_rejected() {
        assert!(DeepSeekClient::new("  ", "https://example.com").is_err());
    }

    #[test]
    fn endpoints_are_joined_without_double_slashes() {
        let client = DeepSeekClient::new("key", "https://example.com/").unwrap();
        assert_eq!(client.endpoint("models"), "https://example.com/models");
        assert_eq!(client.endpoint("/models"), "https://example.com/models");
        assert_eq!(client.base_url(), "https://example.com");
    }

    #[test]
    fn wire_messages_drop_reasoning() {
        let message = ChatMessage::assistant("answer").with_reasoning("private thoughts");
        let wire: WireMessage = (&message).into();
        assert_eq!(wire.role, "assistant");
        assert_eq!(wire.content, "answer");

        let request = ChatRequest::new("deepseek-reasoner", vec![wire]);
        let json = serde_json::to_value(&request).unwrap();
        assert!(
            json.get("messages").unwrap()[0]
                .get("reasoning_content")
                .is_none()
        );
        assert!(json.get("stream_options").is_none());
    }

    #[test]
    fn reasoning_effort_is_omitted_when_disabled() {
        let request = ChatRequest::new("deepseek-chat", vec![]).reasoning_effort(None::<String>);
        let json = serde_json::to_value(&request).unwrap();
        assert!(json.get("reasoning_effort").is_none());
    }

    #[test]
    fn recognises_a_context_length_refusal() {
        // The wording DeepSeek and OpenAI use for an over-long request.
        assert!(is_context_overflow(
            "This model's maximum context length is 131072 tokens. However, you requested \
             140000 tokens (12345 in the messages). Please reduce the length of the messages."
        ));
        assert!(is_context_overflow(
            "Error code: context_length_exceeded (400)"
        ));
        assert!(is_context_overflow("too many tokens in the request"));
    }

    #[test]
    fn does_not_mistake_other_errors_for_an_overflow() {
        assert!(!is_context_overflow(
            "Rate limit reached for requests (429)"
        ));
        assert!(!is_context_overflow(
            "Authentication Fails, Your api key is invalid (401)"
        ));
    }
}
