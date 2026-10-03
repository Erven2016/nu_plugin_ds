//! Integration tests for the DeepSeek client, against a local mock server.
//!
//! These exercise the real HTTP and server-sent-event paths: request shaping, chunked
//! delivery, mid-stream disconnects and API error mapping.

mod support;

use futures_util::StreamExt;
use nu_plugin_ds::api::{
    BalanceProbe, ChatMessage, FunctionCall, Role, StreamEvent, ToolCall, ToolCallAccumulator,
    ToolCallFragment, ToolDefinition, build_chat_request, with_tools,
};
use nu_plugin_ds::config::ThinkingEffort;

use support::{MockResponse, MockServer, sse_frame};

fn client_for(server: &MockServer) -> nu_plugin_ds::api::DeepSeekClient {
    nu_plugin_ds::api::DeepSeekClient::new("test-key-123456", server.base_url())
        .expect("client should build")
}

fn chunk(delta: serde_json::Value) -> String {
    sse_frame(serde_json::json!({
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "delta": delta, "finish_reason": null}],
    }))
}

#[tokio::test]
async fn lists_the_models_with_the_api_key() {
    let server = MockServer::start(vec![MockResponse::json(
        r#"{"object":"list","data":[
            {"id":"deepseek-v4-pro","object":"model","owned_by":"deepseek"},
            {"id":"deepseek-flash","object":"model","owned_by":"deepseek"}
        ]}"#,
    )])
    .await;

    let models = client_for(&server)
        .list_models()
        .await
        .expect("model list should decode");

    assert_eq!(
        models
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
        vec!["deepseek-flash", "deepseek-v4-pro"],
        "models should come back sorted"
    );

    let request = &server.raw_requests()[0];
    assert!(request.starts_with("GET /models "), "got: {request}");
    assert!(
        request
            .to_lowercase()
            .contains("authorization: bearer test-key-123456"),
        "the API key should be sent as a bearer token: {request}"
    );
}

#[tokio::test]
async fn streams_reasoning_content_and_usage() {
    let server = MockServer::start(vec![MockResponse::sse(vec![
        chunk(serde_json::json!({"role": "assistant", "reasoning_content": "let me think"})),
        chunk(serde_json::json!({"reasoning_content": " harder"})),
        chunk(serde_json::json!({"content": "Hello"})),
        chunk(serde_json::json!({"content": ", world"})),
        sse_frame(serde_json::json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })),
        sse_frame(serde_json::json!({
            "choices": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
                "prompt_cache_hit_tokens": 8,
                "prompt_cache_miss_tokens": 4
            }
        })),
        "data: [DONE]\n\n".to_owned(),
    ])])
    .await;

    let messages = vec![ChatMessage::user("hi")];
    let request = build_chat_request(
        "deepseek-v4-pro",
        ThinkingEffort::High,
        Some(256),
        Some(0.9),
        &messages,
        true,
    );

    let mut stream = client_for(&server)
        .stream(&request)
        .await
        .expect("the stream should start");

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("no error expected"));
    }

    assert_eq!(
        events,
        vec![
            StreamEvent::Reasoning("let me think".to_owned()),
            StreamEvent::Reasoning(" harder".to_owned()),
            StreamEvent::Content("Hello".to_owned()),
            StreamEvent::Content(", world".to_owned()),
            StreamEvent::Usage(nu_plugin_ds::api::Usage {
                prompt_tokens: 12,
                completion_tokens: 4,
                total_tokens: 16,
                prompt_cache_hit_tokens: 8,
                prompt_cache_miss_tokens: 4,
            }),
            StreamEvent::Finished(Some("stop".to_owned())),
        ]
    );

    // The request must ask for usage and never send reasoning back.
    let body = server.body_for("/chat/completions");
    assert_eq!(body["stream"], serde_json::json!(true));
    assert_eq!(
        body["stream_options"]["include_usage"],
        serde_json::json!(true)
    );
    assert_eq!(body["reasoning_effort"], serde_json::json!("high"));
    assert_eq!(
        body["messages"],
        serde_json::json!([{"role": "user", "content": "hi"}])
    );
    // The current models are not name-guarded, so a custom temperature is sent as-is.
    assert!(
        body.get("temperature").is_some(),
        "the temperature should be sent: {body}"
    );
}

#[tokio::test]
async fn reports_an_interrupted_stream() {
    let server = MockServer::start(vec![MockResponse::sse_truncated(vec![chunk(
        serde_json::json!({"content": "partial"}),
    )])])
    .await;

    let messages = vec![ChatMessage::user("hi")];
    let request = build_chat_request(
        "deepseek-flash",
        ThinkingEffort::Off,
        None,
        None,
        &messages,
        true,
    );

    let mut stream = client_for(&server)
        .stream(&request)
        .await
        .expect("the stream should start");

    assert_eq!(
        stream.next().await.expect("an event").expect("content"),
        StreamEvent::Content("partial".to_owned())
    );

    let error = stream
        .next()
        .await
        .expect("an error event")
        .expect_err("the truncated stream should fail");
    assert!(
        format!("{error:#}").contains("interrupted"),
        "unexpected error: {error:#}"
    );
}

#[tokio::test]
async fn explains_a_rejected_api_key() {
    let server = MockServer::start(vec![MockResponse::status(
        401,
        r#"{"error":{"message":"Authentication Fails, Your api key is invalid","type":"authentication_error","code":"invalid_request_error"}}"#,
    )])
    .await;

    let error = client_for(&server)
        .list_models()
        .await
        .expect_err("401 should fail");

    let rendered = format!("{error:#}");
    assert!(rendered.contains("401"), "got: {rendered}");
    assert!(
        rendered.contains("Authentication Fails"),
        "the API message should be surfaced: {rendered}"
    );
    assert!(
        rendered.contains("invalid_request_error"),
        "the API error code should be surfaced: {rendered}"
    );
    assert!(
        rendered.contains("DEEPSEEK_API_KEY"),
        "the hint should point at the environment variable: {rendered}"
    );
}

#[tokio::test]
async fn completes_without_streaming() {
    let server = MockServer::start(vec![MockResponse::json(
        r#"{"model":"deepseek-flash","choices":[{"index":0,"message":{"role":"assistant","content":"ls ./"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}}"#,
    )])
    .await;

    let messages = vec![
        ChatMessage::system("be brief"),
        ChatMessage::user("list files"),
    ];
    let request = build_chat_request(
        "deepseek-flash",
        ThinkingEffort::Off,
        Some(64),
        Some(0.0),
        &messages,
        false,
    );

    let response = client_for(&server)
        .complete(&request)
        .await
        .expect("the completion should decode");

    assert_eq!(response.text(), "ls ./");
    assert_eq!(response.usage.expect("usage").total_tokens, 12);

    let body = server.body_for("/chat/completions");
    assert!(body.get("stream_options").is_none());
    assert_eq!(body["messages"][0]["role"], serde_json::json!("system"));
}

#[tokio::test]
async fn streams_a_tool_call_in_pieces() {
    let server = MockServer::start(vec![MockResponse::sse(vec![
        chunk(serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "write_file", "arguments": ""}
            }]
        })),
        chunk(serde_json::json!({
            "tool_calls": [{
                "index": 0,
                "function": {"arguments": "{\"path\":\"a.txt\",\"content\":\"hi"}
            }]
        })),
        chunk(serde_json::json!({
            "tool_calls": [{
                "index": 0,
                "function": {"arguments": "}"}
            }]
        })),
        sse_frame(serde_json::json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        })),
        "data: [DONE]\n\n".to_owned(),
    ])])
    .await;

    let messages = vec![ChatMessage::user("write a file")];
    let request = build_chat_request(
        "deepseek-flash",
        ThinkingEffort::Off,
        None,
        None,
        &messages,
        true,
    );

    let mut stream = client_for(&server)
        .stream(&request)
        .await
        .expect("the stream should start");

    let mut events = Vec::new();
    let mut accumulator = ToolCallAccumulator::default();
    while let Some(event) = stream.next().await {
        let event = event.expect("no error expected");
        if let StreamEvent::ToolCall(fragment) = &event {
            accumulator.push(fragment.clone());
        }
        events.push(event);
    }

    assert_eq!(
        events,
        vec![
            StreamEvent::ToolCall(ToolCallFragment {
                index: 0,
                id: Some("call_1".to_owned()),
                name: Some("write_file".to_owned()),
                // The empty arguments fragment still carries the id and name and must be
                // emitted, or the call could never be assembled.
                arguments: String::new(),
            }),
            StreamEvent::ToolCall(ToolCallFragment {
                index: 0,
                id: None,
                name: None,
                arguments: "{\"path\":\"a.txt\",\"content\":\"hi".to_owned(),
            }),
            StreamEvent::ToolCall(ToolCallFragment {
                index: 0,
                id: None,
                name: None,
                arguments: "}".to_owned(),
            }),
            StreamEvent::Finished(Some("tool_calls".to_owned())),
        ]
    );

    assert_eq!(
        accumulator.finish(),
        vec![ToolCall {
            id: "call_1".to_owned(),
            kind: "function".to_owned(),
            function: FunctionCall {
                name: "write_file".to_owned(),
                arguments: "{\"path\":\"a.txt\",\"content\":\"hi}".to_owned(),
            },
        }]
    );
}

#[tokio::test]
async fn sends_tools_and_the_tool_conversation_back() {
    let server = MockServer::start(vec![
        MockResponse::json(
            r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"done"}}]}"#,
        ),
        MockResponse::json(
            r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"done"}}]}"#,
        ),
    ])
    .await;

    // First request: offer a tool.
    let messages = vec![ChatMessage::user("write a file")];
    let request = with_tools(
        build_chat_request(
            "deepseek-flash",
            ThinkingEffort::Off,
            None,
            None,
            &messages,
            false,
        ),
        vec![ToolDefinition::function(
            "write_file",
            "Write a file to disk",
            serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }),
        )],
    );

    client_for(&server)
        .complete(&request)
        .await
        .expect("the completion should decode");

    // Second request: the conversation the model produced, with the tool result.
    let mut assistant = ChatMessage::assistant("").with_reasoning("let me think");
    assistant.tool_calls = Some(vec![ToolCall {
        id: "call_1".to_owned(),
        kind: "function".to_owned(),
        function: FunctionCall {
            name: "write_file".to_owned(),
            arguments: "{\"path\":\"a.txt\"}".to_owned(),
        },
    }]);
    let mut tool = ChatMessage::new(Role::Tool, "wrote a.txt");
    tool.tool_call_id = Some("call_1".to_owned());

    let messages = vec![ChatMessage::user("write a file"), assistant, tool];
    let request = build_chat_request(
        "deepseek-flash",
        ThinkingEffort::Off,
        None,
        None,
        &messages,
        false,
    );

    client_for(&server)
        .complete(&request)
        .await
        .expect("the completion should decode");

    let bodies = server.request_bodies();
    assert_eq!(bodies.len(), 2, "both requests should have been recorded");

    let with_tools_body = &bodies[0];
    assert_eq!(with_tools_body["tools"][0]["type"], "function");
    assert_eq!(
        with_tools_body["tools"][0]["function"]["name"],
        "write_file"
    );

    let conversation = &bodies[1];
    assert_eq!(
        conversation["messages"][1]["tool_calls"][0]["function"]["name"],
        "write_file"
    );
    // With tools in play the API requires the thinking back, or it rejects the call.
    assert_eq!(
        conversation["messages"][1]["reasoning_content"],
        serde_json::json!("let me think")
    );
    assert_eq!(conversation["messages"][2]["role"], "tool");
    assert_eq!(conversation["messages"][2]["tool_call_id"], "call_1");
}

#[tokio::test]
async fn fetches_the_account_balance() {
    let server = MockServer::start(vec![MockResponse::json(
        r#"{"is_available":true,"balance_infos":[{"currency":"CNY","total_balance":"110.00","granted_balance":"10.00","topped_up_balance":"100.00"}]}"#,
    )])
    .await;

    let balance = client_for(&server)
        .balance()
        .await
        .expect("the balance should decode");
    assert!(balance.is_available);
    assert_eq!(balance.balance_infos.len(), 1);
    let info = &balance.balance_infos[0];
    assert_eq!(info.currency, "CNY");
    assert_eq!(info.total_balance, "110.00");
    assert_eq!(info.granted_balance, "10.00");

    let request = &server.raw_requests()[0];
    assert!(request.starts_with("GET /user/balance "), "got: {request}");
    assert!(
        request
            .to_lowercase()
            .contains("authorization: bearer test-key-123456"),
        "the API key should be sent: {request}"
    );
}

#[tokio::test]
async fn probes_an_available_balance() {
    let server = MockServer::start(vec![MockResponse::json(
        r#"{"is_available":true,"balance_infos":[{"currency":"CNY","total_balance":"110.00","granted_balance":"10.00","topped_up_balance":"100.00"}]}"#,
    )])
    .await;

    match client_for(&server).probe_balance().await {
        BalanceProbe::Available(info) => {
            assert_eq!(info.currency, "CNY");
            assert_eq!(info.total_balance, "110.00");
            assert_eq!(info.granted_balance, "10.00");
        }
        other => panic!("expected an available balance, got {other:?}"),
    }
}

#[tokio::test]
async fn classifies_a_provider_without_a_balance_endpoint() {
    // A gateway that does not implement `GET /user/balance` answers 404.
    let server =
        MockServer::start(vec![MockResponse::status(404, r#"{"error":"not found"}"#)]).await;

    assert_eq!(
        client_for(&server).probe_balance().await,
        BalanceProbe::Unsupported
    );

    let request = &server.raw_requests()[0];
    assert!(request.starts_with("GET /user/balance "), "got: {request}");
}

#[tokio::test]
async fn keeps_retrying_when_the_balance_request_fails_otherwise() {
    // A 500 says nothing about whether the endpoint exists, so it must not disable it.
    let server = MockServer::start(vec![MockResponse::status(500, "boom")]).await;

    assert!(
        matches!(
            client_for(&server).probe_balance().await,
            BalanceProbe::Failed(_)
        ),
        "a 500 is inconclusive, not a missing endpoint"
    );
}
