//! Server-sent event decoding for streaming chat completions.

use std::pin::Pin;
use std::task::{Context, Poll};

use anyhow::Result;
use futures_util::{Stream, StreamExt};

use super::types::{ChatChunk, StreamEvent, ToolCallFragment};

/// The byte stream handed to us by `reqwest`.
type ByteStream = Pin<Box<dyn Stream<Item = reqwest::Result<Vec<u8>>> + Send>>;

/// One decoded element of the SSE wire format.
#[derive(Debug, PartialEq, Eq)]
enum DecodedEvent {
    /// A comment/keep-alive or a field we do not care about.
    Ignored,
    /// A `data:` payload.
    Data(String),
    /// The `data: [DONE]` sentinel.
    Done,
}

/// Incremental SSE parser.
///
/// All carriage returns are dropped as they arrive: JSON payloads escape literal CRs as
/// `\r`, so a raw CR byte can only ever be part of a line terminator.
#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
    /// How much of `buffer` has already been turned into events. Consuming from a cursor
    /// rather than draining the front keeps this linear: `Vec::drain(..n)` would move the
    /// rest of the buffer on every event.
    start: usize,
    eof: bool,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) {
        self.buffer
            .extend(bytes.iter().copied().filter(|byte| *byte != b'\r'));
    }

    fn mark_eof(&mut self) {
        self.eof = true;
    }

    /// Pull the next complete event out of the buffer.
    fn next_event(&mut self) -> Option<DecodedEvent> {
        if let Some(end) = find_event_boundary(&self.buffer[self.start..]) {
            let event = parse_event(&self.buffer[self.start..self.start + end]);
            self.start += end + 2;
            self.compact();
            return Some(event);
        }
        if self.eof && self.start < self.buffer.len() {
            let event = parse_event(&self.buffer[self.start..]);
            self.start = self.buffer.len();
            self.compact();
            return Some(event);
        }
        None
    }

    /// Drop the consumed prefix once enough of it has piled up, so the buffer cannot grow
    /// without bound during a long stream.
    fn compact(&mut self) {
        if self.start > 0 && (self.start == self.buffer.len() || self.start >= 4096) {
            self.buffer.drain(..self.start);
            self.start = 0;
        }
    }
}

fn find_event_boundary(buffer: &[u8]) -> Option<usize> {
    buffer.windows(2).position(|pair| pair == b"\n\n")
}

fn parse_event(event: &[u8]) -> DecodedEvent {
    let text = String::from_utf8_lossy(event);

    let mut payload = String::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("data:") else {
            continue;
        };
        if !payload.is_empty() {
            payload.push('\n');
        }
        payload.push_str(rest.strip_prefix(' ').unwrap_or(rest));
    }

    if payload.is_empty() {
        DecodedEvent::Ignored
    } else if payload.trim() == "[DONE]" {
        DecodedEvent::Done
    } else {
        DecodedEvent::Data(payload)
    }
}

/// A stream of [`StreamEvent`]s decoded from a streaming chat completion.
///
/// The stream always ends with exactly one [`StreamEvent::Finished`], including when the
/// connection is interrupted or the server omits `[DONE]`.
pub struct EventStream {
    inner: ByteStream,
    decoder: SseDecoder,
    /// No further events will be produced; only the trailing `Finished` remains.
    ended: bool,
    finished_sent: bool,
    finish_reason: Option<String>,
    /// Events decoded from the current chunk that have not been yielded yet.
    pending: std::collections::VecDeque<StreamEvent>,
}

impl EventStream {
    pub(crate) fn new(inner: ByteStream) -> Self {
        EventStream {
            inner,
            decoder: SseDecoder::default(),
            ended: false,
            finished_sent: false,
            finish_reason: None,
            pending: std::collections::VecDeque::new(),
        }
    }
}

impl Stream for EventStream {
    type Item = Result<StreamEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            if let Some(event) = this.pending.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }

            if this.ended {
                if this.finished_sent {
                    return Poll::Ready(None);
                }
                this.finished_sent = true;
                return Poll::Ready(Some(Ok(StreamEvent::Finished(this.finish_reason.clone()))));
            }

            if let Some(event) = this.decoder.next_event() {
                match event {
                    DecodedEvent::Ignored => continue,
                    DecodedEvent::Done => {
                        this.ended = true;
                        continue;
                    }
                    DecodedEvent::Data(payload) => {
                        let chunk: ChatChunk = match serde_json::from_str(&payload) {
                            Ok(chunk) => chunk,
                            Err(err) => {
                                this.ended = true;
                                return Poll::Ready(Some(Err(anyhow::Error::new(err).context(
                                    format!("could not decode a stream chunk: {payload}"),
                                ))));
                            }
                        };

                        if let Some(usage) = chunk.usage {
                            return Poll::Ready(Some(Ok(StreamEvent::Usage(usage))));
                        }

                        let Some(choice) = chunk.choices.into_iter().next() else {
                            continue;
                        };
                        if choice.finish_reason.is_some() {
                            this.finish_reason = choice.finish_reason;
                        }
                        if let Some(reasoning) = non_empty(choice.delta.reasoning_content) {
                            this.pending.push_back(StreamEvent::Reasoning(reasoning));
                        }
                        if let Some(content) = non_empty(choice.delta.content) {
                            this.pending.push_back(StreamEvent::Content(content));
                        }
                        for call in choice.delta.tool_calls {
                            this.pending
                                .push_back(StreamEvent::ToolCall(ToolCallFragment {
                                    index: call.index,
                                    id: call.id,
                                    name: call.function.as_ref().and_then(|f| f.name.clone()),
                                    arguments: call
                                        .function
                                        .and_then(|f| f.arguments)
                                        .unwrap_or_default(),
                                }));
                        }
                    }
                }
                continue;
            }

            match this.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(bytes))) => this.decoder.push(&bytes),
                Poll::Ready(Some(Err(err))) => {
                    this.ended = true;
                    return Poll::Ready(Some(Err(anyhow::Error::new(err)
                        .context("the connection to DeepSeek was interrupted"))));
                }
                Poll::Ready(None) => {
                    this.decoder.mark_eof();
                    this.ended = true;
                }
            }
        }
    }
}

fn non_empty(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.is_empty())
}

/// Turn a `reqwest` byte stream into the boxed representation [`EventStream`] needs.
pub(crate) fn boxed_byte_stream(response: reqwest::Response) -> ByteStream {
    Box::pin(
        response
            .bytes_stream()
            .map(|chunk| chunk.map(|bytes| bytes.to_vec())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoder_for(chunks: &[&str]) -> SseDecoder {
        let mut decoder = SseDecoder::default();
        for chunk in chunks {
            decoder.push(chunk.as_bytes());
        }
        decoder
    }

    #[test]
    fn parses_a_single_event() {
        let mut decoder = decoder_for(&["data: {\"a\":1}\n\n"]);
        assert_eq!(
            decoder.next_event(),
            Some(DecodedEvent::Data("{\"a\":1}".to_owned()))
        );
        assert_eq!(decoder.next_event(), None);
    }

    #[test]
    fn reassembles_events_split_across_chunks() {
        let mut decoder = decoder_for(&["data: {\"a\"", ":1}\n", "\n"]);
        assert_eq!(
            decoder.next_event(),
            Some(DecodedEvent::Data("{\"a\":1}".to_owned()))
        );
    }

    #[test]
    fn handles_crlf_and_comments() {
        let mut decoder = decoder_for(&[": keep-alive\r\n\r\ndata: [DONE]\r\n\r\n"]);
        assert_eq!(decoder.next_event(), Some(DecodedEvent::Ignored));
        assert_eq!(decoder.next_event(), Some(DecodedEvent::Done));
    }

    #[test]
    fn flushes_a_trailing_event_without_a_blank_line() {
        let mut decoder = decoder_for(&["data: [DONE]"]);
        assert_eq!(decoder.next_event(), None);
        decoder.mark_eof();
        assert_eq!(decoder.next_event(), Some(DecodedEvent::Done));
    }
}
