//! A tiny HTTP/1.1 server used to exercise the DeepSeek client without the network.
//!
//! Responses can be queued globally ([`MockServer::start`]) or matched by path
//! ([`MockServer::routed`]), which keeps tests independent of request ordering.
//! Every request is recorded so tests can assert on what was actually sent, and a
//! response can be delivered in chunks so the streaming path is exercised for real.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// One canned HTTP response.
pub struct MockResponse {
    pub status: u16,
    pub content_type: &'static str,
    /// The whole body, sent with `Content-Length`.
    pub body: String,
    /// When set, the body is split into these pieces and sent with chunked encoding,
    /// with a short pause between them, which is how SSE really arrives.
    pub chunks: Option<Vec<String>>,
    /// Drop the connection after the chunks, without the terminating zero-size chunk.
    pub truncate: bool,
}

impl MockResponse {
    pub fn json(body: impl Into<String>) -> Self {
        MockResponse {
            status: 200,
            content_type: "application/json",
            body: body.into(),
            chunks: None,
            truncate: false,
        }
    }

    pub fn status(status: u16, body: impl Into<String>) -> Self {
        MockResponse {
            status,
            content_type: "application/json",
            body: body.into(),
            chunks: None,
            truncate: false,
        }
    }

    pub fn sse(chunks: Vec<String>) -> Self {
        MockResponse {
            status: 200,
            content_type: "text/event-stream",
            body: chunks.concat(),
            chunks: Some(chunks),
            truncate: false,
        }
    }

    /// A stream that dies part way through, as a dropped connection would.
    pub fn sse_truncated(chunks: Vec<String>) -> Self {
        MockResponse {
            status: 200,
            content_type: "text/event-stream",
            body: chunks.concat(),
            chunks: Some(chunks),
            truncate: true,
        }
    }
}

/// A recorded request.
#[derive(Clone)]
pub struct RecordedRequest {
    pub method: String,
    pub path: String,
    pub body: String,
    /// The request exactly as it arrived, headers included.
    pub raw: String,
}

impl RecordedRequest {
    pub fn json(&self) -> Option<serde_json::Value> {
        (!self.body.is_empty())
            .then(|| serde_json::from_str(&self.body).ok())
            .flatten()
    }
}

/// A running mock server.
pub struct MockServer {
    addr: SocketAddr,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
}

impl MockServer {
    /// Answer requests in the order they arrive.
    pub async fn start(responses: Vec<MockResponse>) -> Self {
        Self::spawn(HashMap::new(), VecDeque::from(responses)).await
    }

    /// Answer requests by path: the first route whose key appears in the request path
    /// wins, longest key first. Unmatched requests are refused.
    pub async fn routed(routes: Vec<(&'static str, MockResponse)>) -> Self {
        let mut map: HashMap<String, VecDeque<MockResponse>> = HashMap::new();
        for (path, response) in routes {
            map.entry(path.to_owned()).or_default().push_back(response);
        }
        Self::spawn(map, VecDeque::new()).await
    }

    async fn spawn(
        routes: HashMap<String, VecDeque<MockResponse>>,
        fallback: VecDeque<MockResponse>,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("could not bind the mock server");
        let addr = listener.local_addr().expect("no local address");

        let routes = Arc::new(Mutex::new(routes));
        let fallback = Arc::new(Mutex::new(fallback));
        let requests = Arc::new(Mutex::new(Vec::new()));

        let accept_requests = requests.clone();
        tokio::spawn(async move {
            loop {
                let Ok((socket, _)) = listener.accept().await else {
                    return;
                };
                let routes = routes.clone();
                let fallback = fallback.clone();
                let requests = accept_requests.clone();
                tokio::spawn(async move {
                    let _ = serve(socket, routes, fallback, requests).await;
                });
            }
        });

        MockServer { addr, requests }
    }

    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Every request received so far.
    pub fn recorded(&self) -> Vec<RecordedRequest> {
        self.requests.lock().unwrap().clone()
    }

    /// The paths requested so far, in order.
    pub fn paths(&self) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|request| request.path.clone())
            .collect()
    }

    /// The raw text of every request received so far.
    pub fn raw_requests(&self) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|request| request.raw.clone())
            .collect()
    }

    /// The JSON bodies received so far, ignoring requests without one.
    pub fn request_bodies(&self) -> Vec<serde_json::Value> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter_map(|request| request.json())
            .collect()
    }

    /// The JSON body of the first request whose path contains `needle`.
    pub fn body_for(&self, needle: &str) -> serde_json::Value {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .find(|request| request.path.contains(needle))
            .and_then(|request| request.json())
            .unwrap_or_else(|| panic!("no JSON request matched `{needle}`"))
    }
}

async fn serve(
    mut socket: TcpStream,
    routes: Arc<Mutex<HashMap<String, VecDeque<MockResponse>>>>,
    fallback: Arc<Mutex<VecDeque<MockResponse>>>,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
) -> std::io::Result<()> {
    let raw = read_request(&mut socket).await?;
    let (method, path, body) = split_request(&raw);

    let response = {
        let mut guard = routes.lock().unwrap();
        // Prefer the most specific matching route.
        let key = guard
            .keys()
            .filter(|key| !key.is_empty() && path.contains(key.as_str()))
            .max_by_key(|key| key.len())
            .cloned();

        key.and_then(|key| guard.get_mut(&key).and_then(VecDeque::pop_front))
            .or_else(|| fallback.lock().unwrap().pop_front())
    };

    requests.lock().unwrap().push(RecordedRequest {
        method,
        path,
        body,
        raw,
    });

    let Some(response) = response else {
        return Ok(());
    };

    let reason = match response.status {
        200 => "OK",
        401 => "Unauthorized",
        402 => "Payment Required",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        _ => "Error",
    };

    match response.chunks {
        None => {
            let head = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                response.status,
                reason,
                response.content_type,
                response.body.len()
            );
            socket.write_all(head.as_bytes()).await?;
            socket.write_all(response.body.as_bytes()).await?;
        }
        Some(chunks) => {
            let head = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                response.status, reason, response.content_type
            );
            socket.write_all(head.as_bytes()).await?;
            for chunk in chunks {
                let framed = format!("{:x}\r\n{}\r\n", chunk.len(), chunk);
                socket.write_all(framed.as_bytes()).await?;
                socket.flush().await?;
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            if !response.truncate {
                socket.write_all(b"0\r\n\r\n").await?;
            }
        }
    }

    socket.flush().await?;
    socket.shutdown().await
}

/// Read a complete HTTP request (headers plus `Content-Length` bytes of body).
async fn read_request(socket: &mut TcpStream) -> std::io::Result<String> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];

    loop {
        let read = socket.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);

        if let Some(header_end) = find_subslice(&buffer, b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&buffer[..header_end]).to_lowercase();
            let content_length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if buffer.len() >= header_end + 4 + content_length {
                break;
            }
        }
    }

    Ok(String::from_utf8_lossy(&buffer).into_owned())
}

/// Split a raw request into `(method, path, body)`.
fn split_request(raw: &str) -> (String, String, String) {
    let mut parts = raw.splitn(2, "\r\n");
    let request_line = parts.next().unwrap_or_default();
    let mut pieces = request_line.split_whitespace();
    let method = pieces.next().unwrap_or_default().to_owned();
    let path = pieces.next().unwrap_or_default().to_owned();

    let body = raw
        .split_once("\r\n\r\n")
        .map(|(_, body)| body.to_owned())
        .unwrap_or_default();

    (method, path, body)
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Wrap a JSON object as one SSE `data:` frame.
pub fn sse_frame(value: serde_json::Value) -> String {
    format!("data: {value}\n\n")
}
