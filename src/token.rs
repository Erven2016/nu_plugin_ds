//! Token estimation.
//!
//! The plugin never sees a tokenizer, so it estimates. The numbers are only used for
//! two things: the context gauge in the status bar and deciding when to compact. Real
//! usage figures reported by the API always win when they are available.

use crate::api::ChatMessage;

/// Approximate tokens in a single message, including the chat template overhead.
pub const MESSAGE_OVERHEAD: usize = 4;

/// Estimate the number of tokens in `text`.
///
/// Latin text is assumed to average four characters per token, CJK and other
/// wide characters closer to one token per character. This tracks DeepSeek's published
/// guidance closely enough for budgeting purposes.
pub fn estimate_tokens(text: &str) -> usize {
    let mut tokens = 0.0f32;
    for ch in text.chars() {
        if ch.is_ascii_whitespace() || ch.is_ascii_punctuation() {
            tokens += 0.2;
        } else if ch.is_ascii() {
            tokens += 0.26;
        } else if is_cjk(ch) {
            tokens += 1.0;
        } else {
            tokens += 0.6;
        }
    }
    tokens.ceil() as usize
}

/// Estimate the tokens required to send a whole conversation.
pub fn estimate_messages(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|message| {
            MESSAGE_OVERHEAD
                + estimate_tokens(&message.content)
                + message
                    .reasoning_content
                    .as_deref()
                    .map_or(0, estimate_tokens)
        })
        .sum()
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32,
        0x3000..=0x303F      // CJK symbols and punctuation
        | 0x3040..=0x30FF    // hiragana + katakana
        | 0x3400..=0x4DBF    // CJK extension A
        | 0x4E00..=0x9FFF    // CJK unified ideographs
        | 0xAC00..=0xD7AF    // hangul syllables
        | 0xF900..=0xFAFF    // CJK compatibility ideographs
        | 0xFF00..=0xFFEF    // halfwidth and fullwidth forms
        | 0x20000..=0x2FA1F  // CJK extensions B..F
    )
}

/// Render a token count compactly, e.g. `1.2k`.
pub fn format_tokens(tokens: u32) -> String {
    if tokens < 1_000 {
        tokens.to_string()
    } else if tokens < 1_000_000 {
        format!("{:.1}k", tokens as f64 / 1_000.0)
    } else {
        format!("{:.2}M", tokens as f64 / 1_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_is_roughly_a_quarter_token_per_char() {
        let tokens = estimate_tokens("hello world, this is a test sentence.");
        assert!((5..=12).contains(&tokens), "got {tokens}");
    }

    #[test]
    fn chinese_is_roughly_one_token_per_char() {
        let tokens = estimate_tokens("你好世界");
        assert!((3..=6).contains(&tokens), "got {tokens}");
    }

    #[test]
    fn empty_input_costs_nothing() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_messages(&[]), 0);
    }

    #[test]
    fn message_overhead_is_counted() {
        let message = ChatMessage::user("hi");
        assert_eq!(estimate_messages(&[message]), MESSAGE_OVERHEAD + 1);
    }
}
