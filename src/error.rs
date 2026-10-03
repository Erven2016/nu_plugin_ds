//! Conversion of internal errors into nushell's [`LabeledError`].
//!
//! Internally the plugin uses [`anyhow`] to attach context while an error bubbles up.
//! At the command boundary the whole chain is rendered into a single error the user
//! can act on.

use nu_protocol::{LabeledError, Span};

/// Render an [`anyhow::Error`] (and its context chain) as a [`LabeledError`].
///
/// The first line of the rendered chain becomes the error's headline and also labels
/// the span of the command call. Remaining lines are surfaced as the error's help text.
pub fn labeled(err: anyhow::Error, span: Span) -> LabeledError {
    let rendered = format!("{err:#}");
    let mut lines = rendered.lines();

    let headline = lines.next().unwrap_or("unknown error").to_owned();
    let help = lines
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();

    let error = LabeledError::new(headline.clone()).with_label(headline, span);
    if help.is_empty() {
        error
    } else {
        error.with_help(help.join("\n"))
    }
}

/// Shorthand for turning a `Result` into one carrying a [`LabeledError`].
pub trait IntoLabeled<T> {
    fn labeled(self, span: Span) -> Result<T, LabeledError>;
}

impl<T> IntoLabeled<T> for anyhow::Result<T> {
    fn labeled(self, span: Span) -> Result<T, LabeledError> {
        self.map_err(|err| labeled(err, span))
    }
}
