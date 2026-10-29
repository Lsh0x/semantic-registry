//! The error type of the crate.

use thiserror::Error as ThisError;

/// Everything that can fail in the registry, its stores and its embedders.
#[derive(ThisError, Debug)]
pub enum Error {
    /// The embedder could not produce a vector.
    #[error("Embedding generation failed: {0}")]
    EmbeddingFailed(String),

    /// The store failed to read or write.
    #[error("Storage engine error: {0}")]
    StoreError(String),

    /// Stored or downloaded data could not be parsed.
    #[error("Failed to parse entry: {0}")]
    ParseError(String),

    /// A network or file transfer failed, such as a model download.
    #[error("I/O or Network transport error: {0}")]
    TransportError(String),

    /// A background task failed.
    #[error("Internal registry error: {0}")]
    Internal(String),

    #[cfg(feature = "candle")]
    /// Candle failed during inference.
    #[error("Candle ML error: {0}")]
    Candle(#[from] candle_core::Error),
}

/// `Result` with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
