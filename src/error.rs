use thiserror::Error as ThisError;

#[derive(ThisError, Debug)]
pub enum Error {
    #[error("Embedding generation failed: {0}")]
    EmbeddingFailed(String),

    #[error("Storage engine error: {0}")]
    StoreError(String),

    #[error("Failed to parse entry: {0}")]
    ParseError(String),

    #[error("I/O or Network transport error: {0}")]
    TransportError(String),

    #[error("Internal registry error: {0}")]
    Internal(String),

    #[cfg(feature = "candle")]
    #[error("Candle ML error: {0}")]
    Candle(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
