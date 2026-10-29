//! Built-in embedders. Implement [`Embedder`](crate::traits::Embedder) for any other model.

pub mod dummy;

/// Embedding cache over tiercache (feature `cache`).
#[cfg(feature = "cache")]
pub mod cached;

/// Local inference with Candle (feature `candle`).
#[cfg(feature = "candle")]
pub mod candle;

pub use dummy::DummyEmbedder;

#[cfg(feature = "candle")]
pub use candle::CandleEmbedder;

#[cfg(feature = "cache")]
pub use cached::CachedEmbedder;
