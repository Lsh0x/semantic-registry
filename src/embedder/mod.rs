pub mod dummy;

#[cfg(feature = "cache")]
pub mod cached;

#[cfg(feature = "candle")]
pub mod candle;

pub use dummy::DummyEmbedder;

#[cfg(feature = "candle")]
pub use candle::CandleEmbedder;

#[cfg(feature = "cache")]
pub use cached::CachedEmbedder;
