pub mod dummy;

#[cfg(feature = "candle")]
pub mod candle;

pub use dummy::DummyEmbedder;

#[cfg(feature = "candle")]
pub use candle::CandleEmbedder;
