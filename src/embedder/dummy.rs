//! A placeholder embedder for tests.

use crate::error::Result;
use crate::traits::Embedder;
use async_trait::async_trait;

/// A constant vector for every text: for tests and wiring, useless for ranking.
pub struct DummyEmbedder {
    /// Length of the vectors returned.
    pub dimension: usize,
}

impl DummyEmbedder {
    /// Vectors of `dimension` components.
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl Embedder for DummyEmbedder {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Ok(vec![0.1f32; self.dimension])
    }
}
