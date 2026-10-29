use crate::errors::Result;
use crate::traits::Embedder;
use async_trait::async_trait;

/// Embedder factice pour les tests unitaires et le développement local
pub struct DummyEmbedder {
    pub dimension: usize,
}

impl DummyEmbedder {
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
