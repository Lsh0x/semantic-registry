#![allow(dead_code)]

use std::sync::Arc;

use async_trait::async_trait;
use semantic_registry::error::{Error, Result};
use semantic_registry::prelude::*;
use semantic_registry::{CosineRetriever, InMemoryStore};

/// Deterministic embedder: the text is the vector itself, e.g. `"1,0,0"`, so tests
/// control cosine similarities exactly.
pub struct VecEmbedder;

#[async_trait]
impl Embedder for VecEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        text.split(',')
            .map(|x| x.trim().parse::<f32>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| Error::EmbeddingFailed(format!("{text:?}: {e}")))
    }
}

pub fn entry(id: &str, base: &str) -> Entry<()> {
    Entry::new(id, base, ())
}

pub fn registry(config: RegistryConfig) -> (SemanticRegistry<()>, Arc<InMemoryStore<()>>) {
    registry_with_strategy(config, ScoringStrategy::MaxSim)
}

pub fn registry_with_strategy(
    config: RegistryConfig,
    strategy: ScoringStrategy,
) -> (SemanticRegistry<()>, Arc<InMemoryStore<()>>) {
    let store = Arc::new(InMemoryStore::new());
    let embedder = Arc::new(VecEmbedder);
    let retriever =
        Arc::new(CosineRetriever::new(store.clone(), embedder.clone()).with_strategy(strategy));
    let registry = SemanticRegistry::new(store.clone(), embedder, retriever).with_config(config);
    (registry, store)
}

/// Permissive config: only the guards a test sets explicitly are active
pub fn permissive() -> RegistryConfig {
    RegistryConfig {
        max_examples_per_entry: 10,
        dedup_similarity_threshold: 0.99,
        min_feedback_score: 0.0,
        ..RegistryConfig::default()
    }
}

/// [`VecEmbedder`] that records the texts of every `embed_batch` call
#[derive(Default)]
pub struct RecordingEmbedder {
    pub batches: std::sync::Mutex<Vec<Vec<String>>>,
}

#[async_trait]
impl Embedder for RecordingEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.batches.lock().unwrap().push(vec![text.to_string()]);
        VecEmbedder.embed(text).await
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.batches.lock().unwrap().push(texts.to_vec());
        VecEmbedder.embed_batch(texts).await
    }
}
