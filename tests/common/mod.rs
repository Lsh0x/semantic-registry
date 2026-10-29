#![allow(dead_code)]

use std::sync::Arc;

use async_trait::async_trait;
use semantic_registry::error::{Error, Result};
use semantic_registry::prelude::*;
use semantic_registry::{CosineRetriever, InMemoryStore};

/// Embedder déterministe : le texte est le vecteur lui-même, ex. `"1,0,0"`.
/// Permet de contrôler exactement les similarités cosinus dans les tests.
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

/// Config permissive : seuls les garde-fous explicitement testés sont actifs
pub fn permissive() -> RegistryConfig {
    RegistryConfig {
        max_examples_per_entry: 10,
        dedup_similarity_threshold: 0.99,
        min_feedback_score: 0.0,
        ..RegistryConfig::default()
    }
}
