#![allow(dead_code)]

use std::sync::Arc;

use async_trait::async_trait;
use tools_rag::defaults::cosine_retriever::CosineRetriever;
use tools_rag::defaults::memory_store::InMemoryToolStore;
use tools_rag::errors::{Result, ToolRagError};
use tools_rag::registry::ToolRegistry;
use tools_rag::traits::{Embedder, ToolFormatter};
use tools_rag::types::{RegistryConfig, ScoringStrategy, ToolDefinition};

/// Embedder déterministe : le texte est le vecteur lui-même, ex. `"1,0,0"`.
/// Permet de contrôler exactement les similarités cosinus dans les tests.
pub struct VecEmbedder;

#[async_trait]
impl Embedder for VecEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        text.split(',')
            .map(|x| x.trim().parse::<f32>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| ToolRagError::EmbeddingFailed(format!("{text:?}: {e}")))
    }
}

/// Formatter qui n'embarque que la description (le vecteur canonique) puis les exemples
pub struct PlainFormatter;

impl ToolFormatter for PlainFormatter {
    fn format(&self, tool: &ToolDefinition) -> Vec<String> {
        std::iter::once(tool.description.clone())
            .chain(tool.examples.iter().cloned())
            .collect()
    }
}

pub fn tool(name: &str, base: &str) -> ToolDefinition {
    ToolDefinition::new(name, base, serde_json::json!({ "type": "object" }))
}

pub fn registry(config: RegistryConfig) -> (ToolRegistry, Arc<InMemoryToolStore>) {
    registry_with_strategy(config, ScoringStrategy::MaxSim)
}

pub fn registry_with_strategy(
    config: RegistryConfig,
    strategy: ScoringStrategy,
) -> (ToolRegistry, Arc<InMemoryToolStore>) {
    let store = Arc::new(InMemoryToolStore::new());
    let embedder = Arc::new(VecEmbedder);
    let retriever =
        Arc::new(CosineRetriever::new(store.clone(), embedder.clone()).with_strategy(strategy));
    let registry = ToolRegistry::new(store.clone(), embedder, retriever, Box::new(PlainFormatter))
        .with_config(config);
    (registry, store)
}

/// Config permissive : seuls les garde-fous explicitement testés sont actifs
pub fn permissive() -> RegistryConfig {
    RegistryConfig {
        max_examples_per_tool: 10,
        dedup_similarity_threshold: 0.99,
        min_feedback_score: 0.0,
        ..RegistryConfig::default()
    }
}
