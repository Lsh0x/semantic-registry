use crate::errors::Result;
use crate::traits::{Embedder, ToolRetriever, ToolStore};
use crate::types::{ScoringStrategy, SearchResult};
use async_trait::async_trait;
use std::sync::Arc;

pub struct CosineRetriever {
    store: Arc<dyn ToolStore>,
    embedder: Arc<dyn Embedder>,
    strategy: ScoringStrategy,
}

impl CosineRetriever {
    pub fn new(store: Arc<dyn ToolStore>, embedder: Arc<dyn Embedder>) -> Self {
        Self {
            store,
            embedder,
            strategy: ScoringStrategy::default(),
        }
    }

    pub fn with_strategy(mut self, strategy: ScoringStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Score d'un outil selon la stratégie. Par convention, `embeddings[0]` est le chunk
    /// canonique (nom + description) et `embeddings[1..]` les exemples.
    pub fn score(&self, query_vec: &[f32], embeddings: &[Vec<f32>]) -> f32 {
        self.score_with_match(query_vec, embeddings).0
    }

    /// Score et index de l'exemple gagnant (`None` si la description canonique l'emporte)
    pub fn score_with_match(
        &self,
        query_vec: &[f32],
        embeddings: &[Vec<f32>],
    ) -> (f32, Option<usize>) {
        let Some((base, examples)) = embeddings.split_first() else {
            return (0.0, None);
        };
        let base_score = Self::cosine_similarity(query_vec, base).max(0.0);
        let best_example = examples
            .iter()
            .map(|emb| Self::cosine_similarity(query_vec, emb))
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let (score, example_wins) = self
            .strategy
            .combine_with_source(base_score, best_example.map(|(_, s)| s));
        (score, best_example.filter(|_| example_wins).map(|(i, _)| i))
    }

    /// Rendu public pour être utilisé par les garde-fous de dédoublonnage sémantique
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

#[async_trait]
impl ToolRetriever for CosineRetriever {
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let query_vec = self.embedder.embed(query).await?;
        let all_tools = self.store.list_all().await?;

        let mut scored: Vec<SearchResult> = all_tools
            .into_iter()
            .map(|(tool, embeddings)| {
                let (score, matched_example) = self.score_with_match(&query_vec, &embeddings);
                SearchResult {
                    tool,
                    score,
                    matched_example,
                }
            })
            .collect();

        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(limit);

        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_basics() {
        assert!((CosineRetriever::cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
        assert_eq!(
            CosineRetriever::cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]),
            0.0
        );
        assert!((CosineRetriever::cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_degenerate_inputs_are_zero() {
        assert_eq!(CosineRetriever::cosine_similarity(&[], &[]), 0.0);
        assert_eq!(CosineRetriever::cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
        assert_eq!(
            CosineRetriever::cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]),
            0.0
        );
    }
}
