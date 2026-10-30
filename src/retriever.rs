use async_trait::async_trait;
use std::sync::Arc;

use crate::error::Result;
use crate::traits::{Embedder, EntryStore, Payload, Retriever};
use crate::types::{ScoringStrategy, SearchResult};

/// Cosine similarity; 0.0 for empty or zero vectors, or vectors of different lengths.
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
        // f32 rounding can exceed 1.0 for identical vectors
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

/// Brute-force cosine search over every chunk in the store.
pub struct CosineRetriever<P> {
    store: Arc<dyn EntryStore<P>>,
    embedder: Arc<dyn Embedder>,
    strategy: ScoringStrategy,
}

impl<P: Payload> CosineRetriever<P> {
    /// Searches `store`, embedding queries with `embedder` (the one used for the entries).
    pub fn new(store: Arc<dyn EntryStore<P>>, embedder: Arc<dyn Embedder>) -> Self {
        Self {
            store,
            embedder,
            strategy: ScoringStrategy::default(),
        }
    }

    /// Replaces the scoring strategy (default: [`ScoringStrategy::MaxSim`]).
    pub fn with_strategy(mut self, strategy: ScoringStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Score of an entry under the strategy. `embeddings[0]` is the canonical chunk and
    /// `embeddings[1..]` the other chunks, scored like examples.
    pub fn score(&self, query_vec: &[f32], embeddings: &[Vec<f32>]) -> f32 {
        self.score_with_match(query_vec, embeddings).0
    }

    /// Score, and index in `embeddings[1..]` of the winning chunk (`None` when the canonical
    /// chunk wins).
    pub fn score_with_match(
        &self,
        query_vec: &[f32],
        embeddings: &[Vec<f32>],
    ) -> (f32, Option<usize>) {
        let Some((base, examples)) = embeddings.split_first() else {
            return (0.0, None);
        };
        let base_score = cosine_similarity(query_vec, base).max(0.0);
        let best_example = examples
            .iter()
            .map(|emb| cosine_similarity(query_vec, emb))
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let (score, example_wins) = self
            .strategy
            .combine_with_source(base_score, best_example.map(|(_, s)| s));
        (score, best_example.filter(|_| example_wins).map(|(i, _)| i))
    }
}

#[async_trait]
impl<P: Payload> Retriever<P> for CosineRetriever<P> {
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<SearchResult<P>>> {
        let query_vec = self.embedder.embed(query).await?;
        let all_entries = self.store.list_all().await?;

        let mut scored: Vec<SearchResult<P>> = all_entries
            .into_iter()
            .map(|(entry, embeddings)| {
                let (score, matched) = self.score_with_match(&query_vec, &embeddings);
                // Content chunks sit before the examples: a content win names no example
                let matched_example = matched.and_then(|i| i.checked_sub(entry.content.len()));
                SearchResult {
                    entry,
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
        assert!((cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_degenerate_inputs_are_zero() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn identical_vectors_never_exceed_one() {
        let v = vec![0.1f32; 384];
        assert!(cosine_similarity(&v, &v) <= 1.0);
    }
}
