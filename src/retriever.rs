use async_trait::async_trait;
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use crate::error::{Error, Result};
use crate::traits::{AllEntries, Embedder, EntryStore, IndexSink, Payload, Retriever, View};
use crate::types::{Entry, ScoringStrategy, SearchResult};

/// An entry and its embeddings, as the store holds them (`EntryStore::list_all`)
type Record<P> = (Entry<P>, Vec<Vec<f32>>);

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
///
/// By default each search reads every entry with [`EntryStore::list_all`].
/// [`with_index`](Self::with_index) keeps them in memory instead: filled once by
/// [`load`](Self::load), then kept entry by entry through [`IndexSink`].
pub struct CosineRetriever<P> {
    store: Arc<dyn EntryStore<P>>,
    embedder: Arc<dyn Embedder>,
    strategy: ScoringStrategy,
    /// The entries by id, when indexed
    index: Option<RwLock<BTreeMap<String, Record<P>>>>,
}

fn poisoned(e: impl std::fmt::Display) -> Error {
    Error::Internal(format!("Index lock poisoned: {e}"))
}

impl<P: Payload> CosineRetriever<P> {
    /// Searches `store`, embedding queries with `embedder` (the one used for the entries).
    pub fn new(store: Arc<dyn EntryStore<P>>, embedder: Arc<dyn Embedder>) -> Self {
        Self {
            store,
            embedder,
            strategy: ScoringStrategy::default(),
            index: None,
        }
    }

    /// Searches an index kept in memory instead of reading the store at each search.
    ///
    /// The index starts empty: fill it once with [`load`](Self::load), then report each
    /// change through [`IndexSink`], for example by wrapping the store the registry writes
    /// to in an [`IndexedStore`](crate::store::IndexedStore). Entries with equal scores
    /// come in id order.
    pub fn with_index(mut self) -> Self {
        self.index = Some(RwLock::new(BTreeMap::new()));
        self
    }

    /// Fills the index from [`EntryStore::list_all`], replacing what it held, and returns
    /// how many entries it holds. Call it once, at start. Without an index, does nothing.
    pub async fn load(&self) -> Result<usize> {
        let Some(index) = &self.index else {
            return Ok(0);
        };
        let entries: BTreeMap<String, Record<P>> = self
            .store
            .list_all()
            .await?
            .into_iter()
            .map(|record| (record.0.id.clone(), record))
            .collect();
        let len = entries.len();
        *index.write().map_err(poisoned)? = entries;
        Ok(len)
    }

    /// The indexed entries, in id order; empty without an index.
    pub fn indexed(&self) -> Result<Vec<Record<P>>> {
        let Some(index) = &self.index else {
            return Ok(Vec::new());
        };
        Ok(index.read().map_err(poisoned)?.values().cloned().collect())
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

impl<P: Payload> CosineRetriever<P> {
    /// The best `limit` of the `records` that `view` admits, best first; equal scores keep
    /// the records' order.
    fn rank<'a>(
        &self,
        query_vec: &[f32],
        records: impl Iterator<Item = &'a Record<P>>,
        limit: usize,
        view: &dyn View,
    ) -> Vec<SearchResult<P>> {
        let mut scored: Vec<(f32, Option<usize>, &Entry<P>)> = records
            .filter(|(entry, _)| view.admits(&entry.id))
            .map(|(entry, embeddings)| {
                let (score, matched) = self.score_with_match(query_vec, embeddings);
                // Content chunks sit before the examples: a content win names no example
                let matched_example = matched.and_then(|i| i.checked_sub(entry.content.len()));
                (score, matched_example, entry)
            })
            .collect();

        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored
            .into_iter()
            .take(limit)
            .map(|(score, matched_example, entry)| SearchResult {
                entry: entry.clone(),
                score,
                matched_example,
            })
            .collect()
    }
}

#[async_trait]
impl<P: Payload> Retriever<P> for CosineRetriever<P> {
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<SearchResult<P>>> {
        self.retrieve_in(query, limit, &AllEntries).await
    }

    /// Drops the entries the view does not admit before ranking, then cuts to `limit`.
    async fn retrieve_in(
        &self,
        query: &str,
        limit: usize,
        view: &dyn View,
    ) -> Result<Vec<SearchResult<P>>> {
        let query_vec = self.embedder.embed(query).await?;
        match &self.index {
            Some(index) => {
                let index = index.read().map_err(poisoned)?;
                Ok(self.rank(&query_vec, index.values(), limit, view))
            }
            None => {
                let all_entries = self.store.list_all().await?;
                Ok(self.rank(&query_vec, all_entries.iter(), limit, view))
            }
        }
    }
}

/// Keeps the index, when there is one; does nothing otherwise.
#[async_trait]
impl<P: Payload> IndexSink for CosineRetriever<P> {
    async fn upsert(&self, id: &str) -> Result<()> {
        let Some(index) = &self.index else {
            return Ok(());
        };
        let record = match self.store.get(id).await? {
            Some(entry) => self
                .store
                .get_embeddings(id)
                .await?
                .map(|embeddings| (entry, embeddings)),
            None => None,
        };
        let mut index = index.write().map_err(poisoned)?;
        match record {
            Some(record) => index.insert(id.to_string(), record),
            None => index.remove(id),
        };
        Ok(())
    }

    async fn forget(&self, id: &str) -> Result<()> {
        if let Some(index) = &self.index {
            index.write().map_err(poisoned)?.remove(id);
        }
        Ok(())
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
