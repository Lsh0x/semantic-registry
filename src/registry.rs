use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::formatter::DefaultFormatter;
use crate::retriever::{cosine_similarity, CosineRetriever};
use crate::store::InMemoryStore;
use crate::traits::{Embedder, EntryStore, Formatter, Payload, Retriever};
use crate::types::{
    Entry, EvictionPolicy, FeedbackDecision, FeedbackOutcome, RegistryConfig, SearchResult,
};

/// Entries searched by semantic similarity, which learn from feedback.
pub struct SemanticRegistry<P> {
    store: Arc<dyn EntryStore<P>>,
    embedder: Arc<dyn Embedder>,
    retriever: Arc<dyn Retriever<P>>,
    formatter: Box<dyn Formatter<P>>,
    config: RegistryConfig,
    usage: UsageTracker,
}

impl<P: Payload> SemanticRegistry<P> {
    /// A registry over your own store and retriever. Pass the same type twice when one
    /// backend does both.
    pub fn new(
        store: Arc<dyn EntryStore<P>>,
        embedder: Arc<dyn Embedder>,
        retriever: Arc<dyn Retriever<P>>,
    ) -> Self {
        Self {
            store,
            embedder,
            retriever,
            formatter: Box::new(DefaultFormatter),
            config: RegistryConfig::default(),
            usage: UsageTracker::default(),
        }
    }

    /// An in-memory registry with brute-force cosine search: the shortest setup.
    pub fn in_memory(embedder: Arc<dyn Embedder>) -> Self {
        let store: Arc<dyn EntryStore<P>> = Arc::new(InMemoryStore::new());
        let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
        Self::new(store, embedder, retriever)
    }

    /// Replaces what gets embedded (default: `text`, then each example as is).
    pub fn with_formatter(mut self, formatter: impl Formatter<P> + 'static) -> Self {
        self.formatter = Box::new(formatter);
        self
    }

    /// Replaces the policy (default: [`RegistryConfig::default`]).
    pub fn with_config(mut self, config: RegistryConfig) -> Self {
        self.config = config;
        self
    }

    /// The current policy.
    pub fn config(&self) -> &RegistryConfig {
        &self.config
    }

    /// Adds the entry, or replaces the one with the same id.
    ///
    /// Chunks whose text has not changed since the previous version keep their vectors;
    /// the others are embedded in one [`Embedder::embed_batch`] call. This assumes the
    /// formatter and the model are the same as when the previous version was stored: after
    /// changing either, [`remove`](Self::remove) entries before registering them again.
    ///
    /// Replacing an entry also replaces its examples, including those learned through
    /// feedback.
    pub async fn register(&self, entry: Entry<P>) -> Result<()> {
        let chunks = self.formatter.format(&entry);
        let mut known = self.previous_vectors(&entry.id).await?;

        let missing: Vec<String> = chunks
            .iter()
            .filter(|chunk| !known.contains_key(*chunk))
            .cloned()
            .collect();
        if !missing.is_empty() {
            let vectors = self.embedder.embed_batch(&missing).await?;
            known.extend(missing.into_iter().zip(vectors));
        }

        let embeddings = chunks
            .iter()
            .map(|chunk| known.get(chunk).cloned().unwrap_or_default())
            .collect();
        self.store.insert(entry, embeddings).await
    }

    /// Removes the entry, its examples and their usage history. Returns whether it existed.
    pub async fn remove(&self, id: &str) -> Result<bool> {
        let removed = self.store.remove(id).await?;
        self.usage.forget_entry(id);
        Ok(removed)
    }

    /// Vectors of the stored version of an entry, keyed by chunk text
    async fn previous_vectors(&self, id: &str) -> Result<HashMap<String, Vec<f32>>> {
        let (Some(previous), Some(embeddings)) = (
            self.store.get(id).await?,
            self.store.get_embeddings(id).await?,
        ) else {
            return Ok(HashMap::new());
        };
        Ok(self
            .formatter
            .format(&previous)
            .into_iter()
            .zip(embeddings)
            .collect())
    }

    /// At most `limit` entries, best first, without those below `min_search_score`.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult<P>>> {
        let mut results = self.retriever.retrieve(query, limit).await?;
        results.retain(|r| r.score >= self.config.min_search_score);
        for r in &results {
            if let Some(example) = r.matched_example.and_then(|i| r.entry.examples.get(i)) {
                self.usage.touch(&r.entry.id, example);
            }
        }
        Ok(results)
    }

    /// Learns from a choice: on success the query becomes an example of the chosen entry,
    /// on correction an example of the right one, behind the guards of [`RegistryConfig`].
    ///
    /// Reading, evicting and appending are separate store calls: concurrent feedback on the
    /// same entry can briefly exceed `max_examples_per_entry`.
    pub async fn record_feedback(
        &self,
        user_query: &str,
        selected_id: &str,
        outcome: FeedbackOutcome,
    ) -> Result<FeedbackDecision> {
        let config = &self.config;
        let (target_id, is_correction) = match outcome {
            FeedbackOutcome::Success => (selected_id, false),
            FeedbackOutcome::CorrectedTo { ref correct_id } => (correct_id.as_str(), true),
        };

        if config.max_examples_per_entry == 0 {
            return Ok(FeedbackDecision::Full);
        }

        let entry = match self.store.get(target_id).await? {
            Some(e) => e,
            None => return Ok(FeedbackDecision::EntryNotFound),
        };

        let existing_embeddings = match self.store.get_embeddings(target_id).await? {
            Some(e) => e,
            None => return Ok(FeedbackDecision::EntryNotFound),
        };

        let query_embedding = self
            .embedder
            .embed(&self.formatter.example_text(&entry, user_query))
            .await?;

        // 1. Semantic dedup; the best similarity doubles as the confidence score
        let best_similarity = existing_embeddings
            .iter()
            .map(|emb| cosine_similarity(&query_embedding, emb))
            .fold(0.0f32, f32::max);

        if best_similarity >= config.dedup_similarity_threshold {
            return Ok(FeedbackDecision::Duplicate {
                similarity: best_similarity,
            });
        }

        // 2. A success on a query far from the entry is an unreliable signal
        if !is_correction && best_similarity < config.min_feedback_score {
            return Ok(FeedbackDecision::BelowMinScore {
                score: best_similarity,
            });
        }

        // 3. Evict until there is room
        let mut examples = entry.examples;
        let mut embeddings = existing_embeddings;
        let mut evicted = Vec::new();

        while examples.len() >= config.max_examples_per_entry {
            let index = match config.eviction_policy {
                EvictionPolicy::RejectWhenFull => return Ok(FeedbackDecision::Full),
                EvictionPolicy::Fifo => 0,
                EvictionPolicy::MostRedundantWithBase => {
                    most_redundant_with_base(&embeddings).min(examples.len() - 1)
                }
                EvictionPolicy::LeastRecentlyUsed => {
                    self.usage.least_recently_used(target_id, &examples)
                }
            };
            self.store.remove_example(target_id, index).await?;
            let example = examples.remove(index);
            self.usage.forget(target_id, &example);
            evicted.push(example);
            if index + 1 < embeddings.len() {
                embeddings.remove(index + 1);
            }
        }

        // 4. Persist
        self.store
            .append_example(target_id, user_query.to_string(), query_embedding)
            .await?;
        self.usage.touch(target_id, user_query);

        Ok(FeedbackDecision::Added { evicted })
    }
}

/// Index in `examples` of the example most similar to the canonical chunk `embeddings[0]`.
fn most_redundant_with_base(embeddings: &[Vec<f32>]) -> usize {
    let Some((base, examples)) = embeddings.split_first() else {
        return 0;
    };
    examples
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, cosine_similarity(base, emb)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(i, _)| i)
}

/// Last use of each example (logical clock), per entry.
#[derive(Default)]
struct UsageTracker {
    clock: AtomicU64,
    last_used: Mutex<HashMap<String, HashMap<String, u64>>>,
}

impl UsageTracker {
    fn touch(&self, entry: &str, example: &str) {
        let tick = self.clock.fetch_add(1, Ordering::Relaxed) + 1;
        if let Ok(mut map) = self.last_used.lock() {
            map.entry(entry.to_string())
                .or_default()
                .insert(example.to_string(), tick);
        }
    }

    fn forget_entry(&self, entry: &str) {
        if let Ok(mut map) = self.last_used.lock() {
            map.remove(entry);
        }
    }

    fn forget(&self, entry: &str, example: &str) {
        if let Ok(mut map) = self.last_used.lock() {
            if let Some(examples) = map.get_mut(entry) {
                examples.remove(example);
            }
        }
    }

    /// The example never used or used least recently; on a tie, the oldest.
    fn least_recently_used(&self, entry: &str, examples: &[String]) -> usize {
        let Ok(map) = self.last_used.lock() else {
            return 0;
        };
        let used = map.get(entry);
        examples
            .iter()
            .enumerate()
            .min_by_key(|(_, ex)| used.and_then(|u| u.get(ex.as_str())).copied().unwrap_or(0))
            .map_or(0, |(i, _)| i)
    }
}
