use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::formatter::DefaultFormatter;
use crate::retriever::{cosine_similarity, CosineRetriever};
use crate::store::InMemoryStore;
use crate::traits::{Embedder, EntryStore, Formatter, Payload, Retriever};
use crate::types::{
    Entry, EvictionPolicy, FeedbackDecision, FeedbackOutcome, OnMissing, RegistryConfig,
    SearchResult, SyncReport,
};

/// Entries searched by semantic similarity, which learn from feedback.
pub struct SemanticRegistry<P> {
    store: Arc<dyn EntryStore<P>>,
    embedder: Arc<dyn Embedder>,
    retriever: Arc<dyn Retriever<P>>,
    formatter: Box<dyn Formatter<P>>,
    config: RegistryConfig,
    usage: UsageTracker,
    locks: EntryLocks,
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
            locks: EntryLocks::default(),
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
        let lock = self.locks.get(&entry.id);
        let _guard = lock.lock().await;
        self.insert_locked(entry).await
    }

    /// `register` for a caller already holding the entry's lock
    async fn insert_locked(&self, mut entry: Entry<P>) -> Result<()> {
        entry.content = self.formatter.content_chunks(&entry);
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
        let lock = self.locks.get(id);
        let _guard = lock.lock().await;
        self.remove_locked(id).await
    }

    /// `remove` for a caller already holding the entry's lock
    async fn remove_locked(&self, id: &str) -> Result<bool> {
        let removed = self.store.remove(id).await?;
        self.usage.forget_entry(id);
        Ok(removed)
    }

    /// Makes the registry match a source's current list of entries, keeping what was learned.
    ///
    /// - An unknown id is registered.
    /// - A known id takes the incoming `text` and payload and **keeps its stored examples**,
    ///   learned ones included; incoming examples it does not have yet are appended. Only
    ///   changed chunks are embedded. An identical entry is left untouched.
    /// - A stored entry the list no longer contains is removed or kept, per `on_missing`.
    ///
    /// Every stored entry is considered part of the source: when one registry holds several
    /// sources, use [`sync_scoped`](Self::sync_scoped). If an id appears twice in `entries`,
    /// the last one wins. Examples removed from the source stay: remove them by hand.
    pub async fn sync(&self, entries: Vec<Entry<P>>, on_missing: OnMissing) -> Result<SyncReport>
    where
        P: PartialEq,
    {
        self.sync_scoped(entries, on_missing, |_| true).await
    }

    /// [`sync`](Self::sync) for one source among several: only stored entries for which
    /// `in_scope` returns `true` can be reported missing, removed or kept. Incoming entries are
    /// handled whatever the scope.
    pub async fn sync_scoped<F>(
        &self,
        entries: Vec<Entry<P>>,
        on_missing: OnMissing,
        in_scope: F,
    ) -> Result<SyncReport>
    where
        P: PartialEq,
        F: Fn(&Entry<P>) -> bool + Send + Sync,
    {
        let mut report = SyncReport::default();
        let incoming = last_occurrence_wins(entries);
        let incoming_ids: HashSet<String> = incoming.iter().map(|e| e.id.clone()).collect();

        for entry in incoming {
            let id = entry.id.clone();
            let lock = self.locks.get(&id);
            let _guard = lock.lock().await;
            match self.store.get(&id).await? {
                None => {
                    self.insert_locked(entry).await?;
                    report.added.push(id);
                }
                Some(stored) => {
                    let examples = merge_examples(&stored.examples, entry.examples);
                    if stored.text == entry.text
                        && stored.payload == entry.payload
                        && stored.examples == examples
                    {
                        report.unchanged.push(id);
                    } else {
                        self.insert_locked(Entry { examples, ..entry }).await?;
                        report.updated.push(id);
                    }
                }
            }
        }

        for (stored, _) in self.store.list_all().await? {
            if incoming_ids.contains(&stored.id) || !in_scope(&stored) {
                continue;
            }
            match on_missing {
                OnMissing::Keep => report.missing.push(stored.id),
                OnMissing::Remove => {
                    let lock = self.locks.get(&stored.id);
                    let _guard = lock.lock().await;
                    if self.remove_locked(&stored.id).await? {
                        report.removed.push(stored.id);
                    }
                }
            }
        }

        Ok(report)
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
                let at = self.usage.touch(&r.entry.id, example);
                self.store
                    .record_example_use(&r.entry.id, example, at)
                    .await?;
            }
        }
        Ok(results)
    }

    /// Learns from a choice: on success the query becomes an example of the chosen entry,
    /// on correction an example of the right one, behind the guards of [`RegistryConfig`].
    ///
    /// Feedback, [`register`](Self::register) and [`remove`](Self::remove) on the same entry
    /// run one at a time within this registry; different entries proceed in parallel. Several
    /// registries (or processes) sharing one store are not coordinated.
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

        // Read, evict and append as one step for this entry: concurrent feedback on it
        // waits instead of evicting from a stale snapshot (#16)
        let lock = self.locks.get(target_id);
        let _guard = lock.lock().await;

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
        let content_len = entry.content.len();
        let mut embeddings = existing_embeddings;
        let mut evicted = Vec::new();
        let persisted_uses = match config.eviction_policy {
            EvictionPolicy::LeastRecentlyUsed
                if examples.len() >= config.max_examples_per_entry =>
            {
                self.store.example_uses(target_id).await?
            }
            _ => HashMap::new(),
        };

        while examples.len() >= config.max_examples_per_entry {
            let index = match config.eviction_policy {
                EvictionPolicy::RejectWhenFull => return Ok(FeedbackDecision::Full),
                EvictionPolicy::Fifo => 0,
                EvictionPolicy::MostRedundantWithBase => {
                    most_redundant_with_base(&embeddings, content_len).min(examples.len() - 1)
                }
                EvictionPolicy::LeastRecentlyUsed => {
                    self.usage
                        .least_recently_used(target_id, &examples, &persisted_uses)
                }
            };
            self.store.remove_example(target_id, index).await?;
            let example = examples.remove(index);
            self.usage.forget(target_id, &example);
            evicted.push(example);
            if 1 + content_len + index < embeddings.len() {
                embeddings.remove(1 + content_len + index);
            }
        }

        // 4. Persist
        self.store
            .append_example(target_id, user_query.to_string(), query_embedding)
            .await?;
        let at = self.usage.touch(target_id, user_query);
        self.store
            .record_example_use(target_id, user_query, at)
            .await?;

        Ok(FeedbackDecision::Added { evicted })
    }
}

/// Index in `examples` of the example most similar to the canonical chunk `embeddings[0]`;
/// the `content_len` content chunks that follow it are not examples.
fn most_redundant_with_base(embeddings: &[Vec<f32>], content_len: usize) -> usize {
    let Some((base, rest)) = embeddings.split_first() else {
        return 0;
    };
    rest.iter()
        .skip(content_len)
        .enumerate()
        .map(|(i, emb)| (i, cosine_similarity(base, emb)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(i, _)| i)
}

/// Last use of each example, per entry, in microseconds since the Unix epoch.
///
/// The clock never goes backwards within a process, and stays comparable with uses a store
/// persisted in an earlier run.
#[derive(Default)]
struct UsageTracker {
    clock: AtomicU64,
    last_used: Mutex<HashMap<String, HashMap<String, u64>>>,
}

impl UsageTracker {
    /// A time strictly later than any this tracker returned before
    fn now(&self) -> u64 {
        let wall = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_micros()).unwrap_or(u64::MAX));
        let mut previous = self.clock.load(Ordering::Relaxed);
        loop {
            let next = wall.max(previous + 1);
            match self.clock.compare_exchange_weak(
                previous,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return next,
                Err(current) => previous = current,
            }
        }
    }

    /// Marks the example as used now, and returns that time
    fn touch(&self, entry: &str, example: &str) -> u64 {
        let at = self.now();
        if let Ok(mut map) = self.last_used.lock() {
            map.entry(entry.to_string())
                .or_default()
                .insert(example.to_string(), at);
        }
        at
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

    /// The example never used or used least recently, from this run's uses and those the store
    /// persisted; on a tie, the oldest.
    fn least_recently_used(
        &self,
        entry: &str,
        examples: &[String],
        persisted: &HashMap<String, u64>,
    ) -> usize {
        let Ok(map) = self.last_used.lock() else {
            return 0;
        };
        let used = map.get(entry);
        examples
            .iter()
            .enumerate()
            .min_by_key(|(_, ex)| {
                let here = used.and_then(|u| u.get(ex.as_str())).copied();
                let stored = persisted.get(ex.as_str()).copied();
                here.max(stored).unwrap_or(0)
            })
            .map_or(0, |(i, _)| i)
    }
}

/// One async lock per entry id, so operations on the same entry run one at a time.
///
/// Locks are created on first use and kept: dropping one while a task still waits on it
/// would let a newcomer take a fresh lock and run alongside that task. One small lock per
/// id ever used is the cost.
#[derive(Default)]
struct EntryLocks(Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>);

impl EntryLocks {
    fn get(&self, id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        locks.entry(id.to_string()).or_default().clone()
    }
}

/// Keeps the last entry for each id, in the order of their first appearance
fn last_occurrence_wins<P>(entries: Vec<Entry<P>>) -> Vec<Entry<P>> {
    let mut position: HashMap<String, usize> = HashMap::new();
    let mut unique: Vec<Entry<P>> = Vec::new();
    for entry in entries {
        match position.get(&entry.id) {
            Some(&i) => unique[i] = entry,
            None => {
                position.insert(entry.id.clone(), unique.len());
                unique.push(entry);
            }
        }
    }
    unique
}

/// Stored examples first, in order, then incoming ones not already there
fn merge_examples(stored: &[String], incoming: Vec<String>) -> Vec<String> {
    let mut merged = stored.to_vec();
    for example in incoming {
        if !merged.contains(&example) {
            merged.push(example);
        }
    }
    merged
}
