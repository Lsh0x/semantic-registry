//! Extension points: plug in your own embedding model, storage or search engine.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::BuildHasher;

use async_trait::async_trait;

use crate::error::Result;
use crate::types::{Entry, SearchResult};

/// Caller data attached to an entry. The library never reads it.
pub trait Payload: Clone + Send + Sync + 'static {}

impl<T: Clone + Send + Sync + 'static> Payload for T {}

/// Decides which text gets embedded for an entry.
///
/// The canonical chunk comes from [`canonical_text`](Self::canonical_text), then one chunk
/// per item of [`content_chunks`](Self::content_chunks) (stored in `entry.content`, never
/// evicted), then one per example from [`example_text`](Self::example_text), in order:
/// scoring and eviction rely on that order. Examples learned through feedback also go through `example_text`.
pub trait Formatter<P>: Send + Sync {
    /// Text of the canonical chunk.
    fn canonical_text(&self, entry: &Entry<P>) -> String;

    /// Text of one example chunk. Defaults to the example as is.
    fn example_text(&self, _entry: &Entry<P>, example: &str) -> String {
        example.to_string()
    }

    /// Content chunks of the entry, cut from its payload or text. Defaults to none.
    fn content_chunks(&self, _entry: &Entry<P>) -> Vec<String> {
        Vec::new()
    }

    /// Every chunk of the entry: the canonical one first, then one per `entry.content` item
    /// (as is), then one per example.
    fn format(&self, entry: &Entry<P>) -> Vec<String> {
        std::iter::once(self.canonical_text(entry))
            .chain(entry.content.iter().cloned())
            .chain(entry.examples.iter().map(|e| self.example_text(entry, e)))
            .collect()
    }
}

/// Storage for entries and their embeddings.
///
/// Invariant: `embeddings[0]` is the canonical chunk, `embeddings[1..=k]` belong to
/// `entry.content` (`k = entry.content.len()`) and `embeddings[1 + k + i]` belongs to
/// `entry.examples[i]`, so `remove_example(id, i)` removes both.
#[async_trait]
pub trait EntryStore<P: Payload>: Send + Sync {
    /// Inserts the entry, replacing any entry with the same id.
    async fn insert(&self, entry: Entry<P>, embeddings: Vec<Vec<f32>>) -> Result<()>;
    /// Removes the entry and its embeddings. Returns whether it existed.
    async fn remove(&self, id: &str) -> Result<bool>;
    /// The entry with this id, if any.
    async fn get(&self, id: &str) -> Result<Option<Entry<P>>>;
    /// The embeddings of the entry, in chunk order.
    async fn get_embeddings(&self, id: &str) -> Result<Option<Vec<Vec<f32>>>>;
    /// Every entry with its embeddings.
    async fn list_all(&self) -> Result<Vec<(Entry<P>, Vec<Vec<f32>>)>>;
    /// Appends an example and its embedding. Unknown ids are ignored.
    async fn append_example(&self, id: &str, example: String, embedding: Vec<f32>) -> Result<()>;
    /// Removes `examples[index]` and its embedding. Unknown ids and indexes are ignored.
    async fn remove_example(&self, id: &str, index: usize) -> Result<()>;

    /// Records that `example` of entry `id` was last used at `at`: it won a search, or was just
    /// added. `at` is in microseconds since the Unix epoch and increases within a process.
    ///
    /// Feeds [`EvictionPolicy::LeastRecentlyUsed`](crate::types::EvictionPolicy) across
    /// restarts. The default persists nothing: the registry then remembers uses until it
    /// restarts. A store may defer the write, for example to its next change.
    async fn record_example_use(&self, _id: &str, _example: &str, _at: u64) -> Result<()> {
        Ok(())
    }

    /// The last recorded use of each example of entry `id`, by example text. Defaults to none.
    async fn example_uses(&self, _id: &str) -> Result<HashMap<String, u64>> {
        Ok(HashMap::new())
    }
}

/// Turns text into a vector. Every entry and query must go through the same model.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embeds one text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embeds several texts, in order. Defaults to one `embed` call per text: override it
    /// when the model or API can process a batch at once.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(texts.len());
        for text in texts {
            vectors.push(self.embed(text).await?);
        }
        Ok(vectors)
    }
}

/// Ranks entries for a query.
#[async_trait]
pub trait Retriever<P: Payload>: Send + Sync {
    /// The best `limit` entries for `query`, best first.
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<SearchResult<P>>>;

    /// The best `limit` entries for `query` among those `view` admits, best first.
    ///
    /// The view applies before the cut to `limit`: a caller who may read few entries
    /// still gets a full page of them, not what is left of a page once the others are
    /// dropped. Override it to apply the view inside the ranking.
    ///
    /// The default asks [`retrieve`](Self::retrieve) for `limit` entries, keeps those the
    /// view admits and, while that leaves fewer than `limit` and the retriever had more to
    /// give, asks again for twice as many. The answer is exact; it costs one more ranking
    /// each time the request doubles.
    async fn retrieve_in(
        &self,
        query: &str,
        limit: usize,
        view: &dyn View,
    ) -> Result<Vec<SearchResult<P>>> {
        let mut asked = limit;
        loop {
            let hits = self.retrieve(query, asked).await?;
            let exhausted = hits.len() < asked;
            let mut admitted: Vec<SearchResult<P>> = hits
                .into_iter()
                .filter(|hit| view.admits(&hit.entry.id))
                .collect();
            if admitted.len() >= limit || exhausted || asked == usize::MAX {
                admitted.truncate(limit);
                return Ok(admitted);
            }
            asked = asked.saturating_mul(2);
        }
    }
}

/// The entries a caller may be answered, by id: a search returns only ids its view admits.
///
/// The library decides no right: the service builds the view from its caller's rights and
/// reads what a hit names under those rights again. Implemented for [`AllEntries`], sets of
/// ids and closures `Fn(&str) -> bool`.
pub trait View: Send + Sync {
    /// Whether entry `id` may be returned.
    fn admits(&self, id: &str) -> bool;
}

/// The view that admits every entry: [`Retriever::retrieve_in`] then answers as
/// [`Retriever::retrieve`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllEntries;

impl View for AllEntries {
    fn admits(&self, _id: &str) -> bool {
        true
    }
}

impl<S: BuildHasher + Send + Sync> View for HashSet<String, S> {
    fn admits(&self, id: &str) -> bool {
        self.contains(id)
    }
}

impl View for BTreeSet<String> {
    fn admits(&self, id: &str) -> bool {
        self.contains(id)
    }
}

impl<F: Fn(&str) -> bool + Send + Sync> View for F {
    fn admits(&self, id: &str) -> bool {
        self(id)
    }
}

/// Told when an entry changes, so that a retriever keeps its own index entry by entry
/// instead of reading every entry again.
///
/// [`IndexedStore`](crate::store::IndexedStore) calls it after each change it writes; a
/// store may also call it itself. A retriever that does not implement it is unaffected.
#[async_trait]
pub trait IndexSink: Send + Sync {
    /// Entry `id` was inserted or changed: replaced, an example appended or removed. Read
    /// it again from the store; an id the store no longer holds is forgotten.
    async fn upsert(&self, id: &str) -> Result<()>;

    /// Entry `id` was removed.
    async fn forget(&self, id: &str) -> Result<()>;
}
