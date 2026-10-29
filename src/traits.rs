//! Extension points: plug in your own embedding model, storage or search engine.

use async_trait::async_trait;

use crate::error::Result;
use crate::types::{Entry, SearchResult};

/// Caller data attached to an entry. The library never reads it.
pub trait Payload: Clone + Send + Sync + 'static {}

impl<T: Clone + Send + Sync + 'static> Payload for T {}

/// Decides which text gets embedded for an entry.
///
/// The canonical chunk comes from [`canonical_text`](Self::canonical_text), then one chunk
/// per example from [`example_text`](Self::example_text), in order: scoring and eviction
/// rely on that order. Examples learned through feedback also go through `example_text`.
pub trait Formatter<P>: Send + Sync {
    /// Text of the canonical chunk.
    fn canonical_text(&self, entry: &Entry<P>) -> String;

    /// Text of one example chunk. Defaults to the example as is.
    fn example_text(&self, _entry: &Entry<P>, example: &str) -> String {
        example.to_string()
    }

    /// Every chunk of the entry: the canonical one first, then one per example.
    fn format(&self, entry: &Entry<P>) -> Vec<String> {
        std::iter::once(self.canonical_text(entry))
            .chain(entry.examples.iter().map(|e| self.example_text(entry, e)))
            .collect()
    }
}

/// Storage for entries and their embeddings.
///
/// Invariant: `embeddings[0]` is the canonical chunk and `embeddings[i + 1]` belongs to
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
}
