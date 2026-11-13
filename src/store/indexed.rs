use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::traits::{EntryStore, IndexSink, Payload};
use crate::types::Entry;

/// A store that tells an [`IndexSink`] about each change it writes.
///
/// Every call goes to the inner store; after a successful `insert`, `append_example` or
/// `remove_example` the sink gets [`upsert`](IndexSink::upsert), after a successful
/// `remove` it gets [`forget`](IndexSink::forget). Reads and usage records are not
/// reported. A failed write reports nothing.
///
/// Give the retriever the inner store, not this one: the retriever is usually the sink,
/// and holding this store from it would make a reference cycle.
pub struct IndexedStore<P> {
    inner: Arc<dyn EntryStore<P>>,
    sink: Arc<dyn IndexSink>,
}

impl<P> IndexedStore<P> {
    /// `inner`, reporting its changes to `sink`.
    pub fn new(inner: Arc<dyn EntryStore<P>>, sink: Arc<dyn IndexSink>) -> Self {
        Self { inner, sink }
    }
}

#[async_trait]
impl<P: Payload> EntryStore<P> for IndexedStore<P> {
    async fn insert(&self, entry: Entry<P>, embeddings: Vec<Vec<f32>>) -> Result<()> {
        let id = entry.id.clone();
        self.inner.insert(entry, embeddings).await?;
        self.sink.upsert(&id).await
    }

    async fn remove(&self, id: &str) -> Result<bool> {
        let removed = self.inner.remove(id).await?;
        self.sink.forget(id).await?;
        Ok(removed)
    }

    async fn get(&self, id: &str) -> Result<Option<Entry<P>>> {
        self.inner.get(id).await
    }

    async fn get_embeddings(&self, id: &str) -> Result<Option<Vec<Vec<f32>>>> {
        self.inner.get_embeddings(id).await
    }

    async fn list_all(&self) -> Result<Vec<(Entry<P>, Vec<Vec<f32>>)>> {
        self.inner.list_all().await
    }

    async fn append_example(&self, id: &str, example: String, embedding: Vec<f32>) -> Result<()> {
        self.inner.append_example(id, example, embedding).await?;
        self.sink.upsert(id).await
    }

    async fn remove_example(&self, id: &str, index: usize) -> Result<()> {
        self.inner.remove_example(id, index).await?;
        self.sink.upsert(id).await
    }

    async fn record_example_use(&self, id: &str, example: &str, at: u64) -> Result<()> {
        self.inner.record_example_use(id, example, at).await
    }

    async fn example_uses(&self, id: &str) -> Result<HashMap<String, u64>> {
        self.inner.example_uses(id).await
    }
}
