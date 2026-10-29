use async_trait::async_trait;

use crate::error::Result;
use crate::types::{Entry, SearchResult};

/// Données utilisateur attachées à une entrée, jamais lues par la bibliothèque
pub trait Payload: Clone + Send + Sync + 'static {}

impl<T: Clone + Send + Sync + 'static> Payload for T {}

/// Décide du texte embarqué pour une entrée.
///
/// Le chunk canonique vient de [`canonical_text`](Self::canonical_text), puis un chunk par
/// exemple via [`example_text`](Self::example_text), dans l'ordre : le scoring et l'éviction
/// reposent sur cet ordre. Les exemples appris par feedback passent aussi par `example_text`.
pub trait Formatter<P>: Send + Sync {
    fn canonical_text(&self, entry: &Entry<P>) -> String;

    fn example_text(&self, _entry: &Entry<P>, example: &str) -> String {
        example.to_string()
    }

    fn format(&self, entry: &Entry<P>) -> Vec<String> {
        std::iter::once(self.canonical_text(entry))
            .chain(entry.examples.iter().map(|e| self.example_text(entry, e)))
            .collect()
    }
}

/// Stockage des entrées et de leurs embeddings.
///
/// Invariant : `embeddings[0]` est le chunk canonique et `embeddings[i + 1]` correspond à
/// `entry.examples[i]`. `remove_example(id, i)` supprime donc les deux.
#[async_trait]
pub trait EntryStore<P: Payload>: Send + Sync {
    async fn insert(&self, entry: Entry<P>, embeddings: Vec<Vec<f32>>) -> Result<()>;
    async fn get(&self, id: &str) -> Result<Option<Entry<P>>>;
    async fn get_embeddings(&self, id: &str) -> Result<Option<Vec<Vec<f32>>>>;
    async fn list_all(&self) -> Result<Vec<(Entry<P>, Vec<Vec<f32>>)>>;
    async fn append_example(&self, id: &str, example: String, embedding: Vec<f32>) -> Result<()>;
    async fn remove_example(&self, id: &str, index: usize) -> Result<()>;
}

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

#[async_trait]
pub trait Retriever<P: Payload>: Send + Sync {
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<SearchResult<P>>>;
}
