use crate::errors::Result;
use crate::types::{SearchResult, ToolDefinition};
use async_trait::async_trait;

pub trait ToolFormatter: Send + Sync {
    fn format(&self, tool: &ToolDefinition) -> Vec<String>;
}

/// Stockage des outils et de leurs embeddings.
///
/// Invariant : `embeddings[0]` est le chunk canonique et `embeddings[i + 1]` correspond à
/// `tool.examples[i]`. `remove_example(name, i)` supprime donc les deux.
#[async_trait]
pub trait ToolStore: Send + Sync {
    async fn insert(&self, tool: ToolDefinition, embeddings: Vec<Vec<f32>>) -> Result<()>;
    async fn get(&self, name: &str) -> Result<Option<ToolDefinition>>;
    async fn get_embeddings(&self, tool_name: &str) -> Result<Option<Vec<Vec<f32>>>>;
    async fn list_all(&self) -> Result<Vec<(ToolDefinition, Vec<Vec<f32>>)>>;
    async fn append_example(
        &self,
        tool_name: &str,
        example: String,
        embedding: Vec<f32>,
    ) -> Result<()>;
    async fn remove_example(&self, tool_name: &str, index: usize) -> Result<()>;
}

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

#[async_trait]
pub trait ToolRetriever: Send + Sync {
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>>;
}
