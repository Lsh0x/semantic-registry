use async_trait::async_trait;
use std::sync::RwLock;

use crate::errors::{Result, ToolRagError};
use crate::traits::ToolStore;
use crate::types::ToolDefinition;

pub struct InMemoryToolStore {
    storage: RwLock<Vec<(ToolDefinition, Vec<Vec<f32>>)>>,
}

impl InMemoryToolStore {
    pub fn new() -> Self {
        Self {
            storage: RwLock::new(Vec::new()),
        }
    }
}

impl Default for InMemoryToolStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolStore for InMemoryToolStore {
    async fn insert(&self, tool: ToolDefinition, embeddings: Vec<Vec<f32>>) -> Result<()> {
        let mut store = self
            .storage
            .write()
            .map_err(|e| ToolRagError::StoreError(format!("Write lock poisoned: {e}")))?;
        store.retain(|(t, _)| t.name != tool.name);
        store.push((tool, embeddings));
        Ok(())
    }

    async fn get(&self, name: &str) -> Result<Option<ToolDefinition>> {
        let store = self
            .storage
            .read()
            .map_err(|e| ToolRagError::StoreError(format!("Read lock poisoned: {e}")))?;
        Ok(store
            .iter()
            .find(|(t, _)| t.name == name)
            .map(|(t, _)| t.clone()))
    }

    async fn get_embeddings(&self, tool_name: &str) -> Result<Option<Vec<Vec<f32>>>> {
        let store = self
            .storage
            .read()
            .map_err(|e| ToolRagError::StoreError(format!("Read lock poisoned: {e}")))?;
        Ok(store
            .iter()
            .find(|(t, _)| t.name == tool_name)
            .map(|(_, emb)| emb.clone()))
    }

    async fn list_all(&self) -> Result<Vec<(ToolDefinition, Vec<Vec<f32>>)>> {
        let store = self
            .storage
            .read()
            .map_err(|e| ToolRagError::StoreError(format!("Read lock poisoned: {e}")))?;
        Ok(store.clone())
    }

    async fn append_example(
        &self,
        tool_name: &str,
        example: String,
        embedding: Vec<f32>,
    ) -> Result<()> {
        let mut store = self
            .storage
            .write()
            .map_err(|e| ToolRagError::StoreError(format!("Write lock poisoned: {e}")))?;

        if let Some((tool, embeddings)) = store.iter_mut().find(|(t, _)| t.name == tool_name) {
            tool.examples.push(example);
            embeddings.push(embedding);
        }
        Ok(())
    }

    async fn remove_example(&self, tool_name: &str, index: usize) -> Result<()> {
        let mut store = self
            .storage
            .write()
            .map_err(|e| ToolRagError::StoreError(format!("Write lock poisoned: {e}")))?;

        if let Some((tool, embeddings)) = store.iter_mut().find(|(t, _)| t.name == tool_name) {
            if index < tool.examples.len() {
                tool.examples.remove(index);
                if index + 1 < embeddings.len() {
                    embeddings.remove(index + 1);
                }
            }
        }
        Ok(())
    }
}
