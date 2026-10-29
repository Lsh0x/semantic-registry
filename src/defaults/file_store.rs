use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::RwLock;
use tokio::fs;

use crate::errors::{Result, ToolRagError};
use crate::traits::ToolStore;
use crate::types::ToolDefinition;

#[derive(Serialize, Deserialize)]
struct StoredRecord {
    tool: ToolDefinition,
    embeddings: Vec<Vec<f32>>,
}

pub struct JsonToolStore {
    file_path: PathBuf,
    storage: RwLock<Vec<(ToolDefinition, Vec<Vec<f32>>)>>,
}

impl JsonToolStore {
    pub async fn load_or_create(path: impl Into<PathBuf>) -> Result<Self> {
        let file_path = path.into();
        let mut storage_vec = Vec::new();

        if file_path.exists() {
            let content = fs::read_to_string(&file_path)
                .await
                .map_err(|e| ToolRagError::StoreError(e.to_string()))?;

            if !content.trim().is_empty() {
                let records: Vec<StoredRecord> = serde_json::from_str(&content)
                    .map_err(|e| ToolRagError::ParseError(e.to_string()))?;
                storage_vec = records
                    .into_iter()
                    .map(|r| (r.tool, r.embeddings))
                    .collect();
            }
        }

        Ok(Self {
            file_path,
            storage: RwLock::new(storage_vec),
        })
    }

    async fn save_to_disk(&self) -> Result<()> {
        let records: Vec<StoredRecord> = {
            let store = self
                .storage
                .read()
                .map_err(|e| ToolRagError::StoreError(format!("Read lock poisoned: {e}")))?;
            store
                .iter()
                .cloned()
                .map(|(tool, embeddings)| StoredRecord { tool, embeddings })
                .collect()
        };

        let json = serde_json::to_string_pretty(&records)
            .map_err(|e| ToolRagError::ParseError(e.to_string()))?;

        if let Some(parent) = self.file_path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| ToolRagError::StoreError(e.to_string()))?;
        }

        fs::write(&self.file_path, json)
            .await
            .map_err(|e| ToolRagError::StoreError(e.to_string()))?;

        Ok(())
    }
}

#[async_trait]
impl ToolStore for JsonToolStore {
    async fn insert(&self, tool: ToolDefinition, embeddings: Vec<Vec<f32>>) -> Result<()> {
        {
            let mut store = self
                .storage
                .write()
                .map_err(|e| ToolRagError::StoreError(format!("Write lock poisoned: {e}")))?;
            store.retain(|(t, _)| t.name != tool.name);
            store.push((tool, embeddings));
        }
        self.save_to_disk().await
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
        let modified = {
            let mut store = self
                .storage
                .write()
                .map_err(|e| ToolRagError::StoreError(format!("Write lock poisoned: {e}")))?;

            if let Some((tool, embeddings)) = store.iter_mut().find(|(t, _)| t.name == tool_name) {
                tool.examples.push(example);
                embeddings.push(embedding);
                true
            } else {
                false
            }
        };

        if modified {
            self.save_to_disk().await?;
        }
        Ok(())
    }

    async fn remove_example(&self, tool_name: &str, index: usize) -> Result<()> {
        let modified = {
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
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };

        if modified {
            self.save_to_disk().await?;
        }
        Ok(())
    }
}
