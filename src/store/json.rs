use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::RwLock;
use tokio::fs;

use crate::error::{Error, Result};
use crate::store::memory::{append, remove, remove_entry, Record};
use crate::traits::{EntryStore, Payload};
use crate::types::Entry;

#[derive(Serialize, Deserialize)]
struct StoredRecord<P> {
    entry: Entry<P>,
    embeddings: Vec<Vec<f32>>,
}

/// Entries persisted to a JSON file, rewritten on each change. Embeddings are stored too,
/// so reloading recomputes nothing.
pub struct JsonStore<P> {
    file_path: PathBuf,
    storage: RwLock<Vec<Record<P>>>,
}

fn poisoned(e: impl std::fmt::Display) -> Error {
    Error::StoreError(format!("Lock poisoned: {e}"))
}

impl<P: Payload + Serialize + DeserializeOwned> JsonStore<P> {
    /// Loads the file if it exists, or starts empty. The file is created on the first change.
    ///
    /// # Errors
    /// The file cannot be read, or is not a valid store.
    pub async fn load_or_create(path: impl Into<PathBuf>) -> Result<Self> {
        let file_path = path.into();
        let mut storage_vec = Vec::new();

        if file_path.exists() {
            let content = fs::read_to_string(&file_path)
                .await
                .map_err(|e| Error::StoreError(e.to_string()))?;

            if !content.trim().is_empty() {
                let records: Vec<StoredRecord<P>> =
                    serde_json::from_str(&content).map_err(|e| Error::ParseError(e.to_string()))?;
                storage_vec = records
                    .into_iter()
                    .map(|r| (r.entry, r.embeddings))
                    .collect();
            }
        }

        Ok(Self {
            file_path,
            storage: RwLock::new(storage_vec),
        })
    }

    async fn save_to_disk(&self) -> Result<()> {
        let json = {
            let store = self.storage.read().map_err(poisoned)?;
            let records: Vec<StoredRecord<&P>> = store
                .iter()
                .map(|(entry, embeddings)| StoredRecord {
                    entry: Entry {
                        id: entry.id.clone(),
                        text: entry.text.clone(),
                        examples: entry.examples.clone(),
                        payload: &entry.payload,
                    },
                    embeddings: embeddings.clone(),
                })
                .collect();
            serde_json::to_string_pretty(&records).map_err(|e| Error::ParseError(e.to_string()))?
        };

        if let Some(parent) = self.file_path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| Error::StoreError(e.to_string()))?;
        }

        fs::write(&self.file_path, json)
            .await
            .map_err(|e| Error::StoreError(e.to_string()))?;

        Ok(())
    }
}

#[async_trait]
impl<P: Payload + Serialize + DeserializeOwned> EntryStore<P> for JsonStore<P> {
    async fn insert(&self, entry: Entry<P>, embeddings: Vec<Vec<f32>>) -> Result<()> {
        {
            let mut store = self.storage.write().map_err(poisoned)?;
            store.retain(|(e, _)| e.id != entry.id);
            store.push((entry, embeddings));
        }
        self.save_to_disk().await
    }

    async fn remove(&self, id: &str) -> Result<bool> {
        let removed = {
            let mut store = self.storage.write().map_err(poisoned)?;
            remove_entry(&mut store, id)
        };
        if removed {
            self.save_to_disk().await?;
        }
        Ok(removed)
    }

    async fn get(&self, id: &str) -> Result<Option<Entry<P>>> {
        let store = self.storage.read().map_err(poisoned)?;
        Ok(store
            .iter()
            .find(|(e, _)| e.id == id)
            .map(|(e, _)| e.clone()))
    }

    async fn get_embeddings(&self, id: &str) -> Result<Option<Vec<Vec<f32>>>> {
        let store = self.storage.read().map_err(poisoned)?;
        Ok(store
            .iter()
            .find(|(e, _)| e.id == id)
            .map(|(_, emb)| emb.clone()))
    }

    async fn list_all(&self) -> Result<Vec<(Entry<P>, Vec<Vec<f32>>)>> {
        let store = self.storage.read().map_err(poisoned)?;
        Ok(store.clone())
    }

    async fn append_example(&self, id: &str, example: String, embedding: Vec<f32>) -> Result<()> {
        let modified = {
            let mut store = self.storage.write().map_err(poisoned)?;
            append(&mut store, id, example, embedding)
        };
        if modified {
            self.save_to_disk().await?;
        }
        Ok(())
    }

    async fn remove_example(&self, id: &str, index: usize) -> Result<()> {
        let modified = {
            let mut store = self.storage.write().map_err(poisoned)?;
            remove(&mut store, id, index)
        };
        if modified {
            self.save_to_disk().await?;
        }
        Ok(())
    }
}
