use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
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
    /// Last use of each example, by text (`EntryStore::record_example_use`)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    last_used: HashMap<String, u64>,
}

type Uses = HashMap<String, HashMap<String, u64>>;

/// Entries persisted to a JSON file, rewritten on each change. Embeddings are stored too,
/// so reloading recomputes nothing.
///
/// Example uses (for `LeastRecentlyUsed`) are kept in memory and written with the next
/// change, so searching never rewrites the file. Call [`flush`](Self::flush) before exiting
/// to keep the uses recorded since the last change.
pub struct JsonStore<P> {
    file_path: PathBuf,
    storage: RwLock<Vec<Record<P>>>,
    uses: RwLock<Uses>,
    uses_dirty: AtomicBool,
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
        let mut uses = Uses::new();

        if file_path.exists() {
            let content = fs::read_to_string(&file_path)
                .await
                .map_err(|e| Error::StoreError(e.to_string()))?;

            if !content.trim().is_empty() {
                let records: Vec<StoredRecord<P>> =
                    serde_json::from_str(&content).map_err(|e| Error::ParseError(e.to_string()))?;
                for record in records {
                    if !record.last_used.is_empty() {
                        uses.insert(record.entry.id.clone(), record.last_used);
                    }
                    storage_vec.push((record.entry, record.embeddings));
                }
            }
        }

        Ok(Self {
            file_path,
            storage: RwLock::new(storage_vec),
            uses: RwLock::new(uses),
            uses_dirty: AtomicBool::new(false),
        })
    }

    /// Writes the file if example uses were recorded since the last write.
    ///
    /// # Errors
    /// The file cannot be written.
    pub async fn flush(&self) -> Result<()> {
        if self.uses_dirty.load(Ordering::Acquire) {
            self.save_to_disk().await?;
        }
        Ok(())
    }

    async fn save_to_disk(&self) -> Result<()> {
        let json = {
            let store = self.storage.read().map_err(poisoned)?;
            let uses = self.uses.read().map_err(poisoned)?;
            self.uses_dirty.store(false, Ordering::Release);
            let records: Vec<StoredRecord<&P>> = store
                .iter()
                .map(|(entry, embeddings)| StoredRecord {
                    entry: Entry {
                        id: entry.id.clone(),
                        text: entry.text.clone(),
                        examples: entry.examples.clone(),
                        content: entry.content.clone(),
                        payload: &entry.payload,
                    },
                    embeddings: embeddings.clone(),
                    // Only the uses of examples the entry still has
                    last_used: uses
                        .get(&entry.id)
                        .map(|u| {
                            u.iter()
                                .filter(|(example, _)| entry.examples.contains(example))
                                .map(|(example, at)| (example.clone(), *at))
                                .collect()
                        })
                        .unwrap_or_default(),
                })
                .collect();
            serde_json::to_string_pretty(&records).map_err(|e| Error::ParseError(e.to_string()))?
        };

        if let Some(parent) = self.file_path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| Error::StoreError(e.to_string()))?;
        }

        if let Err(e) = fs::write(&self.file_path, json).await {
            // The uses serialized above were not written: keep them pending
            self.uses_dirty.store(true, Ordering::Release);
            return Err(Error::StoreError(e.to_string()));
        }

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
        self.uses.write().map_err(poisoned)?.remove(id);
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

    async fn record_example_use(&self, id: &str, example: &str, at: u64) -> Result<()> {
        let mut uses = self.uses.write().map_err(poisoned)?;
        let slot = uses
            .entry(id.to_string())
            .or_default()
            .entry(example.to_string())
            .or_insert(0);
        *slot = (*slot).max(at);
        self.uses_dirty.store(true, Ordering::Release);
        Ok(())
    }

    async fn example_uses(&self, id: &str) -> Result<HashMap<String, u64>> {
        let uses = self.uses.read().map_err(poisoned)?;
        Ok(uses.get(id).cloned().unwrap_or_default())
    }
}
