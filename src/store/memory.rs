use async_trait::async_trait;
use std::sync::RwLock;

use crate::error::{Error, Result};
use crate::traits::{EntryStore, Payload};
use crate::types::Entry;

pub(crate) type Record<P> = (Entry<P>, Vec<Vec<f32>>);

pub struct InMemoryStore<P> {
    storage: RwLock<Vec<Record<P>>>,
}

impl<P> InMemoryStore<P> {
    pub fn new() -> Self {
        Self {
            storage: RwLock::new(Vec::new()),
        }
    }
}

impl<P> Default for InMemoryStore<P> {
    fn default() -> Self {
        Self::new()
    }
}

fn poisoned(e: impl std::fmt::Display) -> Error {
    Error::StoreError(format!("Lock poisoned: {e}"))
}

#[async_trait]
impl<P: Payload> EntryStore<P> for InMemoryStore<P> {
    async fn insert(&self, entry: Entry<P>, embeddings: Vec<Vec<f32>>) -> Result<()> {
        let mut store = self.storage.write().map_err(poisoned)?;
        store.retain(|(e, _)| e.id != entry.id);
        store.push((entry, embeddings));
        Ok(())
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
        let mut store = self.storage.write().map_err(poisoned)?;
        append(&mut store, id, example, embedding);
        Ok(())
    }

    async fn remove_example(&self, id: &str, index: usize) -> Result<()> {
        let mut store = self.storage.write().map_err(poisoned)?;
        remove(&mut store, id, index);
        Ok(())
    }
}

/// Ajoute un exemple et son embedding ; `false` si l'entrée n'existe pas
pub(crate) fn append<P>(
    store: &mut [Record<P>],
    id: &str,
    example: String,
    embedding: Vec<f32>,
) -> bool {
    match store.iter_mut().find(|(e, _)| e.id == id) {
        Some((entry, embeddings)) => {
            entry.examples.push(example);
            embeddings.push(embedding);
            true
        }
        None => false,
    }
}

/// Supprime l'exemple `index` et son embedding ; `false` si rien n'a changé
pub(crate) fn remove<P>(store: &mut [Record<P>], id: &str, index: usize) -> bool {
    match store.iter_mut().find(|(e, _)| e.id == id) {
        Some((entry, embeddings)) if index < entry.examples.len() => {
            entry.examples.remove(index);
            if index + 1 < embeddings.len() {
                embeddings.remove(index + 1);
            }
            true
        }
        _ => false,
    }
}
