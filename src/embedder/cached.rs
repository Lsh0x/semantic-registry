use std::sync::Arc;

use async_trait::async_trait;
use tiercache::{Cache, Lru};

use crate::errors::Result;
use crate::traits::Embedder;

/// Embedder qui mémorise les vecteurs dans un [`tiercache::Cache`].
///
/// Évite de recalculer l'embedding d'une même requête entre `search()` et
/// `record_feedback()`, ou lors d'un nouvel `register()` du même outil.
/// Les clés sont `"{namespace}/{texte}"` : changer de modèle implique de changer
/// de namespace, sinon des vecteurs d'un autre modèle seraient servis.
pub struct CachedEmbedder {
    inner: Arc<dyn Embedder>,
    cache: Cache,
    namespace: String,
}

impl CachedEmbedder {
    /// Cache en mémoire (LRU de `capacity` entrées)
    pub fn new(inner: Arc<dyn Embedder>, namespace: impl Into<String>, capacity: usize) -> Self {
        Self::with_cache(
            inner,
            namespace,
            Cache::new(vec![Box::new(Lru::new(capacity))]),
        )
    }

    /// Cache multi-niveaux fourni par l'appelant (LRU, Redis, base…)
    pub fn with_cache(
        inner: Arc<dyn Embedder>,
        namespace: impl Into<String>,
        cache: Cache,
    ) -> Self {
        Self {
            inner,
            cache,
            namespace: namespace.into(),
        }
    }

    fn key(&self, text: &str) -> String {
        format!("{}/{}", self.namespace, text)
    }
}

fn encode(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn decode(bytes: &[u8]) -> Option<Vec<f32>> {
    let (chunks, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return None;
    }
    Some(chunks.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

#[async_trait]
impl Embedder for CachedEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let key = self.key(text);
        if let Some(vector) = self.cache.get(&key).await.as_deref().and_then(decode) {
            return Ok(vector);
        }
        // Absent ou illisible (valeur écrite par autre chose) : on recalcule et on écrase
        let vector = self.inner.embed(text).await?;
        self.cache.put(&key, &encode(&vector)).await;
        Ok(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counting(AtomicUsize);

    #[async_trait]
    impl Embedder for Counting {
        async fn embed(&self, text: &str) -> Result<Vec<f32>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(vec![text.len() as f32, -0.5])
        }
    }

    #[tokio::test]
    async fn second_call_is_served_from_cache() {
        let inner = Arc::new(Counting(AtomicUsize::new(0)));
        let cached = CachedEmbedder::new(inner.clone(), "test", 8);

        assert_eq!(cached.embed("abc").await.unwrap(), [3.0, -0.5]);
        assert_eq!(cached.embed("abc").await.unwrap(), [3.0, -0.5]);
        assert_eq!(inner.0.load(Ordering::SeqCst), 1);

        cached.embed("other").await.unwrap();
        assert_eq!(inner.0.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn keys_are_namespaced_by_model() {
        let inner: Arc<dyn Embedder> = Arc::new(Counting(AtomicUsize::new(0)));
        let a = CachedEmbedder::new(inner.clone(), "model-a", 8);
        let b = CachedEmbedder::new(inner, "model-b", 8);
        assert_eq!(a.key("q"), "model-a/q");
        assert_ne!(a.key("q"), b.key("q"));
    }

    #[test]
    fn encoding_roundtrips_and_rejects_truncated_bytes() {
        let v = vec![1.5, -2.25, f32::MIN_POSITIVE];
        assert_eq!(decode(&encode(&v)).unwrap(), v);
        assert!(decode(&[0, 1, 2]).is_none());
    }
}
