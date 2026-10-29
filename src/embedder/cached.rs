use std::sync::Arc;

use async_trait::async_trait;
use tiercache::{Cache, Lru};

use crate::error::{Error, Result};
use crate::traits::Embedder;

/// An [`Embedder`] that remembers vectors in a [`tiercache::Cache`].
///
/// Saves re-embedding the same query between `search()` and `record_feedback()`.
/// Keys are `"{namespace}/{text}"`: change the namespace when you change the model, or
/// vectors from the old model would be served.
pub struct CachedEmbedder {
    inner: Arc<dyn Embedder>,
    cache: Cache,
    namespace: String,
}

impl CachedEmbedder {
    /// An in-process LRU cache of `capacity` vectors.
    pub fn new(inner: Arc<dyn Embedder>, namespace: impl Into<String>, capacity: usize) -> Self {
        Self::with_cache(
            inner,
            namespace,
            Cache::new(vec![Box::new(Lru::new(capacity))]),
        )
    }

    /// A cache you build: several levels, such as memory in front of Redis.
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
        let mut vectors = self.embed_batch(&[text.to_string()]).await?;
        Ok(vectors.pop().unwrap_or_default())
    }

    /// Serves hits from the cache and embeds all misses in one call to the inner embedder.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut vectors: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
        for text in texts {
            // A missing or unreadable value (written by something else) is a miss
            let cached = self.cache.get(&self.key(text)).await;
            vectors.push(cached.as_deref().and_then(decode));
        }

        let misses: Vec<String> = texts
            .iter()
            .zip(&vectors)
            .filter(|(_, v)| v.is_none())
            .map(|(t, _)| t.clone())
            .collect();
        if !misses.is_empty() {
            let mut computed = self.inner.embed_batch(&misses).await?.into_iter();
            for (text, slot) in texts.iter().zip(vectors.iter_mut()) {
                if slot.is_none() {
                    let vector = computed.next().ok_or_else(|| {
                        Error::EmbeddingFailed("inner embedder returned too few vectors".into())
                    })?;
                    self.cache.put(&self.key(text), &encode(&vector)).await;
                    *slot = Some(vector);
                }
            }
        }

        Ok(vectors.into_iter().map(Option::unwrap_or_default).collect())
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

    #[tokio::test]
    async fn batch_embeds_only_the_misses_in_one_call() {
        struct Batches(std::sync::Mutex<Vec<usize>>);

        #[async_trait]
        impl Embedder for Batches {
            async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
                unreachable!("CachedEmbedder must go through embed_batch")
            }
            async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                self.0.lock().unwrap().push(texts.len());
                Ok(texts.iter().map(|t| vec![t.len() as f32]).collect())
            }
        }

        let inner = Arc::new(Batches(std::sync::Mutex::new(Vec::new())));
        let cached = CachedEmbedder::new(inner.clone(), "test", 8);
        cached.embed("bb").await.unwrap();

        let texts = ["a".to_string(), "bb".to_string(), "ccc".to_string()];
        let vectors = cached.embed_batch(&texts).await.unwrap();

        assert_eq!(vectors, [vec![1.0], vec![2.0], vec![3.0]]);
        // One call for "bb", then one call for the two misses
        assert_eq!(*inner.0.lock().unwrap(), [1, 2]);
    }

    #[test]
    fn encoding_roundtrips_and_rejects_truncated_bytes() {
        let v = vec![1.5, -2.25, f32::MIN_POSITIVE];
        assert_eq!(decode(&encode(&v)).unwrap(), v);
        assert!(decode(&[0, 1, 2]).is_none());
    }
}
