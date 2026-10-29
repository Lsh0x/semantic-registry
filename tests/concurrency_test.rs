//! Regression test for #16: concurrent feedback on one entry.

mod common;

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use semantic_registry::error::Result;
use semantic_registry::prelude::*;
use semantic_registry::{CosineRetriever, InMemoryStore};

/// [`common::VecEmbedder`] that yields before answering, like a network call,
/// so concurrent calls interleave between their store reads and writes
struct YieldingEmbedder;

#[async_trait]
impl Embedder for YieldingEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        tokio::task::yield_now().await;
        common::VecEmbedder.embed(text).await
    }
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_feedback_evicts_and_reports_consistently() {
    let seeded = ["0,1,0", "0,0,1"];
    let queries: [&'static str; 4] = ["0,1,1", "1,1,0", "1,0,1", "1,1,1"];

    for _ in 0..20 {
        let store = Arc::new(InMemoryStore::<()>::new());
        let embedder = Arc::new(YieldingEmbedder);
        let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
        let registry = Arc::new(
            SemanticRegistry::new(store.clone(), embedder, retriever).with_config(RegistryConfig {
                max_examples_per_entry: 2,
                dedup_similarity_threshold: 0.999,
                min_feedback_score: 0.0,
                ..RegistryConfig::default()
            }),
        );
        registry
            .register(
                Entry::new("x", "1,0,0", ())
                    .with_examples(seeded.iter().map(|s| s.to_string()).collect()),
            )
            .await
            .unwrap();

        let handles: Vec<_> = queries
            .iter()
            .map(|&query| {
                let registry = registry.clone();
                tokio::spawn(async move {
                    registry
                        .record_feedback(query, "x", FeedbackOutcome::Success)
                        .await
                        .unwrap()
                })
            })
            .collect();

        let mut evicted = Vec::new();
        for handle in handles {
            match handle.await.unwrap() {
                FeedbackDecision::Added { evicted: e } => evicted.extend(e),
                other => panic!("every query is new and above the score floor: {other:?}"),
            }
        }

        let entry = store.get("x").await.unwrap().unwrap();
        let remaining: BTreeSet<String> = entry.examples.iter().cloned().collect();
        let evicted_set: BTreeSet<String> = evicted.iter().cloned().collect();

        // Each reported eviction happened once, and nothing vanished unreported
        assert_eq!(
            evicted.len(),
            evicted_set.len(),
            "an example reported evicted twice"
        );
        assert!(
            remaining.is_disjoint(&evicted_set),
            "a reported eviction is still there"
        );
        let mut all = set(&seeded);
        all.extend(set(&queries));
        let accounted: BTreeSet<String> = remaining.union(&evicted_set).cloned().collect();
        assert_eq!(accounted, all, "an example vanished without being reported");
        assert_eq!(entry.examples.len(), 2, "the cap holds");

        // Vectors stay aligned with examples
        let vectors = store.get_embeddings("x").await.unwrap().unwrap();
        assert_eq!(vectors.len(), entry.examples.len() + 1);
        for (example, vector) in entry.examples.iter().zip(&vectors[1..]) {
            assert_eq!(*vector, common::VecEmbedder.embed(example).await.unwrap());
        }
    }
}
