//! `sync` and `sync_scoped` (#18).

mod common;

use std::sync::Arc;

use async_trait::async_trait;
use common::{RecordingEmbedder, VecEmbedder};
use semantic_registry::error::Result;
use semantic_registry::prelude::*;
use semantic_registry::{CosineRetriever, InMemoryStore};

type Registry = SemanticRegistry<String>;

fn setup() -> (Registry, Arc<InMemoryStore<String>>, Arc<RecordingEmbedder>) {
    let store = Arc::new(InMemoryStore::new());
    let embedder = Arc::new(RecordingEmbedder::default());
    let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
    let registry = SemanticRegistry::new(store.clone(), embedder.clone(), retriever).with_config(
        RegistryConfig {
            dedup_similarity_threshold: 0.99,
            min_feedback_score: 0.0,
            ..RegistryConfig::default()
        },
    );
    (registry, store, embedder)
}

fn entry(id: &str, text: &str, payload: &str) -> Entry<String> {
    Entry::new(id, text, payload.to_string())
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

async fn examples_of(store: &InMemoryStore<String>, id: &str) -> Vec<String> {
    store.get(id).await.unwrap().unwrap().examples
}

#[tokio::test]
async fn sync_adds_updates_keeps_unchanged_and_removes() {
    let (registry, store, _) = setup();
    registry.register(entry("a", "1,0,0", "v1")).await.unwrap();
    registry.register(entry("b", "0,1,0", "v1")).await.unwrap();
    registry.register(entry("c", "0,0,1", "v1")).await.unwrap();

    let report = registry
        .sync(
            vec![
                entry("a", "1,0,0", "v2"), // payload changed
                entry("c", "0,0,1", "v1"), // identical
                entry("d", "1,1,0", "v1"), // new
            ],
            OnMissing::Remove,
        )
        .await
        .unwrap();

    assert_eq!(report.added, ["d"]);
    assert_eq!(report.updated, ["a"]);
    assert_eq!(report.unchanged, ["c"]);
    assert_eq!(report.removed, ["b"]);
    assert!(report.missing.is_empty());
    assert_eq!(store.get("a").await.unwrap().unwrap().payload, "v2");
    assert!(store.get("b").await.unwrap().is_none());
}

#[tokio::test]
async fn learned_examples_survive_a_sync() {
    let (registry, store, _) = setup();
    registry
        .register(entry("a", "1,0,0", "v1").with_examples(strings(&["0,1,0"])))
        .await
        .unwrap();
    let decision = registry
        .record_feedback("0,0,1", "a", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert!(decision.is_added());

    // The source still lists its seeded example and adds a new one
    let report = registry
        .sync(
            vec![entry("a", "1,0,0", "v1").with_examples(strings(&["0,1,0", "1,1,0"]))],
            OnMissing::Remove,
        )
        .await
        .unwrap();

    assert_eq!(report.updated, ["a"]);
    // Stored examples first (seeded, learned), then the new one from the source
    assert_eq!(examples_of(&store, "a").await, ["0,1,0", "0,0,1", "1,1,0"]);
}

#[tokio::test]
async fn an_unchanged_sync_embeds_and_writes_nothing() {
    let (registry, _, embedder) = setup();
    let entries = vec![
        entry("a", "1,0,0", "v1").with_examples(strings(&["0,1,0"])),
        entry("b", "0,0,1", "v1"),
    ];
    registry
        .sync(entries.clone(), OnMissing::Remove)
        .await
        .unwrap();
    let batches_after_first = embedder.batches.lock().unwrap().len();

    let report = registry.sync(entries, OnMissing::Remove).await.unwrap();

    assert_eq!(report.unchanged, ["a", "b"]);
    assert!(report.added.is_empty() && report.updated.is_empty() && report.removed.is_empty());
    assert_eq!(embedder.batches.lock().unwrap().len(), batches_after_first);
}

#[tokio::test]
async fn a_changed_text_embeds_only_the_text() {
    let (registry, _, embedder) = setup();
    registry
        .sync(
            vec![entry("a", "1,0,0", "v1").with_examples(strings(&["0,1,0", "0,0,1"]))],
            OnMissing::Remove,
        )
        .await
        .unwrap();

    registry
        .sync(
            vec![entry("a", "1,1,1", "v1").with_examples(strings(&["0,1,0", "0,0,1"]))],
            OnMissing::Remove,
        )
        .await
        .unwrap();

    let batches = embedder.batches.lock().unwrap();
    assert_eq!(batches.last().unwrap(), &strings(&["1,1,1"]));
}

#[tokio::test]
async fn keep_reports_missing_entries_and_leaves_them() {
    let (registry, store, _) = setup();
    registry.register(entry("a", "1,0,0", "v1")).await.unwrap();
    registry.register(entry("b", "0,1,0", "v1")).await.unwrap();

    let report = registry
        .sync(vec![entry("a", "1,0,0", "v1")], OnMissing::Keep)
        .await
        .unwrap();

    assert_eq!(report.missing, ["b"]);
    assert!(report.removed.is_empty());
    assert!(store.get("b").await.unwrap().is_some());
}

#[tokio::test]
async fn a_scoped_sync_leaves_other_sources_alone() {
    let (registry, store, _) = setup();
    for id in ["github/a", "github/b", "files/c"] {
        registry.register(entry(id, "1,0,0", "v1")).await.unwrap();
    }

    let report = registry
        .sync_scoped(
            vec![entry("github/a", "1,0,0", "v1")],
            OnMissing::Remove,
            |stored| stored.id.starts_with("github/"),
        )
        .await
        .unwrap();

    assert_eq!(report.removed, ["github/b"]);
    assert!(store.get("files/c").await.unwrap().is_some());
}

#[tokio::test]
async fn the_last_occurrence_of_an_id_wins() {
    let (registry, store, _) = setup();
    let report = registry
        .sync(
            vec![entry("a", "1,0,0", "first"), entry("a", "1,0,0", "second")],
            OnMissing::Remove,
        )
        .await
        .unwrap();

    assert_eq!(report.added, ["a"]);
    assert_eq!(store.get("a").await.unwrap().unwrap().payload, "second");
}

/// Yields before answering, so the sync and the feedback interleave
struct YieldingEmbedder;

#[async_trait]
impl Embedder for YieldingEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        tokio::task::yield_now().await;
        VecEmbedder.embed(text).await
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sync_keeps_learned_examples_under_concurrent_feedback() {
    for _ in 0..50 {
        concurrent_sync_and_feedback().await;
    }
}

async fn concurrent_sync_and_feedback() {
    let store = Arc::new(InMemoryStore::<String>::new());
    let embedder = Arc::new(YieldingEmbedder);
    let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
    let registry = Arc::new(
        SemanticRegistry::new(store.clone(), embedder, retriever).with_config(RegistryConfig {
            dedup_similarity_threshold: 0.99,
            min_feedback_score: 0.0,
            ..RegistryConfig::default()
        }),
    );
    registry.register(entry("a", "1,0,0", "v1")).await.unwrap();

    let feedback = {
        let registry = registry.clone();
        tokio::spawn(async move {
            registry
                .record_feedback("0,1,0", "a", FeedbackOutcome::Success)
                .await
                .unwrap()
        })
    };
    let sync = {
        let registry = registry.clone();
        tokio::spawn(async move {
            registry
                .sync(vec![entry("a", "1,1,0", "v2")], OnMissing::Remove)
                .await
                .unwrap()
        })
    };
    assert!(feedback.await.unwrap().is_added());
    sync.await.unwrap();

    // Whichever ran first, the learned example and the new text and payload all survive
    let stored = store.get("a").await.unwrap().unwrap();
    assert_eq!(stored.examples, ["0,1,0"]);
    assert_eq!(
        (stored.text.as_str(), stored.payload.as_str()),
        ("1,1,0", "v2")
    );
}
