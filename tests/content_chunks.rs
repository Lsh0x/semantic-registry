mod common;

use std::sync::Arc;

use common::*;
use semantic_registry::prelude::*;
use semantic_registry::{CosineRetriever, InMemoryStore, JsonStore};

/// Two content chunks, whatever the entry
struct ContentFormatter;

impl Formatter<()> for ContentFormatter {
    fn canonical_text(&self, entry: &Entry<()>) -> String {
        entry.text.clone()
    }

    fn content_chunks(&self, _entry: &Entry<()>) -> Vec<String> {
        vec!["0,1,0".into(), "0,0,1".into()]
    }
}

fn setup(config: RegistryConfig) -> (SemanticRegistry<()>, Arc<InMemoryStore<()>>) {
    let store = Arc::new(InMemoryStore::new());
    let embedder = Arc::new(VecEmbedder);
    let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
    let registry = SemanticRegistry::new(store.clone(), embedder, retriever)
        .with_formatter(ContentFormatter)
        .with_config(config);
    (registry, store)
}

fn v(text: &str) -> Vec<f32> {
    text.split(',').map(|x| x.parse().unwrap()).collect()
}

#[tokio::test]
async fn register_stores_canonical_then_content_then_examples() {
    let (registry, store) = setup(permissive());
    let e = entry("a", "1,0,0").with_examples(vec!["1,1,0".into(), "1,2,0".into()]);
    registry.register(e).await.unwrap();

    let stored = store.get("a").await.unwrap().unwrap();
    assert_eq!(stored.content, ["0,1,0", "0,0,1"]);
    let embeddings = store.get_embeddings("a").await.unwrap().unwrap();
    let expected: Vec<Vec<f32>> = ["1,0,0", "0,1,0", "0,0,1", "1,1,0", "1,2,0"]
        .into_iter()
        .map(v)
        .collect();
    assert_eq!(embeddings, expected);
}

#[tokio::test]
async fn eviction_takes_examples_never_content() {
    let config = RegistryConfig {
        max_examples_per_entry: 1,
        eviction_policy: EvictionPolicy::MostRedundantWithBase,
        ..permissive()
    };
    let (registry, store) = setup(config);
    let examples = vec!["1,3,0".into(), "1,1,0".into(), "1,2,0".into()];
    registry
        .register(entry("a", "1,0,0").with_examples(examples))
        .await
        .unwrap();

    let decision = registry
        .record_feedback("1,0,5", "a", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert!(matches!(decision, FeedbackDecision::Added { ref evicted } if evicted.len() == 3));

    let stored = store.get("a").await.unwrap().unwrap();
    assert_eq!(stored.content, ["0,1,0", "0,0,1"]);
    assert_eq!(stored.examples, ["1,0,5"]);
    let embeddings = store.get_embeddings("a").await.unwrap().unwrap();
    let expected: Vec<Vec<f32>> = ["1,0,0", "0,1,0", "0,0,1", "1,0,5"]
        .into_iter()
        .map(v)
        .collect();
    assert_eq!(embeddings, expected);
}

#[tokio::test]
async fn eviction_leaves_remaining_examples_aligned() {
    let config = RegistryConfig {
        max_examples_per_entry: 1,
        eviction_policy: EvictionPolicy::MostRedundantWithBase,
        ..permissive()
    };
    let (registry, store) = setup(config);
    // redundancy with the base: b (0.71) > c (0.45) > a (0.32)
    let examples = vec!["1,3,0".into(), "1,1,0".into(), "1,2,0".into()];
    registry
        .register(entry("a", "1,0,0").with_examples(examples))
        .await
        .unwrap();
    registry
        .record_feedback("1,0,5", "a", FeedbackOutcome::Success)
        .await
        .unwrap();
    let embeddings = store.get_embeddings("a").await.unwrap().unwrap();
    let stored = store.get("a").await.unwrap().unwrap();
    for (i, example) in stored.examples.iter().enumerate() {
        assert_eq!(embeddings[3 + i], v(example));
    }
    assert_eq!(embeddings.len(), 4);
}

#[tokio::test]
async fn eviction_order_ignores_content_when_ranking_examples() {
    let config = RegistryConfig {
        max_examples_per_entry: 2,
        eviction_policy: EvictionPolicy::MostRedundantWithBase,
        ..permissive()
    };
    let (registry, store) = setup(config);
    let examples = vec!["1,3,0".into(), "1,1,0".into(), "1,2,0".into()];
    registry
        .register(entry("a", "1,0,0").with_examples(examples))
        .await
        .unwrap();
    // 3 examples, cap 2: evicts b, then c (cap reached again before adding), keeps a
    registry
        .record_feedback("1,0,5", "a", FeedbackOutcome::Success)
        .await
        .unwrap();
    let stored = store.get("a").await.unwrap().unwrap();
    assert_eq!(stored.examples, ["1,3,0", "1,0,5"]);
    let embeddings = store.get_embeddings("a").await.unwrap().unwrap();
    assert_eq!(embeddings[3], v("1,3,0"));
    assert_eq!(embeddings[4], v("1,0,5"));
}

#[tokio::test]
async fn learned_example_lands_after_content_and_earlier_examples() {
    let (registry, store) = setup(permissive());
    registry
        .register(entry("a", "1,0,0").with_examples(vec!["1,1,0".into()]))
        .await
        .unwrap();
    registry
        .record_feedback("1,0,5", "a", FeedbackOutcome::Success)
        .await
        .unwrap();
    let embeddings = store.get_embeddings("a").await.unwrap().unwrap();
    assert_eq!(embeddings.len(), 5);
    assert_eq!(embeddings[3], v("1,1,0"));
    assert_eq!(embeddings[4], v("1,0,5"));
}

#[tokio::test]
async fn re_register_does_not_re_embed_unchanged_content() {
    let store = Arc::new(InMemoryStore::new());
    let embedder = Arc::new(RecordingEmbedder::default());
    let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
    let registry = SemanticRegistry::new(store.clone(), embedder.clone(), retriever)
        .with_formatter(ContentFormatter);

    registry.register(entry("a", "1,0,0")).await.unwrap();
    {
        let batches = embedder.batches.lock().unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], ["1,0,0", "0,1,0", "0,0,1"]);
    }

    registry.register(entry("a", "1,0,1")).await.unwrap();
    let batches = embedder.batches.lock().unwrap();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[1], ["1,0,1"]);
}

#[tokio::test]
async fn a_stored_entry_without_content_loads() {
    let json = r#"{"id":"a","text":"1,0,0","payload":null}"#;
    let e: Entry<()> = serde_json::from_str(json).unwrap();
    assert!(e.content.is_empty());

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("entries.json");
    let file = r#"[{"entry":{"id":"a","text":"1,0,0","examples":["1,1,0"],"payload":null},"embeddings":[[1.0,0.0,0.0],[1.0,1.0,0.0]]}]"#;
    std::fs::write(&path, file).unwrap();
    let store = JsonStore::<()>::load_or_create(&path).await.unwrap();
    let loaded = store.get("a").await.unwrap().unwrap();
    assert!(loaded.content.is_empty());
    assert_eq!(loaded.examples, ["1,1,0"]);
}
