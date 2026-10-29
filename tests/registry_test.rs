mod common;

use std::sync::Arc;

use common::{entry, permissive, registry, registry_with_strategy, VecEmbedder};
use semantic_registry::prelude::*;
use semantic_registry::InMemoryStore;

async fn examples_of(store: &InMemoryStore<()>, id: &str) -> Vec<String> {
    store.get(id).await.unwrap().unwrap().examples
}

#[tokio::test]
async fn in_memory_registry_returns_payloads() {
    let registry = SemanticRegistry::in_memory(Arc::new(VecEmbedder));
    registry
        .register(Entry::new(
            "weather",
            "1,0,0",
            serde_json::json!({ "endpoint": "/w" }),
        ))
        .await
        .unwrap();
    registry
        .register(Entry::new(
            "fx",
            "0,1,0",
            serde_json::json!({ "endpoint": "/fx" }),
        ))
        .await
        .unwrap();

    let hits = registry.search("1,0.2,0", 1).await.unwrap();
    assert_eq!(hits[0].entry.id, "weather");
    assert_eq!(hits[0].entry.payload["endpoint"], "/w");
}

#[tokio::test]
async fn search_ranks_by_similarity_and_respects_limit() {
    let (registry, _) = registry(permissive());
    registry.register(entry("x", "1,0,0")).await.unwrap();
    registry.register(entry("y", "0,1,0")).await.unwrap();
    registry.register(entry("xy", "1,1,0")).await.unwrap();

    let results = registry.search("1,0.1,0", 2).await.unwrap();
    let names: Vec<_> = results.iter().map(|r| r.entry.id.as_str()).collect();
    assert_eq!(names, ["x", "xy"]);
    assert!(results[0].score > results[1].score);
}

#[tokio::test]
async fn search_drops_results_below_min_search_score() {
    let (registry, _) = registry(RegistryConfig {
        min_search_score: 0.5,
        ..permissive()
    });
    registry.register(entry("x", "1,0,0")).await.unwrap();
    registry.register(entry("y", "0,1,0")).await.unwrap();

    let results = registry.search("1,0,0", 10).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].entry.id, "x");
}

#[tokio::test]
async fn reregistering_an_entry_replaces_it() {
    let (registry, store) = registry(permissive());
    registry.register(entry("x", "1,0,0")).await.unwrap();
    registry.register(entry("x", "0,1,0")).await.unwrap();

    let all = store.list_all().await.unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].1, vec![vec![0.0, 1.0, 0.0]]);
}

#[tokio::test]
async fn success_feedback_adds_example_and_improves_ranking() {
    let (registry, store) = registry(permissive());
    registry.register(entry("logs", "1,0,0")).await.unwrap();
    registry
        .register(entry("metrics", "0.6,0.8,0"))
        .await
        .unwrap();

    // Before feedback, "metrics" wins this query
    let before = registry.search("0.5,1,0.3", 1).await.unwrap();
    assert_eq!(before[0].entry.id, "metrics");

    let decision = registry
        .record_feedback(
            "0.5,1,0.3",
            "metrics",
            FeedbackOutcome::CorrectedTo {
                correct_id: "logs".into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(decision, FeedbackDecision::Added { evicted: vec![] });
    assert_eq!(examples_of(&store, "logs").await, ["0.5,1,0.3"]);
    assert!(examples_of(&store, "metrics").await.is_empty());

    let after = registry.search("0.5,1,0.3", 1).await.unwrap();
    assert_eq!(after[0].entry.id, "logs");
    assert!((after[0].score - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn duplicate_query_is_rejected() {
    let (registry, store) = registry(RegistryConfig {
        dedup_similarity_threshold: 0.9,
        ..permissive()
    });
    registry.register(entry("x", "1,0,0")).await.unwrap();

    let decision = registry
        .record_feedback("1,0.1,0", "x", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert!(matches!(decision, FeedbackDecision::Duplicate { similarity } if similarity > 0.99));
    assert!(examples_of(&store, "x").await.is_empty());
}

#[tokio::test]
async fn duplicate_check_covers_learned_examples() {
    let (registry, _) = registry(RegistryConfig {
        dedup_similarity_threshold: 0.9,
        ..permissive()
    });
    registry.register(entry("x", "1,0,0")).await.unwrap();

    let first = registry
        .record_feedback("0,1,0", "x", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert!(first.is_added());
    let second = registry
        .record_feedback("0,1,0.05", "x", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert!(matches!(second, FeedbackDecision::Duplicate { .. }));
}

#[tokio::test]
async fn success_below_min_feedback_score_is_rejected() {
    let (registry, store) = registry(RegistryConfig {
        min_feedback_score: 0.4,
        ..permissive()
    });
    registry.register(entry("x", "1,0,0")).await.unwrap();

    let decision = registry
        .record_feedback("0.2,1,0", "x", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert!(matches!(decision, FeedbackDecision::BelowMinScore { score } if score < 0.4));
    assert!(examples_of(&store, "x").await.is_empty());
}

#[tokio::test]
async fn correction_bypasses_min_feedback_score() {
    let (registry, store) = registry(RegistryConfig {
        min_feedback_score: 0.4,
        ..permissive()
    });
    registry.register(entry("x", "1,0,0")).await.unwrap();
    registry.register(entry("y", "0,0,1")).await.unwrap();

    let decision = registry
        .record_feedback(
            "0,1,0",
            "y",
            FeedbackOutcome::CorrectedTo {
                correct_id: "x".into(),
            },
        )
        .await
        .unwrap();
    assert!(decision.is_added());
    assert_eq!(examples_of(&store, "x").await, ["0,1,0"]);
}

#[tokio::test]
async fn unknown_entry_is_reported() {
    let (registry, _) = registry(permissive());
    let decision = registry
        .record_feedback("1,0,0", "missing", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert_eq!(decision, FeedbackDecision::EntryNotFound);
}

#[tokio::test]
async fn zero_max_examples_disables_feedback() {
    let (registry, store) = registry(RegistryConfig {
        max_examples_per_entry: 0,
        ..permissive()
    });
    registry.register(entry("x", "1,0,0")).await.unwrap();

    let decision = registry
        .record_feedback("0,1,0", "x", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert_eq!(decision, FeedbackDecision::Full);
    assert!(examples_of(&store, "x").await.is_empty());
}

#[tokio::test]
async fn fifo_evicts_oldest_example() {
    let (registry, store) = registry(RegistryConfig {
        max_examples_per_entry: 2,
        eviction_policy: EvictionPolicy::Fifo,
        ..permissive()
    });
    registry.register(entry("x", "1,0,0")).await.unwrap();

    for q in ["0,1,0", "0,0,1"] {
        let d = registry
            .record_feedback(q, "x", FeedbackOutcome::Success)
            .await
            .unwrap();
        assert!(d.is_added());
    }
    let decision = registry
        .record_feedback("0,1,1", "x", FeedbackOutcome::Success)
        .await
        .unwrap();

    assert_eq!(
        decision,
        FeedbackDecision::Added {
            evicted: vec!["0,1,0".into()]
        }
    );
    assert_eq!(examples_of(&store, "x").await, ["0,0,1", "0,1,1"]);
    // Embeddings stay aligned with examples
    assert_eq!(
        store.get_embeddings("x").await.unwrap().unwrap(),
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0]
        ]
    );
}

#[tokio::test]
async fn most_redundant_with_base_evicts_closest_to_description() {
    let (registry, store) = registry(RegistryConfig {
        max_examples_per_entry: 2,
        eviction_policy: EvictionPolicy::MostRedundantWithBase,
        ..permissive()
    });
    let seeded = entry("x", "1,0,0").with_examples(vec!["0.1,0.9,0".into(), "0.9,0.1,0".into()]);
    registry.register(seeded).await.unwrap();

    let decision = registry
        .record_feedback("0,0,1", "x", FeedbackOutcome::Success)
        .await
        .unwrap();

    assert_eq!(
        decision,
        FeedbackDecision::Added {
            evicted: vec!["0.9,0.1,0".into()]
        }
    );
    assert_eq!(examples_of(&store, "x").await, ["0.1,0.9,0", "0,0,1"]);
}

#[tokio::test]
async fn reject_when_full_leaves_store_untouched() {
    let (registry, store) = registry(RegistryConfig {
        max_examples_per_entry: 1,
        eviction_policy: EvictionPolicy::RejectWhenFull,
        ..permissive()
    });
    registry
        .register(entry("x", "1,0,0").with_examples(vec!["0,1,0".into()]))
        .await
        .unwrap();

    let decision = registry
        .record_feedback("0,0,1", "x", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert_eq!(decision, FeedbackDecision::Full);
    assert_eq!(examples_of(&store, "x").await, ["0,1,0"]);
}

#[tokio::test]
async fn eviction_shrinks_entries_registered_above_the_cap() {
    let (registry, store) = registry(RegistryConfig {
        max_examples_per_entry: 2,
        ..permissive()
    });
    let seeded =
        entry("x", "1,0,0").with_examples(vec!["0,1,0".into(), "0,0,1".into(), "0,1,1".into()]);
    registry.register(seeded).await.unwrap();

    let decision = registry
        .record_feedback("1,1,1", "x", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert_eq!(
        decision,
        FeedbackDecision::Added {
            evicted: vec!["0,1,0".into(), "0,0,1".into()]
        }
    );
    assert_eq!(examples_of(&store, "x").await, ["0,1,1", "1,1,1"]);
    assert_eq!(store.get_embeddings("x").await.unwrap().unwrap().len(), 3);
}

#[tokio::test]
async fn weighted_max_sim_dampens_example_heavy_entries() {
    let setup = |strategy| async move {
        let (registry, _) = registry_with_strategy(permissive(), strategy);
        // "canonical": text close to the query (cos ≈ 0.70)
        registry
            .register(entry("canonical", "0.7,0.714,0"))
            .await
            .unwrap();
        // "magnet": off-topic text, but a learned example close to the query (cos = 0.8)
        registry
            .register(entry("magnet", "0,1,0").with_examples(vec!["0.8,0.6,0".into()]))
            .await
            .unwrap();
        registry.search("1,0,0", 1).await.unwrap()[0]
            .entry
            .id
            .clone()
    };

    assert_eq!(setup(ScoringStrategy::MaxSim).await, "magnet");
    assert_eq!(
        setup(ScoringStrategy::WeightedMaxSim {
            example_weight: 0.8
        })
        .await,
        "canonical"
    );
}

#[tokio::test]
async fn search_reports_which_example_matched() {
    let (registry, _) = registry(permissive());
    registry
        .register(entry("x", "1,0,0").with_examples(vec!["0,1,0".into(), "0,0,1".into()]))
        .await
        .unwrap();

    assert_eq!(
        registry.search("0,0.1,1", 1).await.unwrap()[0].matched_example,
        Some(1)
    );
    assert_eq!(
        registry.search("1,0,0.1", 1).await.unwrap()[0].matched_example,
        None
    );
}

#[tokio::test]
async fn least_recently_used_evicts_example_not_hit_by_searches() {
    let (registry, store) = registry(RegistryConfig {
        max_examples_per_entry: 2,
        eviction_policy: EvictionPolicy::LeastRecentlyUsed,
        ..permissive()
    });
    registry
        .register(entry("x", "1,0,0").with_examples(vec!["0,1,0".into(), "0,0,1".into()]))
        .await
        .unwrap();

    // The oldest example ("0,1,0") is the only one used, so "0,0,1" goes
    registry.search("0,1,0.1", 1).await.unwrap();
    let decision = registry
        .record_feedback("0,1,1", "x", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert_eq!(
        decision,
        FeedbackDecision::Added {
            evicted: vec!["0,0,1".into()]
        }
    );

    // The example just added counts as used: "0,1,0", touched earlier, goes next
    let decision = registry
        .record_feedback("-1,1,0", "x", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert_eq!(
        decision,
        FeedbackDecision::Added {
            evicted: vec!["0,1,0".into()]
        }
    );
    assert_eq!(examples_of(&store, "x").await, ["0,1,1", "-1,1,0"]);
}

#[tokio::test]
async fn least_recently_used_without_history_behaves_like_fifo() {
    let (registry, store) = registry(RegistryConfig {
        max_examples_per_entry: 2,
        eviction_policy: EvictionPolicy::LeastRecentlyUsed,
        ..permissive()
    });
    registry
        .register(entry("x", "1,0,0").with_examples(vec!["0,1,0".into(), "0,0,1".into()]))
        .await
        .unwrap();

    registry
        .record_feedback("0,1,1", "x", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert_eq!(examples_of(&store, "x").await, ["0,0,1", "0,1,1"]);
}

struct Shift;

impl Formatter<()> for Shift {
    fn canonical_text(&self, entry: &Entry<()>) -> String {
        entry.text.clone()
    }

    // Examples are embedded with one extra component: "a,b" -> "a,b,1"
    fn example_text(&self, _entry: &Entry<()>, example: &str) -> String {
        format!("{example},1")
    }
}

#[tokio::test]
async fn feedback_examples_go_through_the_formatter() {
    let store = Arc::new(InMemoryStore::new());
    let embedder = Arc::new(VecEmbedder);
    let retriever = Arc::new(semantic_registry::CosineRetriever::new(
        store.clone(),
        embedder.clone(),
    ));
    let registry = SemanticRegistry::new(store.clone(), embedder, retriever)
        .with_config(permissive())
        .with_formatter(Shift);
    registry
        .register(entry("x", "1,0,0").with_examples(vec!["0,1".into()]))
        .await
        .unwrap();

    registry
        .record_feedback("1,1", "x", FeedbackOutcome::Success)
        .await
        .unwrap();

    // Seeded and learned examples are embedded the same way; the text stays raw
    assert_eq!(
        store.get_embeddings("x").await.unwrap().unwrap(),
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0]
        ]
    );
    assert_eq!(examples_of(&store, "x").await, ["0,1", "1,1"]);
}

fn recording_registry() -> (
    SemanticRegistry<()>,
    Arc<InMemoryStore<()>>,
    Arc<common::RecordingEmbedder>,
) {
    let store = Arc::new(InMemoryStore::new());
    let embedder = Arc::new(common::RecordingEmbedder::default());
    let retriever = Arc::new(semantic_registry::CosineRetriever::new(
        store.clone(),
        embedder.clone(),
    ));
    let registry =
        SemanticRegistry::new(store.clone(), embedder.clone(), retriever).with_config(permissive());
    (registry, store, embedder)
}

#[tokio::test]
async fn register_embeds_all_chunks_in_one_batch() {
    let (registry, _, embedder) = recording_registry();
    registry
        .register(entry("x", "1,0,0").with_examples(vec!["0,1,0".into(), "0,0,1".into()]))
        .await
        .unwrap();

    assert_eq!(
        *embedder.batches.lock().unwrap(),
        [vec!["1,0,0", "0,1,0", "0,0,1"]]
    );
}

#[tokio::test]
async fn reregistering_reuses_vectors_of_unchanged_chunks() {
    let (registry, store, embedder) = recording_registry();
    registry
        .register(entry("x", "1,0,0").with_examples(vec!["0,1,0".into()]))
        .await
        .unwrap();

    // Same text and example, one new example: only the new one is embedded
    registry
        .register(entry("x", "1,0,0").with_examples(vec!["0,1,0".into(), "0,0,1".into()]))
        .await
        .unwrap();
    // Nothing changed: nothing is embedded
    registry
        .register(entry("x", "1,0,0").with_examples(vec!["0,1,0".into(), "0,0,1".into()]))
        .await
        .unwrap();

    assert_eq!(
        *embedder.batches.lock().unwrap(),
        [vec!["1,0,0", "0,1,0"], vec!["0,0,1"]]
    );
    assert_eq!(
        store.get_embeddings("x").await.unwrap().unwrap(),
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0]
        ]
    );
}

#[tokio::test]
async fn reregistering_reuses_vectors_learned_through_feedback() {
    let (registry, _, embedder) = recording_registry();
    registry.register(entry("x", "1,0,0")).await.unwrap();
    registry
        .record_feedback("0,1,0", "x", FeedbackOutcome::Success)
        .await
        .unwrap();

    // Re-registering with the learned example kept: no new embedding
    registry
        .register(entry("x", "1,0,0").with_examples(vec!["0,1,0".into()]))
        .await
        .unwrap();

    assert_eq!(
        *embedder.batches.lock().unwrap(),
        [vec!["1,0,0"], vec!["0,1,0"]]
    );
}

#[tokio::test]
async fn remove_deletes_the_entry_and_its_vectors() {
    let (registry, store) = registry(permissive());
    registry.register(entry("x", "1,0,0")).await.unwrap();
    registry.register(entry("y", "0,1,0")).await.unwrap();

    assert!(registry.remove("x").await.unwrap());
    assert!(!registry.remove("x").await.unwrap());

    assert!(store.get("x").await.unwrap().is_none());
    assert!(store.get_embeddings("x").await.unwrap().is_none());
    let hits = registry.search("1,0,0", 10).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].entry.id, "y");
    assert_eq!(
        registry
            .record_feedback("1,0,0", "x", FeedbackOutcome::Success)
            .await
            .unwrap(),
        FeedbackDecision::EntryNotFound
    );
}

#[tokio::test]
async fn removed_entry_starts_fresh_when_registered_again() {
    let (registry, store, embedder) = recording_registry();
    registry.register(entry("x", "1,0,0")).await.unwrap();
    registry.remove("x").await.unwrap();
    registry.register(entry("x", "1,0,0")).await.unwrap();

    // No stale vectors: the text is embedded again
    assert_eq!(embedder.batches.lock().unwrap().len(), 2);
    assert!(store.get("x").await.unwrap().unwrap().examples.is_empty());
}

#[tokio::test]
async fn least_recently_used_survives_a_restart_with_a_persistent_store() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("entries.json");
    let config = RegistryConfig {
        max_examples_per_entry: 2,
        eviction_policy: EvictionPolicy::LeastRecentlyUsed,
        ..permissive()
    };
    let open = |config: RegistryConfig| {
        let path = path.clone();
        async move {
            let store = Arc::new(
                semantic_registry::JsonStore::<()>::load_or_create(path)
                    .await
                    .unwrap(),
            );
            let embedder = Arc::new(VecEmbedder);
            let retriever = Arc::new(semantic_registry::CosineRetriever::new(
                store.clone(),
                embedder.clone(),
            ));
            let registry =
                SemanticRegistry::new(store.clone(), embedder, retriever).with_config(config);
            (registry, store)
        }
    };

    // First run: the oldest example is the one searches keep finding
    {
        let (registry, store) = open(config.clone()).await;
        registry
            .register(entry("x", "1,0,0").with_examples(vec!["0,1,0".into(), "0,0,1".into()]))
            .await
            .unwrap();
        let hit = registry.search("0,1,0.1", 1).await.unwrap().remove(0);
        assert_eq!(hit.matched_example, Some(0));
        store.flush().await.unwrap();
    }

    // Second run, fresh registry: without the persisted use, LRU would fall back to the
    // oldest example ("0,1,0"); with it, the never-used "0,0,1" goes
    let (registry, store) = open(config).await;
    let decision = registry
        .record_feedback("0,1,1", "x", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert_eq!(
        decision,
        FeedbackDecision::Added {
            evicted: vec!["0,0,1".into()]
        }
    );
    assert_eq!(
        store.get("x").await.unwrap().unwrap().examples,
        ["0,1,0", "0,1,1"]
    );
}
