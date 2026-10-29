mod common;

use std::sync::Arc;

use common::{permissive, registry, registry_with_strategy, tool};
use serde_json::json;
use tools_rag::defaults::cosine_retriever::CosineRetriever;
use tools_rag::defaults::memory_store::InMemoryToolStore;
use tools_rag::embedder::DummyEmbedder;
use tools_rag::formatter::DefaultToolFormatter;
use tools_rag::registry::ToolRegistry;
use tools_rag::traits::ToolStore;
use tools_rag::types::{
    EvictionPolicy, FeedbackDecision, FeedbackOutcome, RegistryConfig, ScoringStrategy,
    ToolDefinition,
};

async fn examples_of(store: &InMemoryToolStore, name: &str) -> Vec<String> {
    store.get(name).await.unwrap().unwrap().examples
}

#[tokio::test]
async fn test_tool_registration_and_search() {
    let store = Arc::new(InMemoryToolStore::new());
    let embedder = Arc::new(DummyEmbedder::new(384));
    let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
    let formatter = Box::new(DefaultToolFormatter);

    let registry = ToolRegistry::new(store, embedder, retriever, formatter);

    let sample_tool = ToolDefinition::new(
        "get_weather",
        "Fetch weather info for a city",
        json!({
            "type": "object",
            "properties": {
                "location": { "type": "string", "description": "City name" }
            }
        }),
    );

    registry.register(sample_tool).await.unwrap();

    let results = registry.search("weather in Paris", 1).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].tool.name, "get_weather");
}

#[tokio::test]
async fn search_ranks_by_similarity_and_respects_limit() {
    let (registry, _) = registry(permissive());
    registry.register(tool("x", "1,0,0")).await.unwrap();
    registry.register(tool("y", "0,1,0")).await.unwrap();
    registry.register(tool("xy", "1,1,0")).await.unwrap();

    let results = registry.search("1,0.1,0", 2).await.unwrap();
    let names: Vec<_> = results.iter().map(|r| r.tool.name.as_str()).collect();
    assert_eq!(names, ["x", "xy"]);
    assert!(results[0].score > results[1].score);
}

#[tokio::test]
async fn search_drops_results_below_min_search_score() {
    let (registry, _) = registry(RegistryConfig {
        min_search_score: 0.5,
        ..permissive()
    });
    registry.register(tool("x", "1,0,0")).await.unwrap();
    registry.register(tool("y", "0,1,0")).await.unwrap();

    let results = registry.search("1,0,0", 10).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].tool.name, "x");
}

#[tokio::test]
async fn reregistering_a_tool_replaces_it() {
    let (registry, store) = registry(permissive());
    registry.register(tool("x", "1,0,0")).await.unwrap();
    registry.register(tool("x", "0,1,0")).await.unwrap();

    let all = store.list_all().await.unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].1, vec![vec![0.0, 1.0, 0.0]]);
}

#[tokio::test]
async fn success_feedback_adds_example_and_improves_ranking() {
    let (registry, store) = registry(permissive());
    registry.register(tool("logs", "1,0,0")).await.unwrap();
    registry
        .register(tool("metrics", "0.6,0.8,0"))
        .await
        .unwrap();

    // Avant feedback, "metrics" gagne sur cette requête
    let before = registry.search("0.5,1,0.3", 1).await.unwrap();
    assert_eq!(before[0].tool.name, "metrics");

    let decision = registry
        .record_feedback(
            "0.5,1,0.3",
            "metrics",
            FeedbackOutcome::CorrectedTo {
                correct_tool_name: "logs".into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(decision, FeedbackDecision::Added { evicted: vec![] });
    assert_eq!(examples_of(&store, "logs").await, ["0.5,1,0.3"]);
    assert!(examples_of(&store, "metrics").await.is_empty());

    let after = registry.search("0.5,1,0.3", 1).await.unwrap();
    assert_eq!(after[0].tool.name, "logs");
    assert!((after[0].score - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn duplicate_query_is_rejected() {
    let (registry, store) = registry(RegistryConfig {
        dedup_similarity_threshold: 0.9,
        ..permissive()
    });
    registry.register(tool("x", "1,0,0")).await.unwrap();

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
    registry.register(tool("x", "1,0,0")).await.unwrap();

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
    registry.register(tool("x", "1,0,0")).await.unwrap();

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
    registry.register(tool("x", "1,0,0")).await.unwrap();
    registry.register(tool("y", "0,0,1")).await.unwrap();

    let decision = registry
        .record_feedback(
            "0,1,0",
            "y",
            FeedbackOutcome::CorrectedTo {
                correct_tool_name: "x".into(),
            },
        )
        .await
        .unwrap();
    assert!(decision.is_added());
    assert_eq!(examples_of(&store, "x").await, ["0,1,0"]);
}

#[tokio::test]
async fn unknown_tool_is_reported() {
    let (registry, _) = registry(permissive());
    let decision = registry
        .record_feedback("1,0,0", "missing", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert_eq!(decision, FeedbackDecision::ToolNotFound);
}

#[tokio::test]
async fn zero_max_examples_disables_feedback() {
    let (registry, store) = registry(RegistryConfig {
        max_examples_per_tool: 0,
        ..permissive()
    });
    registry.register(tool("x", "1,0,0")).await.unwrap();

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
        max_examples_per_tool: 2,
        eviction_policy: EvictionPolicy::Fifo,
        ..permissive()
    });
    registry.register(tool("x", "1,0,0")).await.unwrap();

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
    // Les embeddings restent alignés sur les exemples
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
        max_examples_per_tool: 2,
        eviction_policy: EvictionPolicy::MostRedundantWithBase,
        ..permissive()
    });
    let seeded = tool("x", "1,0,0").with_examples(vec!["0.1,0.9,0".into(), "0.9,0.1,0".into()]);
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
        max_examples_per_tool: 1,
        eviction_policy: EvictionPolicy::RejectWhenFull,
        ..permissive()
    });
    registry
        .register(tool("x", "1,0,0").with_examples(vec!["0,1,0".into()]))
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
async fn eviction_shrinks_tools_registered_above_the_cap() {
    let (registry, store) = registry(RegistryConfig {
        max_examples_per_tool: 2,
        ..permissive()
    });
    let seeded =
        tool("x", "1,0,0").with_examples(vec!["0,1,0".into(), "0,0,1".into(), "0,1,1".into()]);
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
async fn weighted_max_sim_dampens_example_heavy_tools() {
    let setup = |strategy| async move {
        let (registry, _) = registry_with_strategy(permissive(), strategy);
        // "canonical" : description proche de la requête (cos ≈ 0.70)
        registry
            .register(tool("canonical", "0.7,0.714,0"))
            .await
            .unwrap();
        // "magnet" : description hors-sujet, mais un exemple appris proche (cos = 0.8)
        registry
            .register(tool("magnet", "0,1,0").with_examples(vec!["0.8,0.6,0".into()]))
            .await
            .unwrap();
        registry.search("1,0,0", 1).await.unwrap()[0]
            .tool
            .name
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
async fn meta_tool_is_search_tools() {
    let (registry, _) = registry(permissive());
    let meta = registry.meta_tool_definition();
    assert_eq!(meta.name, "search_tools");
    assert_eq!(meta.parameters["required"], json!(["query"]));
}

#[tokio::test]
async fn search_reports_which_example_matched() {
    let (registry, _) = registry(permissive());
    registry
        .register(tool("x", "1,0,0").with_examples(vec!["0,1,0".into(), "0,0,1".into()]))
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
        max_examples_per_tool: 2,
        eviction_policy: EvictionPolicy::LeastRecentlyUsed,
        ..permissive()
    });
    registry
        .register(tool("x", "1,0,0").with_examples(vec!["0,1,0".into(), "0,0,1".into()]))
        .await
        .unwrap();

    // Le plus ancien exemple ("0,1,0") est le seul à servir : c'est "0,0,1" qui part
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

    // L'exemple tout juste ajouté compte comme utilisé : "0,1,0", touché avant, part ensuite
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
        max_examples_per_tool: 2,
        eviction_policy: EvictionPolicy::LeastRecentlyUsed,
        ..permissive()
    });
    registry
        .register(tool("x", "1,0,0").with_examples(vec!["0,1,0".into(), "0,0,1".into()]))
        .await
        .unwrap();

    registry
        .record_feedback("0,1,1", "x", FeedbackOutcome::Success)
        .await
        .unwrap();
    assert_eq!(examples_of(&store, "x").await, ["0,0,1", "0,1,1"]);
}
