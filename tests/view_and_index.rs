//! The caller's view reaches the ranking (`search_in`, `retrieve_in`), and an index kept
//! entry by entry through `IndexSink` equals one rebuilt from `list_all`.

mod common;

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use common::{entry, permissive, VecEmbedder};
use semantic_registry::error::Result;
use semantic_registry::prelude::*;
use semantic_registry::{CosineRetriever, InMemoryStore, IndexedStore};

/// A registry over a plain retriever (list_all at each search)
fn plain() -> SemanticRegistry<()> {
    common::registry(permissive()).0
}

/// A registry over an indexed retriever kept through the sink, and that retriever
fn indexed(
    config: RegistryConfig,
) -> (
    SemanticRegistry<()>,
    Arc<CosineRetriever<()>>,
    Arc<InMemoryStore<()>>,
) {
    let inner = Arc::new(InMemoryStore::new());
    let embedder = Arc::new(VecEmbedder);
    let retriever = Arc::new(CosineRetriever::new(inner.clone(), embedder.clone()).with_index());
    let store = Arc::new(IndexedStore::new(inner.clone(), retriever.clone()));
    let registry = SemanticRegistry::new(store, embedder, retriever.clone()).with_config(config);
    (registry, retriever, inner)
}

/// `count` entries `e0…`, each further from the query `"1,0"` than the one before
async fn fill(registry: &SemanticRegistry<()>, count: usize) {
    for i in 0..count {
        registry
            .register(entry(&format!("e{i}"), &format!("10,{i}")))
            .await
            .unwrap();
    }
}

fn ids(results: &[SearchResult<()>]) -> Vec<&str> {
    results.iter().map(|r| r.entry.id.as_str()).collect()
}

fn view(ids: &[&str]) -> HashSet<String> {
    ids.iter().map(|s| s.to_string()).collect()
}

/// Both registries, filled with the same entries
async fn both(count: usize) -> [SemanticRegistry<()>; 2] {
    let plain = plain();
    let (indexed, _, _) = indexed(permissive());
    fill(&plain, count).await;
    fill(&indexed, count).await;
    [plain, indexed]
}

#[tokio::test]
async fn a_view_admitting_one_of_a_hundred_returns_it_at_limit_one() {
    for registry in both(100).await {
        let only = view(&["e99"]);
        let hits = registry.search_in("1,0", 1, &only).await.unwrap();
        assert_eq!(ids(&hits), ["e99"]);

        // Ranking everything then filtering returns an empty page
        let mut filtered = registry.search("1,0", 1).await.unwrap();
        filtered.retain(|r| only.admits(&r.entry.id));
        assert!(filtered.is_empty());
    }
}

#[tokio::test]
async fn a_view_returns_the_best_of_its_subset_not_a_filtered_page() {
    for registry in both(10).await {
        let subset = view(&["e9", "e7", "e2", "e5"]);
        let hits = registry.search_in("1,0", 3, &subset).await.unwrap();
        assert_eq!(ids(&hits), ["e2", "e5", "e7"]);

        let mut filtered = registry.search("1,0", 3).await.unwrap();
        filtered.retain(|r| subset.admits(&r.entry.id));
        assert_eq!(ids(&filtered), ["e2"]);
    }
}

#[tokio::test]
async fn a_closure_is_a_view() {
    for registry in both(10).await {
        let odd = |id: &str| id.ends_with(['1', '3', '5', '7', '9']);
        let hits = registry.search_in("1,0", 2, &odd).await.unwrap();
        assert_eq!(ids(&hits), ["e1", "e3"]);
    }
}

#[tokio::test]
async fn an_empty_view_returns_nothing() {
    for registry in both(10).await {
        let none = view(&[]);
        assert!(registry
            .search_in("1,0", 10, &none)
            .await
            .unwrap()
            .is_empty());
        let nobody = |_: &str| false;
        assert!(registry
            .search_in("1,0", usize::MAX, &nobody)
            .await
            .unwrap()
            .is_empty());
    }
}

#[tokio::test]
async fn limit_zero_returns_nothing() {
    for registry in both(10).await {
        assert!(registry
            .search_in("1,0", 0, &AllEntries)
            .await
            .unwrap()
            .is_empty());
    }
}

#[tokio::test]
async fn a_view_admitting_everything_answers_as_search() {
    for registry in both(10).await {
        registry
            .register(entry("x", "0,1").with_examples(vec!["10,3".into()]))
            .await
            .unwrap();
        for limit in [1, 3, 11, 50] {
            let old = registry.search("10,3", limit).await.unwrap();
            let new = registry
                .search_in("10,3", limit, &AllEntries)
                .await
                .unwrap();
            assert_eq!(old, new);
        }
        let hit = &registry.search_in("10,3", 1, &AllEntries).await.unwrap()[0];
        assert_eq!(hit.entry.id, "e3");
    }
}

#[tokio::test]
async fn a_removed_id_is_not_found_even_when_the_view_admits_it() {
    for registry in both(5).await {
        assert!(registry.remove("e0").await.unwrap());
        let hits = registry
            .search_in("1,0", 5, &view(&["e0", "e4"]))
            .await
            .unwrap();
        assert_eq!(ids(&hits), ["e4"]);
    }
}

/// A 0.2 retriever: only `retrieve`, so `retrieve_in` is the default
struct OldRetriever(CosineRetriever<()>);

#[async_trait]
impl Retriever<()> for OldRetriever {
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<SearchResult<()>>> {
        self.0.retrieve(query, limit).await
    }
}

#[tokio::test]
async fn the_default_retrieve_in_widens_until_the_page_is_full() {
    let store = Arc::new(InMemoryStore::new());
    let embedder = Arc::new(VecEmbedder);
    let old = Arc::new(OldRetriever(CosineRetriever::new(
        store.clone(),
        embedder.clone(),
    )));
    let registry = SemanticRegistry::new(store, embedder, old.clone()).with_config(permissive());
    fill(&registry, 100).await;

    let hits = registry.search_in("1,0", 1, &view(&["e99"])).await.unwrap();
    assert_eq!(ids(&hits), ["e99"]);
    let subset = view(&["e9", "e7", "e2", "e5"]);
    let hits = registry.search_in("1,0", 3, &subset).await.unwrap();
    assert_eq!(ids(&hits), ["e2", "e5", "e7"]);
    assert!(registry
        .search_in("1,0", 3, &view(&[]))
        .await
        .unwrap()
        .is_empty());
    assert!(registry
        .search_in("1,0", 0, &AllEntries)
        .await
        .unwrap()
        .is_empty());
    assert_eq!(
        registry.search("1,0", 7).await.unwrap(),
        registry.search_in("1,0", 7, &AllEntries).await.unwrap()
    );
}

#[tokio::test]
async fn a_retriever_without_the_sink_still_sees_every_change() {
    // The plain retriever reads the store at each search, as in 0.2
    let registry = plain();
    fill(&registry, 3).await;
    registry.register(entry("e1", "0,1")).await.unwrap();
    let hits = registry.search_in("0,1", 1, &AllEntries).await.unwrap();
    assert_eq!(ids(&hits), ["e1"]);
}

#[tokio::test]
async fn feedback_learned_through_the_sink_reaches_the_next_search() {
    let (registry, _, _) = indexed(permissive());
    registry.register(entry("a", "1,0")).await.unwrap();
    registry.register(entry("b", "0,1")).await.unwrap();
    assert_eq!(ids(&registry.search("0.9,0.1", 1).await.unwrap()), ["a"]);

    let decision = registry
        .record_feedback(
            "0.9,0.1",
            "a",
            FeedbackOutcome::CorrectedTo {
                correct_id: "b".into(),
            },
        )
        .await
        .unwrap();
    assert!(matches!(decision, FeedbackDecision::Added { .. }));

    let hit = &registry.search("0.9,0.1", 1).await.unwrap()[0];
    assert_eq!(hit.entry.id, "b");
    assert_eq!(hit.matched_example, Some(0));
}

#[tokio::test]
async fn an_evicted_example_leaves_the_index() {
    let config = RegistryConfig {
        max_examples_per_entry: 1,
        eviction_policy: EvictionPolicy::Fifo,
        ..permissive()
    };
    let (registry, _, _) = indexed(config);
    registry.register(entry("a", "1,0")).await.unwrap();
    registry.register(entry("b", "0,1")).await.unwrap();
    let to_b = || FeedbackOutcome::CorrectedTo {
        correct_id: "b".into(),
    };
    registry
        .record_feedback("1,0.1", "a", to_b())
        .await
        .unwrap();
    registry
        .record_feedback("0.5,1", "a", to_b())
        .await
        .unwrap();

    // "1,0.1" was evicted from b: a wins it again
    assert_eq!(ids(&registry.search("1,0.1", 1).await.unwrap()), ["a"]);
}

/// Deterministic pseudo-random numbers
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, below: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % below
    }

    fn vector(&mut self) -> String {
        format!("{},{},{}", self.next(5) + 1, self.next(5), self.next(5))
    }
}

#[tokio::test]
async fn the_index_kept_through_the_sink_equals_one_rebuilt_from_list_all() {
    let config = RegistryConfig {
        max_examples_per_entry: 2,
        eviction_policy: EvictionPolicy::Fifo,
        dedup_similarity_threshold: 0.999,
        min_feedback_score: 0.0,
        ..RegistryConfig::default()
    };
    let (registry, kept, inner) = indexed(config);
    let mut rng = Lcg(176);
    for _ in 0..300 {
        let id = format!("e{}", rng.next(12));
        match rng.next(7) {
            0..=2 => {
                let examples = (0..rng.next(3)).map(|_| rng.vector()).collect();
                let e = entry(&id, &rng.vector()).with_examples(examples);
                registry.register(e).await.unwrap();
            }
            3 => {
                registry.remove(&id).await.unwrap();
            }
            4 => {
                let query = rng.vector();
                registry
                    .record_feedback(&query, &id, FeedbackOutcome::Success)
                    .await
                    .unwrap();
            }
            _ => {
                let query = rng.vector();
                let outcome = FeedbackOutcome::CorrectedTo { correct_id: id };
                registry
                    .record_feedback(&query, "e0", outcome)
                    .await
                    .unwrap();
            }
        }
    }

    let rebuilt = CosineRetriever::new(inner.clone(), Arc::new(VecEmbedder)).with_index();
    let held = rebuilt.load().await.unwrap();
    assert!(held > 0, "the sequence must leave entries");
    let kept_entries = kept.indexed().unwrap();
    assert_eq!(kept_entries.len(), held);
    for (a, b) in kept_entries.iter().zip(rebuilt.indexed().unwrap()) {
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1, "embeddings of {}", a.0.id);
    }

    for _ in 0..20 {
        let query = rng.vector();
        let a = kept.retrieve(&query, usize::MAX).await.unwrap();
        let b = rebuilt.retrieve(&query, usize::MAX).await.unwrap();
        assert_eq!(a, b, "query {query}");
    }
}

#[tokio::test]
async fn load_fills_the_index_once_and_unindexed_retrievers_ignore_the_sink() {
    let store = Arc::new(InMemoryStore::new());
    let embedder = Arc::new(VecEmbedder);
    store
        .insert(entry("a", "1,0"), vec![vec![1.0, 0.0]])
        .await
        .unwrap();

    let indexed = CosineRetriever::new(store.clone(), embedder.clone()).with_index();
    assert!(indexed.retrieve("1,0", 5).await.unwrap().is_empty());
    assert_eq!(indexed.load().await.unwrap(), 1);
    assert_eq!(ids(&indexed.retrieve("1,0", 5).await.unwrap()), ["a"]);
    // An upsert of an id the store does not hold forgets it
    indexed.upsert("ghost").await.unwrap();
    store.remove("a").await.unwrap();
    indexed.upsert("a").await.unwrap();
    assert!(indexed.retrieve("1,0", 5).await.unwrap().is_empty());

    let plain = CosineRetriever::new(store.clone(), embedder);
    assert_eq!(plain.load().await.unwrap(), 0);
    plain.upsert("a").await.unwrap();
    plain.forget("a").await.unwrap();
    assert!(plain.indexed().unwrap().is_empty());
    // Without an index, each search reads the store
    store
        .insert(entry("b", "1,0"), vec![vec![1.0, 0.0]])
        .await
        .unwrap();
    assert_eq!(ids(&plain.retrieve("1,0", 5).await.unwrap()), ["b"]);
}

#[tokio::test]
async fn each_write_of_the_indexed_store_reaches_the_index() {
    let inner: Arc<InMemoryStore<()>> = Arc::new(InMemoryStore::new());
    let retriever =
        Arc::new(CosineRetriever::new(inner.clone(), Arc::new(VecEmbedder)).with_index());
    let store = IndexedStore::new(inner.clone(), retriever.clone());
    let snapshot = || retriever.indexed().unwrap();

    store
        .insert(entry("a", "1,0"), vec![vec![1.0, 0.0]])
        .await
        .unwrap();
    assert_eq!(snapshot(), inner.list_all().await.unwrap());
    store
        .append_example("a", "0,1".into(), vec![0.0, 1.0])
        .await
        .unwrap();
    assert_eq!(snapshot(), inner.list_all().await.unwrap());
    store.remove_example("a", 0).await.unwrap();
    assert_eq!(snapshot(), inner.list_all().await.unwrap());
    assert!(store.remove("a").await.unwrap());
    assert!(snapshot().is_empty());
    // Writes to unknown ids leave the index as the store
    store
        .append_example("b", "0,1".into(), vec![0.0, 1.0])
        .await
        .unwrap();
    store.remove_example("b", 3).await.unwrap();
    assert!(!store.remove("b").await.unwrap());
    assert!(snapshot().is_empty());
}
