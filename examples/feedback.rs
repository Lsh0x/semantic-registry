//! The feedback loop, with an embedding cache so feedback reuses the search's vectors.
//!
//! cargo run --features candle,cache --example feedback

use std::sync::Arc;

use semantic_registry::embedder::{CachedEmbedder, CandleEmbedder};
use semantic_registry::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let embedder = Arc::new(CachedEmbedder::new(
        Arc::new(CandleEmbedder::minilm_v2().await?),
        "all-MiniLM-L6-v2",
        1024,
    ));
    let registry: SemanticRegistry<()> =
        SemanticRegistry::in_memory(embedder).with_config(RegistryConfig {
            max_examples_per_entry: 5,
            ..RegistryConfig::default()
        });

    // No example yet: only the description is indexed
    let id = "get_k8s_pod_logs";
    registry
        .register(Entry::new(
            id,
            "Fetches stdout and stderr logs for a Kubernetes pod in a given namespace.",
            (),
        ))
        .await?;

    let query = "My payment pod is crashing, I need stdout logs";
    let before = registry.search(query, 1).await?;
    println!("1. search {query:?}");
    println!("   -> {} ({:.4})", before[0].entry.id, before[0].score);

    let decision = registry
        .record_feedback(query, id, FeedbackOutcome::Success)
        .await?;
    println!("2. feedback Success -> {decision:?}");

    let similar = "Payment container crashing, get stdout output";
    let after = registry.search(similar, 1).await?;
    println!("3. search {similar:?}");
    println!(
        "   -> {} ({:.4}, matched example {:?})",
        after[0].entry.id, after[0].score, after[0].matched_example
    );

    let again = registry
        .record_feedback(query, id, FeedbackOutcome::Success)
        .await?;
    println!("4. same feedback again -> {again:?}");

    Ok(())
}
