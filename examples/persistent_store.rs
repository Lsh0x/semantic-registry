//! Persist entries and their embeddings to a JSON file: the second run embeds nothing.
//!
//! cargo run --features candle --example persistent_store

use std::sync::Arc;

use semantic_registry::embedder::CandleEmbedder;
use semantic_registry::prelude::*;
use semantic_registry::{CosineRetriever, JsonStore};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "./target/semantic_registry_cache.json";
    let store = Arc::new(JsonStore::<String>::load_or_create(path).await?);
    let embedder = Arc::new(CandleEmbedder::minilm_v2().await?);
    let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
    let registry = SemanticRegistry::new(store.clone(), embedder, retriever);

    let id = "convert_currency";
    if store.get(id).await?.is_none() {
        println!("{id} is not on disk yet: embedding it and saving to {path}");
        let entry = Entry::new(
            id,
            "Convert monetary amounts between foreign currencies.",
            "https://api.example.com/fx".to_string(), // payload: anything serializable
        )
        .with_examples(vec![
            "How much is $100 USD in EUR?".to_string(),
            "Convert 50 GBP to JPY".to_string(),
        ]);
        registry.register(entry).await?;
    } else {
        println!("{id} loaded from {path}: no embedding computed");
    }

    let query = "I need to exchange 500 dollars to euros";
    let hit = registry.search(query, 1).await?.remove(0);
    println!(
        "{query:?} -> {} ({:.4}), payload {}",
        hit.entry.id, hit.score, hit.entry.payload
    );

    Ok(())
}
