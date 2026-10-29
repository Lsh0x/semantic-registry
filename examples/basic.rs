//! Register entries with examples, then search them by intent.
//!
//! cargo run --features candle --example basic

use std::sync::Arc;

use semantic_registry::embedder::CandleEmbedder;
use semantic_registry::prelude::*;
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading all-MiniLM-L6-v2...");
    let embedder = Arc::new(CandleEmbedder::minilm_v2().await?);

    // The payload is whatever the caller needs back: here, a JSON Schema
    let registry: SemanticRegistry<Value> = SemanticRegistry::in_memory(embedder);

    let entries = vec![
        Entry::new(
            "get_weather",
            "Fetch current weather conditions or forecasts for a location.",
            json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
        )
        .with_examples(vec![
            "What's the weather like in Tokyo right now?".to_string(),
            "Is it going to rain in Paris tomorrow?".to_string(),
        ]),
        Entry::new(
            "convert_currency",
            "Convert monetary amounts between currencies.",
            json!({ "type": "object", "properties": { "amount": { "type": "number" } } }),
        )
        .with_examples(vec![
            "How much is $100 in euros?".to_string(),
            "Convert 5000 JPY to British pounds".to_string(),
        ]),
        Entry::new(
            "search_github_issues",
            "Search repository issues and pull requests on GitHub.",
            json!({ "type": "object", "properties": { "repo": { "type": "string" } } }),
        )
        .with_examples(vec![
            "Find memory leak issues in tokio-rs/tokio".to_string(),
            "Has anyone reported a panic in version 0.2?".to_string(),
        ]),
    ];

    for entry in entries {
        registry.register(entry).await?;
    }

    let queries = [
        "Do I need an umbrella in London today?",
        "I need to pay 2000 Yen, how many dollars is that?",
        "Check if there are open bug reports for connection timeouts",
    ];

    for query in queries {
        println!("\nQuery: {query}");
        for hit in registry.search(query, 2).await? {
            let (id, score, matched) = (&hit.entry.id, hit.score, hit.matched_example);
            println!("  {id:<22} {score:.3}  matched example: {matched:?}");
        }
    }

    Ok(())
}
