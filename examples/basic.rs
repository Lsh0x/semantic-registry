use serde_json::json;
use std::sync::Arc;

use tools_rag::defaults::cosine_retriever::CosineRetriever;
use tools_rag::defaults::memory_store::InMemoryToolStore;
use tools_rag::embedder::CandleEmbedder;
use tools_rag::formatter::DefaultToolFormatter;
use tools_rag::registry::ToolRegistry;
use tools_rag::types::ToolDefinition;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading Candle embedder model (all-MiniLM-L6-v2)...");
    let embedder = Arc::new(CandleEmbedder::minilm_v2().await?);

    let store = Arc::new(InMemoryToolStore::new());
    let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
    let formatter = Box::new(DefaultToolFormatter);

    let registry = ToolRegistry::new(store, embedder, retriever, formatter);

    // 1. Weather Tool
    let weather_tool = ToolDefinition::new(
        "get_weather",
        "Fetch current weather conditions or forecasts for a specific location.",
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" },
                "units": { "type": "string", "enum": ["metric", "imperial"] }
            },
            "required": ["city"]
        }),
    )
    .with_examples(vec![
        "What's the weather like in Tokyo right now?".to_string(),
        "Is it going to rain in Paris tomorrow?".to_string(),
        "Check current temperature in New York in Fahrenheit".to_string(),
    ]);

    // 2. Currency Conversion Tool
    let currency_tool = ToolDefinition::new(
        "convert_currency",
        "Convert monetary amounts between foreign currencies.",
        json!({
            "type": "object",
            "properties": {
                "amount": { "type": "number", "description": "Amount to convert" },
                "from": { "type": "string", "description": "Source ISO currency code (e.g. USD)" },
                "to": { "type": "string", "description": "Target ISO currency code (e.g. EUR)" }
            },
            "required": ["amount", "from", "to"]
        }),
    )
    .with_examples(vec![
        "How much is $100 USD in Euros?".to_string(),
        "Convert 5000 JPY to British Pounds".to_string(),
        "What is the current exchange rate from CAD to AUD?".to_string(),
    ]);

    // 3. GitHub Issues Search Tool
    let github_tool = ToolDefinition::new(
        "search_github_issues",
        "Search repository issues and pull requests on GitHub.",
        json!({
            "type": "object",
            "properties": {
                "repo": { "type": "string", "description": "Repository in format owner/repo" },
                "query": { "type": "string", "description": "Search keyword or error message" },
                "state": { "type": "string", "enum": ["open", "closed", "all"] }
            },
            "required": ["repo", "query"]
        }),
    )
    .with_examples(vec![
        "Find memory leak issues in tokio-rs/tokio".to_string(),
        "Search for open pull requests related to authentication".to_string(),
        "Has anyone reported a panic in version 0.2?".to_string(),
    ]);

    // Register all tools
    registry.register(weather_tool).await?;
    registry.register(currency_tool).await?;
    registry.register(github_tool).await?;

    println!("Registered 3 distinct tools into the registry.\n");

    // Test queries to demonstrate vector routing
    let test_queries = vec![
        "Do I need an umbrella in London today?",
        "I need to pay 2000 Yen, how many dollars is that?",
        "Check if there are any open bug reports for connection timeouts in the repo",
    ];

    for query in test_queries {
        println!("User Query: \"{}\"", query);
        let results = registry.search(query, 1).await?;

        if let Some(res) = results.first() {
            println!(
                "  -> Selected Tool: {} (Score: {:.4})\n",
                res.tool.name, res.score
            );
        } else {
            println!("  -> No matching tool found.\n");
        }
    }

    Ok(())
}
