use serde_json::json;
use std::sync::Arc;

use tools_rag::defaults::cosine_retriever::CosineRetriever;
use tools_rag::defaults::memory_store::InMemoryToolStore;
use tools_rag::embedder::{CachedEmbedder, CandleEmbedder};
use tools_rag::formatter::DefaultToolFormatter;
use tools_rag::registry::ToolRegistry;
use tools_rag::types::{FeedbackOutcome, RegistryConfig, ToolDefinition};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Initialisation de l'Embedder Candle...");
    // Le cache évite de ré-embedder la requête entre search() et record_feedback()
    let embedder = Arc::new(CachedEmbedder::new(
        Arc::new(CandleEmbedder::minilm_v2().await?),
        "all-MiniLM-L6-v2",
        1024,
    ));
    let store = Arc::new(InMemoryToolStore::new());
    let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
    // Configuration personnalisée des garde-fous
    let config = RegistryConfig {
        max_examples_per_tool: 5,
        ..RegistryConfig::default()
    };
    let registry = ToolRegistry::new(store, embedder, retriever, Box::new(DefaultToolFormatter))
        .with_config(config);

    // Outil brut sans aucun exemple
    let tool_name = "get_k8s_pod_logs";
    let tool = ToolDefinition::new(
        tool_name,
        "Fetches stdout and stderr logs for a Kubernetes pod in a given namespace.",
        json!({
            "type": "object",
            "properties": {
                "namespace": { "type": "string" },
                "pod_name": { "type": "string" }
            }
        }),
    );

    registry.register(tool).await?;

    let query = "My payment pod is crashing, I need stdout logs";
    println!("\n2. Première recherche pour : \"{}\"", query);

    let res_before = registry.search(query, 1).await?;
    if let Some(res) = res_before.first() {
        println!(
            "   -> Outil trouvé : {} (Score initial: {:.4})",
            res.tool.name, res.score
        );
    }

    println!("\n3. Enregistrement du feedback (Succès d'exécution)...");
    let added = registry
        .record_feedback(query, tool_name, FeedbackOutcome::Success)
        .await?;
    let similar_query = "Payment container crashing, get stdout output";

    println!("   -> Décision : {:?}", added);
    println!("\n4. Deuxième recherche avec une tournure similaire...");
    println!("     -> query: {}", similar_query);
    let res_after = registry.search(similar_query, 1).await?;
    if let Some(res) = res_after.first() {
        println!(
            "   -> Outil trouvé : {} (Nouveau Score MaxSim: {:.4})",
            res.tool.name, res.score
        );
    }

    println!("\n5. Test de dédoublonnage sémantique (tentative d'ajout d'une requête quasi-identique)...");
    let duplicate_added = registry
        .record_feedback(query, tool_name, FeedbackOutcome::Success)
        .await?;
    println!(
        "   -> Décision : {:?} (attendu : Duplicate)",
        duplicate_added
    );

    Ok(())
}
