use serde_json::json;
use std::sync::Arc;

use tools_rag::defaults::cosine_retriever::CosineRetriever;
use tools_rag::defaults::file_store::JsonToolStore;
use tools_rag::embedder::CandleEmbedder;
use tools_rag::formatter::DefaultToolFormatter;
use tools_rag::registry::ToolRegistry;
use tools_rag::traits::ToolStore;
use tools_rag::types::ToolDefinition;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store_path = "./target/tools_cache.json";

    // 1. Initialisation du store JSON sur disque
    println!("1. Chargement du store disque : '{}'...", store_path);
    let store = Arc::new(JsonToolStore::load_or_create(store_path).await?);

    println!("2. Chargement de l'embedder Candle...");
    let embedder = Arc::new(CandleEmbedder::minilm_v2().await?);
    let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
    let formatter = Box::new(DefaultToolFormatter);

    let registry = ToolRegistry::new(store.clone(), embedder, retriever, formatter);

    // 2. Vérification avant enregistrement
    let tool_name = "convert_currency";

    if store.get(tool_name).await?.is_none() {
        println!(
            "\n[CACHE MISS] L'outil '{}' n'existe pas encore.",
            tool_name
        );
        println!("--> Génération des embeddings et sauvegarde sur disque...");

        let tool = ToolDefinition::new(
            tool_name,
            "Convert monetary amounts between foreign currencies.",
            json!({
                "type": "object",
                "properties": {
                    "amount": { "type": "number" },
                    "from": { "type": "string" },
                    "to": { "type": "string" }
                }
            }),
        )
        .with_examples(vec![
            "How much is $100 USD in EUR?".to_string(),
            "Convert 50 GBP to JPY".to_string(),
        ]);

        registry.register(tool).await?;
        println!("--> Outil enregistré et indexé dans '{}'.", store_path);
    } else {
        println!(
            "\n[CACHE HIT] L'outil '{}' est déjà chargé depuis le fichier !",
            tool_name
        );
        println!("--> Zéro calcul d'embedding effectué.");
    }

    // 3. Test de recherche immédiat
    let query = "I need to exchange 500 dollars to euros";
    println!("\n3. Recherche vectorielle pour : \"{}\"", query);

    let results = registry.search(query, 1).await?;
    if let Some(res) = results.first() {
        println!(
            "   -> Outil trouvé : {} (Score MaxSim: {:.4})",
            res.tool.name, res.score
        );
    }

    Ok(())
}
