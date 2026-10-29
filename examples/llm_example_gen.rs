use serde_json::json;
use std::sync::Arc;

use semantic_registry::embedder::CandleEmbedder;
use semantic_registry::prelude::*;
use serde_json::Value;

// Importation de Candle pour la génération de texte locale
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use hf_hub::api::tokio::Api;
use tokenizers::Tokenizer;

/// Charge un modèle Qwen2.5-0.5B local via Candle et génère 3 exemples d'utilisation
async fn generate_examples_with_candle_llm(
    tool: &Entry<Value>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    println!("  [LLM Candle] Génération pour '{}'...", tool.id);

    let api = Api::new()?;
    let repo = api.model("Qwen/Qwen2.5-0.5B-Instruct".to_string());

    let config_file = repo.get("config.json").await?;
    let tokenizer_file = repo.get("tokenizer.json").await?;
    let weights_file = repo.get("model.safetensors").await?;

    let config: Config = serde_json::from_str(&tokio::fs::read_to_string(config_file).await?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(|e| e.to_string())?;
    let device = Device::Cpu;
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[weights_file],
            candle_core::DType::F32,
            &device,
        )?
    };

    let mut model = ModelForCausalLM::new(&config, vb)?;

    // Construction du prompt ChatML
    let prompt_text = format!(
        "<|im_start|>user\n\
         Generate exactly 3 short user questions for the tool '{}': {}\n\
         Format: One question per line starting with '- '. No other text.<|im_end|>\n\
         <|im_start|>assistant\n- ",
        tool.id, tool.text
    );

    let tokens = tokenizer
        .encode(prompt_text.as_str(), true)
        .map_err(|e| e.to_string())?;
    let mut input_tokens = tokens.get_ids().to_vec();

    let mut logits_processor = LogitsProcessor::new(299792458, Some(0.7), None);
    let mut generated_text = String::from("- ");

    for index in 0..128 {
        let context_size = if index > 0 { 1 } else { input_tokens.len() };
        let start_pos = input_tokens.len() - context_size;
        let input_tensor = Tensor::new(&input_tokens[start_pos..], &device)?.unsqueeze(0)?;

        let logits = model.forward(&input_tensor, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;

        let next_token = logits_processor.sample(&logits)?;
        input_tokens.push(next_token);

        if next_token == 151643 || next_token == 151645 {
            break;
        }

        if let Ok(piece) = tokenizer.decode(&[next_token], false) {
            generated_text.push_str(&piece);
        }
    }

    let examples: Vec<String> = generated_text
        .lines()
        .map(|line| line.trim_start_matches('-').trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();

    Ok(examples)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Initialisation de l'Embedder Candle (all-MiniLM-L6-v2)...");
    let embedder = Arc::new(CandleEmbedder::minilm_v2().await?);

    // Registre 1 : SANS exemples (description seule)
    let registry_no_ex: SemanticRegistry<Value> = SemanticRegistry::in_memory(embedder.clone());

    // Registre 2 : AVEC exemples synthétiques
    let registry_with_ex: SemanticRegistry<Value> = SemanticRegistry::in_memory(embedder.clone());

    // Définition de 3 outils distincts
    let raw_tools = vec![
        Entry::new(
            "query_database",
            "Executes a read-only SQL query against the analytical PostgreSQL database.",
            json!({
                "type": "object",
                "properties": {
                    "sql": { "type": "string", "description": "PostgreSQL query" }
                },
                "required": ["sql"]
            }),
        ),
        Entry::new(
            "get_k8s_pod_logs",
            "Fetches stdout and stderr logs for a Kubernetes pod in a given namespace.",
            json!({
                "type": "object",
                "properties": {
                    "namespace": { "type": "string" },
                    "pod_name": { "type": "string" }
                },
                "required": ["namespace", "pod_name"]
            }),
        ),
        Entry::new(
            "search_github_issues",
            "Searches repository issues and pull requests on GitHub for bug reports and error logs.",
            json!({
                "type": "object",
                "properties": {
                    "repo": { "type": "string" },
                    "query": { "type": "string" }
                },
                "required": ["repo", "query"]
            }),
        ),
    ];

    println!("\n2. Indexation des outils et génération d'exemples LLM...");
    for tool in raw_tools {
        // Enregistrement dans le registre SANS exemples
        registry_no_ex.register(tool.clone()).await?;

        // Génération des exemples pour le registre AVEC exemples
        let mut enriched_tool = tool.clone();
        match generate_examples_with_candle_llm(&tool).await {
            Ok(examples) => {
                println!("     Exemples retenus :");
                for ex in &examples {
                    println!("       * \"{ex}\"");
                }
                enriched_tool = enriched_tool.with_examples(examples);
            }
            Err(e) => {
                eprintln!("     Erreur LLM ({e}), utilisation d'un fallback statique.");
                enriched_tool =
                    enriched_tool.with_examples(vec![format!("How do I use {}?", tool.id)]);
            }
        }

        registry_with_ex.register(enriched_tool).await?;
    }

    println!("\n============================================================");
    println!(" 3. COMPARAISON DES RECHERCHES (SANS EXEMPLES vs AVEC EXEMPLES)");
    println!("============================================================\n");

    let test_queries = vec![
        "Find total revenue grouped by month for 2025",
        "Why is my payment container crashing in the cluster?",
        "Has anyone reported a memory leak issue in the repository?",
    ];

    for query in test_queries {
        println!("Requête utilisateur : \"{}\"", query);

        let res_no_ex = registry_no_ex.search(query, 1).await?;
        let res_with_ex = registry_with_ex.search(query, 1).await?;

        if let Some(r1) = res_no_ex.first() {
            println!(
                "  [SANS Exemples] Outil : {:<20} | Score : {:.4}",
                r1.entry.id, r1.score
            );
        }
        if let Some(r2) = res_with_ex.first() {
            println!(
                "  [AVEC Exemples] Outil : {:<20} | Score : {:.4} (MaxSim)",
                r2.entry.id, r2.score
            );
        }
        println!();
    }

    Ok(())
}
