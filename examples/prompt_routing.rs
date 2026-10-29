//! Pick the right prompt template for a request, with a fallback when nothing fits.
//!
//! cargo run --features candle --example prompt_routing

use std::sync::Arc;

use semantic_registry::embedder::CandleEmbedder;
use semantic_registry::prelude::*;

/// The payload: what the caller needs once a prompt is chosen
#[derive(Clone, Debug)]
struct Prompt {
    system: &'static str,
    template: &'static str,
}

impl Prompt {
    fn render(&self, input: &str) -> String {
        format!(
            "[system] {}\n[user]   {}",
            self.system,
            self.template.replace("{input}", input)
        )
    }
}

const FALLBACK: Prompt = Prompt {
    system: "You are a helpful assistant.",
    template: "{input}",
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let embedder = Arc::new(CandleEmbedder::minilm_v2().await?);
    let registry: SemanticRegistry<Prompt> =
        SemanticRegistry::in_memory(embedder).with_config(RegistryConfig {
            // Below this score no template fits: use the generic prompt instead
            min_search_score: 0.30,
            ..RegistryConfig::default()
        });

    let prompts = vec![
        Entry::new(
            "code_review",
            "Review source code for bugs, readability and security issues.",
            Prompt {
                system: "You are a senior engineer doing a code review.",
                template: "Review this code and list concrete issues:\n{input}",
            },
        )
        .with_examples(vec![
            "can you review this code".to_string(),
            "is this function safe".to_string(),
            "find the bugs in this snippet".to_string(),
        ]),
        Entry::new(
            "explain_sql",
            "Explain what a SQL query does, step by step, in plain language.",
            Prompt {
                system: "You explain SQL to non-experts.",
                template: "Explain this query step by step:\n{input}",
            },
        )
        .with_examples(vec![
            "explain this sql query".to_string(),
            "what does this select statement return".to_string(),
        ]),
        Entry::new(
            "commit_message",
            "Write a concise git commit message from a diff.",
            Prompt {
                system: "You write Conventional Commits messages.",
                template: "Write a commit message for this diff:\n{input}",
            },
        )
        .with_examples(vec![
            "write a commit message".to_string(),
            "give me a commit message for these changes".to_string(),
        ]),
    ];
    for prompt in prompts {
        registry.register(prompt).await?;
    }

    let requests = [
        "is this rust function safe? fn get(v: &[u8], i: usize) -> u8 { v[i] }",
        "what does SELECT name FROM users WHERE age > 30 GROUP BY name do",
        "give me a commit message, I renamed the crate and its types",
        "what's a good name for a cat",
    ];

    for request in requests {
        match registry.search(request, 1).await?.into_iter().next() {
            Some(hit) => {
                println!("\n{request}\n  -> {} ({:.3})", hit.entry.id, hit.score);
                println!("{}", hit.entry.payload.render(request));
            }
            None => {
                println!("\n{request}\n  -> no template fits, fallback");
                println!("{}", FALLBACK.render(request));
            }
        }
    }

    // A performance question lands on `explain_sql`; the user says it is a review
    let request = "why is this query slow";
    let picked = registry.search(request, 1).await?.remove(0).entry.id;
    let decision = registry
        .record_feedback(
            request,
            &picked,
            FeedbackOutcome::CorrectedTo {
                correct_id: "code_review".to_string(),
            },
        )
        .await?;
    println!("\n{request}\n  picked {picked}, corrected to code_review: {decision:?}");

    // The correction is now an example of `code_review`, so close phrasings follow it
    for request in [request, "how do I make this query faster"] {
        let hit = registry.search(request, 1).await?.remove(0);
        println!("  {request:?} -> {} ({:.3})", hit.entry.id, hit.score);
    }

    Ok(())
}
