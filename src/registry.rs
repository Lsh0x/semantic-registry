use std::sync::Arc;

use crate::defaults::cosine_retriever::CosineRetriever;
use crate::errors::Result;
use crate::meta::MetaTool;
use crate::traits::{Embedder, ToolFormatter, ToolRetriever, ToolStore};
use crate::types::{
    EvictionPolicy, FeedbackDecision, FeedbackOutcome, RegistryConfig, SearchResult, ToolDefinition,
};

pub struct ToolRegistry {
    store: Arc<dyn ToolStore>,
    embedder: Arc<dyn Embedder>,
    retriever: Arc<dyn ToolRetriever>,
    formatter: Box<dyn ToolFormatter>,
    config: RegistryConfig,
}

impl ToolRegistry {
    pub fn new(
        store: Arc<dyn ToolStore>,
        embedder: Arc<dyn Embedder>,
        retriever: Arc<dyn ToolRetriever>,
        formatter: Box<dyn ToolFormatter>,
    ) -> Self {
        Self {
            store,
            embedder,
            retriever,
            formatter,
            config: RegistryConfig::default(),
        }
    }

    pub fn with_config(mut self, config: RegistryConfig) -> Self {
        self.config = config;
        self
    }

    pub fn config(&self) -> &RegistryConfig {
        &self.config
    }

    pub async fn register(&self, tool: ToolDefinition) -> Result<()> {
        let text_chunks = self.formatter.format(&tool);
        let mut embeddings = Vec::with_capacity(text_chunks.len());

        for chunk in text_chunks {
            let emb = self.embedder.embed(&chunk).await?;
            embeddings.push(emb);
        }

        self.store.insert(tool, embeddings).await
    }

    /// Renvoie au plus `limit` outils, en écartant ceux sous `min_search_score`
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let mut results = self.retriever.retrieve(query, limit).await?;
        results.retain(|r| r.score >= self.config.min_search_score);
        Ok(results)
    }

    /// Absorbe une requête utilisateur comme exemple de l'outil validé.
    ///
    /// La lecture, l'éviction et l'ajout ne sont pas atomiques : des feedbacks concurrents
    /// sur le même outil peuvent dépasser brièvement `max_examples_per_tool`.
    pub async fn record_feedback(
        &self,
        user_query: &str,
        selected_tool_name: &str,
        outcome: FeedbackOutcome,
    ) -> Result<FeedbackDecision> {
        let config = &self.config;
        let (target_tool_name, is_correction) = match outcome {
            FeedbackOutcome::Success => (selected_tool_name, false),
            FeedbackOutcome::CorrectedTo {
                ref correct_tool_name,
            } => (correct_tool_name.as_str(), true),
        };

        if config.max_examples_per_tool == 0 {
            return Ok(FeedbackDecision::Full);
        }

        let tool = match self.store.get(target_tool_name).await? {
            Some(t) => t,
            None => return Ok(FeedbackDecision::ToolNotFound),
        };

        let existing_embeddings = match self.store.get_embeddings(target_tool_name).await? {
            Some(e) => e,
            None => return Ok(FeedbackDecision::ToolNotFound),
        };

        let query_embedding = self.embedder.embed(user_query).await?;

        // 1. Dédoublonnage sémantique (le meilleur score sert aussi de score de confiance)
        let best_similarity = existing_embeddings
            .iter()
            .map(|emb| CosineRetriever::cosine_similarity(&query_embedding, emb))
            .fold(0.0f32, f32::max);

        if best_similarity >= config.dedup_similarity_threshold {
            return Ok(FeedbackDecision::Duplicate {
                similarity: best_similarity,
            });
        }

        // 2. Filtre d'ingestion : un succès sur une requête très éloignée est un signal peu fiable
        if !is_correction && best_similarity < config.min_feedback_score {
            return Ok(FeedbackDecision::BelowMinScore {
                score: best_similarity,
            });
        }

        // 3. Éviction jusqu'à libérer une place
        let mut examples = tool.examples;
        let mut embeddings = existing_embeddings;
        let mut evicted = Vec::new();

        while examples.len() >= config.max_examples_per_tool {
            let index = match config.eviction_policy {
                EvictionPolicy::RejectWhenFull => return Ok(FeedbackDecision::Full),
                EvictionPolicy::Fifo => 0,
                EvictionPolicy::MostRedundantWithBase => {
                    most_redundant_with_base(&embeddings).min(examples.len() - 1)
                }
            };
            self.store.remove_example(target_tool_name, index).await?;
            evicted.push(examples.remove(index));
            if index + 1 < embeddings.len() {
                embeddings.remove(index + 1);
            }
        }

        // 4. Persistance
        self.store
            .append_example(target_tool_name, user_query.to_string(), query_embedding)
            .await?;

        Ok(FeedbackDecision::Added { evicted })
    }

    pub fn meta_tool_definition(&self) -> ToolDefinition {
        MetaTool::spec()
    }
}

/// Index (dans `examples`) de l'exemple le plus similaire au chunk canonique `embeddings[0]`
fn most_redundant_with_base(embeddings: &[Vec<f32>]) -> usize {
    let Some((base, examples)) = embeddings.split_first() else {
        return 0;
    };
    examples
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, CosineRetriever::cosine_similarity(base, emb)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(i, _)| i)
}
