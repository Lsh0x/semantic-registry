use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::Result;
use crate::formatter::DefaultFormatter;
use crate::retriever::{cosine_similarity, CosineRetriever};
use crate::store::InMemoryStore;
use crate::traits::{Embedder, EntryStore, Formatter, Payload, Retriever};
use crate::types::{
    Entry, EvictionPolicy, FeedbackDecision, FeedbackOutcome, RegistryConfig, SearchResult,
};

/// Registre d'entrées recherchées par similarité sémantique, enrichi par le feedback
pub struct SemanticRegistry<P> {
    store: Arc<dyn EntryStore<P>>,
    embedder: Arc<dyn Embedder>,
    retriever: Arc<dyn Retriever<P>>,
    formatter: Box<dyn Formatter<P>>,
    config: RegistryConfig,
    usage: UsageTracker,
}

impl<P: Payload> SemanticRegistry<P> {
    pub fn new(
        store: Arc<dyn EntryStore<P>>,
        embedder: Arc<dyn Embedder>,
        retriever: Arc<dyn Retriever<P>>,
    ) -> Self {
        Self {
            store,
            embedder,
            retriever,
            formatter: Box::new(DefaultFormatter),
            config: RegistryConfig::default(),
            usage: UsageTracker::default(),
        }
    }

    /// Registre en mémoire avec recherche cosinus : le point de départ le plus court
    pub fn in_memory(embedder: Arc<dyn Embedder>) -> Self {
        let store: Arc<dyn EntryStore<P>> = Arc::new(InMemoryStore::new());
        let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
        Self::new(store, embedder, retriever)
    }

    /// Remplace le texte embarqué (par défaut : `text`, puis chaque exemple)
    pub fn with_formatter(mut self, formatter: impl Formatter<P> + 'static) -> Self {
        self.formatter = Box::new(formatter);
        self
    }

    pub fn with_config(mut self, config: RegistryConfig) -> Self {
        self.config = config;
        self
    }

    pub fn config(&self) -> &RegistryConfig {
        &self.config
    }

    pub async fn register(&self, entry: Entry<P>) -> Result<()> {
        let text_chunks = self.formatter.format(&entry);
        let mut embeddings = Vec::with_capacity(text_chunks.len());

        for chunk in text_chunks {
            let emb = self.embedder.embed(&chunk).await?;
            embeddings.push(emb);
        }

        self.store.insert(entry, embeddings).await
    }

    /// Renvoie au plus `limit` entrées, en écartant celles sous `min_search_score`
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult<P>>> {
        let mut results = self.retriever.retrieve(query, limit).await?;
        results.retain(|r| r.score >= self.config.min_search_score);
        for r in &results {
            if let Some(example) = r.matched_example.and_then(|i| r.entry.examples.get(i)) {
                self.usage.touch(&r.entry.id, example);
            }
        }
        Ok(results)
    }

    /// Absorbe une requête utilisateur comme exemple de l'entrée validée.
    ///
    /// La lecture, l'éviction et l'ajout ne sont pas atomiques : des feedbacks concurrents
    /// sur la même entrée peuvent dépasser brièvement `max_examples_per_entry`.
    pub async fn record_feedback(
        &self,
        user_query: &str,
        selected_id: &str,
        outcome: FeedbackOutcome,
    ) -> Result<FeedbackDecision> {
        let config = &self.config;
        let (target_id, is_correction) = match outcome {
            FeedbackOutcome::Success => (selected_id, false),
            FeedbackOutcome::CorrectedTo { ref correct_id } => (correct_id.as_str(), true),
        };

        if config.max_examples_per_entry == 0 {
            return Ok(FeedbackDecision::Full);
        }

        let entry = match self.store.get(target_id).await? {
            Some(e) => e,
            None => return Ok(FeedbackDecision::EntryNotFound),
        };

        let existing_embeddings = match self.store.get_embeddings(target_id).await? {
            Some(e) => e,
            None => return Ok(FeedbackDecision::EntryNotFound),
        };

        let query_embedding = self
            .embedder
            .embed(&self.formatter.example_text(&entry, user_query))
            .await?;

        // 1. Dédoublonnage sémantique (le meilleur score sert aussi de score de confiance)
        let best_similarity = existing_embeddings
            .iter()
            .map(|emb| cosine_similarity(&query_embedding, emb))
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
        let mut examples = entry.examples;
        let mut embeddings = existing_embeddings;
        let mut evicted = Vec::new();

        while examples.len() >= config.max_examples_per_entry {
            let index = match config.eviction_policy {
                EvictionPolicy::RejectWhenFull => return Ok(FeedbackDecision::Full),
                EvictionPolicy::Fifo => 0,
                EvictionPolicy::MostRedundantWithBase => {
                    most_redundant_with_base(&embeddings).min(examples.len() - 1)
                }
                EvictionPolicy::LeastRecentlyUsed => {
                    self.usage.least_recently_used(target_id, &examples)
                }
            };
            self.store.remove_example(target_id, index).await?;
            let example = examples.remove(index);
            self.usage.forget(target_id, &example);
            evicted.push(example);
            if index + 1 < embeddings.len() {
                embeddings.remove(index + 1);
            }
        }

        // 4. Persistance
        self.store
            .append_example(target_id, user_query.to_string(), query_embedding)
            .await?;
        self.usage.touch(target_id, user_query);

        Ok(FeedbackDecision::Added { evicted })
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
        .map(|(i, emb)| (i, cosine_similarity(base, emb)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(i, _)| i)
}

/// Dernier usage de chaque exemple (horloge logique), par entrée
#[derive(Default)]
struct UsageTracker {
    clock: AtomicU64,
    last_used: Mutex<HashMap<String, HashMap<String, u64>>>,
}

impl UsageTracker {
    fn touch(&self, entry: &str, example: &str) {
        let tick = self.clock.fetch_add(1, Ordering::Relaxed) + 1;
        if let Ok(mut map) = self.last_used.lock() {
            map.entry(entry.to_string())
                .or_default()
                .insert(example.to_string(), tick);
        }
    }

    fn forget(&self, entry: &str, example: &str) {
        if let Ok(mut map) = self.last_used.lock() {
            if let Some(examples) = map.get_mut(entry) {
                examples.remove(example);
            }
        }
    }

    /// Exemple jamais utilisé ou utilisé le moins récemment ; à égalité, le plus ancien
    fn least_recently_used(&self, entry: &str, examples: &[String]) -> usize {
        let Ok(map) = self.last_used.lock() else {
            return 0;
        };
        let used = map.get(entry);
        examples
            .iter()
            .enumerate()
            .min_by_key(|(_, ex)| used.and_then(|u| u.get(ex.as_str())).copied().unwrap_or(0))
            .map_or(0, |(i, _)| i)
    }
}
