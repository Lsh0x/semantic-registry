use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Résultat d'une recherche d'outil avec son score de similarité
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResult {
    pub tool: ToolDefinition,
    pub score: f32,
    /// Index (dans `tool.examples`) de l'exemple qui a donné le score, `None` si c'est la
    /// description canonique. Alimente `EvictionPolicy::LeastRecentlyUsed`.
    #[serde(default)]
    pub matched_example: Option<usize>,
}

/// Définition canonique d'un outil
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            examples: Vec::new(),
            metadata: None,
        }
    }

    pub fn with_examples(mut self, examples: Vec<String>) -> Self {
        self.examples = examples;
        self
    }
}

/// Issue d'une exécution d'outil pour le retour d'expérience
#[derive(Debug, Clone, PartialEq)]
pub enum FeedbackOutcome {
    /// L'outil sélectionné était correct
    Success,
    /// L'outil sélectionné était incorrect, spécifie le nom de l'outil qui aurait dû être choisi
    CorrectedTo { correct_tool_name: String },
}

/// Politique appliquée quand un outil a atteint `max_examples_per_tool`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    /// Supprime le plus ancien exemple (ring buffer)
    #[default]
    Fifo,
    /// Supprime l'exemple le plus proche de la description canonique (le moins informatif)
    MostRedundantWithBase,
    /// Supprime l'exemple qui a gagné une recherche le moins récemment (jamais = le plus ancien).
    /// L'historique d'usage vit dans le `ToolRegistry`, en mémoire : il repart de zéro au
    /// redémarrage, où la politique se comporte comme `Fifo` jusqu'aux premières recherches.
    LeastRecentlyUsed,
    /// Conservateur : refuse le nouvel exemple tant que le quota est atteint
    RejectWhenFull,
}

/// Combinaison des scores de la description canonique et des exemples d'un outil
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ScoringStrategy {
    /// MaxSim pur : `max(score_base, max(score_exemples))`
    #[default]
    MaxSim,
    /// MaxSim pondéré : `max(score_base, max(score_exemples) * example_weight)`.
    /// Un poids < 1.0 limite l'effet « outil aimant » des outils qui accumulent du feedback.
    WeightedMaxSim { example_weight: f32 },
}

impl ScoringStrategy {
    /// Calcule le score d'un outil. `base` est la similarité avec le chunk canonique,
    /// `best_example` la meilleure similarité parmi les exemples (si l'outil en a).
    pub fn combine(&self, base: f32, best_example: Option<f32>) -> f32 {
        self.combine_with_source(base, best_example).0
    }

    /// Comme [`combine`](Self::combine), et indique si c'est l'exemple qui l'emporte
    pub fn combine_with_source(&self, base: f32, best_example: Option<f32>) -> (f32, bool) {
        let weighted = match (self, best_example) {
            (_, None) => return (base, false),
            (Self::MaxSim, Some(ex)) => ex,
            (Self::WeightedMaxSim { example_weight }, Some(ex)) => ex * example_weight,
        };
        if weighted > base {
            (weighted, true)
        } else {
            (base, false)
        }
    }
}

/// Configuration de la politique métier du [`ToolRegistry`](crate::registry::ToolRegistry)
#[derive(Debug, Clone, PartialEq)]
pub struct RegistryConfig {
    /// Plafond d'exemples conservés par outil (par défaut : 10). `0` désactive le feedback.
    pub max_examples_per_tool: usize,
    /// Similarité cosinus à partir de laquelle une requête est jugée redondante (par défaut : 0.85)
    pub dedup_similarity_threshold: f32,
    /// Score MaxSim minimal entre la requête et l'outil pour accepter un feedback `Success`
    /// (par défaut : 0.40). Ne s'applique pas à `CorrectedTo` : une correction porte justement
    /// sur un outil que la recherche a mal classé.
    pub min_feedback_score: f32,
    /// Politique d'éviction quand le plafond est atteint (par défaut : `Fifo`)
    pub eviction_policy: EvictionPolicy,
    /// Les résultats de `search()` sous ce score sont écartés (par défaut : 0.0, aucun filtre)
    pub min_search_score: f32,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_examples_per_tool: 10,
            dedup_similarity_threshold: 0.85,
            min_feedback_score: 0.40,
            eviction_policy: EvictionPolicy::Fifo,
            min_search_score: 0.0,
        }
    }
}

/// Décision prise par `record_feedback`, pour l'observabilité côté appelant
#[derive(Debug, Clone, PartialEq)]
pub enum FeedbackDecision {
    /// La requête a été ajoutée comme exemple
    Added {
        /// Exemples supprimés pour respecter le plafond
        evicted: Vec<String>,
    },
    /// Rejetée : trop proche d'un chunk existant
    Duplicate { similarity: f32 },
    /// Rejetée : la requête est trop éloignée de l'outil pour être un signal fiable
    BelowMinScore { score: f32 },
    /// Rejetée : quota atteint avec `RejectWhenFull`, ou feedback désactivé (`max_examples_per_tool == 0`)
    Full,
    /// L'outil cible n'existe pas dans le store
    ToolNotFound,
}

impl FeedbackDecision {
    pub fn is_added(&self) -> bool {
        matches!(self, Self::Added { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_without_examples_is_base_score() {
        assert_eq!(ScoringStrategy::MaxSim.combine(0.4, None), 0.4);
        let weighted = ScoringStrategy::WeightedMaxSim {
            example_weight: 0.5,
        };
        assert_eq!(weighted.combine(0.4, None), 0.4);
    }

    #[test]
    fn combine_applies_weight_to_examples_only() {
        assert_eq!(ScoringStrategy::MaxSim.combine(0.4, Some(0.9)), 0.9);
        let weighted = ScoringStrategy::WeightedMaxSim {
            example_weight: 0.5,
        };
        assert_eq!(weighted.combine(0.4, Some(0.9)), 0.45);
        assert_eq!(weighted.combine(0.6, Some(0.9)), 0.6);
    }

    #[test]
    fn tool_definition_serde_skips_empty_fields() {
        let tool = ToolDefinition::new("t", "d", serde_json::json!({}));
        let json = serde_json::to_value(&tool).unwrap();
        assert!(json.get("examples").is_none());
        assert!(json.get("metadata").is_none());
        let back: ToolDefinition = serde_json::from_value(json).unwrap();
        assert_eq!(back, tool);
    }
}
