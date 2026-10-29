use serde::{Deserialize, Serialize};

/// Une entrée du registre : ce que la bibliothèque indexe, et le `payload` qu'elle rend tel quel
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Entry<P> {
    /// Identifiant unique (nom d'outil, clé de skill, nom de table…)
    pub id: String,
    /// Texte du chunk canonique
    pub text: String,
    /// Formulations d'utilisateurs : fournies à l'enregistrement, puis enrichies par le feedback
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    /// Données de l'appelant (schéma d'outil, config, `()`…), jamais lues par la bibliothèque
    pub payload: P,
}

impl<P> Entry<P> {
    pub fn new(id: impl Into<String>, text: impl Into<String>, payload: P) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            examples: Vec::new(),
            payload,
        }
    }

    pub fn with_examples(mut self, examples: Vec<String>) -> Self {
        self.examples = examples;
        self
    }
}

/// Résultat d'une recherche avec son score de similarité
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResult<P> {
    pub entry: Entry<P>,
    pub score: f32,
    /// Index (dans `entry.examples`) de l'exemple qui a donné le score, `None` si c'est le
    /// chunk canonique. Alimente `EvictionPolicy::LeastRecentlyUsed`.
    #[serde(default)]
    pub matched_example: Option<usize>,
}

/// Retour d'expérience sur une entrée choisie à partir d'une recherche
#[derive(Debug, Clone, PartialEq)]
pub enum FeedbackOutcome {
    /// L'entrée choisie était la bonne
    Success,
    /// L'entrée choisie était mauvaise : `correct_id` est celle qui aurait dû l'être
    CorrectedTo { correct_id: String },
}

/// Politique appliquée quand une entrée a atteint `max_examples_per_entry`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    /// Supprime le plus ancien exemple (ring buffer)
    #[default]
    Fifo,
    /// Supprime l'exemple le plus proche du chunk canonique (le moins informatif)
    MostRedundantWithBase,
    /// Supprime l'exemple qui a gagné une recherche le moins récemment (jamais = le plus ancien).
    /// L'historique d'usage vit dans le `SemanticRegistry`, en mémoire : il repart de zéro au
    /// redémarrage, où la politique se comporte comme `Fifo` jusqu'aux premières recherches.
    LeastRecentlyUsed,
    /// Conservateur : refuse le nouvel exemple tant que le quota est atteint
    RejectWhenFull,
}

/// Combinaison des scores du chunk canonique et des exemples d'une entrée
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ScoringStrategy {
    /// MaxSim pur : `max(score_base, max(score_exemples))`
    #[default]
    MaxSim,
    /// MaxSim pondéré : `max(score_base, max(score_exemples) * example_weight)`.
    /// Un poids < 1.0 limite l'effet « aimant » des entrées qui accumulent du feedback.
    WeightedMaxSim { example_weight: f32 },
}

impl ScoringStrategy {
    /// Calcule le score d'une entrée. `base` est la similarité avec le chunk canonique,
    /// `best_example` la meilleure similarité parmi les exemples (si l'entrée en a).
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

/// Configuration de la politique métier du [`SemanticRegistry`](crate::registry::SemanticRegistry)
#[derive(Debug, Clone, PartialEq)]
pub struct RegistryConfig {
    /// Plafond d'exemples conservés par entrée (par défaut : 10). `0` désactive le feedback.
    pub max_examples_per_entry: usize,
    /// Similarité cosinus à partir de laquelle une requête est jugée redondante (par défaut : 0.85)
    pub dedup_similarity_threshold: f32,
    /// Score MaxSim minimal entre la requête et l'entrée pour accepter un feedback `Success`
    /// (par défaut : 0.30). Ne s'applique pas à `CorrectedTo` : une correction porte justement
    /// sur une entrée que la recherche a mal classée.
    pub min_feedback_score: f32,
    /// Politique d'éviction quand le plafond est atteint (par défaut : `Fifo`)
    pub eviction_policy: EvictionPolicy,
    /// Les résultats de `search()` sous ce score sont écartés (par défaut : 0.0, aucun filtre)
    pub min_search_score: f32,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_examples_per_entry: 10,
            dedup_similarity_threshold: 0.85,
            min_feedback_score: 0.30,
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
    /// Rejetée : la requête est trop éloignée de l'entrée pour être un signal fiable
    BelowMinScore { score: f32 },
    /// Rejetée : quota atteint avec `RejectWhenFull`, ou feedback désactivé (`max_examples_per_entry == 0`)
    Full,
    /// L'entrée cible n'existe pas dans le store
    EntryNotFound,
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
    fn entry_serde_roundtrips_and_skips_empty_examples() {
        let entry = Entry::new("t", "d", serde_json::json!({ "k": 1 }));
        let json = serde_json::to_value(&entry).unwrap();
        assert!(json.get("examples").is_none());
        let back: Entry<serde_json::Value> = serde_json::from_value(json).unwrap();
        assert_eq!(back, entry);
    }
}
