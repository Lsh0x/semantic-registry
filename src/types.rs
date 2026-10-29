use serde::{Deserialize, Serialize};

/// A registry entry: what the library indexes, plus a `payload` it hands back untouched.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Entry<P> {
    /// Unique key: a tool name, a skill id, a table name…
    pub id: String,
    /// Text of the canonical chunk.
    pub text: String,
    /// Phrasings users might type: seeded at registration, then grown by feedback.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    /// Caller data (a tool schema, a config, `()`…). Never read by the library.
    pub payload: P,
}

impl<P> Entry<P> {
    /// An entry without examples.
    pub fn new(id: impl Into<String>, text: impl Into<String>, payload: P) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            examples: Vec::new(),
            payload,
        }
    }

    /// Replaces the examples.
    pub fn with_examples(mut self, examples: Vec<String>) -> Self {
        self.examples = examples;
        self
    }
}

/// A search hit and its similarity score.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResult<P> {
    /// The matching entry, payload included.
    pub entry: Entry<P>,
    /// Similarity with the query, per the [`ScoringStrategy`].
    pub score: f32,
    /// Index in `entry.examples` of the example that produced the score, or `None` when the
    /// canonical chunk did. Feeds `EvictionPolicy::LeastRecentlyUsed`.
    #[serde(default)]
    pub matched_example: Option<usize>,
}

/// What happened with an entry chosen from a search.
#[derive(Debug, Clone, PartialEq)]
pub enum FeedbackOutcome {
    /// The chosen entry was the right one.
    Success,
    /// The chosen entry was wrong: `correct_id` is the one that should have been chosen.
    CorrectedTo {
        /// Id of the entry that should have been chosen.
        correct_id: String,
    },
}

/// What to do when an entry has reached `max_examples_per_entry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    /// Evict the oldest example (ring buffer).
    #[default]
    Fifo,
    /// Evict the example closest to the canonical chunk, the one that adds least.
    MostRedundantWithBase,
    /// Evict the example that has gone longest without winning a search (never = oldest).
    /// Usage history lives in the `SemanticRegistry`, in memory: it restarts empty, and the
    /// policy behaves like `Fifo` until searches build it back.
    LeastRecentlyUsed,
    /// Conservative: refuse new examples while the quota is reached.
    RejectWhenFull,
}

/// How the canonical chunk's score and the examples' scores combine into an entry's score.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ScoringStrategy {
    /// Plain MaxSim: `max(canonical, best_example)`.
    #[default]
    MaxSim,
    /// Weighted MaxSim: `max(canonical, best_example * example_weight)`.
    /// A weight below 1.0 keeps entries that accumulate feedback from becoming magnets.
    WeightedMaxSim {
        /// Multiplier applied to example scores, usually below 1.0.
        example_weight: f32,
    },
}

impl ScoringStrategy {
    /// Score of an entry. `base` is the similarity with the canonical chunk, `best_example`
    /// the best similarity among the examples, if the entry has any.
    pub fn combine(&self, base: f32, best_example: Option<f32>) -> f32 {
        self.combine_with_source(base, best_example).0
    }

    /// Like [`combine`](Self::combine), and whether the example won.
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

/// The policy of a [`SemanticRegistry`](crate::registry::SemanticRegistry).
#[derive(Debug, Clone, PartialEq)]
pub struct RegistryConfig {
    /// Example cap per entry (default: 10). `0` disables feedback.
    pub max_examples_per_entry: usize,
    /// Cosine similarity from which a query counts as a duplicate (default: 0.85).
    pub dedup_similarity_threshold: f32,
    /// Minimum MaxSim score between the query and the entry to accept a `Success` feedback
    /// (default: 0.30). Does not apply to `CorrectedTo`: a correction is about an entry the
    /// search ranked badly, so a low score is expected.
    pub min_feedback_score: f32,
    /// What to evict once the cap is reached (default: `Fifo`).
    pub eviction_policy: EvictionPolicy,
    /// `search()` drops results below this score (default: 0.0, no cutoff).
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

/// What `record_feedback` did, and why.
#[derive(Debug, Clone, PartialEq)]
pub enum FeedbackDecision {
    /// The query was added as an example.
    Added {
        /// Examples evicted to stay under the cap.
        evicted: Vec<String>,
    },
    /// Rejected: too close to an existing chunk.
    Duplicate {
        /// Similarity with the closest chunk.
        similarity: f32,
    },
    /// Rejected: the query is too far from the entry to be a reliable signal.
    BelowMinScore {
        /// The query's best similarity with the entry.
        score: f32,
    },
    /// Rejected: quota reached with `RejectWhenFull`, or feedback disabled
    /// (`max_examples_per_entry == 0`).
    Full,
    /// The target entry is not in the store.
    EntryNotFound,
}

impl FeedbackDecision {
    /// Whether the query was added as an example.
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
