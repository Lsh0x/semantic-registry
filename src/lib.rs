//! Registre sémantique : des entrées nommées, recherchées par le sens, qui apprennent
//! des requêtes validées ou corrigées.
//!
//! La bibliothèque ne sait pas ce que sont les entrées (outils, skills, tables SQL…) :
//! elle indexe `text` et `examples`, et rend le `payload` tel quel.

pub mod embedder;
pub mod error;
pub mod formatter;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod registry;
pub mod retriever;
pub mod store;
pub mod traits;
pub mod types;

pub use registry::SemanticRegistry;
pub use retriever::CosineRetriever;
pub use store::{InMemoryStore, JsonStore};

pub mod prelude {
    pub use crate::registry::SemanticRegistry;
    pub use crate::traits::{Embedder, EntryStore, Formatter, Payload, Retriever};
    pub use crate::types::{
        Entry, EvictionPolicy, FeedbackDecision, FeedbackOutcome, RegistryConfig, ScoringStrategy,
        SearchResult,
    };
}
