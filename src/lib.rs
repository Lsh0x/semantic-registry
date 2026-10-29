//! A registry of named entries searched by meaning, which learns from validated and
//! corrected queries.
//!
//! The library does not know what the entries are (tools, skills, SQL tables…): it
//! indexes `text` and `examples`, and hands the `payload` back untouched.
//!
//! See the [README](https://github.com/Lsh0x/semantic-registry#readme) for a tour.

#![warn(missing_docs)]

pub mod embedder;
pub mod error;
/// The default [`Formatter`](traits::Formatter).
pub mod formatter;
#[cfg(feature = "mcp")]
pub mod mcp;
/// The [`SemanticRegistry`].
pub mod registry;
/// The built-in [`CosineRetriever`].
pub mod retriever;
pub mod store;
pub mod traits;
/// Entries, search results, configuration and feedback types.
pub mod types;

pub use registry::SemanticRegistry;
pub use retriever::CosineRetriever;
pub use store::{InMemoryStore, JsonStore};

/// The types most programs need: `use semantic_registry::prelude::*;`
pub mod prelude {
    pub use crate::registry::SemanticRegistry;
    pub use crate::traits::{Embedder, EntryStore, Formatter, Payload, Retriever};
    pub use crate::types::{
        Entry, EvictionPolicy, FeedbackDecision, FeedbackOutcome, RegistryConfig, ScoringStrategy,
        SearchResult,
    };
}
