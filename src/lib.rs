pub mod defaults;
pub mod embedder;
pub mod errors;
pub mod formatter;
pub mod meta;
pub mod parsers;
pub mod registry;
pub mod traits;
pub mod types;

pub use defaults::file_store::JsonToolStore;
pub use defaults::memory_store::InMemoryToolStore;

pub mod prelude {
    pub use crate::registry::ToolRegistry;
    pub use crate::traits::{Embedder, ToolFormatter, ToolRetriever, ToolStore};
    pub use crate::types::{
        EvictionPolicy, FeedbackDecision, FeedbackOutcome, RegistryConfig, ScoringStrategy,
        SearchResult, ToolDefinition,
    };
}
