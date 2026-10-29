//! Built-in stores: in memory, or persisted to a JSON file.

/// [`JsonStore`]: entries persisted to a JSON file.
pub mod json;
/// [`InMemoryStore`]: entries kept in memory.
pub mod memory;

pub use json::JsonStore;
pub use memory::InMemoryStore;
