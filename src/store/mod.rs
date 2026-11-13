//! Built-in stores: in memory, or persisted to a JSON file, and a wrapper that reports
//! changes to an index.

/// [`IndexedStore`]: any store, reporting its changes to an [`IndexSink`](crate::traits::IndexSink).
pub mod indexed;
/// [`JsonStore`]: entries persisted to a JSON file.
pub mod json;
/// [`InMemoryStore`]: entries kept in memory.
pub mod memory;

pub use indexed::IndexedStore;
pub use json::JsonStore;
pub use memory::InMemoryStore;
