# Changelog

All notable changes to this project will be documented in this file.
Maintained by [release-plz](https://release-plz.dev/) from
[conventional commits](https://www.conventionalcommits.org/).

## [0.2.0](https://github.com/Lsh0x/semantic-registry/compare/semantic-registry-v0.1.0...semantic-registry-v0.2.0) - 2026-10-29

### Added

- `SemanticRegistry::remove(id)`: deletes an entry, its vectors and its usage
  history, and returns whether it existed
- `Embedder::embed_batch`, with a sequential default. `CandleEmbedder` batches
  for real with masked mean pooling: same vectors as one by one, 7.6x faster
  on 64 texts. `CachedEmbedder` sends only its misses, in one call
- `register` embeds an entry's new chunks in one batch and reuses the stored
  vectors of unchanged chunks: re-registering an unchanged entry embeds
  nothing
- `mcp::search_definition(name, description)` for any search meta-tool
  (`find_agent`, `search_skills`…); `search_tools_definition()` is now a
  shortcut for it
- Every doc comment is in English, and `missing_docs` is enforced

### Breaking

- `EntryStore` has a new required method `remove`
- `EvictionPolicy` and `FeedbackDecision` are `#[non_exhaustive]`: a `match`
  on them outside the crate needs a `_ =>` arm

## [0.1.0](https://github.com/Lsh0x/semantic-registry/releases/tag/semantic-registry-v0.1.0) - 2026-10-29

First release, generalized from [`tools-rag`](https://crates.io/crates/tools-rag)
0.2.0: the registry no longer assumes its entries are tools.

### Added

- `SemanticRegistry<P>` over `Entry<P> { id, text, examples, payload }`: the
  payload is the caller's and is never read; `SemanticRegistry::in_memory`
  for the shortest setup
- `EntryStore<P>`, `Retriever<P>`, `Formatter<P>` and `Embedder` traits;
  `InMemoryStore`, `JsonStore`, `CosineRetriever` with
  `ScoringStrategy::{MaxSim, WeightedMaxSim}`
- Feedback loop configured by `RegistryConfig`: semantic deduplication,
  minimum confidence score (corrections exempt), per-entry cap with
  `EvictionPolicy::{Fifo, LeastRecentlyUsed, MostRedundantWithBase,
  RejectWhenFull}`, and a `FeedbackDecision` result
- `Formatter::example_text` also applies to examples learned through feedback
- Features: `candle` (`CandleEmbedder`, all-MiniLM-L6-v2 on CPU), `cache`
  (`CachedEmbedder` over tiercache), `mcp` (`parse_tools_list`,
  `search_tools_definition`); none enabled by default

### Migrating from tools-rag 0.2

| tools-rag | semantic-registry |
|---|---|
| `ToolRegistry::new(store, embedder, retriever, formatter)` | `SemanticRegistry::new(store, embedder, retriever)`, then `.with_formatter(..)` if needed |
| `ToolDefinition::new(name, description, parameters)` | `Entry::new(id, text, payload)` |
| `result.tool.name` | `result.entry.id` |
| `ToolStore`, `ToolRetriever`, `ToolFormatter` | `EntryStore<P>`, `Retriever<P>`, `Formatter<P>` |
| `InMemoryToolStore`, `JsonToolStore` | `InMemoryStore<P>`, `JsonStore<P>` |
| `CorrectedTo { correct_tool_name }` | `CorrectedTo { correct_id }` |
| `FeedbackDecision::ToolNotFound` | `FeedbackDecision::EntryNotFound` |
| `max_examples_per_tool` | `max_examples_per_entry` |
| `McpParser::parse_tools_list` | `mcp::parse_tools_list` (feature `mcp`), payload = the MCP tool object |
| `registry.meta_tool_definition()` | `mcp::search_tools_definition()` (feature `mcp`) |
| `ToolRagError` | `error::Error` |
