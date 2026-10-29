# Changelog

All notable changes to this project will be documented in this file.
Maintained by [release-plz](https://release-plz.dev/) from
[conventional commits](https://www.conventionalcommits.org/).

## [0.2.0](https://github.com/Lsh0x/tools-rag/compare/v0.1.0...v0.2.0) - 2026-10-29

### Fixed

- **`CandleEmbedder` returned near-identical vectors.** The all-MiniLM-L6-v2
  tokenizer pads every input to 128 tokens and mean pooling included the
  padding, so unrelated sentences scored around 0.75. Padding is now
  disabled: unrelated tools drop to ~0.15, correct matches sit at 0.45-0.55.
  Embeddings persisted with 0.1.0 (`JsonToolStore`) must be recomputed.
- Default `min_feedback_score` lowered from 0.40 to 0.30, calibrated on the
  corrected embeddings
- Cosine similarity clamped to [-1, 1]

### Added

- `EvictionPolicy::LeastRecentlyUsed`: evicts the example that has gone
  longest without winning a search
- `SearchResult::matched_example`: index of the example that won the search
- `CachedEmbedder` over [tiercache](https://crates.io/crates/tiercache)
  (feature `cache`), so `record_feedback` reuses the vector computed by
  `search`
- README: architecture diagram, complete feedback example, external backend
  guide (Meilisearch), MCP integration

### Breaking

- `SearchResult` has a new public field `matched_example`: struct literals
  in custom `ToolRetriever`s must set it (`None` if unknown)
- `EvictionPolicy` has a new variant: exhaustive `match`es need an arm for
  `LeastRecentlyUsed`

## [0.1.0](https://github.com/Lsh0x/tools-rag/releases/tag/v0.1.0) - 2026-10-29

First release.

### Added

- `ToolRegistry` with pluggable `Embedder`, `ToolStore`, `ToolRetriever` and
  `ToolFormatter` traits, and a `search_tools` meta-tool definition
- `CandleEmbedder` (all-MiniLM-L6-v2 on CPU, opt-in feature `candle`),
  `DummyEmbedder`; no feature is enabled by default
- `InMemoryToolStore` and `JsonToolStore`
- `CosineRetriever` with `ScoringStrategy::{MaxSim, WeightedMaxSim}`
- Feedback loop configured by `RegistryConfig`: semantic deduplication,
  minimum confidence score (corrections exempt), per-tool cap with
  `EvictionPolicy::{Fifo, MostRedundantWithBase, RejectWhenFull}`, and a
  `FeedbackDecision` result explaining each outcome
- Search cutoff `min_search_score`
- `McpParser` for MCP `tools/list` responses
- CI: fmt, clippy `-D warnings`, test matrix (default and
  `--all-features`), rustdoc `-D warnings`, MSRV 1.88, cargo-deny,
  coverage
