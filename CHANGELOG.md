# Changelog

All notable changes to this project will be documented in this file.
Maintained by [release-plz](https://release-plz.dev/) from
[conventional commits](https://www.conventionalcommits.org/).

## [0.1.0](https://github.com/Lsh0x/tools-rag/releases/tag/v0.1.0) - 2026-10-29

First release.

### Added

- `ToolRegistry` with pluggable `Embedder`, `ToolStore`, `ToolRetriever` and
  `ToolFormatter` traits, and a `search_tools` meta-tool definition
- `CandleEmbedder` (all-MiniLM-L6-v2 on CPU, feature `candle`, default on),
  `DummyEmbedder`
- `InMemoryToolStore` and `JsonToolStore`
- `CosineRetriever` with `ScoringStrategy::{MaxSim, WeightedMaxSim}`
- Feedback loop configured by `RegistryConfig`: semantic deduplication,
  minimum confidence score (corrections exempt), per-tool cap with
  `EvictionPolicy::{Fifo, MostRedundantWithBase, RejectWhenFull}`, and a
  `FeedbackDecision` result explaining each outcome
- Search cutoff `min_search_score`
- `McpParser` for MCP `tools/list` responses
- CI: fmt, clippy `-D warnings`, test matrix (default and
  `--no-default-features`), rustdoc `-D warnings`, MSRV 1.88, cargo-deny,
  coverage
