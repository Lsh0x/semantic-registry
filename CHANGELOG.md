# Changelog

All notable changes to this project will be documented in this file.
Maintained by [release-plz](https://release-plz.dev/) from
[conventional commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added

- `ToolRegistry` with pluggable `Embedder`, `ToolStore`, `ToolRetriever` and
  `ToolFormatter` traits, and a `search_tools` meta-tool definition
- `CandleEmbedder` (all-MiniLM-L6-v2, feature `candle`), `DummyEmbedder`
- `InMemoryToolStore` and `JsonToolStore`
- `CosineRetriever` with `ScoringStrategy::{MaxSim, WeightedMaxSim}`
- Feedback loop configured by `RegistryConfig`: semantic deduplication,
  minimum confidence score, per-tool cap with `EvictionPolicy::{Fifo,
  MostRedundantWithBase, RejectWhenFull}`, and a `FeedbackDecision` result
- Search cutoff `min_search_score`
- `McpParser` for MCP `tools/list` responses
