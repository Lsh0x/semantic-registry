# Changelog

All notable changes to this project will be documented in this file.
Maintained by [release-plz](https://release-plz.dev/) from
[conventional commits](https://www.conventionalcommits.org/).

## [Unreleased]

## [0.1.0](https://github.com/Lsh0x/tools-rag/releases/tag/v0.1.0) - 2026-10-29

### Added

- make the candle embedder opt-in
- semantic tool registry with configurable feedback loop

### Fixed

- *(deps)* upgrade candle to 0.11, tokenizers to 0.23, hf-hub to 0.5

### Other

- add CI, cargo-deny, dependabot and release-plz

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
