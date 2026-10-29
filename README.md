# tools-rag

[![CI](https://github.com/Lsh0x/tools-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Lsh0x/tools-rag/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/tools-rag.svg)](https://crates.io/crates/tools-rag)
[![docs.rs](https://docs.rs/tools-rag/badge.svg)](https://docs.rs/tools-rag)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Semantic tool retrieval for LLM agents.**

An agent with hundreds of tools cannot put every schema in its context window.
`tools-rag` embeds tool definitions, exposes a single `search_tools` meta-tool,
and returns only the tools relevant to the user's intent. It learns from usage:
validated or corrected queries become examples that sharpen future searches.

```toml
[dependencies]
tools-rag = "0.1"                                       # core only: bring your own Embedder
# tools-rag = { version = "0.1", features = ["candle"] } # + local MiniLM embedder
```

## Quick start

With the `candle` feature:

```rust
use std::sync::Arc;
use serde_json::json;
use tools_rag::defaults::cosine_retriever::CosineRetriever;
use tools_rag::embedder::CandleEmbedder;
use tools_rag::formatter::DefaultToolFormatter;
use tools_rag::prelude::*;
use tools_rag::InMemoryToolStore;

#[tokio::main]
async fn main() -> tools_rag::errors::Result<()> {
let embedder = Arc::new(CandleEmbedder::minilm_v2().await?); // all-MiniLM-L6-v2, CPU
let store = Arc::new(InMemoryToolStore::new());
let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
let registry = ToolRegistry::new(store, embedder, retriever, Box::new(DefaultToolFormatter));

registry
    .register(ToolDefinition::new(
        "get_weather",
        "Fetch current weather conditions for a location.",
        json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
    ))
    .await?;

let hits = registry.search("is it raining in Lyon?", 3).await?;
Ok(())
}
```

## How it works

- **Chunks.** A `ToolFormatter` turns each tool into one canonical chunk
  (name, description, parameters) plus one chunk per example. Each chunk is
  embedded separately.
- **MaxSim scoring.** A tool's score is its best chunk similarity, so one
  well-phrased example is enough to surface it. `ScoringStrategy::WeightedMaxSim`
  discounts examples relative to the canonical description, which keeps tools
  that accumulate feedback from becoming "magnets".
- **Feedback loop.** `record_feedback(query, tool, outcome)` stores the query
  as a new example of the right tool, behind guards set in `RegistryConfig`:

| Field | Default | Effect |
|---|---|---|
| `max_examples_per_tool` | `10` | Cap per tool; `0` disables feedback |
| `dedup_similarity_threshold` | `0.85` | Reject queries this close to an existing chunk |
| `min_feedback_score` | `0.40` | Reject a `Success` on a query far from the tool (corrections are exempt) |
| `eviction_policy` | `Fifo` | `Fifo`, `MostRedundantWithBase` or `RejectWhenFull` |
| `min_search_score` | `0.0` | Drop search results below this score |

  It returns a `FeedbackDecision` stating whether the example was added,
  what it evicted, or why it was rejected.

## Pluggable parts

Every stage is a trait: `Embedder`, `ToolStore`, `ToolRetriever`,
`ToolFormatter`. Built-ins:

- `CandleEmbedder` (feature `candle`): local inference, no API key.
- `InMemoryToolStore`, `JsonToolStore` (persists to a JSON file).
- `CosineRetriever`: brute-force cosine scoring, fine up to a few thousand tools.
- `McpParser`: converts an MCP `tools/list` response into `ToolDefinition`s.

No feature is enabled by default: the core depends only on `serde`,
`serde_json`, `async-trait`, `thiserror` and `tokio` (`fs` only). Implement
`Embedder` over any API or model, or enable `candle` for local inference.

## Examples

```bash
cargo run --features candle --example basic
cargo run --features candle --example feedback
cargo run --features candle --example persistent_store
cargo run --features candle --example llm_example_gen   # generates tool examples with Qwen2.5-0.5B
```

The first run downloads the models from the Hugging Face Hub.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
