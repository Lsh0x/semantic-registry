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

![tools-rag architecture: register, search and feedback flows](https://raw.githubusercontent.com/Lsh0x/tools-rag/main/assets/architecture.svg)

- [Install](#install)
- [Quick start](#quick-start)
- [Learning from feedback](#learning-from-feedback)
- [Configuration](#configuration)
- [Scoring](#scoring)
- [Embedders](#embedders)
- [Stores](#stores)
- [MCP and the `search_tools` meta-tool](#mcp-and-the-search_tools-meta-tool)
- [Custom formatter](#custom-formatter)
- [Limits](#limits)
- [Examples](#examples)

## Install

```toml
[dependencies]
tools-rag = { version = "0.2", features = ["candle", "cache"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

| Feature | Adds | Dependencies |
|---|---|---|
| *(none)* | Registry, stores, retriever, feedback loop, MCP parser | `serde`, `serde_json`, `async-trait`, `thiserror`, `tokio` (`fs`) |
| `candle` | `CandleEmbedder`: all-MiniLM-L6-v2 on CPU, no API key | candle, tokenizers, hf-hub |
| `cache` | `CachedEmbedder`: embedding cache over [tiercache](https://crates.io/crates/tiercache) | tiercache (no dependencies) |

No feature is enabled by default. Without `candle`, bring your own
[`Embedder`](#your-own-embedder).

## Quick start

Each tool is a name, a description, a JSON Schema for its parameters, and
optional **examples**: phrasings a user might type. Examples are embedded on
their own, so a tool is found even when the query shares no words with its
description.

```rust
use std::sync::Arc;

use serde_json::json;
use tools_rag::defaults::cosine_retriever::CosineRetriever;
use tools_rag::embedder::CandleEmbedder;
use tools_rag::formatter::DefaultToolFormatter;
use tools_rag::prelude::*;
use tools_rag::InMemoryToolStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Downloads all-MiniLM-L6-v2 from the Hugging Face Hub on first run
    let embedder = Arc::new(CandleEmbedder::minilm_v2().await?);
    let store = Arc::new(InMemoryToolStore::new());
    let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
    let registry = ToolRegistry::new(store, embedder, retriever, Box::new(DefaultToolFormatter));

    let weather = ToolDefinition::new(
        "get_weather",
        "Fetch current weather conditions or forecasts for a location.",
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        }),
    )
    .with_examples(vec![
        "Is it going to rain in Paris tomorrow?".to_string(),
        "Do I need a coat in Oslo today?".to_string(),
    ]);

    let currency = ToolDefinition::new(
        "convert_currency",
        "Convert monetary amounts between currencies.",
        json!({
            "type": "object",
            "properties": {
                "amount": { "type": "number" },
                "from": { "type": "string", "description": "ISO code, e.g. USD" },
                "to": { "type": "string", "description": "ISO code, e.g. EUR" }
            },
            "required": ["amount", "from", "to"]
        }),
    )
    .with_examples(vec![
        "How much is $100 in euros?".to_string(),
        "Convert 5000 JPY to pounds".to_string(),
    ]);

    registry.register(weather).await?;
    registry.register(currency).await?;

    let query = "should I take an umbrella in London?";
    for hit in registry.search(query, 2).await? {
        let (name, score, matched) = (&hit.tool.name, hit.score, hit.matched_example);
        println!("{name:<18} {score:.3}  matched example: {matched:?}");
    }
    Ok(())
}
```

```text
get_weather        0.484  matched example: Some(1)
convert_currency   0.165  matched example: Some(0)
```

`matched_example` is the index in `tool.examples` of the chunk that won, or
`None` when the canonical description scored best.

## Learning from feedback

Once the agent has used a tool, tell the registry whether it was the right
one. The query becomes a new example of the correct tool, behind guards that
stop the registry from filling with duplicates or noise.

The query was already embedded by `search()`; `CachedEmbedder` makes
`record_feedback()` reuse that vector instead of running the model again.

```rust
use std::sync::Arc;

use serde_json::json;
use tools_rag::defaults::cosine_retriever::CosineRetriever;
use tools_rag::embedder::{CachedEmbedder, CandleEmbedder};
use tools_rag::formatter::DefaultToolFormatter;
use tools_rag::prelude::*;
use tools_rag::InMemoryToolStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let embedder = Arc::new(CachedEmbedder::new(
        Arc::new(CandleEmbedder::minilm_v2().await?),
        "all-MiniLM-L6-v2", // cache namespace: change it when the model changes
        1024,               // in-process LRU capacity
    ));
    let store = Arc::new(InMemoryToolStore::new());
    let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
    let registry = ToolRegistry::new(store, embedder, retriever, Box::new(DefaultToolFormatter))
        .with_config(RegistryConfig {
            max_examples_per_tool: 20,
            eviction_policy: EvictionPolicy::LeastRecentlyUsed,
            ..RegistryConfig::default()
        });

    registry
        .register(
            ToolDefinition::new(
                "get_pod_logs",
                "Fetch stdout and stderr logs of a Kubernetes pod.",
                json!({ "type": "object", "properties": { "pod": { "type": "string" } } }),
            )
            .with_examples(vec!["show me the logs of the api pod".to_string()]),
        )
        .await?;
    registry
        .register(ToolDefinition::new(
            "rollback_deployment",
            "Roll a Kubernetes deployment back to its previous revision.",
            json!({ "type": "object", "properties": { "deployment": { "type": "string" } } }),
        ))
        .await?;

    // 1. The agent picked the right tool: absorb the phrasing as an example.
    let query = "the payment pod keeps crashing, I need its output";
    let best = registry.search(query, 1).await?.remove(0);
    let decision = registry
        .record_feedback(query, &best.tool.name, FeedbackOutcome::Success)
        .await?;
    println!("{} -> {decision:?}", best.tool.name);

    // 2. The same feedback again is rejected as a duplicate.
    let again = registry
        .record_feedback(query, &best.tool.name, FeedbackOutcome::Success)
        .await?;
    println!("again -> {again:?}");

    // 3. The agent picked the wrong tool: the user names the right one.
    let query = "undo yesterday's checkout release";
    let wrong = registry.search(query, 1).await?.remove(0);
    let decision = registry
        .record_feedback(
            query,
            &wrong.tool.name,
            FeedbackOutcome::CorrectedTo {
                correct_tool_name: "rollback_deployment".to_string(),
            },
        )
        .await?;

    match decision {
        FeedbackDecision::Added { evicted } => println!("learned, evicted {evicted:?}"),
        FeedbackDecision::Duplicate { similarity } => println!("already known ({similarity:.2})"),
        FeedbackDecision::BelowMinScore { score } => println!("too far from the tool ({score:.2})"),
        FeedbackDecision::Full => println!("example quota reached"),
        FeedbackDecision::ToolNotFound => println!("no such tool"),
    }

    let after = registry.search(query, 1).await?.remove(0);
    println!("now: {} ({:.3})", after.tool.name, after.score);
    Ok(())
}
```

```text
get_pod_logs -> Added { evicted: [] }
again -> Duplicate { similarity: 1.0 }
learned, evicted []
now: rollback_deployment (1.000)
```

### The guards

`record_feedback` checks, in order:

1. **Tool exists.** Otherwise `ToolNotFound`. With `max_examples_per_tool == 0`
   feedback is disabled and every call returns `Full`.
2. **Not a duplicate.** If the query is within `dedup_similarity_threshold` of
   any chunk of the tool (description or example), it brings nothing new:
   `Duplicate`.
3. **Confident enough.** A `Success` on a query far from the tool (below
   `min_feedback_score`) is more likely a user giving up than a real match:
   `BelowMinScore`. **Corrections skip this check**: they concern a tool the
   search ranked badly, so a low score is expected.
4. **Room left.** At `max_examples_per_tool`, the eviction policy frees a
   slot, or `RejectWhenFull` answers `Full`.

### Eviction policies

| Policy | Evicts | Use when |
|---|---|---|
| `Fifo` *(default)* | The oldest example | Usage drifts over time |
| `LeastRecentlyUsed` | The example that has gone longest without winning a search | Some phrasings are common, others one-offs |
| `MostRedundantWithBase` | The example closest to the description, so the one that adds least | You want examples to cover distinct phrasings |
| `RejectWhenFull` | Nothing: new examples are refused | Curated examples must never be replaced |

`LeastRecentlyUsed` relies on `SearchResult::matched_example`: every search
that returns a tool through one of its examples marks that example as used.
A newly added example counts as used. History lives in memory, so after a
restart the policy behaves like `Fifo` until searches build it back.

## Configuration

`RegistryConfig` holds the whole business policy; set it once with
`with_config`.

| Field | Default | Effect |
|---|---|---|
| `max_examples_per_tool` | `10` | Example cap per tool; `0` disables feedback |
| `dedup_similarity_threshold` | `0.85` | Reject feedback this close to an existing chunk |
| `min_feedback_score` | `0.30` | Reject a `Success` on a query below this score (corrections exempt) |
| `eviction_policy` | `Fifo` | See [eviction policies](#eviction-policies) |
| `min_search_score` | `0.0` | `search()` drops results below this score instead of always filling `limit` |

```rust
let config = RegistryConfig {
    max_examples_per_tool: 5,
    min_search_score: 0.35, // an empty result beats an off-topic tool
    eviction_policy: EvictionPolicy::MostRedundantWithBase,
    ..RegistryConfig::default()
};
let registry = ToolRegistry::new(store, embedder, retriever, formatter).with_config(config);
```

The thresholds are cosine similarities: tune them for your embedding model.
The defaults suit all-MiniLM-L6-v2.

## Scoring

The `DefaultToolFormatter` turns a tool into chunks: one canonical chunk
(`name: description | Parameters: …`) and one per example. A tool's score is
its best chunk (MaxSim), so one well-phrased example is enough to surface it.

The flip side: a tool that has accumulated many examples through feedback
can start winning queries that belong to others. `WeightedMaxSim` discounts
examples relative to the canonical description:

```text
MaxSim          score = max(base, best_example)
WeightedMaxSim  score = max(base, best_example × example_weight)
```

```rust
let retriever = Arc::new(
    CosineRetriever::new(store.clone(), embedder.clone())
        .with_strategy(ScoringStrategy::WeightedMaxSim { example_weight: 0.85 }),
);
```

`CosineRetriever` scores every tool on each search, which is fine up to a
few thousand tools. Past that, implement `ToolRetriever` over an ANN index.

## Embedders

### Candle (feature `candle`)

`CandleEmbedder::minilm_v2()` loads sentence-transformers/all-MiniLM-L6-v2
from the Hugging Face Hub (cached locally after the first download) and runs
it on CPU in a blocking thread: 384 dimensions, mean pooling, L2-normalized.

### Cache (feature `cache`)

`CachedEmbedder` wraps any `Embedder`. Keys are `"{namespace}/{text}"`, so
two models never share vectors. `new` uses an in-process LRU; `with_cache`
takes any [tiercache](https://crates.io/crates/tiercache) `Cache`, for
example memory in front of Redis shared between instances:

```rust
use tiercache::{Cache, Lru};

let cache = Cache::new(vec![
    Box::new(Lru::new(4096)), // in-process
    Box::new(my_redis),       // your tiercache::CacheStore
]);
let embedder = Arc::new(CachedEmbedder::with_cache(inner, "all-MiniLM-L6-v2", cache));
```

A failing cache level never fails an embedding: it counts as a miss.

### Your own embedder

Any API or model works. Implement one method:

```rust
use async_trait::async_trait;
use tools_rag::errors::{Result, ToolRagError};
use tools_rag::traits::Embedder;

struct ApiEmbedder {
    client: MyHttpClient,
}

#[async_trait]
impl Embedder for ApiEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.client
            .embeddings(text)
            .await
            .map_err(|e| ToolRagError::EmbeddingFailed(e.to_string()))
    }
}
```

All tools and queries must go through the same model: vectors from
different models are not comparable.

## Stores

| Store | Persistence |
|---|---|
| `InMemoryToolStore` | None |
| `JsonToolStore` | A JSON file, rewritten on each change; reloaded with embeddings, so a restart recomputes nothing |

```rust
let store = Arc::new(JsonToolStore::load_or_create("./data/tools.json").await?);

// Embed only the tools that are not on disk yet
if store.get("convert_currency").await?.is_none() {
    registry.register(currency_tool).await?;
}
```

To use a database, implement `ToolStore`. One invariant: `embeddings[0]` is
the canonical chunk and `embeddings[i + 1]` belongs to `tool.examples[i]`;
`remove_example(name, i)` removes both.

## MCP and the `search_tools` meta-tool

`McpParser` turns an MCP `tools/list` response into `ToolDefinition`s
(`inputSchema` becomes `parameters`):

```rust
use tools_rag::parsers::mcp::McpParser;

let response: serde_json::Value = mcp_client.list_tools().await?;
for tool in McpParser::parse_tools_list(&response) {
    registry.register(tool).await?;
}
```

Instead of handing the LLM every tool, hand it one:
`registry.meta_tool_definition()` returns `search_tools(query)`. When the
model calls it, run `registry.search(query, k)` and expose the returned tools
for the next turn. Once a tool has run, call `record_feedback`.

## Custom formatter

The formatter decides what text gets embedded. Replace it to add
parameters, tags or a language prefix:

```rust
use tools_rag::traits::ToolFormatter;
use tools_rag::types::ToolDefinition;

struct DescriptionOnly;

impl ToolFormatter for DescriptionOnly {
    fn format(&self, tool: &ToolDefinition) -> Vec<String> {
        // First chunk: the canonical text. Then one chunk per example, in order.
        std::iter::once(tool.description.clone())
            .chain(tool.examples.iter().cloned())
            .collect()
    }
}
```

Keep that order: scoring and eviction rely on it.

## Limits

- **Feedback is not atomic.** Reading, evicting and appending are separate
  store calls; concurrent feedback on the same tool can briefly exceed the
  cap.
- **LRU history is in memory**, and restarts empty.
- **`JsonToolStore` rewrites the whole file** on each change: fine for
  hundreds of tools, not for a write-heavy service.
- **Brute-force search** in `CosineRetriever`, linear in the number of tools.

## Examples

```bash
cargo run --features candle --example basic             # register with examples, search
cargo run --features candle,cache --example feedback    # feedback loop with the cache
cargo run --features candle --example persistent_store  # JsonToolStore across runs
cargo run --features candle --example llm_example_gen   # generate examples with Qwen2.5-0.5B
```

## MSRV

Rust 1.88, checked in CI against the committed `Cargo.lock`.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
