# semantic-registry

[![CI](https://github.com/Lsh0x/semantic-registry/actions/workflows/ci.yml/badge.svg)](https://github.com/Lsh0x/semantic-registry/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/semantic-registry.svg)](https://crates.io/crates/semantic-registry)
[![docs.rs](https://docs.rs/semantic-registry/badge.svg)](https://docs.rs/semantic-registry)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**A registry of named entries searched by meaning, which learns from usage.**

You register entries: an id, a text describing it, a few example phrasings,
and a payload of your own. You search them with a user's words. When you tell
the registry which entry was right, the query becomes a new example and
similar queries find it next time.

What an entry *is* is up to you. LLM agents use it to pick what to put in the
context window instead of everything:

- **Tools**: hundreds of tool schemas, only the relevant ones exposed per turn
- **Skills or agents**: which specialist handles the request
- **Prompts**: which system prompt or template fits the task
- **Schemas**: which SQL tables or API endpoints a question is about

The library never reads the payload: it indexes `text` and `examples`, and
hands the payload back.

![semantic-registry architecture: register, search and feedback flows](https://raw.githubusercontent.com/Lsh0x/semantic-registry/main/assets/architecture.svg)

> Formerly [`tools-rag`](https://crates.io/crates/tools-rag), which only
> handled tool definitions.

- [Install](#install)
- [Quick start](#quick-start)
- [Entries and payloads](#entries-and-payloads)
- [Learning from feedback](#learning-from-feedback)
- [Configuration](#configuration)
- [Scoring](#scoring)
- [Embedders](#embedders)
- [Stores](#stores)
- [Plugging in an external backend](#plugging-in-an-external-backend)
- [MCP tools](#mcp-tools)
- [Custom formatter](#custom-formatter)
- [Limits](#limits)
- [Examples](#examples)

## Install

```toml
[dependencies]
semantic-registry = { version = "0.1", features = ["candle", "cache"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

| Feature | Adds | Dependencies |
|---|---|---|
| *(none)* | Registry, stores, retriever, feedback loop | `serde`, `serde_json`, `async-trait`, `thiserror`, `tokio` (`fs`) |
| `candle` | `CandleEmbedder`: all-MiniLM-L6-v2 on CPU, no API key | candle, tokenizers, hf-hub |
| `cache` | `CachedEmbedder`: embedding cache over [tiercache](https://crates.io/crates/tiercache) | tiercache (no dependencies) |
| `mcp` | Ingest MCP `tools/list`, the `search_tools` definition | none |

No feature is enabled by default. Without `candle`, bring your own
[`Embedder`](#your-own-embedder).

## Quick start

```rust
use std::sync::Arc;

use semantic_registry::embedder::CandleEmbedder;
use semantic_registry::prelude::*;
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Downloads all-MiniLM-L6-v2 from the Hugging Face Hub on first run
    let embedder = Arc::new(CandleEmbedder::minilm_v2().await?);
    let registry: SemanticRegistry<Value> = SemanticRegistry::in_memory(embedder);

    let weather = Entry::new(
        "get_weather",
        "Fetch current weather conditions or forecasts for a location.",
        json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
    )
    .with_examples(vec![
        "Is it going to rain in Paris tomorrow?".to_string(),
        "Do I need a coat in Oslo today?".to_string(),
    ]);

    let currency = Entry::new(
        "convert_currency",
        "Convert monetary amounts between currencies.",
        json!({ "type": "object", "properties": { "amount": { "type": "number" } } }),
    )
    .with_examples(vec![
        "How much is $100 in euros?".to_string(),
        "Convert 5000 JPY to pounds".to_string(),
    ]);

    registry.register(weather).await?;
    registry.register(currency).await?;

    let query = "should I take an umbrella in London?";
    for hit in registry.search(query, 2).await? {
        let (id, score, matched) = (&hit.entry.id, hit.score, hit.matched_example);
        println!("{id:<18} {score:.3}  matched example: {matched:?}");
    }
    Ok(())
}
```

```text
get_weather        0.484  matched example: Some(1)
convert_currency   0.165  matched example: Some(0)
```

Each hit carries the whole `Entry`, payload included (`hit.entry.payload`).
`matched_example` is the index in `entry.examples` of the chunk that won, or
`None` when the canonical `text` scored best.

## Entries and payloads

```rust
pub struct Entry<P> {
    pub id: String,            // unique key: a tool name, a skill id, a table name
    pub text: String,          // the canonical description, embedded as is
    pub examples: Vec<String>, // phrasings users might type; feedback adds more
    pub payload: P,            // yours: never read by the library
}
```

`P` is any `Clone + Send + Sync + 'static` type: your own struct, a
`serde_json::Value`, or `()` when the id is enough. `JsonStore` also needs it
to be `Serialize + DeserializeOwned`.

Examples matter: they are embedded one by one, so an entry is found even when
the query shares no words with its `text`. Seed a few by hand, or generate
them with an LLM (see `examples/llm_example_gen.rs`).

**Several collections.** The registry knows nothing about kinds. Keeping tools
and skills apart is two registries, each with its own config; ranking them
together is one registry over an enum payload. Share the embedder between
registries so the model loads once and the cache serves both:

```rust
let tools: SemanticRegistry<ToolSpec> = SemanticRegistry::in_memory(embedder.clone());
let skills: SemanticRegistry<SkillSpec> = SemanticRegistry::in_memory(embedder.clone());
```

### Example: picking a prompt

[`examples/prompt_routing.rs`](examples/prompt_routing.rs) keeps prompt
templates in the payload, picks one per request, and falls back to a generic
prompt when nothing scores above `min_search_score`:

```rust
#[derive(Clone, Debug)]
struct Prompt {
    system: &'static str,
    template: &'static str,
}

let registry: SemanticRegistry<Prompt> =
    SemanticRegistry::in_memory(embedder).with_config(RegistryConfig {
        min_search_score: 0.30, // below this, no template fits
        ..RegistryConfig::default()
    });

match registry.search(request, 1).await?.into_iter().next() {
    Some(hit) => println!("{}", hit.entry.payload.render(request)),
    None => println!("{}", FALLBACK.render(request)),
}
```

```text
is this rust function safe? fn get(v: &[u8], i: usize) -> u8 { v[i] }
  -> code_review (0.437)
what does SELECT name FROM users WHERE age > 30 GROUP BY name do
  -> explain_sql (0.404)
give me a commit message, I renamed the crate and its types
  -> commit_message (0.533)
what's a good name for a cat
  -> no template fits, fallback

why is this query slow
  picked explain_sql, corrected to code_review: Added { evicted: [] }
  "why is this query slow" -> code_review (1.000)
  "how do I make this query faster" -> code_review (0.895)
```

The last lines show a correction at work: once the user says a performance
question belongs to `code_review`, close phrasings follow it.

## Learning from feedback

Once the caller has used an entry, tell the registry whether it was the right
one. The query becomes a new example of the correct entry, behind guards that
stop the registry from filling with duplicates or noise.

The query was already embedded by `search()`; `CachedEmbedder` makes
`record_feedback()` reuse that vector instead of running the model again.

```rust
use std::sync::Arc;

use semantic_registry::embedder::{CachedEmbedder, CandleEmbedder};
use semantic_registry::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let embedder = Arc::new(CachedEmbedder::new(
        Arc::new(CandleEmbedder::minilm_v2().await?),
        "all-MiniLM-L6-v2", // cache namespace: change it when the model changes
        1024,               // in-process LRU capacity
    ));
    let registry: SemanticRegistry<()> =
        SemanticRegistry::in_memory(embedder).with_config(RegistryConfig {
            max_examples_per_entry: 20,
            eviction_policy: EvictionPolicy::LeastRecentlyUsed,
            ..RegistryConfig::default()
        });

    registry
        .register(
            Entry::new(
                "get_pod_logs",
                "Fetch stdout and stderr logs of a Kubernetes pod.",
                (),
            )
            .with_examples(vec!["show me the logs of the api pod".to_string()]),
        )
        .await?;
    registry
        .register(Entry::new(
            "rollback_deployment",
            "Roll a Kubernetes deployment back to its previous revision.",
            (),
        ))
        .await?;

    // 1. The right entry was picked: absorb the phrasing as an example.
    let query = "the payment pod keeps crashing, I need its output";
    let best = registry.search(query, 1).await?.remove(0);
    let decision = registry
        .record_feedback(query, &best.entry.id, FeedbackOutcome::Success)
        .await?;
    println!("{} -> {decision:?}", best.entry.id);

    // 2. The same feedback again is rejected as a duplicate.
    let again = registry
        .record_feedback(query, &best.entry.id, FeedbackOutcome::Success)
        .await?;
    println!("again -> {again:?}");

    // 3. The wrong entry was picked: the user names the right one.
    let query = "undo yesterday's checkout release";
    let wrong = registry.search(query, 1).await?.remove(0);
    let decision = registry
        .record_feedback(
            query,
            &wrong.entry.id,
            FeedbackOutcome::CorrectedTo {
                correct_id: "rollback_deployment".to_string(),
            },
        )
        .await?;

    match decision {
        FeedbackDecision::Added { evicted } => println!("learned, evicted {evicted:?}"),
        FeedbackDecision::Duplicate { similarity } => println!("already known ({similarity:.2})"),
        FeedbackDecision::BelowMinScore { score } => println!("too far off ({score:.2})"),
        FeedbackDecision::Full => println!("example quota reached"),
        FeedbackDecision::EntryNotFound => println!("no such entry"),
    }

    let after = registry.search(query, 1).await?.remove(0);
    println!("now: {} ({:.3})", after.entry.id, after.score);
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

1. **Entry exists.** Otherwise `EntryNotFound`. With
   `max_examples_per_entry == 0` feedback is disabled and every call returns
   `Full`.
2. **Not a duplicate.** If the query is within `dedup_similarity_threshold` of
   any chunk of the entry (text or example), it brings nothing new:
   `Duplicate`.
3. **Confident enough.** A `Success` on a query far from the entry (below
   `min_feedback_score`) is more likely a user giving up than a real match:
   `BelowMinScore`. **Corrections skip this check**: they concern an entry the
   search ranked badly, so a low score is expected.
4. **Room left.** At `max_examples_per_entry`, the eviction policy frees a
   slot, or `RejectWhenFull` answers `Full`.

### Eviction policies

| Policy | Evicts | Use when |
|---|---|---|
| `Fifo` *(default)* | The oldest example | Usage drifts over time |
| `LeastRecentlyUsed` | The example that has gone longest without winning a search | Some phrasings are common, others one-offs |
| `MostRedundantWithBase` | The example closest to `text`, so the one that adds least | You want examples to cover distinct phrasings |
| `RejectWhenFull` | Nothing: new examples are refused | Curated examples must never be replaced |

`LeastRecentlyUsed` relies on `SearchResult::matched_example`: every search
that returns an entry through one of its examples marks that example as used.
A newly added example counts as used. History lives in memory, so after a
restart the policy behaves like `Fifo` until searches build it back.

## Configuration

`RegistryConfig` holds the whole policy; set it once with `with_config`.

| Field | Default | Effect |
|---|---|---|
| `max_examples_per_entry` | `10` | Example cap per entry; `0` disables feedback |
| `dedup_similarity_threshold` | `0.85` | Reject feedback this close to an existing chunk |
| `min_feedback_score` | `0.30` | Reject a `Success` on a query below this score (corrections exempt) |
| `eviction_policy` | `Fifo` | See [eviction policies](#eviction-policies) |
| `min_search_score` | `0.0` | `search()` drops results below this score instead of always filling `limit` |

```rust
let config = RegistryConfig {
    max_examples_per_entry: 5,
    min_search_score: 0.35, // an empty result beats an off-topic entry
    eviction_policy: EvictionPolicy::MostRedundantWithBase,
    ..RegistryConfig::default()
};
let registry: SemanticRegistry<()> = SemanticRegistry::in_memory(embedder).with_config(config);
```

The thresholds are cosine similarities: tune them for your embedding model.
The defaults suit all-MiniLM-L6-v2, where right matches score 0.45-0.60 and
unrelated entries around 0.15.

## Scoring

Each entry is embedded as chunks: `text`, then one chunk per example. An
entry's score is its best chunk (MaxSim), so one well-phrased example is
enough to surface it.

The flip side: an entry that has accumulated many examples through feedback
can start winning queries that belong to others. `WeightedMaxSim` discounts
examples relative to the canonical text:

```text
MaxSim          score = max(text, best_example)
WeightedMaxSim  score = max(text, best_example × example_weight)
```

`SemanticRegistry::in_memory` uses plain MaxSim. For the weighted variant,
build the parts yourself:

```rust
use semantic_registry::{CosineRetriever, InMemoryStore};

let store: Arc<dyn EntryStore<()>> = Arc::new(InMemoryStore::new());
let retriever = Arc::new(
    CosineRetriever::new(store.clone(), embedder.clone())
        .with_strategy(ScoringStrategy::WeightedMaxSim { example_weight: 0.85 }),
);
let registry = SemanticRegistry::new(store, embedder, retriever);
```

`CosineRetriever` scores every entry on each search, which is fine up to a
few thousand entries. Past that, use an
[external backend](#plugging-in-an-external-backend).

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
use semantic_registry::error::{Error, Result};
use semantic_registry::traits::Embedder;

struct ApiEmbedder {
    client: MyHttpClient,
}

#[async_trait]
impl Embedder for ApiEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.client
            .embeddings(text)
            .await
            .map_err(|e| Error::EmbeddingFailed(e.to_string()))
    }
}
```

Every entry and query must go through the same model: vectors from
different models are not comparable.

## Stores

| Store | Persistence |
|---|---|
| `InMemoryStore` | None |
| `JsonStore` | A JSON file, rewritten on each change; reloaded with embeddings, so a restart recomputes nothing |

```rust
let store = Arc::new(JsonStore::<MyPayload>::load_or_create("./data/entries.json").await?);
let retriever = Arc::new(CosineRetriever::new(store.clone(), embedder.clone()));
let registry = SemanticRegistry::new(store.clone(), embedder, retriever);

// Embed only the entries that are not on disk yet
if store.get("convert_currency").await?.is_none() {
    registry.register(currency).await?;
}
```

To use a database, implement `EntryStore` (see
[external backends](#plugging-in-an-external-backend)). One invariant:
`embeddings[0]` is the canonical chunk and `embeddings[i + 1]` belongs to
`entry.examples[i]`; `remove_example(id, i)` removes both.

## Plugging in an external backend

`EntryStore` and `Retriever` are separate traits, and the registry does not
know about `CosineRetriever`. A search engine that stores vectors can take
over both: implement the two traits on one type and pass it twice.

```rust
let backend = Arc::new(MeiliBackend { client, embedder: embedder.clone() });
let registry = SemanticRegistry::new(backend.clone(), embedder, backend);
```

The registry, the feedback guards and the cache keep working unchanged.
Below, [Meilisearch](https://www.meilisearch.com) as an example
([`meilisearch-sdk`](https://crates.io/crates/meilisearch-sdk) 0.33), with a
`serde_json::Value` payload.

### Layout

Two indexes:

- `entries`: one document per `Entry`, primary key `id`.
- `chunks`: one document per chunk, the text or one example, with its
  vector.

Storing chunks as separate documents gives MaxSim for free: searching with
`distinct = entry` keeps each entry's best chunk, and that chunk tells which
example matched, so `LeastRecentlyUsed` keeps working.

```json
{
  "filterableAttributes": ["entry", "example"],
  "embedders": {
    "semantic-registry": { "source": "userProvided", "dimensions": 384 }
  }
}
```

`userProvided` means this library computes the vectors and Meilisearch only
stores and searches them. Keep it that way: the feedback guards compare the
query vector with the entry's vectors in Rust, so both sides must come from
the same model.

### Retriever

```rust
use std::sync::Arc;

use async_trait::async_trait;
use meilisearch_sdk::client::Client;
use meilisearch_sdk::errors::{Error as MeiliError, ErrorCode, MeilisearchError};
use semantic_registry::error::{Error, Result};
use semantic_registry::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One document per chunk in the `chunks` index
#[derive(Serialize, Deserialize)]
struct Chunk {
    id: String,
    entry: String,
    /// `None` for the canonical chunk, the example text otherwise
    example: Option<String>,
    #[serde(rename = "_vectors", default, skip_serializing_if = "Value::is_null")]
    vectors: Value,
}

pub struct MeiliBackend {
    client: Client,
    embedder: Arc<dyn Embedder>,
}

fn store_err(e: MeiliError) -> Error {
    Error::StoreError(e.to_string())
}

#[async_trait]
impl Retriever<Value> for MeiliBackend {
    async fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<SearchResult<Value>>> {
        let vector = self.embedder.embed(query).await?;
        let hits = self
            .client
            .index("chunks")
            .search()
            .with_vector(&vector)
            .with_hybrid("semantic-registry", 1.0) // 1.0 = vectors only; lower it to mix in keywords
            .with_distinct("entry") // best chunk per entry: MaxSim, done by Meilisearch
            .with_show_ranking_score(true)
            .with_limit(limit)
            .execute::<Chunk>()
            .await
            .map_err(store_err)?
            .hits;

        let mut results = Vec::with_capacity(hits.len());
        for hit in hits {
            let Some(entry) = self.get(&hit.result.entry).await? else {
                continue;
            };
            let matched_example = hit
                .result
                .example
                .and_then(|text| entry.examples.iter().position(|e| *e == text));
            results.push(SearchResult {
                entry,
                score: hit.ranking_score.unwrap_or(0.0) as f32,
                matched_example,
            });
        }
        Ok(results)
    }
}

#[async_trait]
impl EntryStore<Value> for MeiliBackend {
    async fn get(&self, id: &str) -> Result<Option<Entry<Value>>> {
        match self.client.index("entries").get_document(id).await {
            Ok(entry) => Ok(Some(entry)),
            Err(MeiliError::Meilisearch(MeilisearchError {
                error_code: ErrorCode::DocumentNotFound,
                ..
            })) => Ok(None),
            Err(e) => Err(store_err(e)),
        }
    }

    // insert, get_embeddings, list_all, append_example, remove_example:
    // see the table below
}
```

### Store

| Method | Meilisearch |
|---|---|
| `insert` | Replace the document in `entries`; delete its chunks (filter `entry = "…"`); add one chunk per embedding, `example: None` for the first |
| `get` | `get_document` on `entries` (above) |
| `get_embeddings` | `get_documents` on `chunks` with filter `entry = "…"` and `retrieve_vectors`; canonical chunk first, then examples in `entry.examples` order |
| `append_example` | Add the text to `entry.examples` in `entries`; add one chunk |
| `remove_example(i)` | Remove `entry.examples[i]` in `entries`; delete the chunk with filter `entry = "…" AND example = "…"` |
| `list_all` | Not called when Meilisearch is the retriever: only `CosineRetriever` uses it |

Chunk ids must be unique and contain only `a-z A-Z 0-9 - _`: derive them from
the entry id and a hash of the example text.

### What changes

- **Scores.** `_rankingScore` is not a raw cosine, especially in hybrid mode.
  Recalibrate `min_search_score`, `min_feedback_score` and
  `dedup_similarity_threshold` on your data.
- **Hybrid search.** A `semantic_ratio` below `1.0` mixes keyword ranking in:
  useful when queries name things exactly (`kubectl`, `S3`).
- **`WeightedMaxSim`** has no Meilisearch equivalent. To keep it, fetch more
  hits with `retrieve_vectors` and rescore them in Rust.
- **One lookup per hit.** `retrieve` fetches each `Entry` with `get`; for large
  `limit`s, fetch them in one `get_documents` call with an `id IN [...]`
  filter.

## MCP tools

The `mcp` feature covers the protocol-agnostic parts of using the registry
for MCP tools. The library is not an MCP server or client: transport,
authentication, and how the tools found are made callable stay with your
application.

**Ingesting tools.** `parse_tools_list` turns a `tools/list` response into
entries: `id` is the tool name, `text` its name, description and parameters,
and `payload` the full MCP tool object, `inputSchema` included.

```rust
use semantic_registry::mcp;

let response: serde_json::Value = mcp_client.list_tools().await?;
for entry in mcp::parse_tools_list(&response) {
    registry.register(entry).await?;
}
```

**Exposing search.** Instead of handing the model every tool, hand it one:
`mcp::search_tools_definition()` is the MCP definition of `search_tools(query)`.
Declare it in your agent loop or MCP server and answer its calls with
`registry.search`. The payloads are the MCP tool objects, ready to return:

```rust
/// Answers a `search_tools` call, whatever transport delivered it
async fn handle_search_tools(
    registry: &SemanticRegistry<Value>,
    arguments: &Value,
) -> semantic_registry::error::Result<Value> {
    let query = arguments["query"].as_str().unwrap_or_default();
    let hits = registry.search(query, 5).await?;
    let tools: Vec<&Value> = hits.iter().map(|hit| &hit.entry.payload).collect();
    Ok(json!({ "tools": tools }))
}
```

The model then calls the tools it found. In an MCP server, either add them to
your `tools/list` and send `notifications/tools/list_changed`, or expose a
generic `call_tool(name, arguments)` that forwards to the upstream server;
the second works with clients that ignore list changes. Once a tool has run,
call `record_feedback`.

## Custom formatter

The formatter decides what text gets embedded: `canonical_text` for the entry,
`example_text` for each example. The default embeds `text` and the examples
as they are. Replace it to add the prefixes some models expect, or to render
fields from the payload:

```rust
use semantic_registry::traits::Formatter;

struct E5Prefixes;

impl<P> Formatter<P> for E5Prefixes {
    fn canonical_text(&self, entry: &Entry<P>) -> String {
        format!("passage: {}", entry.text)
    }

    fn example_text(&self, _entry: &Entry<P>, example: &str) -> String {
        format!("query: {example}")
    }
}

let registry: SemanticRegistry<()> =
    SemanticRegistry::in_memory(embedder).with_formatter(E5Prefixes);
```

Examples learned through feedback go through `example_text` too, so seeded
and learned examples are embedded the same way. Search queries are embedded
as given: wrap your `Embedder` if the model needs a prefix on them.

## Limits

- **Feedback is not atomic.** Reading, evicting and appending are separate
  store calls; concurrent feedback on the same entry can briefly exceed the
  cap.
- **LRU history is in memory**, and restarts empty.
- **`JsonStore` rewrites the whole file** on each change: fine for hundreds
  of entries, not for a write-heavy service.
- **Brute-force search** in `CosineRetriever`, linear in the number of
  entries.

## Examples

```bash
cargo run --features candle --example basic             # register with examples, search
cargo run --features candle,cache --example feedback    # feedback loop with the cache
cargo run --features candle --example persistent_store  # JsonStore across runs
cargo run --features candle --example prompt_routing    # pick a prompt template, with fallback
cargo run --features candle --example llm_example_gen   # generate examples with Qwen2.5-0.5B
```

## MSRV

Rust 1.88, checked in CI against the committed `Cargo.lock`.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
