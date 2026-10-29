use semantic_registry::prelude::*;
use semantic_registry::{InMemoryStore, JsonStore};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct Payload {
    endpoint: String,
}

fn sample(id: &str) -> Entry<Payload> {
    let payload = Payload {
        endpoint: "/t".into(),
    };
    Entry::new(id, "desc", payload).with_examples(vec!["a".into(), "b".into()])
}

fn embeddings() -> Vec<Vec<f32>> {
    vec![vec![0.0], vec![1.0], vec![2.0]]
}

async fn exercise_store(store: &dyn EntryStore<Payload>) {
    store.insert(sample("t"), embeddings()).await.unwrap();
    assert!(store.get("missing").await.unwrap().is_none());
    assert!(store.get_embeddings("missing").await.unwrap().is_none());

    store
        .append_example("t", "c".into(), vec![3.0])
        .await
        .unwrap();
    store.remove_example("t", 0).await.unwrap();
    // Out-of-bounds index: ignored
    store.remove_example("t", 42).await.unwrap();
    // Unknown entry: ignored
    store
        .append_example("missing", "x".into(), vec![9.0])
        .await
        .unwrap();

    let entry = store.get("t").await.unwrap().unwrap();
    assert_eq!(entry.examples, ["b", "c"]);
    assert_eq!(entry.payload.endpoint, "/t");
    assert_eq!(
        store.get_embeddings("t").await.unwrap().unwrap(),
        vec![vec![0.0], vec![2.0], vec![3.0]]
    );
    assert_eq!(store.list_all().await.unwrap().len(), 1);

    store.insert(sample("other"), embeddings()).await.unwrap();
    assert!(store.remove("other").await.unwrap());
    assert!(!store.remove("other").await.unwrap());
    assert!(store.get("other").await.unwrap().is_none());
    assert_eq!(store.list_all().await.unwrap().len(), 1);
}

#[tokio::test]
async fn in_memory_store_keeps_examples_and_embeddings_aligned() {
    exercise_store(&InMemoryStore::new()).await;
}

#[tokio::test]
async fn json_store_persists_across_reloads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested").join("entries.json");

    {
        let store = JsonStore::<Payload>::load_or_create(&path).await.unwrap();
        exercise_store(&store).await;
    }

    let reloaded = JsonStore::<Payload>::load_or_create(&path).await.unwrap();
    let all = reloaded.list_all().await.unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].0.examples, ["b", "c"]);
    assert_eq!(all[0].1, vec![vec![0.0], vec![2.0], vec![3.0]]);
}

#[tokio::test]
async fn json_store_treats_empty_file_as_empty_store() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("entries.json");
    std::fs::write(&path, "  \n").unwrap();

    let store = JsonStore::<Payload>::load_or_create(&path).await.unwrap();
    assert!(store.list_all().await.unwrap().is_empty());
}

#[tokio::test]
async fn json_store_rejects_corrupted_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("entries.json");
    std::fs::write(&path, "{not json").unwrap();

    assert!(JsonStore::<Payload>::load_or_create(&path).await.is_err());
}
