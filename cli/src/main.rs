#![forbid(unsafe_code)]

use clap::{Parser, Subcommand};
use core::embedding::LocalHashEmbeddingProvider;
use core::llm::{LocalSentenceLlmProvider, Message, Role};
use core::memory::Memory;
use core::vector_store::{InMemoryVectorStore, VectorRecord, VectorStore};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "memoria")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Add {
        content: String,
        #[arg(long)]
        user_id: Option<String>,
        #[arg(long)]
        agent_id: Option<String>,
        #[arg(long)]
        run_id: Option<String>,
    },
    Search {
        query: String,
        #[arg(long)]
        user_id: Option<String>,
        #[arg(long)]
        agent_id: Option<String>,
        #[arg(long)]
        run_id: Option<String>,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
    },
    Get {
        id: String,
    },
    List {
        #[arg(long, default_value_t = 0)]
        offset: usize,
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },
}

fn store_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Path::new(&home).join(".memoria").join("store.json")
}

fn load_store(path: &Path) -> InMemoryVectorStore {
    let store = InMemoryVectorStore::new();
    if let Ok(data) = fs::read_to_string(path) {
        if let Ok(records) = serde_json::from_str::<Vec<VectorRecord>>(&data) {
            for record in records {
                let _ = store.insert(record);
            }
        }
    }
    store
}

fn save_store<L, E, V>(memory: &Memory<L, E, V>, path: &Path)
where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    let Ok(ids) = memory.list(0, usize::MAX) else {
        return;
    };
    let records: Vec<VectorRecord> = ids.iter().filter_map(|id| memory.get(id).ok().flatten()).collect();
    let Ok(data) = serde_json::to_string_pretty(&records) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(path, data);
}

fn build_scope(user_id: Option<String>, agent_id: Option<String>, run_id: Option<String>) -> HashMap<String, String> {
    let mut scope = HashMap::new();
    if let Some(v) = user_id {
        scope.insert("user_id".to_string(), v);
    }
    if let Some(v) = agent_id {
        scope.insert("agent_id".to_string(), v);
    }
    if let Some(v) = run_id {
        scope.insert("run_id".to_string(), v);
    }
    scope
}

fn main() {
    let cli = Cli::parse();
    let path = store_path();
    let store = load_store(&path);
    let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), store);

    match cli.command {
        Command::Add { content, user_id, agent_id, run_id } => {
            let scope = build_scope(user_id, agent_id, run_id);
            match memory.add(&[Message::new(Role::User, content)], scope) {
                Ok(ids) => {
                    for id in &ids {
                        println!("{id}");
                    }
                    save_store(&memory, &path);
                }
                Err(err) => {
                    eprintln!("error: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::Search { query, user_id, agent_id, run_id, top_k } => {
            let scope = build_scope(user_id, agent_id, run_id);
            match memory.search(&query, top_k, &scope) {
                Ok(results) => {
                    for result in results {
                        let content = result.payload.get("content").map_or("", String::as_str);
                        println!("{}\t{}\t{content}", result.id, result.score);
                    }
                }
                Err(err) => {
                    eprintln!("error: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::Get { id } => match memory.get(&id) {
            Ok(Some(record)) => {
                let content = record.payload.get("content").map_or("", String::as_str);
                println!("{}\t{content}", record.id);
            }
            Ok(None) => {
                eprintln!("not found: {id}");
                std::process::exit(1);
            }
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        },
        Command::List { offset, limit } => match memory.list(offset, limit) {
            Ok(ids) => {
                for id in ids {
                    println!("{id}");
                }
            }
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_scope_includes_only_provided_keys() {
        let scope = build_scope(Some("alice".to_string()), None, None);
        assert_eq!(scope.len(), 1);
        assert_eq!(scope.get("user_id"), Some(&"alice".to_string()));
    }

    #[test]
    fn build_scope_with_nothing_provided_is_empty() {
        let scope = build_scope(None, None, None);
        assert!(scope.is_empty());
    }

    #[test]
    fn save_then_load_round_trips_records() {
        let dir = std::env::temp_dir().join(format!("memoria-cli-test-{}", std::process::id()));
        let path = dir.join("store.json");

        let store = InMemoryVectorStore::new();
        let mut payload = HashMap::new();
        payload.insert("user_id".to_string(), "alice".to_string());
        payload.insert("content".to_string(), "Alice is an engineer.".to_string());
        store.insert(VectorRecord::new("rec-1".to_string(), vec![1.0, 2.0, 3.0], payload)).expect("insert should succeed");
        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), store);

        save_store(&memory, &path);
        let loaded = load_store(&path);

        let record = loaded.get("rec-1").expect("get should succeed").expect("record should round-trip");
        assert_eq!(record.payload.get("content"), Some(&"Alice is an engineer.".to_string()));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_store_with_no_file_present_is_empty() {
        let dir = std::env::temp_dir().join(format!("memoria-cli-test-missing-{}", std::process::id()));
        let path = dir.join("store.json");

        let loaded = load_store(&path);
        let ids = loaded.list(0, usize::MAX).expect("list should succeed");
        assert!(ids.is_empty());
    }
}
