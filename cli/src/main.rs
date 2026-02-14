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
#[command(name = "memoria", about = "A local-first memory layer for AI agents")]
struct Cli {
    #[arg(long, global = true, help = "Override the local store's file path")]
    store_path: Option<String>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(about = "Extract facts from a message and store them under a scope")]
    Add {
        #[arg(help = "The message to extract facts from")]
        content: String,
        #[arg(long, help = "Scope this memory to a user")]
        user_id: Option<String>,
        #[arg(long, help = "Scope this memory to an agent")]
        agent_id: Option<String>,
        #[arg(long, help = "Scope this memory to a run")]
        run_id: Option<String>,
    },
    #[command(about = "Search stored memories within a scope")]
    Search {
        #[arg(help = "The search query")]
        query: String,
        #[arg(long, help = "Search within a user's scope")]
        user_id: Option<String>,
        #[arg(long, help = "Search within an agent's scope")]
        agent_id: Option<String>,
        #[arg(long, help = "Search within a run's scope")]
        run_id: Option<String>,
        #[arg(long, default_value_t = 10, help = "Maximum number of results")]
        top_k: usize,
    },
    #[command(about = "Fetch a single memory by id")]
    Get {
        #[arg(help = "The memory's id")]
        id: String,
    },
    #[command(about = "List stored memory ids")]
    List {
        #[arg(long, default_value_t = 0, help = "Number of ids to skip")]
        offset: usize,
        #[arg(long, default_value_t = 100, help = "Maximum number of ids to return")]
        limit: usize,
    },
    #[command(about = "Update a memory's content and/or metadata")]
    Update {
        #[arg(help = "The memory's id")]
        id: String,
        #[arg(long, help = "Replace the memory's content")]
        content: Option<String>,
        #[arg(long = "set", value_parser = parse_key_value, help = "Set a metadata KEY=VALUE pair (repeatable)")]
        set: Vec<(String, String)>,
    },
    #[command(about = "Delete a memory by id (idempotent)")]
    Delete {
        #[arg(help = "The memory's id")]
        id: String,
    },
    #[command(about = "Create the local store file if it doesn't exist yet")]
    Init,
    #[command(about = "Show the active store path and providers")]
    Whoami,
    #[command(about = "Check that every provider is reachable")]
    Status,
}

#[derive(serde::Deserialize)]
struct FileConfig {
    store_path: Option<String>,
}

fn config_file_path(home: &str) -> PathBuf {
    Path::new(home).join(".memoria").join("config.json")
}

fn read_config_file(path: &Path) -> Option<FileConfig> {
    let data = fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

fn resolve_store_path(cli_override: Option<&str>, env_override: Option<&str>, file_config: Option<FileConfig>, home: &str) -> PathBuf {
    if let Some(p) = cli_override {
        return PathBuf::from(p);
    }
    if let Some(p) = env_override {
        return PathBuf::from(p);
    }
    if let Some(p) = file_config.and_then(|c| c.store_path) {
        return PathBuf::from(p);
    }
    Path::new(home).join(".memoria").join("store.json")
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

fn parse_key_value(s: &str) -> Result<(String, String), String> {
    s.split_once('=')
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .ok_or_else(|| format!("expected KEY=VALUE, got {s:?}"))
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
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let env_override = std::env::var("MEMORIA_STORE_PATH").ok();
    let file_config = read_config_file(&config_file_path(&home));
    let path = resolve_store_path(cli.store_path.as_deref(), env_override.as_deref(), file_config, &home);
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
        Command::Update { id, content, set } => {
            let metadata = if set.is_empty() { None } else { Some(set.into_iter().collect::<HashMap<_, _>>()) };
            match memory.update(&id, content.as_deref(), metadata) {
                Ok(()) => save_store(&memory, &path),
                Err(err) => {
                    eprintln!("error: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::Delete { id } => match memory.delete(&id) {
            Ok(()) => save_store(&memory, &path),
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        },
        Command::Init => {
            save_store(&memory, &path);
            println!("initialized local store at {}", path.display());
        }
        Command::Whoami => {
            println!("store: {}", path.display());
            println!("llm provider: LocalSentenceLlmProvider (local, non-AI; see ADR-12)");
            println!("embedding provider: LocalHashEmbeddingProvider (local, non-AI; see ADR-12)");
        }
        Command::Status => match memory.health_check() {
            Ok(()) => println!("ok"),
            Err(err) => {
                eprintln!("unhealthy: {err}");
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

    #[test]
    fn parse_key_value_splits_on_first_equals() {
        assert_eq!(parse_key_value("source=chat"), Ok(("source".to_string(), "chat".to_string())));
    }

    #[test]
    fn parse_key_value_rejects_missing_equals() {
        assert!(parse_key_value("no-equals-here").is_err());
    }

    #[test]
    fn parse_key_value_keeps_everything_after_first_equals() {
        assert_eq!(parse_key_value("url=http://x=y"), Ok(("url".to_string(), "http://x=y".to_string())));
    }

    #[test]
    fn resolve_store_path_cli_flag_wins_over_everything() {
        let path = resolve_store_path(Some("/from/cli"), Some("/from/env"), Some(FileConfig { store_path: Some("/from/config".to_string()) }), "/home");
        assert_eq!(path, PathBuf::from("/from/cli"));
    }

    #[test]
    fn resolve_store_path_env_wins_over_config_file() {
        let path = resolve_store_path(None, Some("/from/env"), Some(FileConfig { store_path: Some("/from/config".to_string()) }), "/home");
        assert_eq!(path, PathBuf::from("/from/env"));
    }

    #[test]
    fn resolve_store_path_config_file_wins_over_default() {
        let path = resolve_store_path(None, None, Some(FileConfig { store_path: Some("/from/config".to_string()) }), "/home");
        assert_eq!(path, PathBuf::from("/from/config"));
    }

    #[test]
    fn resolve_store_path_falls_back_to_default() {
        let path = resolve_store_path(None, None, None, "/home");
        assert_eq!(path, PathBuf::from("/home/.memoria/store.json"));
    }

    #[test]
    fn resolve_store_path_falls_back_to_default_when_config_file_has_no_store_path() {
        let path = resolve_store_path(None, None, Some(FileConfig { store_path: None }), "/home");
        assert_eq!(path, PathBuf::from("/home/.memoria/store.json"));
    }

    #[test]
    fn read_config_file_with_no_file_present_returns_none() {
        let dir = std::env::temp_dir().join(format!("memoria-cli-test-noconfig-{}", std::process::id()));
        let path = dir.join("config.json");
        assert!(read_config_file(&path).is_none());
    }

    #[test]
    fn read_config_file_reads_store_path_field() {
        let dir = std::env::temp_dir().join(format!("memoria-cli-test-config-{}", std::process::id()));
        let path = dir.join("config.json");
        fs::create_dir_all(&dir).expect("create_dir_all should succeed");
        fs::write(&path, r#"{"store_path": "/custom/store.json"}"#).expect("write should succeed");

        let config = read_config_file(&path).expect("config should be read");
        assert_eq!(config.store_path, Some("/custom/store.json".to_string()));

        let _ = fs::remove_dir_all(&dir);
    }
}
