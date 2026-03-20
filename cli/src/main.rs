#![forbid(unsafe_code)]
#![allow(clippy::multiple_crate_versions)]

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::Shell;
use core::embedding::LocalHashEmbeddingProvider;
use core::llm::{LocalSentenceLlmProvider, Message, Role};
use core::memory::Memory;
use core::vector_store::{InMemoryVectorStore, VectorRecord, VectorStore};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "memoria", version = env!("CARGO_PKG_VERSION"), about = "A local-first memory layer for AI agents")]
struct Cli {
    #[arg(long, global = true, help = "Override the local store's file path")]
    store_path: Option<String>,
    #[arg(long, global = true, help = "Print output as JSON instead of plain text")]
    json: bool,
    #[arg(long, global = true, help = "Suppress normal output; errors still print to stderr")]
    quiet: bool,
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
        #[arg(long, help = "Store CONTENT verbatim instead of extracting facts from it")]
        no_infer: bool,
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
        #[arg(long, help = "Exclude results whose score exceeds this value")]
        threshold: Option<f32>,
        #[arg(long, help = "Rerank results using the configured reranker (MEMORIA_RERANKER)")]
        rerank: bool,
        #[arg(long, value_parser = parse_filter_arg, help = "Filter results with a JSON filter expression (ADR-30)")]
        filter: Option<core::filter::FilterExpr>,
    },
    #[command(about = "Fetch a single memory by id")]
    Get {
        #[arg(help = "The memory's id")]
        id: String,
    },
    #[command(about = "List stored memory ids within a scope")]
    List {
        #[arg(long, help = "List within a user's scope")]
        user_id: Option<String>,
        #[arg(long, help = "List within an agent's scope")]
        agent_id: Option<String>,
        #[arg(long, help = "List within a run's scope")]
        run_id: Option<String>,
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
    #[command(about = "Show the add/delete change log for a single memory")]
    History {
        #[arg(help = "The memory's id")]
        id: String,
        #[arg(long, default_value_t = 0, help = "Number of entries to skip")]
        offset: usize,
        #[arg(long, default_value_t = 100, help = "Maximum number of entries to return")]
        limit: usize,
    },
    #[command(about = "Create the local store file if it doesn't exist yet")]
    Init,
    #[command(about = "Show the active store path and providers")]
    Whoami,
    #[command(about = "Check that every provider is reachable")]
    Status,
    #[command(about = "Print a shell completion script to stdout")]
    Completions {
        #[arg(help = "Which shell to generate the completion script for")]
        shell: Shell,
    },
}

fn write_completions(shell: Shell, writer: &mut dyn std::io::Write) {
    let mut cmd = Cli::command();
    let name = cmd.get_name().to_string();
    clap_complete::generate(shell, &mut cmd, name, writer);
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
    let Ok(ids) = memory.list_all(0, usize::MAX, true, None) else {
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

fn resolve_history_path(store_path: &Path) -> PathBuf {
    store_path.with_file_name("history.json")
}

fn load_history(path: &Path) -> HashMap<String, Vec<core::memory::HistoryEntry>> {
    fs::read_to_string(path).ok().and_then(|data| serde_json::from_str(&data).ok()).unwrap_or_default()
}

fn save_history<L, E, V>(memory: &Memory<L, E, V>, path: &Path)
where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    let Ok(data) = serde_json::to_string_pretty(&memory.history_snapshot()) else {
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

fn parse_filter_arg(s: &str) -> Result<core::filter::FilterExpr, String> {
    let value: serde_json::Value = serde_json::from_str(s).map_err(|err| format!("malformed filter JSON: {err}"))?;
    core::filter::parse_filter_expr(&value).map_err(|err| format!("malformed filters: {err}"))
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

fn print_ids(ids: &[String], json: bool, quiet: bool) {
    if quiet {
        return;
    }
    if json {
        println!("{}", serde_json::to_string(ids).unwrap_or_default());
    } else {
        for id in ids {
            println!("{id}");
        }
    }
}

fn print_search_results(results: &[core::vector_store::SearchResult], json: bool, quiet: bool) {
    if quiet {
        return;
    }
    if json {
        println!("{}", serde_json::to_string(results).unwrap_or_default());
    } else {
        for result in results {
            let content = result.payload.get("content").map_or("", String::as_str);
            println!("{}\t{}\t{content}", result.id, result.score);
        }
    }
}

fn print_record(record: &VectorRecord, json: bool, quiet: bool) {
    if quiet {
        return;
    }
    if json {
        println!("{}", serde_json::to_string(record).unwrap_or_default());
    } else {
        let content = record.payload.get("content").map_or("", String::as_str);
        println!("{}\t{content}", record.id);
    }
}

fn print_history(entries: &[core::memory::HistoryEntry], json: bool, quiet: bool) {
    if quiet {
        return;
    }
    if json {
        println!("{}", serde_json::to_string(entries).unwrap_or_default());
    } else {
        for entry in entries {
            let event = match entry.event {
                core::memory::HistoryEvent::Added => "added",
                core::memory::HistoryEvent::Deleted => "deleted",
            };
            println!("{event}\t{}", entry.content);
        }
    }
}

#[derive(serde::Serialize)]
struct WhoamiInfo {
    store: String,
    llm_provider: String,
    embedding_provider: String,
    reranker: Option<String>,
}

fn resolve_llm_provider(
    provider_choice: Option<&str>,
    model: Option<String>,
    base_url: Option<String>,
    candle_cache_dir: Option<String>,
) -> Result<(Box<dyn core::llm::LlmProvider + Send + Sync>, String), String> {
    match provider_choice.unwrap_or("local") {
        "local" => Ok((
            Box::new(LocalSentenceLlmProvider::new()),
            "LocalSentenceLlmProvider (local, non-AI; see ADR-12)".to_string(),
        )),
        "ollama" => {
            #[cfg(feature = "ollama")]
            {
                let model = model.unwrap_or_else(|| "qwen2.5:0.5b".to_string());
                let resolved_base_url = base_url.clone().unwrap_or_else(|| "http://localhost:11434".to_string());
                let config = core::llm::LlmConfig { model: model.clone(), base_url, api_key: None, temperature: None };
                let provider = core::llm::OllamaLlmProvider::from_config(config).map_err(|err| err.to_string())?;
                let label = format!("OllamaLlmProvider (model={model}, base_url={resolved_base_url}; see ADR-25)");
                Ok((Box::new(provider), label))
            }
            #[cfg(not(feature = "ollama"))]
            {
                let _ = (model, base_url, candle_cache_dir);
                Err("MEMORIA_LLM_PROVIDER=ollama requires the cli binary to be built with --features ollama".to_string())
            }
        }
        "candle" => {
            #[cfg(feature = "candle")]
            {
                let model = model.unwrap_or_else(|| "qwen2.5-0.5b-instruct-q4_0".to_string());
                let config = core::llm::LlmConfig { model: model.clone(), base_url, api_key: None, temperature: None };
                let cache_dir = candle_cache_dir.map(std::path::PathBuf::from);
                let provider = core::llm::CandleLlmProvider::from_config(&config, cache_dir).map_err(|err| err.to_string())?;
                let label = format!("CandleLlmProvider (model={model}; see ADR-40)");
                Ok((Box::new(provider), label))
            }
            #[cfg(not(feature = "candle"))]
            {
                let _ = (model, base_url, candle_cache_dir);
                Err("MEMORIA_LLM_PROVIDER=candle requires the cli binary to be built with --features candle".to_string())
            }
        }
        other => Err(format!("unknown MEMORIA_LLM_PROVIDER value {other:?} (expected \"local\", \"ollama\", or \"candle\")")),
    }
}

fn resolve_embedding_provider(
    provider_choice: Option<&str>,
    model: Option<String>,
    base_url: Option<String>,
    fastembed_cache_dir: Option<String>,
) -> Result<(Box<dyn core::embedding::EmbeddingProvider>, String), String> {
    match provider_choice.unwrap_or("local") {
        "local" => Ok((
            Box::new(LocalHashEmbeddingProvider::new()),
            "LocalHashEmbeddingProvider (local, non-AI; see ADR-12)".to_string(),
        )),
        "ollama" => {
            #[cfg(feature = "ollama")]
            {
                let model = model.unwrap_or_else(|| "nomic-embed-text".to_string());
                let resolved_base_url = base_url.clone().unwrap_or_else(|| "http://localhost:11434".to_string());
                let config = core::embedding::EmbeddingConfig { model: model.clone(), base_url, api_key: None, dimensions: None };
                let provider = core::embedding::OllamaEmbeddingProvider::from_config(config).map_err(|err| err.to_string())?;
                let label = format!("OllamaEmbeddingProvider (model={model}, base_url={resolved_base_url}; see ADR-24)");
                Ok((Box::new(provider), label))
            }
            #[cfg(not(feature = "ollama"))]
            {
                let _ = (model, base_url, fastembed_cache_dir);
                Err("MEMORIA_EMBEDDING_PROVIDER=ollama requires the cli binary to be built with --features ollama".to_string())
            }
        }
        "fastembed" => {
            #[cfg(feature = "fastembed")]
            {
                let model = model.unwrap_or_else(|| "all-MiniLM-L6-v2".to_string());
                let config = core::embedding::EmbeddingConfig { model: model.clone(), base_url: None, api_key: None, dimensions: None };
                let cache_dir = fastembed_cache_dir.clone().map(std::path::PathBuf::from);
                let provider = core::embedding::FastEmbedEmbeddingProvider::from_config(&config, cache_dir).map_err(|err| err.to_string())?;
                let cache_label = fastembed_cache_dir.unwrap_or_else(|| ".fastembed_cache (default)".to_string());
                let label = format!("FastEmbedEmbeddingProvider (model={model}, cache_dir={cache_label}; see ADR-32)");
                Ok((Box::new(provider), label))
            }
            #[cfg(not(feature = "fastembed"))]
            {
                let _ = (model, base_url, fastembed_cache_dir);
                Err("MEMORIA_EMBEDDING_PROVIDER=fastembed requires the cli binary to be built with --features fastembed".to_string())
            }
        }
        other => Err(format!("unknown MEMORIA_EMBEDDING_PROVIDER value {other:?} (expected \"local\", \"ollama\", or \"fastembed\")")),
    }
}

type RerankerResolution = Result<Option<(Box<dyn core::reranker::Reranker + Send + Sync>, String)>, String>;

#[allow(clippy::too_many_arguments)]
fn resolve_reranker(
    choice: Option<&str>,
    llm_choice: Option<&str>,
    llm_model: Option<String>,
    llm_base_url: Option<String>,
    llm_candle_cache_dir: Option<String>,
    fastembed_cache_dir: Option<String>,
) -> RerankerResolution {
    match choice {
        None => Ok(None),
        Some("local") => Ok(Some((
            Box::new(core::reranker::LocalOverlapReranker::new()),
            "LocalOverlapReranker (local, non-AI; see ADR-36)".to_string(),
        ))),
        Some("llm") => {
            let (llm_provider, llm_label) = resolve_llm_provider(llm_choice, llm_model, llm_base_url, llm_candle_cache_dir)?;
            let label = format!("LlmReranker (llm={llm_label}; see ADR-36)");
            Ok(Some((Box::new(core::reranker::LlmReranker::new(llm_provider)), label)))
        }
        Some("fastembed") => {
            #[cfg(feature = "fastembed")]
            {
                let cache_dir = fastembed_cache_dir.clone().map(std::path::PathBuf::from);
                let reranker = core::reranker::FastEmbedReranker::new(cache_dir).map_err(|err| err.to_string())?;
                let cache_label = fastembed_cache_dir.unwrap_or_else(|| ".fastembed_cache (default)".to_string());
                let label = format!("FastEmbedReranker (model=BAAI/bge-reranker-base, cache_dir={cache_label}; see ADR-50)");
                Ok(Some((Box::new(reranker), label)))
            }
            #[cfg(not(feature = "fastembed"))]
            {
                let _ = (fastembed_cache_dir,);
                Err("MEMORIA_RERANKER=fastembed requires the cli binary to be built with --features fastembed".to_string())
            }
        }
        Some(other) => Err(format!("unknown MEMORIA_RERANKER value {other:?} (expected \"local\", \"llm\", or \"fastembed\")")),
    }
}

fn main() {
    let cli = Cli::parse();
    if let Command::Completions { shell } = &cli.command {
        write_completions(*shell, &mut std::io::stdout());
        return;
    }
    let json = cli.json;
    let quiet = cli.quiet;
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let env_override = std::env::var("MEMORIA_STORE_PATH").ok();
    let file_config = read_config_file(&config_file_path(&home));
    let path = resolve_store_path(cli.store_path.as_deref(), env_override.as_deref(), file_config, &home);
    let store = load_store(&path);

    let llm_provider_choice = std::env::var("MEMORIA_LLM_PROVIDER").ok();
    let (llm_provider, llm_label) = match resolve_llm_provider(
        llm_provider_choice.as_deref(),
        std::env::var("MEMORIA_LLM_MODEL").ok(),
        std::env::var("MEMORIA_LLM_BASE_URL").ok(),
        std::env::var("MEMORIA_CANDLE_CACHE_DIR").ok(),
    ) {
        Ok(resolved) => resolved,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };
    let embedding_provider_choice = std::env::var("MEMORIA_EMBEDDING_PROVIDER").ok();
    let (embedding_provider, embedding_label) = match resolve_embedding_provider(
        embedding_provider_choice.as_deref(),
        std::env::var("MEMORIA_EMBEDDING_MODEL").ok(),
        std::env::var("MEMORIA_EMBEDDING_BASE_URL").ok(),
        std::env::var("MEMORIA_FASTEMBED_CACHE_DIR").ok(),
    ) {
        Ok(resolved) => resolved,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };

    let reranker_choice = std::env::var("MEMORIA_RERANKER").ok();
    let reranker = match resolve_reranker(
        reranker_choice.as_deref(),
        llm_provider_choice.as_deref(),
        std::env::var("MEMORIA_LLM_MODEL").ok(),
        std::env::var("MEMORIA_LLM_BASE_URL").ok(),
        std::env::var("MEMORIA_CANDLE_CACHE_DIR").ok(),
        std::env::var("MEMORIA_FASTEMBED_CACHE_DIR").ok(),
    ) {
        Ok(resolved) => resolved,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };
    let reranker_label = reranker.as_ref().map(|(_, label)| label.clone());

    let mut memory = Memory::new(llm_provider, embedding_provider, store);
    if let Some((reranker, _)) = reranker {
        memory = memory.with_reranker(reranker);
    }
    let history_path = resolve_history_path(&path);
    memory.load_history_snapshot(load_history(&history_path));

    run(cli.command, &memory, &path, &history_path, json, quiet, &llm_label, &embedding_label, reranker_label.as_deref());
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn run<L, E, V>(
    command: Command,
    memory: &Memory<L, E, V>,
    path: &Path,
    history_path: &Path,
    json: bool,
    quiet: bool,
    llm_label: &str,
    embedding_label: &str,
    reranker_label: Option<&str>,
) where
    L: core::llm::LlmProvider,
    E: core::embedding::EmbeddingProvider,
    V: core::vector_store::VectorStore,
{
    match command {
        Command::Add { content, user_id, agent_id, run_id, no_infer } => {
            let scope = build_scope(user_id, agent_id, run_id);
            match memory.add(&[Message::new(Role::User, content)], scope, !no_infer) {
                Ok(ids) => {
                    print_ids(&ids, json, quiet);
                    save_store(memory, path);
                    save_history(memory, history_path);
                }
                Err(err) => {
                    eprintln!("error: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::Search { query, user_id, agent_id, run_id, top_k, threshold, rerank, filter } => {
            let scope = build_scope(user_id, agent_id, run_id);
            match memory.search(&query, top_k, &scope, threshold, true, filter.as_ref(), rerank) {
                Ok(results) => print_search_results(&results, json, quiet),
                Err(err) => {
                    eprintln!("error: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::Get { id } => match memory.get(&id) {
            Ok(Some(record)) => print_record(&record, json, quiet),
            Ok(None) => {
                eprintln!("not found: {id}");
                std::process::exit(1);
            }
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        },
        Command::List { user_id, agent_id, run_id, offset, limit } => {
            let scope = build_scope(user_id, agent_id, run_id);
            match memory.list(&scope, offset, limit, true, None) {
                Ok(ids) => print_ids(&ids, json, quiet),
                Err(err) => {
                    eprintln!("error: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::Update { id, content, set } => {
            let metadata = if set.is_empty() { None } else { Some(set.into_iter().collect::<HashMap<_, _>>()) };
            match memory.update(&id, content.as_deref(), metadata) {
                Ok(()) => save_store(memory, path),
                Err(err) => {
                    eprintln!("error: {err}");
                    std::process::exit(1);
                }
            }
        }
        Command::Delete { id } => match memory.delete(&id) {
            Ok(()) => {
                save_store(memory, path);
                save_history(memory, history_path);
            }
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        },
        Command::History { id, offset, limit } => match memory.history(&id, offset, limit) {
            Ok(entries) => print_history(&entries, json, quiet),
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        },
        Command::Init => {
            save_store(memory, path);
            if !quiet {
                println!("initialized local store at {}", path.display());
            }
        }
        Command::Whoami => {
            if !quiet {
                let info = WhoamiInfo {
                    store: path.display().to_string(),
                    llm_provider: llm_label.to_string(),
                    embedding_provider: embedding_label.to_string(),
                    reranker: reranker_label.map(ToString::to_string),
                };
                if json {
                    println!("{}", serde_json::to_string(&info).unwrap_or_default());
                } else {
                    println!("store: {}", info.store);
                    println!("llm provider: {}", info.llm_provider);
                    println!("embedding provider: {}", info.embedding_provider);
                    if let Some(reranker) = &info.reranker {
                        println!("reranker: {reranker}");
                    }
                }
            }
        }
        Command::Status => match memory.health_check() {
            Ok(()) => {
                if !quiet {
                    println!("ok");
                }
            }
            Err(err) => {
                eprintln!("unhealthy: {err}");
                std::process::exit(1);
            }
        },
        Command::Completions { .. } => unreachable!("main() handles and returns on Command::Completions before run() is ever called"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_llm_provider_defaults_to_local() {
        let (_, label) = resolve_llm_provider(None, None, None, None).expect("expected a provider");
        assert!(label.contains("LocalSentenceLlmProvider"), "got: {label}");
    }

    #[test]
    fn resolve_llm_provider_rejects_an_unknown_choice() {
        let result = resolve_llm_provider(Some("bogus"), None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_embedding_provider_defaults_to_local() {
        let (_, label) = resolve_embedding_provider(None, None, None, None).expect("expected a provider");
        assert!(label.contains("LocalHashEmbeddingProvider"), "got: {label}");
    }

    #[test]
    fn resolve_embedding_provider_rejects_an_unknown_choice() {
        let result = resolve_embedding_provider(Some("bogus"), None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_reranker_with_nothing_set_resolves_to_none() {
        let resolved = resolve_reranker(None, None, None, None, None, None).expect("expected a resolution");
        assert!(resolved.is_none(), "no MEMORIA_RERANKER should mean no reranker is configured, not a default one");
    }

    #[test]
    fn resolve_reranker_local_choice_resolves_to_a_real_reranker() {
        let (_, label) = resolve_reranker(Some("local"), None, None, None, None, None).expect("expected a resolution").expect("expected a reranker");
        assert!(label.contains("LocalOverlapReranker"), "got: {label}");
    }

    #[test]
    fn resolve_reranker_llm_choice_resolves_to_a_real_reranker() {
        let (_, label) = resolve_reranker(Some("llm"), None, None, None, None, None).expect("expected a resolution").expect("expected a reranker");
        assert!(label.contains("LlmReranker"), "got: {label}");
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn resolve_reranker_llm_choice_honors_the_underlying_llm_provider_choice() {
        let (_, label) = resolve_reranker(Some("llm"), Some("ollama"), None, None, None, None).expect("expected a resolution").expect("expected a reranker");
        assert!(label.contains("OllamaLlmProvider"), "got: {label}");
    }

    #[test]
    fn resolve_reranker_rejects_an_unknown_choice() {
        assert!(resolve_reranker(Some("bogus"), None, None, None, None, None).is_err());
    }

    #[cfg(not(feature = "fastembed"))]
    #[test]
    fn resolve_reranker_fastembed_choice_fails_clearly_without_the_feature() {
        assert!(resolve_reranker(Some("fastembed"), None, None, None, None, None).is_err());
    }

    #[cfg(feature = "fastembed")]
    #[test]
    #[ignore = "downloads a real ~278MB cross-encoder model + ONNX runtime on first run; needs real network access"]
    fn resolve_reranker_fastembed_choice_builds_with_defaults() {
        let cache_dir = std::env::var("MEMORIA_TEST_FASTEMBED_CACHE_DIR").ok();
        let (_, label) = resolve_reranker(Some("fastembed"), None, None, None, None, cache_dir).expect("expected a resolution").expect("expected a reranker");
        assert!(label.contains("BAAI/bge-reranker-base"), "got: {label}");
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn resolve_llm_provider_ollama_choice_builds_with_defaults() {
        let (_, label) = resolve_llm_provider(Some("ollama"), None, None, None).expect("expected a provider");
        assert!(label.contains("qwen2.5:0.5b"), "got: {label}");
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn resolve_llm_provider_ollama_choice_honors_a_custom_model() {
        let (_, label) = resolve_llm_provider(Some("ollama"), Some("custom-model".to_string()), None, None).expect("expected a provider");
        assert!(label.contains("custom-model"), "got: {label}");
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn resolve_embedding_provider_ollama_choice_builds_with_defaults() {
        let (_, label) = resolve_embedding_provider(Some("ollama"), None, None, None).expect("expected a provider");
        assert!(label.contains("nomic-embed-text"), "got: {label}");
    }

    #[cfg(not(feature = "fastembed"))]
    #[test]
    fn resolve_embedding_provider_fastembed_choice_fails_clearly_without_the_feature() {
        assert!(resolve_embedding_provider(Some("fastembed"), None, None, None).is_err());
    }

    #[cfg(feature = "fastembed")]
    #[test]
    #[ignore = "downloads a real ~188MB model + ONNX runtime on first run; needs real network access"]
    fn resolve_embedding_provider_fastembed_choice_builds_with_defaults() {
        let cache_dir = std::env::var("MEMORIA_TEST_FASTEMBED_CACHE_DIR").ok();
        let (_, label) = resolve_embedding_provider(Some("fastembed"), None, None, cache_dir).expect("expected a provider");
        assert!(label.contains("all-MiniLM-L6-v2"), "got: {label}");
    }

    #[cfg(not(feature = "ollama"))]
    #[test]
    fn resolve_llm_provider_ollama_choice_fails_clearly_without_the_feature() {
        let result = resolve_llm_provider(Some("ollama"), None, None, None);
        assert!(result.is_err());
    }

    #[cfg(not(feature = "candle"))]
    #[test]
    fn resolve_llm_provider_candle_choice_fails_clearly_without_the_feature() {
        assert!(resolve_llm_provider(Some("candle"), None, None, None).is_err());
    }

    #[cfg(feature = "candle")]
    #[test]
    #[ignore = "downloads a real ~430MB GGUF model + tokenizer on first run; needs real network access"]
    fn resolve_llm_provider_candle_choice_builds_with_defaults() {
        let cache_dir = std::env::var("MEMORIA_TEST_CANDLE_CACHE_DIR").ok();
        let (_, label) = resolve_llm_provider(Some("candle"), None, None, cache_dir).expect("expected a provider");
        assert!(label.contains("qwen2.5-0.5b-instruct-q4_0"), "got: {label}");
    }

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
    fn resolve_history_path_is_a_sibling_of_the_store_path() {
        let path = resolve_history_path(&PathBuf::from("/home/alice/.memoria/store.json"));
        assert_eq!(path, PathBuf::from("/home/alice/.memoria/history.json"));
    }

    #[test]
    fn load_history_for_a_missing_file_is_empty_not_an_error() {
        let dir = std::env::temp_dir().join(format!("memoria-cli-history-test-missing-{}", std::process::id()));
        let history = load_history(&dir.join("never-created.json"));
        assert!(history.is_empty());
    }

    #[test]
    fn save_then_load_history_round_trips_real_add_and_delete_events() {
        let dir = std::env::temp_dir().join(format!("memoria-cli-history-test-{}", std::process::id()));
        let history_path = dir.join("history.json");

        let memory = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let ids = memory
            .add(&[Message::new(Role::User, "Alice is an engineer.")], HashMap::from([("user_id".to_string(), "alice".to_string())]), false)
            .expect("add should succeed");
        let id = ids.first().expect("expected an id");
        memory.delete(id).expect("delete should succeed");

        save_history(&memory, &history_path);
        let loaded = load_history(&history_path);

        let entries = loaded.get(id).expect("expected history for this id");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].event, core::memory::HistoryEvent::Added);
        assert_eq!(entries[1].event, core::memory::HistoryEvent::Deleted);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_fresh_memory_loaded_from_a_real_history_snapshot_reports_the_same_history() {
        let dir = std::env::temp_dir().join(format!("memoria-cli-history-test-crossproc-{}", std::process::id()));
        let history_path = dir.join("history.json");

        let first = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        let ids = first
            .add(&[Message::new(Role::User, "Bob likes tea.")], HashMap::from([("user_id".to_string(), "bob".to_string())]), false)
            .expect("add should succeed");
        let id = ids.first().expect("expected an id").clone();
        save_history(&first, &history_path);

        let second = Memory::new(LocalSentenceLlmProvider::new(), LocalHashEmbeddingProvider::new(), InMemoryVectorStore::new());
        second.load_history_snapshot(load_history(&history_path));
        let entries = second.history(&id, 0, usize::MAX).expect("history should succeed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].event, core::memory::HistoryEvent::Added);

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
    fn parse_filter_arg_parses_a_bare_equality_condition() {
        let parsed = parse_filter_arg(r#"{"user_id":"alice"}"#).expect("expected a valid filter");
        assert_eq!(
            parsed,
            core::filter::FilterExpr::Field("user_id".to_string(), core::filter::FilterOp::Eq(core::filter::FilterValue::String("alice".to_string())))
        );
    }

    #[test]
    fn parse_filter_arg_rejects_malformed_json() {
        let err = parse_filter_arg("not json").expect_err("expected a parse error");
        assert!(err.contains("malformed filter JSON"), "got: {err}");
    }

    #[test]
    fn parse_filter_arg_rejects_a_non_object_filter() {
        let err = parse_filter_arg(r#""just a string""#).expect_err("expected a validation error");
        assert!(err.contains("malformed filters"), "got: {err}");
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

    #[test]
    fn write_completions_for_bash_mentions_every_real_subcommand() {
        let mut buf = Vec::new();
        write_completions(Shell::Bash, &mut buf);
        let script = String::from_utf8(buf).expect("completion script should be valid utf8");

        assert!(script.contains("memoria"), "got: {script}");
        for subcommand in ["add", "search", "get", "list", "update", "delete", "history", "init", "whoami", "status", "completions"] {
            assert!(script.contains(subcommand), "bash completion script should mention {subcommand:?}, got: {script}");
        }
    }

    #[test]
    fn write_completions_covers_every_real_shell_clap_complete_supports() {
        for shell in [Shell::Bash, Shell::Elvish, Shell::Fish, Shell::PowerShell, Shell::Zsh] {
            let mut buf = Vec::new();
            write_completions(shell, &mut buf);
            assert!(!buf.is_empty(), "{shell} completion script should not be empty");
        }
    }
}
