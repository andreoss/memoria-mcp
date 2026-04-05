use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

pub const SERVER_NAME: &str = "memoria";

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Client {
    ClaudeCode,
    ClaudeDesktop,
    Cursor,
    Opencode,
    Codex,
}

impl Client {
    pub const ALL: [Self; 5] = [Self::ClaudeCode, Self::ClaudeDesktop, Self::Cursor, Self::Opencode, Self::Codex];

    pub const fn name(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::ClaudeDesktop => "claude-desktop",
            Self::Cursor => "cursor",
            Self::Opencode => "opencode",
            Self::Codex => "codex",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Environment {
    pub home: PathBuf,
    pub xdg_config_home: Option<PathBuf>,
    pub claude_config_dir: Option<PathBuf>,
    pub codex_home: Option<PathBuf>,
    pub macos: bool,
}

impl Environment {
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>, macos: bool) -> Option<Self> {
        let path = |name: &str| lookup(name).filter(|value| !value.is_empty()).map(PathBuf::from);
        Some(Self {
            home: path("HOME")?,
            xdg_config_home: path("XDG_CONFIG_HOME"),
            claude_config_dir: path("CLAUDE_CONFIG_DIR"),
            codex_home: path("CODEX_HOME"),
            macos,
        })
    }

    fn config_home(&self) -> PathBuf {
        self.xdg_config_home.clone().unwrap_or_else(|| self.home.join(".config"))
    }
}

impl Client {
    pub fn config_dir(self, environment: &Environment) -> PathBuf {
        match self {
            Self::ClaudeCode => environment.claude_config_dir.clone().unwrap_or_else(|| environment.home.join(".claude")),
            Self::ClaudeDesktop if environment.macos => environment.home.join("Library").join("Application Support").join("Claude"),
            Self::ClaudeDesktop => environment.config_home().join("Claude"),
            Self::Cursor => environment.home.join(".cursor"),
            Self::Opencode => environment.config_home().join("opencode"),
            Self::Codex => environment.codex_home.clone().unwrap_or_else(|| environment.home.join(".codex")),
        }
    }

    pub fn config_path(self, environment: &Environment) -> PathBuf {
        match self {
            Self::ClaudeCode => environment.claude_config_dir.as_ref().unwrap_or(&environment.home).join(".claude.json"),
            Self::ClaudeDesktop => self.config_dir(environment).join("claude_desktop_config.json"),
            Self::Cursor => self.config_dir(environment).join("mcp.json"),
            Self::Opencode => {
                let json = self.config_dir(environment).join("opencode.json");
                let jsonc = self.config_dir(environment).join("opencode.jsonc");
                if !json.exists() && jsonc.exists() {
                    jsonc
                } else {
                    json
                }
            }
            Self::Codex => self.config_dir(environment).join("config.toml"),
        }
    }

    pub fn is_detected(self, environment: &Environment) -> bool {
        self.config_path(environment).is_file() || self.config_dir(environment).is_dir()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerEntry {
    pub command: String,
    pub env: BTreeMap<String, String>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Edit {
    Added(String),
    Updated(String),
    Removed(String),
    Unchanged,
    Absent,
}

impl Edit {
    pub const fn action(&self) -> &'static str {
        match self {
            Self::Added(_) => "added",
            Self::Updated(_) => "updated",
            Self::Removed(_) => "removed",
            Self::Unchanged => "unchanged",
            Self::Absent => "absent",
        }
    }

    pub fn new_contents(&self) -> Option<&str> {
        match self {
            Self::Added(text) | Self::Updated(text) | Self::Removed(text) => Some(text),
            Self::Unchanged | Self::Absent => None,
        }
    }
}

pub fn upsert(client: Client, existing: Option<&str>, entry: &ServerEntry) -> Result<Edit, String> {
    if client == Client::Codex {
        Ok(upsert_toml(existing.unwrap_or_default(), entry))
    } else {
        upsert_json(client, existing, entry)
    }
}

pub fn remove(client: Client, existing: Option<&str>) -> Result<Edit, String> {
    if client == Client::Codex {
        Ok(remove_toml(existing.unwrap_or_default()))
    } else {
        remove_json(client, existing)
    }
}

const TOML_TABLE: &str = "mcp_servers.memoria";

fn toml_string(value: &str) -> String {
    let mut out = String::from('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if control.is_control() => {
                let _ = write!(out, "\\u{:04X}", u32::from(control));
            }
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

fn toml_key(key: &str) -> String {
    let bare = !key.is_empty() && key.chars().all(|character| character.is_ascii_alphanumeric() || character == '_' || character == '-');
    if bare {
        key.to_owned()
    } else {
        toml_string(key)
    }
}

fn render_toml_block(entry: &ServerEntry, preserved_env: &[String]) -> Vec<String> {
    let mut lines = vec![format!("[{TOML_TABLE}]"), format!("command = {}", toml_string(&entry.command)), "args = []".to_owned()];
    if !entry.env.is_empty() || !preserved_env.is_empty() {
        lines.push(String::new());
        lines.push(format!("[{TOML_TABLE}.env]"));
        lines.extend(entry.env.iter().map(|(key, value)| format!("{} = {}", toml_key(key), toml_string(value))));
        lines.extend(preserved_env.iter().cloned());
    }
    lines
}

fn toml_line_key(line: &str) -> Option<String> {
    let (key, _) = line.split_once('=')?;
    let key = key.trim();
    Some(key.strip_prefix('"').and_then(|inner| inner.strip_suffix('"')).unwrap_or(key).to_owned())
}

fn preserved_toml_env(lines: &[&str], ranges: &[(usize, usize)], entry: &ServerEntry) -> Vec<String> {
    let env_table = format!("{TOML_TABLE}.env");
    let mut preserved = Vec::new();
    for &(start, end) in ranges {
        let mut in_env = false;
        for line in &lines[start..end] {
            if let Some(name) = toml_table_name(line) {
                in_env = name == env_table;
            } else if in_env && !is_blank_or_comment(line) && toml_line_key(line).is_some_and(|key| !entry.env.contains_key(&key)) {
                preserved.push(line.trim().to_owned());
            }
        }
    }
    preserved
}

fn toml_table_name(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let inner = trimmed.strip_prefix("[[").map_or_else(|| trimmed.strip_prefix('['), Some)?;
    let name = &inner[..inner.find(']')?];
    Some(name.chars().filter(|character| !character.is_whitespace()).collect())
}

fn is_memoria_table(name: &str) -> bool {
    name == TOML_TABLE || name.strip_prefix(TOML_TABLE).is_some_and(|rest| rest.starts_with('.'))
}

fn is_blank_or_comment(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.is_empty() || trimmed.starts_with('#')
}

fn memoria_toml_ranges(lines: &[&str]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        if !toml_table_name(lines[index]).is_some_and(|name| is_memoria_table(&name)) {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        while index < lines.len() && toml_table_name(lines[index]).is_none_or(|name| is_memoria_table(&name)) {
            index += 1;
        }
        let mut end = index;
        while end > start + 1 && is_blank_or_comment(lines[end - 1]) {
            end -= 1;
        }
        ranges.push((start, end));
    }
    ranges
}

fn join_toml(lines: &[String]) -> String {
    let mut end = lines.len();
    while end > 0 && lines[end - 1].trim().is_empty() {
        end -= 1;
    }
    if end == 0 {
        return String::new();
    }
    lines[..end].join("\n") + "\n"
}

fn without_ranges(lines: &[&str], ranges: &[(usize, usize)]) -> (Vec<String>, Vec<usize>) {
    let mut kept = Vec::new();
    let mut seams = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if let Some((start, _)) = ranges.iter().find(|(start, end)| (*start..*end).contains(&index)) {
            if index == *start {
                seams.push(kept.len());
            }
            continue;
        }
        kept.push((*line).to_owned());
    }
    (kept, seams)
}

fn upsert_toml(existing: &str, entry: &ServerEntry) -> Edit {
    let lines: Vec<&str> = existing.lines().collect();
    let ranges = memoria_toml_ranges(&lines);
    let (mut kept, seams) = without_ranges(&lines, &ranges);
    let block = render_toml_block(entry, &preserved_toml_env(&lines, &ranges, entry));
    if let Some(&position) = seams.first() {
        kept.splice(position..position, block);
    } else {
        while kept.last().is_some_and(|line| line.trim().is_empty()) {
            kept.pop();
        }
        if !kept.is_empty() {
            kept.push(String::new());
        }
        kept.extend(block);
    }
    let updated = join_toml(&kept);
    if updated == existing {
        Edit::Unchanged
    } else if ranges.is_empty() {
        Edit::Added(updated)
    } else {
        Edit::Updated(updated)
    }
}

fn remove_toml(existing: &str) -> Edit {
    let lines: Vec<&str> = existing.lines().collect();
    let ranges = memoria_toml_ranges(&lines);
    if ranges.is_empty() {
        return Edit::Absent;
    }
    let (mut kept, seams) = without_ranges(&lines, &ranges);
    for &seam in seams.iter().rev() {
        while seam < kept.len() && kept[seam].trim().is_empty() && (seam == 0 || kept[seam - 1].trim().is_empty()) {
            kept.remove(seam);
        }
    }
    Edit::Removed(join_toml(&kept))
}

const fn json_container_key(client: Client) -> &'static str {
    match client {
        Client::Opencode => "mcp",
        _ => "mcpServers",
    }
}

fn env_object(env: &BTreeMap<String, String>) -> Value {
    Value::Object(env.iter().map(|(key, value)| (key.clone(), Value::String(value.clone()))).collect())
}

fn json_entry(client: Client, entry: &ServerEntry) -> Value {
    let mut object = Map::new();
    if client == Client::Opencode {
        object.insert("type".to_owned(), Value::from("local"));
        object.insert("command".to_owned(), Value::Array(vec![Value::from(entry.command.clone())]));
        object.insert("enabled".to_owned(), Value::Bool(true));
        object.insert("environment".to_owned(), env_object(&entry.env));
    } else {
        if client == Client::ClaudeCode {
            object.insert("type".to_owned(), Value::from("stdio"));
        }
        object.insert("command".to_owned(), Value::from(entry.command.clone()));
        object.insert("args".to_owned(), Value::Array(Vec::new()));
        object.insert("env".to_owned(), env_object(&entry.env));
    }
    Value::Object(object)
}

fn parse_json_root(existing: Option<&str>) -> Result<Map<String, Value>, String> {
    let Some(text) = existing.filter(|text| !text.trim().is_empty()) else {
        return Ok(Map::new());
    };
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(root)) => Ok(root),
        Ok(_) => Err("top-level value is not a JSON object".to_owned()),
        Err(error) => Err(format!("not valid JSON ({error})")),
    }
}

fn render_json(root: &Map<String, Value>) -> Result<String, String> {
    serde_json::to_string_pretty(root).map(|text| text + "\n").map_err(|error| error.to_string())
}

const fn json_env_key(client: Client) -> &'static str {
    match client {
        Client::Opencode => "environment",
        _ => "env",
    }
}

fn with_previous_json_env(client: Client, previous: Option<&Value>, entry: &ServerEntry) -> ServerEntry {
    let mut merged = entry.clone();
    if let Some(Value::Object(old_env)) = previous.and_then(|old| old.get(json_env_key(client))) {
        for (key, value) in old_env {
            if let Value::String(text) = value {
                merged.env.entry(key.clone()).or_insert_with(|| text.clone());
            }
        }
    }
    merged
}

fn upsert_json(client: Client, existing: Option<&str>, entry: &ServerEntry) -> Result<Edit, String> {
    let key = json_container_key(client);
    let mut root = parse_json_root(existing)?;
    let Value::Object(servers) = root.entry(key).or_insert_with(|| Value::Object(Map::new())) else {
        return Err(format!("`{key}` is not a JSON object"));
    };
    let desired = json_entry(client, &with_previous_json_env(client, servers.get(SERVER_NAME), entry));
    let previous = servers.insert(SERVER_NAME.to_owned(), desired.clone());
    match previous {
        Some(old) if old == desired => Ok(Edit::Unchanged),
        Some(_) => render_json(&root).map(Edit::Updated),
        None => render_json(&root).map(Edit::Added),
    }
}

fn remove_json(client: Client, existing: Option<&str>) -> Result<Edit, String> {
    let key = json_container_key(client);
    let mut root = parse_json_root(existing)?;
    let removed = match root.get_mut(key) {
        Some(Value::Object(servers)) => servers.remove(SERVER_NAME).is_some(),
        Some(_) => return Err(format!("`{key}` is not a JSON object")),
        None => false,
    };
    if removed {
        render_json(&root).map(Edit::Removed)
    } else {
        Ok(Edit::Absent)
    }
}

#[derive(Debug, clap::Parser)]
#[command(name = "memoria-mcp", version = env!("CARGO_PKG_VERSION"), about = "memoria MCP server; runs on stdio when no subcommand is given")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, clap::Subcommand)]
pub enum Command {
    #[command(about = "Add the memoria server to detected MCP clients' configuration")]
    Install(InstallArgs),
    #[command(about = "Remove the memoria server from detected MCP clients' configuration")]
    Uninstall(UninstallArgs),
}

#[derive(Debug, clap::Args)]
pub struct InstallArgs {
    #[arg(long = "client", value_enum, help = "Configure this client even if not detected (repeatable); default: every detected client")]
    pub clients: Vec<Client>,
    #[arg(long, help = "Report what would change without writing anything")]
    pub dry_run: bool,
    #[arg(long, help = "Server command to write; default: this binary's own path")]
    pub command: Option<PathBuf>,
    #[arg(long = "env", value_parser = parse_env_pair, help = "KEY=VALUE written into the server's environment (repeatable)")]
    pub env: Vec<(String, String)>,
}

#[derive(Debug, clap::Args)]
pub struct UninstallArgs {
    #[arg(long = "client", value_enum, help = "Only this client (repeatable); default: every detected client")]
    pub clients: Vec<Client>,
    #[arg(long, help = "Report what would change without writing anything")]
    pub dry_run: bool,
}

fn parse_env_pair(value: &str) -> Result<(String, String), String> {
    match value.split_once('=') {
        Some((key, assigned)) if !key.is_empty() => Ok((key.to_owned(), assigned.to_owned())),
        _ => Err(format!("expected KEY=VALUE, got {value:?}")),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation<'a> {
    Install(&'a ServerEntry),
    Uninstall,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Report {
    pub client: Client,
    pub path: PathBuf,
    pub outcome: Result<&'static str, String>,
}

impl Report {
    pub fn line(&self) -> String {
        let action = match &self.outcome {
            Ok(action) => (*action).to_owned(),
            Err(reason) => format!("error: {reason}"),
        };
        format!("{}\t{action}\t{}", self.client.name(), self.path.display())
    }
}

pub fn run(operation: Operation<'_>, requested: &[Client], environment: &Environment, dry_run: bool) -> Vec<Report> {
    Client::ALL
        .into_iter()
        .filter(|client| requested.is_empty() || requested.contains(client))
        .map(|client| {
            let path = client.config_path(environment);
            let outcome = if requested.is_empty() && !client.is_detected(environment) {
                Ok("not-detected")
            } else {
                apply(operation, client, &path, dry_run)
            };
            Report { client, path, outcome }
        })
        .collect()
}

fn apply(operation: Operation<'_>, client: Client, path: &Path, dry_run: bool) -> Result<&'static str, String> {
    let existing = match std::fs::read_to_string(path) {
        Ok(text) => Some(text),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(format!("cannot read ({error})")),
    };
    let edit = match operation {
        Operation::Install(entry) => upsert(client, existing.as_deref(), entry)?,
        Operation::Uninstall => remove(client, existing.as_deref())?,
    };
    if let (Some(contents), false) = (edit.new_contents(), dry_run) {
        replace_file(path, existing.is_some(), contents).map_err(|error| format!("cannot write ({error})"))?;
    }
    Ok(edit.action())
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn replace_file(path: &Path, exists: bool, contents: &str) -> std::io::Result<()> {
    if !exists {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        return std::fs::write(path, contents);
    }
    let target = std::fs::canonicalize(path)?;
    std::fs::copy(&target, with_suffix(path, ".bak"))?;
    let permissions = std::fs::metadata(&target)?.permissions();
    let temp = with_suffix(&target, &format!(".memoria-tmp.{}", std::process::id()));
    std::fs::write(&temp, contents)?;
    std::fs::set_permissions(&temp, permissions)?;
    std::fs::rename(&temp, &target)
}

pub fn execute(command: Command) -> i32 {
    let Some(environment) = Environment::from_lookup(|name| std::env::var(name).ok(), cfg!(target_os = "macos")) else {
        eprintln!("HOME is not set; cannot locate client configuration");
        return 1;
    };
    let reports = match command {
        Command::Install(args) => {
            let command_path = match args.command.map_or_else(std::env::current_exe, Ok).and_then(std::fs::canonicalize) {
                Ok(path) => path,
                Err(error) => {
                    eprintln!("cannot resolve the server command path ({error})");
                    return 1;
                }
            };
            let entry = ServerEntry { command: command_path.to_string_lossy().into_owned(), env: args.env.into_iter().collect() };
            run(Operation::Install(&entry), &args.clients, &environment, args.dry_run)
        }
        Command::Uninstall(args) => run(Operation::Uninstall, &args.clients, &environment, args.dry_run),
    };
    for report in &reports {
        println!("{}", report.line());
    }
    i32::from(reports.iter().any(|report| report.outcome.is_err()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn entry() -> ServerEntry {
        ServerEntry { command: "/opt/memoria/mcp".to_owned(), env: BTreeMap::new() }
    }

    fn parsed(edit: &Edit) -> Value {
        serde_json::from_str(edit.new_contents().expect("edit should carry contents")).expect("output should be JSON")
    }

    #[test]
    fn json_upsert_into_missing_file_adds_the_entry() {
        let edit = upsert(Client::Cursor, None, &entry()).expect("upsert should succeed");
        assert_eq!(edit.action(), "added");
        assert_eq!(parsed(&edit)["mcpServers"]["memoria"]["command"], "/opt/memoria/mcp");
        assert_eq!(parsed(&edit)["mcpServers"]["memoria"]["args"], serde_json::json!([]));
    }

    #[test]
    fn json_upsert_into_blank_file_adds_the_entry() {
        let edit = upsert(Client::ClaudeDesktop, Some("  \n"), &entry()).expect("upsert should succeed");
        assert_eq!(edit.action(), "added");
    }

    #[test]
    fn json_upsert_keeps_other_servers_and_top_level_keys() {
        let existing = r#"{"theme":"dark","mcpServers":{"other":{"command":"x"}}}"#;
        let edit = upsert(Client::ClaudeCode, Some(existing), &entry()).expect("upsert should succeed");
        let value = parsed(&edit);
        assert_eq!(value["theme"], "dark");
        assert_eq!(value["mcpServers"]["other"]["command"], "x");
        assert_eq!(value["mcpServers"]["memoria"]["type"], "stdio");
    }

    #[test]
    fn json_upsert_replaces_a_stale_entry() {
        let existing = r#"{"mcpServers":{"memoria":{"command":"/old/path"}}}"#;
        let edit = upsert(Client::Cursor, Some(existing), &entry()).expect("upsert should succeed");
        assert_eq!(edit.action(), "updated");
        assert_eq!(parsed(&edit)["mcpServers"]["memoria"]["command"], "/opt/memoria/mcp");
    }

    #[test]
    fn json_upsert_is_unchanged_on_repeat() {
        let first = upsert(Client::Opencode, None, &entry()).expect("upsert should succeed");
        let second = upsert(Client::Opencode, first.new_contents(), &entry()).expect("upsert should succeed");
        assert_eq!(second, Edit::Unchanged);
    }

    #[test]
    fn json_upsert_writes_env_into_the_entry() {
        let mut with_env = entry();
        with_env.env.insert("MEMORIA_SQLITE_PATH".to_owned(), "/data/m.db".to_owned());
        let claude = parsed(&upsert(Client::ClaudeCode, None, &with_env).expect("upsert should succeed"));
        assert_eq!(claude["mcpServers"]["memoria"]["env"]["MEMORIA_SQLITE_PATH"], "/data/m.db");
        let opencode = parsed(&upsert(Client::Opencode, None, &with_env).expect("upsert should succeed"));
        assert_eq!(opencode["mcp"]["memoria"]["environment"]["MEMORIA_SQLITE_PATH"], "/data/m.db");
    }

    #[test]
    fn opencode_entry_uses_its_own_shape() {
        let value = parsed(&upsert(Client::Opencode, None, &entry()).expect("upsert should succeed"));
        assert_eq!(value["mcp"]["memoria"]["type"], "local");
        assert_eq!(value["mcp"]["memoria"]["command"], serde_json::json!(["/opt/memoria/mcp"]));
        assert_eq!(value["mcp"]["memoria"]["enabled"], true);
        assert!(value.get("mcpServers").is_none());
    }

    #[test]
    fn claude_desktop_and_cursor_entries_carry_no_type() {
        for client in [Client::ClaudeDesktop, Client::Cursor] {
            let value = parsed(&upsert(client, None, &entry()).expect("upsert should succeed"));
            assert!(value["mcpServers"]["memoria"].get("type").is_none(), "{}", client.name());
        }
    }

    #[test]
    fn json_remove_drops_only_the_memoria_entry() {
        let existing = r#"{"mcpServers":{"memoria":{"command":"a"},"other":{"command":"b"}}}"#;
        let edit = remove(Client::Cursor, Some(existing)).expect("remove should succeed");
        assert_eq!(edit.action(), "removed");
        let value = parsed(&edit);
        assert!(value["mcpServers"].get("memoria").is_none());
        assert_eq!(value["mcpServers"]["other"]["command"], "b");
    }

    #[test]
    fn json_remove_when_absent_reports_absent() {
        assert_eq!(remove(Client::Cursor, None), Ok(Edit::Absent));
        assert_eq!(remove(Client::Cursor, Some(r#"{"mcpServers":{}}"#)), Ok(Edit::Absent));
        assert_eq!(remove(Client::Opencode, Some(r#"{"theme":"x"}"#)), Ok(Edit::Absent));
    }

    #[test]
    fn malformed_json_is_an_error_not_a_rewrite() {
        let jsonc = "{\n  // comment\n  \"mcp\": {}\n}";
        let error = upsert(Client::Opencode, Some(jsonc), &entry()).expect_err("JSONC should be refused");
        assert!(error.starts_with("not valid JSON"), "{error}");
        assert!(remove(Client::Opencode, Some(jsonc)).is_err());
        assert!(upsert(Client::Cursor, Some("[1,2]"), &entry()).is_err());
        assert!(upsert(Client::Cursor, Some(r#"{"mcpServers":[]}"#), &entry()).is_err());
    }

    fn toml_text(edit: &Edit) -> &str {
        edit.new_contents().expect("edit should carry contents")
    }

    #[test]
    fn toml_upsert_into_missing_file_adds_the_table() {
        let edit = upsert(Client::Codex, None, &entry()).expect("upsert should succeed");
        assert_eq!(edit.action(), "added");
        assert_eq!(toml_text(&edit), "[mcp_servers.memoria]\ncommand = \"/opt/memoria/mcp\"\nargs = []\n");
    }

    #[test]
    fn toml_upsert_appends_after_existing_content_with_one_blank_line() {
        let existing = "model = \"o4\"\n\n[mcp_servers.other]\ncommand = \"x\"\n\n\n";
        let edit = upsert(Client::Codex, Some(existing), &entry()).expect("upsert should succeed");
        assert_eq!(
            toml_text(&edit),
            "model = \"o4\"\n\n[mcp_servers.other]\ncommand = \"x\"\n\n[mcp_servers.memoria]\ncommand = \"/opt/memoria/mcp\"\nargs = []\n"
        );
    }

    #[test]
    fn toml_upsert_replaces_in_place_and_keeps_neighbors_and_comments() {
        let existing = "# my codex config\n[mcp_servers.memoria]\ncommand = \"/old\"\nargs = [\"--x\"]\n\n[mcp_servers.memoria.env]\nA = \"1\"\n\n# about other\n[mcp_servers.other]\ncommand = \"x\" # inline\n";
        let edit = upsert(Client::Codex, Some(existing), &entry()).expect("upsert should succeed");
        assert_eq!(edit.action(), "updated");
        assert_eq!(
            toml_text(&edit),
            "# my codex config\n[mcp_servers.memoria]\ncommand = \"/opt/memoria/mcp\"\nargs = []\n\n[mcp_servers.memoria.env]\nA = \"1\"\n\n# about other\n[mcp_servers.other]\ncommand = \"x\" # inline\n"
        );
    }

    #[test]
    fn toml_upsert_writes_an_env_sub_table() {
        let mut with_env = entry();
        with_env.env.insert("MEMORIA_SQLITE_PATH".to_owned(), "/data/m.db".to_owned());
        with_env.env.insert("odd key".to_owned(), "v".to_owned());
        let edit = upsert(Client::Codex, None, &with_env).expect("upsert should succeed");
        assert_eq!(
            toml_text(&edit),
            "[mcp_servers.memoria]\ncommand = \"/opt/memoria/mcp\"\nargs = []\n\n[mcp_servers.memoria.env]\nMEMORIA_SQLITE_PATH = \"/data/m.db\"\n\"odd key\" = \"v\"\n"
        );
    }

    #[test]
    fn toml_upsert_is_unchanged_on_repeat() {
        let mut with_env = entry();
        with_env.env.insert("A".to_owned(), "1".to_owned());
        let existing = "[other]\nx = 1\n";
        let first = upsert(Client::Codex, Some(existing), &with_env).expect("upsert should succeed");
        let second = upsert(Client::Codex, first.new_contents(), &with_env).expect("upsert should succeed");
        assert_eq!(second, Edit::Unchanged);
    }

    #[test]
    fn toml_upsert_escapes_quotes_and_backslashes() {
        let odd = ServerEntry { command: "C:\\Program Files\\\"memoria\"\\mcp.exe".to_owned(), env: BTreeMap::new() };
        let edit = upsert(Client::Codex, None, &odd).expect("upsert should succeed");
        assert!(toml_text(&edit).contains("command = \"C:\\\\Program Files\\\\\\\"memoria\\\"\\\\mcp.exe\"\n"), "{}", toml_text(&edit));
    }

    #[test]
    fn toml_upsert_recognizes_spaced_headers_and_ignores_similar_names() {
        let existing = "[ mcp_servers.memoria ]\ncommand = \"/old\"\n\n[mcp_servers.memoria2]\ncommand = \"y\"\n";
        let edit = upsert(Client::Codex, Some(existing), &entry()).expect("upsert should succeed");
        assert_eq!(
            toml_text(&edit),
            "[mcp_servers.memoria]\ncommand = \"/opt/memoria/mcp\"\nargs = []\n\n[mcp_servers.memoria2]\ncommand = \"y\"\n"
        );
    }

    #[test]
    fn toml_remove_drops_the_table_and_its_sub_tables_only() {
        let existing = "model = \"o4\"\n\n[mcp_servers.memoria]\ncommand = \"/m\"\n\n[mcp_servers.memoria.env]\nA = \"1\"\n\n[mcp_servers.other]\ncommand = \"x\"\n";
        let edit = remove(Client::Codex, Some(existing)).expect("remove should succeed");
        assert_eq!(edit.action(), "removed");
        assert_eq!(toml_text(&edit), "model = \"o4\"\n\n[mcp_servers.other]\ncommand = \"x\"\n");
    }

    #[test]
    fn toml_remove_keeps_unrelated_blank_lines() {
        let existing = "a = 1\n\n\nb = 2\n\n[mcp_servers.memoria]\ncommand = \"/m\"\n";
        let edit = remove(Client::Codex, Some(existing)).expect("remove should succeed");
        assert_eq!(toml_text(&edit), "a = 1\n\n\nb = 2\n");
    }

    fn environment(pairs: &[(&str, &str)], macos: bool) -> Environment {
        let map: std::collections::HashMap<String, String> = pairs.iter().map(|(key, value)| ((*key).to_owned(), (*value).to_owned())).collect();
        Environment::from_lookup(|name| map.get(name).cloned(), macos).expect("HOME is set")
    }

    #[test]
    fn environment_requires_a_non_empty_home_and_treats_empty_overrides_as_unset() {
        assert!(Environment::from_lookup(|_| None, false).is_none());
        assert!(Environment::from_lookup(|name| (name == "HOME").then(String::new), false).is_none());
        let env = environment(&[("HOME", "/h"), ("XDG_CONFIG_HOME", ""), ("CODEX_HOME", "")], false);
        assert_eq!(env.xdg_config_home, None);
        assert_eq!(env.codex_home, None);
    }

    #[test]
    fn opencode_prefers_an_existing_jsonc_over_creating_a_second_json() {
        let env = temp_home("opencode-jsonc");
        let dir = env.home.join(".config/opencode");
        std::fs::create_dir_all(&dir).expect("create opencode dir");
        assert_eq!(Client::Opencode.config_path(&env), dir.join("opencode.json"));
        std::fs::write(dir.join("opencode.jsonc"), "{\n  // mine\n}\n").expect("seed jsonc");
        assert_eq!(Client::Opencode.config_path(&env), dir.join("opencode.jsonc"));
        let reports = run(Operation::Install(&entry()), &[], &env, false);
        assert!(reports[3].line().starts_with("opencode\terror: not valid JSON"), "{}", reports[3].line());
        assert!(!dir.join("opencode.json").exists());
        std::fs::write(dir.join("opencode.jsonc"), "{}").expect("seed comment-free jsonc");
        assert_eq!(run(Operation::Install(&entry()), &[], &env, false)[3].outcome, Ok("added"));
        assert!(std::fs::read_to_string(dir.join("opencode.jsonc")).expect("read").contains("memoria"));
        std::fs::write(dir.join("opencode.json"), "{}").expect("seed json");
        assert_eq!(Client::Opencode.config_path(&env), dir.join("opencode.json"));
        let _ = std::fs::remove_dir_all(&env.home);
    }

    #[test]
    fn default_config_paths_on_linux() {
        let env = environment(&[("HOME", "/h")], false);
        let paths: Vec<PathBuf> = Client::ALL.iter().map(|client| client.config_path(&env)).collect();
        assert_eq!(
            paths,
            [
                "/h/.claude.json",
                "/h/.config/Claude/claude_desktop_config.json",
                "/h/.cursor/mcp.json",
                "/h/.config/opencode/opencode.json",
                "/h/.codex/config.toml",
            ]
            .map(PathBuf::from)
        );
    }

    #[test]
    fn override_variables_move_the_config_paths() {
        let env = environment(&[("HOME", "/h"), ("XDG_CONFIG_HOME", "/x"), ("CLAUDE_CONFIG_DIR", "/c"), ("CODEX_HOME", "/k")], false);
        assert_eq!(Client::ClaudeCode.config_path(&env), PathBuf::from("/c/.claude.json"));
        assert_eq!(Client::ClaudeCode.config_dir(&env), PathBuf::from("/c"));
        assert_eq!(Client::ClaudeDesktop.config_path(&env), PathBuf::from("/x/Claude/claude_desktop_config.json"));
        assert_eq!(Client::Opencode.config_path(&env), PathBuf::from("/x/opencode/opencode.json"));
        assert_eq!(Client::Codex.config_path(&env), PathBuf::from("/k/config.toml"));
    }

    #[test]
    fn claude_desktop_uses_application_support_on_macos() {
        let env = environment(&[("HOME", "/Users/a"), ("XDG_CONFIG_HOME", "/x")], true);
        assert_eq!(
            Client::ClaudeDesktop.config_path(&env),
            PathBuf::from("/Users/a/Library/Application Support/Claude/claude_desktop_config.json")
        );
    }

    #[test]
    fn detection_requires_the_config_file_or_its_directory() {
        let home = std::env::temp_dir().join(format!("memoria-mcp-detect-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(home.join(".cursor")).expect("create cursor dir");
        std::fs::write(home.join(".claude.json"), "{}").expect("write claude.json");
        let env = Environment { home: home.clone(), xdg_config_home: None, claude_config_dir: None, codex_home: None, macos: false };
        let detected: Vec<&str> = Client::ALL.iter().filter(|client| client.is_detected(&env)).map(|client| client.name()).collect();
        assert_eq!(detected, ["claude-code", "cursor"]);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn cli_without_a_subcommand_means_serve() {
        assert!(Cli::try_parse_from(["mcp"]).expect("parse").command.is_none());
    }

    #[test]
    fn cli_parses_install_and_uninstall_flags() {
        let cli = Cli::try_parse_from(["mcp", "install", "--client", "claude-code", "--client", "codex", "--env", "A=1=2", "--dry-run", "--command", "/bin/m"])
            .expect("parse");
        let Some(Command::Install(args)) = cli.command else { panic!("expected install") };
        assert_eq!(args.clients, [Client::ClaudeCode, Client::Codex]);
        assert_eq!(args.env, [("A".to_owned(), "1=2".to_owned())]);
        assert!(args.dry_run);
        assert_eq!(args.command, Some(PathBuf::from("/bin/m")));
        let cli = Cli::try_parse_from(["mcp", "uninstall", "--client", "opencode"]).expect("parse");
        assert!(matches!(cli.command, Some(Command::Uninstall(args)) if args.clients == [Client::Opencode]));
    }

    #[test]
    fn cli_rejects_bad_env_pairs_and_unknown_clients() {
        assert!(Cli::try_parse_from(["mcp", "install", "--env", "NOEQUALS"]).is_err());
        assert!(Cli::try_parse_from(["mcp", "install", "--env", "=v"]).is_err());
        assert!(Cli::try_parse_from(["mcp", "install", "--client", "vim"]).is_err());
    }

    fn temp_home(label: &str) -> Environment {
        let home = std::env::temp_dir().join(format!("memoria-mcp-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&home).expect("create home");
        Environment { home, xdg_config_home: None, claude_config_dir: None, codex_home: None, macos: false }
    }

    fn outcomes(reports: &[Report]) -> Vec<(&str, Result<&str, String>)> {
        reports.iter().map(|report| (report.client.name(), report.outcome.clone())).collect()
    }

    #[test]
    fn run_install_then_reinstall_then_uninstall_against_a_temp_home() {
        let env = temp_home("run-cycle");
        std::fs::create_dir_all(env.home.join(".codex")).expect("create codex dir");
        std::fs::write(env.home.join(".claude.json"), r#"{"numStartups":3}"#).expect("seed claude.json");

        let installed = run(Operation::Install(&entry()), &[], &env, false);
        assert_eq!(
            outcomes(&installed),
            [
                ("claude-code", Ok("added")),
                ("claude-desktop", Ok("not-detected")),
                ("cursor", Ok("not-detected")),
                ("opencode", Ok("not-detected")),
                ("codex", Ok("added")),
            ]
            .map(|(name, outcome)| (name, outcome.map_err(|()| String::new())))
        );
        let claude: Value = serde_json::from_str(&std::fs::read_to_string(env.home.join(".claude.json")).expect("read")).expect("json");
        assert_eq!(claude["numStartups"], 3);
        assert_eq!(claude["mcpServers"]["memoria"]["command"], "/opt/memoria/mcp");
        assert_eq!(std::fs::read_to_string(env.home.join(".claude.json.bak")).expect("backup"), r#"{"numStartups":3}"#);
        assert!(std::fs::read_to_string(env.home.join(".codex/config.toml")).expect("read").contains("[mcp_servers.memoria]"));

        let again = run(Operation::Install(&entry()), &[], &env, false);
        assert!(again.iter().filter(|report| report.outcome != Ok("not-detected")).all(|report| report.outcome == Ok("unchanged")));

        let removed = run(Operation::Uninstall, &[], &env, false);
        assert_eq!(removed[0].outcome, Ok("removed"));
        assert_eq!(removed[4].outcome, Ok("removed"));
        let claude: Value = serde_json::from_str(&std::fs::read_to_string(env.home.join(".claude.json")).expect("read")).expect("json");
        assert!(claude["mcpServers"].get("memoria").is_none());
        assert_eq!(std::fs::read_to_string(env.home.join(".codex/config.toml")).expect("read"), "");
        let _ = std::fs::remove_dir_all(&env.home);
    }

    #[test]
    fn run_with_an_explicit_client_creates_its_file_and_skips_the_rest() {
        let env = temp_home("run-explicit");
        let reports = run(Operation::Install(&entry()), &[Client::Cursor], &env, false);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].outcome, Ok("added"));
        assert!(env.home.join(".cursor/mcp.json").is_file());
        assert!(!env.home.join(".cursor/mcp.json.bak").exists());
        let _ = std::fs::remove_dir_all(&env.home);
    }

    #[test]
    fn dry_run_reports_without_writing() {
        let env = temp_home("run-dry");
        let reports = run(Operation::Install(&entry()), &[Client::Opencode], &env, true);
        assert_eq!(reports[0].outcome, Ok("added"));
        assert!(!env.home.join(".config/opencode/opencode.json").exists());
        let _ = std::fs::remove_dir_all(&env.home);
    }

    #[test]
    fn uninstall_of_an_explicit_client_without_a_file_writes_nothing() {
        let env = temp_home("run-uninstall-absent");
        let reports = run(Operation::Uninstall, &[Client::Codex], &env, false);
        assert_eq!(reports[0].outcome, Ok("absent"));
        assert!(!env.home.join(".codex").exists());
        let _ = std::fs::remove_dir_all(&env.home);
    }

    #[test]
    fn a_malformed_file_is_reported_left_untouched_and_does_not_stop_other_clients() {
        let env = temp_home("run-malformed");
        std::fs::create_dir_all(env.home.join(".cursor")).expect("create cursor dir");
        std::fs::write(env.home.join(".cursor/mcp.json"), "{ broken").expect("seed");
        std::fs::create_dir_all(env.home.join(".codex")).expect("create codex dir");
        let reports = run(Operation::Install(&entry()), &[], &env, false);
        assert!(reports[2].line().starts_with("cursor\terror: not valid JSON"), "{}", reports[2].line());
        assert_eq!(std::fs::read_to_string(env.home.join(".cursor/mcp.json")).expect("read"), "{ broken");
        assert_eq!(reports[4].outcome, Ok("added"));
        let _ = std::fs::remove_dir_all(&env.home);
    }

    #[cfg(unix)]
    #[test]
    fn rewrite_keeps_permissions_and_writes_through_a_symlink() {
        use std::os::unix::fs::PermissionsExt;
        let env = temp_home("run-symlink");
        let real = env.home.join("dotfiles-claude.json");
        std::fs::write(&real, "{}").expect("seed");
        std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o600)).expect("chmod");
        std::os::unix::fs::symlink(&real, env.home.join(".claude.json")).expect("symlink");
        let reports = run(Operation::Install(&entry()), &[Client::ClaudeCode], &env, false);
        assert_eq!(reports[0].outcome, Ok("added"));
        assert!(std::fs::symlink_metadata(env.home.join(".claude.json")).expect("stat").file_type().is_symlink());
        assert!(std::fs::read_to_string(&real).expect("read").contains("memoria"));
        assert_eq!(std::fs::metadata(&real).expect("stat").permissions().mode() & 0o777, 0o600);
        let _ = std::fs::remove_dir_all(&env.home);
    }

    #[test]
    fn report_line_is_tab_separated() {
        let report = Report { client: Client::Codex, path: PathBuf::from("/h/.codex/config.toml"), outcome: Ok("added") };
        assert_eq!(report.line(), "codex\tadded\t/h/.codex/config.toml");
    }

    #[test]
    fn json_update_keeps_previous_env_keys_and_lets_new_values_win() {
        let existing = r#"{"mcpServers":{"memoria":{"command":"/old","env":{"MEMORIA_SERVER_URL":"http://s","MEMORIA_API_KEY":"k1"}}}}"#;
        let mut with_env = entry();
        with_env.env.insert("MEMORIA_API_KEY".to_owned(), "k2".to_owned());
        let edit = upsert(Client::ClaudeCode, Some(existing), &with_env).expect("upsert should succeed");
        let value = parsed(&edit);
        assert_eq!(value["mcpServers"]["memoria"]["env"]["MEMORIA_SERVER_URL"], "http://s");
        assert_eq!(value["mcpServers"]["memoria"]["env"]["MEMORIA_API_KEY"], "k2");
        assert_eq!(upsert(Client::ClaudeCode, edit.new_contents(), &entry()), Ok(Edit::Unchanged));
    }

    #[test]
    fn opencode_update_keeps_previous_environment_keys() {
        let existing = r#"{"mcp":{"memoria":{"type":"local","command":["/old"],"environment":{"A":"1"}}}}"#;
        let value = parsed(&upsert(Client::Opencode, Some(existing), &entry()).expect("upsert should succeed"));
        assert_eq!(value["mcp"]["memoria"]["environment"]["A"], "1");
    }

    #[test]
    fn toml_update_keeps_previous_env_lines_and_lets_new_values_win() {
        let existing = "[mcp_servers.memoria]\ncommand = \"/old\"\n\n[mcp_servers.memoria.env]\nMEMORIA_SERVER_URL = \"http://s\"\n\"MEMORIA_API_KEY\" = \"k1\"\n# note\n";
        let mut with_env = entry();
        with_env.env.insert("MEMORIA_API_KEY".to_owned(), "k2".to_owned());
        let edit = upsert(Client::Codex, Some(existing), &with_env).expect("upsert should succeed");
        assert_eq!(
            toml_text(&edit),
            "[mcp_servers.memoria]\ncommand = \"/opt/memoria/mcp\"\nargs = []\n\n[mcp_servers.memoria.env]\nMEMORIA_API_KEY = \"k2\"\nMEMORIA_SERVER_URL = \"http://s\"\n# note\n"
        );
        assert_eq!(upsert(Client::Codex, edit.new_contents(), &entry()), Ok(Edit::Unchanged));
    }

    #[test]
    fn toml_remove_when_absent_reports_absent() {
        assert_eq!(remove(Client::Codex, None), Ok(Edit::Absent));
        assert_eq!(remove(Client::Codex, Some("[mcp_servers.other]\ncommand = \"x\"\n")), Ok(Edit::Absent));
    }
}
