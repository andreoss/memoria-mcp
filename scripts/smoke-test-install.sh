#!/usr/bin/env bash
set -uo pipefail

mcp_bin="${MEMORIA_SMOKE_TEST_MCP_BIN:?MEMORIA_SMOKE_TEST_MCP_BIN must be set to the path of a compiled mcp binary}"
mcp_bin="$(realpath "${mcp_bin}")"

failures=0

check() {
  local description="$1"
  local expected="$2"
  local actual="$3"
  if [[ "${actual}" == "${expected}" ]]; then
    echo "PASS: ${description}"
  else
    echo "FAIL: ${description} (expected ${expected}, got ${actual})"
    failures=$((failures + 1))
  fi
}

action_for() {
  printf '%s\n' "$1" | awk -F '\t' -v client="$2" '$1 == client { print $2 }'
}

home="$(mktemp -d)"
trap 'rm -rf "${home}"' EXIT

run_mcp() {
  env -i PATH="${PATH}" HOME="${home}" "${mcp_bin}" "$@"
}

mkdir -p "${home}/.codex" "${home}/.config/opencode"
printf '{"numStartups":3}' > "${home}/.claude.json"
printf 'model = "o4"\n' > "${home}/.codex/config.toml"

dry="$(run_mcp install --dry-run)"
check "dry-run exits 0" "0" "$?"
check "dry-run plans claude-code" "added" "$(action_for "${dry}" claude-code)"
check "dry-run leaves claude.json alone" '{"numStartups":3}' "$(cat "${home}/.claude.json")"
check "dry-run writes no opencode.json" "false" "$([[ -e "${home}/.config/opencode/opencode.json" ]] && echo true || echo false)"

first="$(run_mcp install --env MEMORIA_SQLITE_PATH=/data/m.db)"
check "install exits 0" "0" "$?"
check "claude-code added" "added" "$(action_for "${first}" claude-code)"
check "opencode added" "added" "$(action_for "${first}" opencode)"
check "codex added" "added" "$(action_for "${first}" codex)"
check "cursor not detected" "not-detected" "$(action_for "${first}" cursor)"
check "claude-desktop not detected" "not-detected" "$(action_for "${first}" claude-desktop)"
check "claude.json keeps other keys" "3" "$(jq -r '.numStartups' "${home}/.claude.json")"
check "claude.json command is the binary" "${mcp_bin}" "$(jq -r '.mcpServers.memoria.command' "${home}/.claude.json")"
check "claude.json env written" "/data/m.db" "$(jq -r '.mcpServers.memoria.env.MEMORIA_SQLITE_PATH' "${home}/.claude.json")"
check "claude.json backup is the original" '{"numStartups":3}' "$(cat "${home}/.claude.json.bak")"
check "opencode command array" "${mcp_bin}" "$(jq -r '.mcp.memoria.command[0]' "${home}/.config/opencode/opencode.json")"
check "codex keeps existing keys" "true" "$(grep -qx 'model = "o4"' "${home}/.codex/config.toml" && echo true || echo false)"
check "codex table written" "true" "$(grep -qx '\[mcp_servers.memoria\]' "${home}/.codex/config.toml" && echo true || echo false)"
check "codex env written" "true" "$(grep -qx 'MEMORIA_SQLITE_PATH = "/data/m.db"' "${home}/.codex/config.toml" && echo true || echo false)"

second="$(run_mcp install --env MEMORIA_SQLITE_PATH=/data/m.db)"
check "reinstall claude-code unchanged" "unchanged" "$(action_for "${second}" claude-code)"
check "reinstall codex unchanged" "unchanged" "$(action_for "${second}" codex)"

explicit="$(run_mcp install --client cursor)"
check "explicit client added" "added" "$(action_for "${explicit}" cursor)"
check "explicit client only" "1" "$(printf '%s\n' "${explicit}" | wc -l | tr -d ' ')"

printf '{ broken' > "${home}/.config/opencode/opencode.json"
broken="$(run_mcp install)"
check "malformed file makes exit non-zero" "1" "$?"
check "malformed file reported" "true" "$([[ "$(action_for "${broken}" opencode)" == "error: not valid JSON"* ]] && echo true || echo false)"
check "malformed file untouched" "{ broken" "$(cat "${home}/.config/opencode/opencode.json")"
rm "${home}/.config/opencode/opencode.json"

removed="$(run_mcp uninstall)"
check "uninstall exits 0" "0" "$?"
check "claude-code removed" "removed" "$(action_for "${removed}" claude-code)"
check "codex removed" "removed" "$(action_for "${removed}" codex)"
check "opencode absent" "absent" "$(action_for "${removed}" opencode)"
check "claude.json entry gone" "null" "$(jq -r '.mcpServers.memoria' "${home}/.claude.json")"
check "codex back to original" 'model = "o4"' "$(cat "${home}/.codex/config.toml")"

check "unknown client rejected" "2" "$(run_mcp install --client vim >/dev/null 2>&1; echo $?)"

if [[ "${failures}" -gt 0 ]]; then
  echo "${failures} check(s) failed"
  exit 1
fi
echo "All install smoke checks passed"
