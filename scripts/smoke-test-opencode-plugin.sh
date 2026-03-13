#!/usr/bin/env bash
set -uo pipefail

server_bin="${MEMORIA_SMOKE_TEST_SERVER_BIN:?MEMORIA_SMOKE_TEST_SERVER_BIN must be set to the path of a compiled server binary}"
mcp_bin="${MEMORIA_SMOKE_TEST_MCP_BIN:?MEMORIA_SMOKE_TEST_MCP_BIN must be set to the path of a compiled mcp binary}"
plugin_dist="${MEMORIA_SMOKE_TEST_PLUGIN_DIST:?MEMORIA_SMOKE_TEST_PLUGIN_DIST must be set to the path of the built plugin dist/index.js}"
ollama_url="${MEMORIA_SMOKE_TEST_OLLAMA_URL:-http://127.0.0.1:11434}"
ollama_model="${MEMORIA_SMOKE_TEST_OLLAMA_MODEL:?MEMORIA_SMOKE_TEST_OLLAMA_MODEL must be set to a real, pulled Ollama model name}"
opencode_bin="${MEMORIA_SMOKE_TEST_OPENCODE_BIN:-opencode}"

failures=0

check() {
  local description="$1"
  local expected="$2"
  local actual="$3"
  if [[ "${actual}" == "${expected}" ]]; then
    echo "PASS: ${description} (got ${actual})"
  else
    echo "FAIL: ${description} (expected ${expected}, got ${actual})"
    failures=$((failures + 1))
  fi
}

contains() {
  if [[ "$1" == *"$2"* ]]; then
    echo true
  else
    echo false
  fi
}

work_dir="$(mktemp -d)"
server_log="${work_dir}/server.log"
run_log="${work_dir}/opencode-run.log"
server_pid=""

cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
  fi
  rm -rf "${work_dir}"
}
trap cleanup EXIT

api_key="smoke-test-opencode-plugin-key"
jwt_secret="smoke-test-opencode-plugin-jwt"

MEMORIA_API_KEY="${api_key}" MEMORIA_JWT_SECRET="${jwt_secret}" "${server_bin}" > "${server_log}" 2>&1 &
server_pid=$!
for _ in $(seq 1 20); do
  curl -s -o /dev/null http://127.0.0.1:8080/health && break
  sleep 0.5
done
check "server started and answers /health" 200 \
  "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health)"

project_dir="${work_dir}/project"
mkdir -p "${project_dir}"
git -C "${project_dir}" init -q
git -C "${project_dir}" remote add origin "https://github.com/memoria-ci/smoke-test-opencode-plugin.git"

cat > "${project_dir}/opencode.json" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "plugin": ["${plugin_dist}"],
  "mcp": {
    "memoria": {
      "type": "local",
      "command": ["${mcp_bin}"],
      "environment": { "MEMORIA_SQLITE_PATH": "${work_dir}/mcp-store.db" },
      "enabled": true
    }
  },
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local Ollama",
      "options": { "baseURL": "${ollama_url}/v1" },
      "models": { "${ollama_model}": { "name": "${ollama_model}" } }
    }
  }
}
EOF

MEMORIA_API_KEY="${api_key}" MEMORIA_SERVER_URL="http://127.0.0.1:8080" \
  timeout 240 "${opencode_bin}" run --model "ollama/${ollama_model}" \
  --dir "${project_dir}" \
  "My favorite programming language is Rust." > "${run_log}" 2>&1
run_exit=$?
check "opencode run completed without crashing" 0 "${run_exit}"

check "the plugin's chat.message hook called POST /search" true \
  "$(contains "$(cat "${server_log}")" "POST /search")"

check "the plugin's event hook called POST /memories (session-idle capture)" true \
  "$(contains "$(cat "${server_log}")" "POST /memories")"

if [[ "${failures}" -gt 0 ]]; then
  echo "-- server log --"
  cat "${server_log}"
  echo "-- opencode run log --"
  cat "${run_log}"
  echo "SMOKE TEST FAILED: ${failures} check(s) failed."
  exit 1
fi

echo "SMOKE TEST PASSED: all checks succeeded."
