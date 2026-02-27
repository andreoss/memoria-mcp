#!/usr/bin/env bash
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

echo "== Scenario 1: server not running (expect non-zero) =="
MEMORIA_SMOKE_TEST_URL="http://127.0.0.1:59999" MEMORIA_SMOKE_TEST_API_KEY="unused" bash scripts/smoke-test-server.sh >/dev/null 2>&1
not_running_rc=$?
echo "exit code: ${not_running_rc}"

echo "== Scenario 2: server running with the right key (expect zero) =="
cargo build --offline -p server >/dev/null 2>&1
scratch_dir="$(mktemp -d)"
MEMORIA_API_KEY=verify-smoke-key \
  MEMORIA_JWT_SECRET=verify-smoke-jwt-secret \
  MEMORIA_STORE_PATH="${scratch_dir}/server-store.json" \
  MEMORIA_AUTH_STORE_PATH="${scratch_dir}/auth-store.json" \
  ./target/debug/server &
server_pid=$!
sleep 0.5
MEMORIA_SMOKE_TEST_API_KEY="verify-smoke-key" bash scripts/smoke-test-server.sh >/dev/null 2>&1
running_rc=$?
kill "${server_pid}" 2>/dev/null || true
wait "${server_pid}" 2>/dev/null || true
rm -rf "${scratch_dir}"
echo "exit code: ${running_rc}"

if [[ "${not_running_rc}" -ne 0 && "${running_rc}" -eq 0 ]]; then
  echo "VERIFY PASSED: server-down detected (rc=${not_running_rc}), healthy run accepted (rc=${running_rc})."
  exit 0
else
  echo "VERIFY FAILED: server-down rc=${not_running_rc}, healthy rc=${running_rc}."
  exit 1
fi
