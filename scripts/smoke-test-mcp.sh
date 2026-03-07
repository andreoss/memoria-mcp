#!/usr/bin/env bash
set -uo pipefail

mcp_bin="${MEMORIA_SMOKE_TEST_MCP_BIN:?MEMORIA_SMOKE_TEST_MCP_BIN must be set to the path of a compiled mcp binary}"

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

contains() {
  if [[ "$1" == *"$2"* ]]; then
    echo true
  else
    echo false
  fi
}

response_for_id() {
  printf '%s\n' "$1" | grep -F "\"jsonrpc\":\"2.0\",\"id\":$2," | head -n1
}

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT
export HOME="${work_dir}"
export MEMORIA_MCP_SECRET="smoke-test-secret"
secret_arg='"secret":"smoke-test-secret"'

init_request='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke-test","version":"0.1"}}}'
initialized_notification='{"jsonrpc":"2.0","method":"notifications/initialized"}'
tools_list_request='{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

add_request="{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"add_memory\",\"arguments\":{\"content\":\"Alice is a backend engineer.\",\"user_id\":\"smoke\",\"infer\":false,${secret_arg}}}}"
unscoped_delete_all_request="{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"delete_all_memories\",\"arguments\":{${secret_arg}}}}"

round1_output="$(printf '%s\n' "${init_request}" "${initialized_notification}" "${tools_list_request}" "${add_request}" "${unscoped_delete_all_request}" | timeout 10 "${mcp_bin}")"

init_line="$(response_for_id "${round1_output}" 1)"
tools_list_line="$(response_for_id "${round1_output}" 2)"
add_line="$(response_for_id "${round1_output}" 3)"
unscoped_delete_all_line="$(response_for_id "${round1_output}" 4)"

check "initialize succeeds" "true" "$(contains "${init_line}" '"protocolVersion"')"
for tool in search_memories get_memory get_memories memory_history add_memory update_memory delete_memory delete_all_memories list_entities; do
  check "tools/list includes ${tool}" "true" "$(contains "${tools_list_line}" "\"name\":\"${tool}\"")"
done
check "add_memory succeeds" "true" "$(contains "${add_line}" '"isError":false')"
record_id="$(printf '%s' "${add_line}" | grep -o 'rec-[0-9]*-[0-9]*-[0-9]*' | head -n1)"
check "add_memory returned a real record id" "true" "$([[ -n "${record_id}" ]] && echo true || echo false)"
check "delete_all_memories with no scope is rejected" "true" "$(contains "${unscoped_delete_all_line}" '"isError":true')"

get_request="{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"tools/call\",\"params\":{\"name\":\"get_memory\",\"arguments\":{\"id\":\"${record_id}\",${secret_arg}}}}"
update_request="{\"jsonrpc\":\"2.0\",\"id\":12,\"method\":\"tools/call\",\"params\":{\"name\":\"update_memory\",\"arguments\":{\"id\":\"${record_id}\",\"content\":\"Alice is a senior backend engineer.\",${secret_arg}}}}"
search_request="{\"jsonrpc\":\"2.0\",\"id\":13,\"method\":\"tools/call\",\"params\":{\"name\":\"search_memories\",\"arguments\":{\"query\":\"engineer\",\"user_id\":\"smoke\",${secret_arg}}}}"
history_request="{\"jsonrpc\":\"2.0\",\"id\":14,\"method\":\"tools/call\",\"params\":{\"name\":\"memory_history\",\"arguments\":{\"id\":\"${record_id}\",${secret_arg}}}}"
entities_request="{\"jsonrpc\":\"2.0\",\"id\":15,\"method\":\"tools/call\",\"params\":{\"name\":\"list_entities\",\"arguments\":{${secret_arg}}}}"
missing_secret_request="{\"jsonrpc\":\"2.0\",\"id\":16,\"method\":\"tools/call\",\"params\":{\"name\":\"get_memory\",\"arguments\":{\"id\":\"${record_id}\"}}}"

round2_output="$(printf '%s\n' "${init_request}" "${initialized_notification}" "${get_request}" "${update_request}" "${search_request}" "${history_request}" "${entities_request}" "${missing_secret_request}" | timeout 10 "${mcp_bin}")"

get_line="$(response_for_id "${round2_output}" 11)"
update_line="$(response_for_id "${round2_output}" 12)"
search_line="$(response_for_id "${round2_output}" 13)"
history_line="$(response_for_id "${round2_output}" 14)"
entities_line="$(response_for_id "${round2_output}" 15)"
missing_secret_line="$(response_for_id "${round2_output}" 16)"

check "a fresh process sees the record a prior process persisted" "true" "$(contains "${get_line}" "${record_id}")"
check "update_memory succeeds and reflects the new content" "true" "$(contains "${update_line}" 'Alice is a senior backend engineer.')"
check "search_memories finds the updated record within scope" "true" "$(contains "${search_line}" "${record_id}")"
check "a fresh process sees the added event in history (history now persists across restarts)" "true" "$(contains "${history_line}" '\"event\":\"added\"')"
check "list_entities reports the smoke user" "true" "$(contains "${entities_line}" '\"entity_id\":\"smoke\"')"
check "a tool call with a missing secret is rejected" "true" "$(contains "${missing_secret_line}" '"isError":true')"

delete_all_request="{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"tools/call\",\"params\":{\"name\":\"delete_all_memories\",\"arguments\":{\"user_id\":\"smoke\",${secret_arg}}}}"
round3_output="$(printf '%s\n' "${init_request}" "${initialized_notification}" "${delete_all_request}" | timeout 10 "${mcp_bin}")"
delete_all_line="$(response_for_id "${round3_output}" 11)"
check "delete_all_memories with a real scope deletes exactly one record" "true" "$(contains "${delete_all_line}" '\"deleted\":1')"

get_after_delete_request="{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"tools/call\",\"params\":{\"name\":\"get_memory\",\"arguments\":{\"id\":\"${record_id}\",${secret_arg}}}}"
round4_output="$(printf '%s\n' "${init_request}" "${initialized_notification}" "${get_after_delete_request}" | timeout 10 "${mcp_bin}")"
get_after_delete_line="$(response_for_id "${round4_output}" 11)"
check "get_memory for the now-deleted id is a real tool error" "true" "$(contains "${get_after_delete_line}" '"isError":true')"

if [[ "${failures}" -gt 0 ]]; then
  echo "SMOKE TEST FAILED: ${failures} check(s) failed."
  exit 1
fi

echo "SMOKE TEST PASSED: all checks succeeded."
