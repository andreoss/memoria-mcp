#!/usr/bin/env bash
set -uo pipefail

cli_bin="${MEMORIA_SMOKE_TEST_CLI_BIN:?MEMORIA_SMOKE_TEST_CLI_BIN must be set to the path of a compiled cli binary}"

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
trap 'rm -rf "${work_dir}"' EXIT
export MEMORIA_STORE_PATH="${work_dir}/store.json"

init_output="$("${cli_bin}" init 2>&1)"
check "init exits 0" "0" "$?"
check "init creates the store file" "true" "$([[ -f "${MEMORIA_STORE_PATH}" ]] && echo true || echo false)"

add_output="$("${cli_bin}" --json add "Alice is a backend engineer." --user-id smoke)"
add_status=$?
check "add exits 0" "0" "${add_status}"
record_id="$(printf '%s' "${add_output}" | grep -o '"[^"]*"' | head -n1 | tr -d '"')"
check "add returned a real record id" "true" "$([[ -n "${record_id}" ]] && echo true || echo false)"

get_output="$("${cli_bin}" --json get "${record_id}")"
check "a fresh process sees the record a prior process persisted" "0" "$?"
check "get returns the real content" "true" "$(contains "${get_output}" "Alice is a backend engineer.")"

search_output="$("${cli_bin}" --json search "engineer" --user-id smoke)"
check "search finds the record within scope" "true" "$(contains "${search_output}" "${record_id}")"

list_output="$("${cli_bin}" --json list)"
check "list includes the record id" "true" "$(contains "${list_output}" "${record_id}")"

"${cli_bin}" update "${record_id}" --content "Alice is a senior backend engineer." --set role=engineer >/dev/null
update_status=$?
check "update exits 0" "0" "${update_status}"
get_after_update_output="$("${cli_bin}" --json get "${record_id}")"
check "a fresh process sees the update a prior process made" "true" "$(contains "${get_after_update_output}" "Alice is a senior backend engineer.")"

whoami_output="$("${cli_bin}" --json whoami)"
check "whoami exits 0" "0" "$?"
check "whoami reports the real store path" "true" "$(contains "${whoami_output}" "${MEMORIA_STORE_PATH}")"

status_output="$("${cli_bin}" status)"
check "status reports ok against the real local providers" "ok" "${status_output}"

"${cli_bin}" get "never-existed" >/dev/null 2>&1
check "get for an unknown id exits non-zero" "1" "$?"

completions_output="$("${cli_bin}" completions bash)"
check "completions exits 0" "0" "$?"
check "completions produces a real bash completion function" "true" "$(contains "${completions_output}" "complete")"

"${cli_bin}" delete "${record_id}" >/dev/null
delete_status=$?
check "delete exits 0" "0" "${delete_status}"
"${cli_bin}" get "${record_id}" >/dev/null 2>&1
check "a fresh process confirms the deleted record is really gone" "1" "$?"

if [[ "${failures}" -gt 0 ]]; then
  echo "SMOKE TEST FAILED: ${failures} check(s) failed."
  exit 1
fi

echo "SMOKE TEST PASSED: all checks succeeded."
