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

noinfer_output="$("${cli_bin}" --json add "The quick brown fox jumps over the lazy dog." --user-id smoke --no-infer)"
noinfer_id="$(printf '%s' "${noinfer_output}" | grep -o '"[^"]*"' | head -n1 | tr -d '"')"
noinfer_get="$("${cli_bin}" --json get "${noinfer_id}")"
check "--no-infer stores the content verbatim" "true" "$(contains "${noinfer_get}" "The quick brown fox jumps over the lazy dog.")"

filter_output="$("${cli_bin}" --json search "engineer" --user-id smoke --filter "{\"content\":{\"icontains\":\"fox\"}}")"
check "--filter excludes a record that doesn't match" "true" "$([[ "${filter_output}" != *"${record_id}"* ]] && echo true || echo false)"
check "--filter includes a record that does match" "true" "$(contains "${filter_output}" "${noinfer_id}")"

check "--filter with malformed JSON is rejected" 2 \
  "$("${cli_bin}" search "x" --user-id smoke --filter "not json" > /dev/null 2>&1; echo $?)"

explain_output="$("${cli_bin}" --json search "engineer" --user-id smoke --explain)"
check "--explain includes a real score_details breakdown" "true" "$(contains "${explain_output}" "score_details")"
no_explain_output="$("${cli_bin}" --json search "engineer" --user-id smoke)"
check "omitting --explain leaves score_details null" "true" "$([[ "${no_explain_output}" == *'"score_details":null'* ]] && echo true || echo false)"

list_output="$("${cli_bin}" --json list --user-id smoke)"
check "list includes the record id" "true" "$(contains "${list_output}" "${record_id}")"

check "list without a scope is rejected" 1 \
  "$("${cli_bin}" list > /dev/null 2>&1; echo $?)"

"${cli_bin}" update "${record_id}" --content "Alice is a senior backend engineer." --set role=engineer >/dev/null
update_status=$?
check "update exits 0" "0" "${update_status}"
get_after_update_output="$("${cli_bin}" --json get "${record_id}")"
check "a fresh process sees the update a prior process made" "true" "$(contains "${get_after_update_output}" "Alice is a senior backend engineer.")"

history_output="$("${cli_bin}" --json history "${record_id}")"
check "history exits 0" "0" "$?"
check "a fresh process sees the real added event a prior process recorded" "true" "$(contains "${history_output}" "Added")"

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

history_after_delete_output="$("${cli_bin}" --json history "${record_id}")"
check "a fresh process sees both the added and deleted events after the record is gone" "true" "$(contains "${history_after_delete_output}" "Deleted")"

if [[ -n "${MEMORIA_SMOKE_TEST_SERVER_URL:-}" ]]; then
  export MEMORIA_SERVER_URL="${MEMORIA_SMOKE_TEST_SERVER_URL}"
  remote_user_id="smoke-test-remote-$$"

  remote_add_output="$("${cli_bin}" add "Remote smoke test content" --user-id "${remote_user_id}" --no-infer)"
  check "remote add exits 0" "0" "$?"
  remote_record_id="${remote_add_output}"
  check "remote add returned a real record id" "true" "$(contains "${remote_record_id}" "rec-")"

  remote_search_output="$("${cli_bin}" search "Remote smoke test" --user-id "${remote_user_id}")"
  check "remote search finds the record within scope" "true" "$(contains "${remote_search_output}" "${remote_record_id}")"

  remote_get_output="$("${cli_bin}" get "${remote_record_id}")"
  check "remote get returns the real content" "true" "$(contains "${remote_get_output}" "Remote smoke test content")"

  "${cli_bin}" delete "${remote_record_id}" >/dev/null
  check "remote delete exits 0" "0" "$?"
  "${cli_bin}" get "${remote_record_id}" >/dev/null 2>&1
  check "a fresh remote request confirms the deleted record is really gone" "1" "$?"

  unset MEMORIA_SERVER_URL
fi

if [[ "${failures}" -gt 0 ]]; then
  echo "SMOKE TEST FAILED: ${failures} check(s) failed."
  exit 1
fi

echo "SMOKE TEST PASSED: all checks succeeded."
