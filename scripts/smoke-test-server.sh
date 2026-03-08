#!/usr/bin/env bash
set -uo pipefail

base_url="${MEMORIA_SMOKE_TEST_URL:-http://127.0.0.1:8080}"
api_key="${MEMORIA_SMOKE_TEST_API_KEY:?MEMORIA_SMOKE_TEST_API_KEY must be set to the key the server was started with}"

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

status_of() {
  curl -s -o /dev/null -w "%{http_code}" "$@"
}

check "GET /health with no token" 200 \
  "$(status_of "${base_url}/health")"

check "POST /memories with no token is rejected" 401 \
  "$(status_of -X POST "${base_url}/memories" -d '{"content":"smoke","user_id":"smoke"}')"

created_response="$(curl -s -H "Authorization: Bearer ${api_key}" -X POST "${base_url}/memories" -d '{"content":"Alice is an engineer.","user_id":"smoke"}')"
record_id="$(printf '%s' "${created_response}" | grep -o '"[^"]*"' | sed -n '2p' | tr -d '"')"
check "POST /memories with a valid token" "true" \
  "$([[ -n "${record_id}" ]] && echo true || echo false)"

check "GET /memories/:id with a valid token" 200 \
  "$(status_of -H "Authorization: Bearer ${api_key}" "${base_url}/memories/${record_id}")"

check "GET /memories with a valid token" 200 \
  "$(status_of -H "Authorization: Bearer ${api_key}" "${base_url}/memories")"

check "PUT /memories/:id with a valid token" 200 \
  "$(status_of -H "Authorization: Bearer ${api_key}" -X PUT "${base_url}/memories/${record_id}" -d '{"content":"Alice is a senior engineer."}')"

check "GET /memories/:id/history with a valid token" 200 \
  "$(status_of -H "Authorization: Bearer ${api_key}" "${base_url}/memories/${record_id}/history")"

check "POST /search with a valid token" 200 \
  "$(status_of -H "Authorization: Bearer ${api_key}" -X POST "${base_url}/search" -d '{"query":"Alice","user_id":"smoke"}')"

check "POST /memories with a malformed body" 400 \
  "$(status_of -H "Authorization: Bearer ${api_key}" -X POST "${base_url}/memories" -d 'not json')"

check "DELETE /memories/:id with a valid token" 200 \
  "$(status_of -H "Authorization: Bearer ${api_key}" -X DELETE "${base_url}/memories/${record_id}")"

bulk_response="$(curl -s -H "Authorization: Bearer ${api_key}" -X POST "${base_url}/memories" -d '{"content":"smoke bulk-delete target.","user_id":"smoke","infer":false}')"
bulk_id="$(printf '%s' "${bulk_response}" | grep -o '"[^"]*"' | sed -n '2p' | tr -d '"')"
check "POST /memories for the bulk-delete target" "true" \
  "$([[ -n "${bulk_id}" ]] && echo true || echo false)"

check "DELETE /memories with a filter, admin token" 200 \
  "$(status_of -H "Authorization: Bearer ${api_key}" -X DELETE "${base_url}/memories" -d '{"filters":{"content":{"contains":"bulk-delete target"}}}')"

check "GET /memories/:id after its bulk deletion is 404" 404 \
  "$(status_of -H "Authorization: Bearer ${api_key}" "${base_url}/memories/${bulk_id}")"

check "POST /reset with an admin token" 200 \
  "$(status_of -H "Authorization: Bearer ${api_key}" -X POST "${base_url}/reset")"

echo "-- burst: 25 rapid requests, expect at least one 429 --"
saw_429=false
for _ in $(seq 1 25); do
  code="$(status_of -H "Authorization: Bearer ${api_key}" "${base_url}/memories")"
  if [[ "${code}" == "429" ]]; then
    saw_429=true
  fi
done
check "burst triggers rate limiting" "true" "$([[ "${saw_429}" == "true" ]] && echo true || echo false)"

if [[ "${failures}" -gt 0 ]]; then
  echo "SMOKE TEST FAILED: ${failures} check(s) failed."
  exit 1
fi

echo "SMOKE TEST PASSED: all checks succeeded."
