#!/usr/bin/env bash
set -euo pipefail

base_ref="${1:-}"
head_ref="${2:-}"

if [[ -z "${base_ref}" || -z "${head_ref}" ]]; then
  if [[ -n "${GITHUB_BASE_REF:-}" && -n "${GITHUB_HEAD_REF:-}" ]]; then
    base_ref="origin/${GITHUB_BASE_REF}"
    head_ref="${GITHUB_HEAD_REF}"
  else
    base_ref="$(git rev-parse --verify origin/master 2>/dev/null || echo 'HEAD~1')"
    head_ref="HEAD"
  fi
fi

git fetch --quiet origin "${base_ref}" 2>/dev/null || true

changed_files="$(git diff --name-only "${base_ref}" "${head_ref}")"

if [[ -z "${changed_files}" ]]; then
  echo "No changed files detected between ${base_ref} and ${head_ref}."
  exit 0
fi

code_dirs=("core/src/" "cli/src/" "server/src/")
doc_file="README.adoc"

touches_code=false
touches_docs=false

while IFS= read -r file; do
  for dir in "${code_dirs[@]}"; do
    if [[ "${file}" == "${dir}"* ]]; then
      touches_code=true
      break
    fi
  done
  if [[ "${file}" == "${doc_file}" ]]; then
    touches_docs=true
  fi
done <<< "${changed_files}"

if [[ "${touches_code}" == true && "${touches_docs}" == false ]]; then
  echo "Docs check failed: changes under core/src/, cli/src/, or server/src/ require a change to ${doc_file} in the same PR."
  echo "Changed files:"
  echo "${changed_files}"
  exit 1
fi

echo "Docs check passed."
