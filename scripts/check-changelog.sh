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

version_bump="$(git diff "${base_ref}" "${head_ref}" -- core/Cargo.toml | grep -E '^\+version = ' || true)"

if [[ -z "${version_bump}" ]]; then
  echo "No version bump in core/Cargo.toml between ${base_ref} and ${head_ref}."
  exit 0
fi

changelog_touched="$(git diff --name-only "${base_ref}" "${head_ref}" -- CHANGELOG.md)"

if [[ -z "${changelog_touched}" ]]; then
  echo "Changelog check failed: core/Cargo.toml's version changed but CHANGELOG.md was not updated in the same span."
  exit 1
fi

echo "Changelog check passed."
