#!/usr/bin/env bash
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

base_branch="$(git rev-parse --abbrev-ref HEAD)"
scratch="verify-changelog-check-scratch"

cleanup() {
  git checkout --quiet "${base_branch}" 2>/dev/null || true
  git branch --delete --force "${scratch}" 2>/dev/null || true
}
trap cleanup EXIT

git checkout --quiet -b "${scratch}"

version_only_violation=0
both_pass=0
no_bump_pass=0

echo "== Scenario 1: version bump, no changelog entry (expect non-zero) =="
sed -i 's/^version = "0.1.0"/version = "0.1.1"/' core/Cargo.toml
git add core/Cargo.toml
git commit --quiet -m "temp: version bump only"
set +e
bash scripts/check-changelog.sh "$(git rev-parse HEAD~1)" "HEAD"
version_only_violation=$?
set -e
echo "exit code: ${version_only_violation}"
git reset --quiet --hard HEAD~1

echo "== Scenario 2: version bump and changelog entry (expect zero) =="
sed -i 's/^version = "0.1.0"/version = "0.1.1"/' core/Cargo.toml
printf '\n## [0.1.1]\n\ntemp entry\n' >> CHANGELOG.md
git add core/Cargo.toml CHANGELOG.md
git commit --quiet -m "temp: version bump and changelog"
set +e
bash scripts/check-changelog.sh "$(git rev-parse HEAD~1)" "HEAD"
both_pass=$?
set -e
echo "exit code: ${both_pass}"
git reset --quiet --hard HEAD~1

echo "== Scenario 3: no version bump at all (expect zero) =="
printf 'temp\n' >> docs/overview.md
git add docs/overview.md
git commit --quiet -m "temp: unrelated change"
set +e
bash scripts/check-changelog.sh "$(git rev-parse HEAD~1)" "HEAD"
no_bump_pass=$?
set -e
echo "exit code: ${no_bump_pass}"
git reset --quiet --hard HEAD~1

if [[ "${version_only_violation}" -ne 0 && "${both_pass}" -eq 0 && "${no_bump_pass}" -eq 0 ]]; then
  echo "VERIFY PASSED: violation detected (rc=${version_only_violation}), both changes accepted (rc=${both_pass}), no-bump accepted (rc=${no_bump_pass})."
  exit 0
else
  echo "VERIFY FAILED: violation rc=${version_only_violation}, both rc=${both_pass}, no-bump rc=${no_bump_pass}."
  exit 1
fi
