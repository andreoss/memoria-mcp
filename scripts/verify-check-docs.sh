#!/usr/bin/env bash
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

base_branch="$(git rev-parse --abbrev-ref HEAD)"
scratch="verify-docs-check-scratch"

cleanup() {
  git checkout --quiet "${base_branch}" 2>/dev/null || true
  git branch --delete --force "${scratch}" 2>/dev/null || true
}
trap cleanup EXIT

git checkout --quiet -b "${scratch}"

code_only_violation=0
both_pass=0

echo "== Scenario 1: change only core/src/ (expect non-zero) =="
printf '// test\n' >> core/src/embedding.rs
git add core/src/embedding.rs
git commit --quiet -m "temp: code only"
set +e
bash scripts/check-docs.sh "$(git rev-parse HEAD~1)" "HEAD"
code_only_violation=$?
set -e
echo "exit code: ${code_only_violation}"
git reset --quiet --hard HEAD~1

echo "== Scenario 2: change core/src/ and docs/ (expect zero) =="
printf '// test\n' >> core/src/embedding.rs
printf 'added\n' >> docs/overview.md
git add core/src/embedding.rs docs/overview.md
git commit --quiet -m "temp: code and docs"
set +e
bash scripts/check-docs.sh "$(git rev-parse HEAD~1)" "HEAD"
both_pass=$?
set -e
echo "exit code: ${both_pass}"
git reset --quiet --hard HEAD~1

if [[ "${code_only_violation}" -ne 0 && "${both_pass}" -eq 0 ]]; then
  echo "VERIFY PASSED: violation detected (rc=${code_only_violation}), clean change accepted (rc=${both_pass})."
  exit 0
else
  echo "VERIFY FAILED: violation rc=${code_only_violation}, clean rc=${both_pass}."
  exit 1
fi
