#!/usr/bin/env bash
# release_gate.sh SHA MAIN_REF OWNER/REPO: a release builds only a commit that is on main and passed CI (called by release.yml).
#   (a) SHA is an ancestor of MAIN_REF (e.g. origin/main, fetched by the caller);
#   (b) the `ci` workflow (ci.yml) has a successful push or workflow_dispatch run for SHA. Pull-request runs do not
#       count: they test the merge with the base branch, not SHA itself. Needs gh with GH_TOKEN (actions: read).
set -euo pipefail
main_ref=$2 repo=$3
sha=$(git rev-parse --verify "$1^{commit}")

if ! git merge-base --is-ancestor "$sha" "$main_ref"; then
  echo "::error::$sha is not on main ($main_ref). Merge it to main first, then tag the merged commit."
  exit 1
fi
echo "$sha is on main ($main_ref)"

runs=$(gh api "repos/$repo/actions/workflows/ci.yml/runs?head_sha=$sha&status=success&per_page=100" \
  --jq '[.workflow_runs[] | select(.event == "push" or .event == "workflow_dispatch") | .html_url] | join(" ")')
if [ -z "$runs" ]; then
  echo "::error::no successful ci run for $sha. Push the commit to main and wait for its ci run to pass (re-run it" \
    "if it failed or was cancelled), then re-run this release workflow."
  exit 1
fi
echo "successful ci run(s) for $sha: $runs"
