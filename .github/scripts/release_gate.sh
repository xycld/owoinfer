#!/usr/bin/env bash
# release_gate.sh [--dry-run] SHA MAIN_REF OWNER/REPO: a release builds only a commit that is on main and passed CI
# (called by release.yml).
#   (a) SHA is an ancestor of MAIN_REF (e.g. origin/main, fetched by the caller);
#   (b) the `ci` workflow (ci.yml) has a successful push or workflow_dispatch run for SHA. Pull-request runs do not
#       count: they test the merge with the base branch, not SHA itself. Needs gh with GH_TOKEN (actions: read).
# --dry-run (a dispatch, which never publishes): failures are warnings and the run goes on.
set -euo pipefail
dry=0
if [ "${1:-}" = "--dry-run" ]; then dry=1; shift; fi
if [ $# -lt 3 ]; then
  echo "usage: release_gate.sh [--dry-run] SHA MAIN_REF OWNER/REPO" >&2
  exit 2
fi
main_ref=$2 repo=$3
sha=$(git rev-parse --verify "$1^{commit}")

# fail WHAT TAG_ADVICE: an error for a tag push, a warning for a dry run.
fail() {
  if [ "$dry" = 1 ]; then
    echo "::warning::dry run: $1; a release tag on this commit would be refused."
  else
    echo "::error::$1. $2"
    exit 1
  fi
}

if git merge-base --is-ancestor "$sha" "$main_ref"; then
  echo "$sha is on main ($main_ref)"
else
  fail "$sha is not on main ($main_ref)" "Merge it to main first, then tag the merged commit."
fi

runs=$(gh api "repos/$repo/actions/workflows/ci.yml/runs?head_sha=$sha&status=success&per_page=100" \
  --jq '[.workflow_runs[] | select(.event == "push" or .event == "workflow_dispatch") | .html_url] | join(" ")')
if [ -n "$runs" ]; then
  echo "successful ci run(s) for $sha: $runs"
else
  fail "no successful ci run for $sha" \
    "Push the commit to main and wait for its ci run to pass (re-run it if it failed or was cancelled), then re-run this release workflow."
fi
