#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'


# determine the current local branch
current_branch=$(git rev-parse --abbrev-ref HEAD)
# try to use the matching remote branch
if git show-ref --verify --quiet "refs/remotes/origin/$current_branch"; then
  remote_ref="origin/$current_branch"
elif git show-ref --verify --quiet refs/remotes/origin/HEAD; then
  remote_ref=origin/HEAD
elif git show-ref --verify --quiet refs/remotes/origin/main; then
  remote_ref=origin/main
elif git show-ref --verify --quiet refs/remotes/origin/master; then
  remote_ref=origin/master
else
  remote_ref=HEAD
fi

# list remote tracked files (not directories)
remote_files=$(git ls-tree -r --name-only "$remote_ref" 2>/dev/null | sed 's/[[:space:]]*$//' | sort | uniq || true)
tmp_remote=$(mktemp)
tmp_local=$(mktemp)
echo "$remote_files" > "$tmp_remote"
find . -type f -not -path "./.git/*" -print | sed 's#^\./##' | sed 's/[[:space:]]*$//' | sort | uniq > "$tmp_local"

comm -23 "$tmp_local" "$tmp_remote"

rm -f "$tmp_remote" "$tmp_local"
