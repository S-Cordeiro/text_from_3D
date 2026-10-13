#!/bin/bash
# Shared Bentley destination for push.sh / pull.sh (run on your local computer).
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/push.sh|pull.sh [--dry-run]' >&2
  exit 2
fi
action="$1"
cd "$(dirname "$0")/.."
name_path="$(basename "$PWD")"
export DATALAB_REMOTE_DIR="${DATALAB_REMOTE_DIR:-$name_path}"
if [[ $# == 2 ]]; then
  exec bash scripts/sync.sh "$action" "$2"
else
  exec bash scripts/sync.sh "$action"
fi
