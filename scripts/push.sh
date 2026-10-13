#!/bin/bash
# Send selected local project files to Bentley: push.sh [--dry-run].
set -euo pipefail
exec bash "$(dirname "$0")/sync-target.sh" push "$@"
