#!/bin/bash
# Retrieve results from Bentley: pull.sh [--dry-run].
set -euo pipefail
exec bash "$(dirname "$0")/sync-target.sh" pull "$@"
