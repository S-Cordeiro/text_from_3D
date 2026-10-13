#!/bin/bash
# Run locally, from any directory. Dry-run first; no delete option.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# -ge 1 && $# -le 2 ]] || { echo "Usage: sync.sh push|pull [--dry-run]" >&2; exit 2; }
action="${1:-}"
remote=bentley
remote_dir="${DATALAB_REMOTE_DIR:-$(basename "$PWD")}"
# Shell-safe relative home path or absolute project path on Bentley.
[[ "$remote_dir" =~ ^[a-zA-Z0-9_./-]+$ && "$remote_dir" != / && "$remote_dir" != . && "$remote_dir" != *..* ]] || {
  echo 'Use a specific remote directory without spaces or ..' >&2; exit 2;
}
[[ "$remote" =~ ^[a-zA-Z0-9_@.-]+$ && "$remote" != -* ]] || exit 2
flags=(-av --itemize-changes)
if [[ "${2:-}" == '--dry-run' ]]; then flags+=(-n); elif [[ -n "${2:-}" ]]; then exit 2; fi
case "$action" in
  push)
    ssh "$remote" "mkdir -p '$remote_dir'"
    rsync "${flags[@]}" --exclude='.venv*/' --exclude='dev/' --exclude='venv/' --exclude='.git/' --exclude='runs/' \
      --exclude='data/' --exclude='demo_results/' --exclude='__pycache__/' --include='*/' --include='*.py' --include='*.sh' \
      --include='*.sbatch' --include='*.json' --include='requirements*.txt' \
      --include='README.md' --exclude='*' ./ "$remote:$remote_dir/"
    ;;
  pull)
    mkdir -p runs logs
    # Filter order matters: never bring back model weights, even in nested dirs.
    rsync "${flags[@]}" --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' \
      --exclude='wandb/' --exclude='__pycache__/' \
      --include='*/' --include='*.json' --include='*.jsonl' --include='*.txt' \
      --include='*.log' --include='*.npz' --include='events.out.tfevents*' \
      --exclude='*' "$remote:$remote_dir/runs/" runs/
    rsync "${flags[@]}" "$remote:$remote_dir/logs/" logs/
    ;;
  *) echo 'Usage: bash scripts/sync.sh push|pull [--dry-run]' >&2; exit 2 ;;
esac
