#!/usr/bin/env bash
#SBATCH --job-name=vision-campaign
#SBATCH --partition=SallesInfo
#SBATCH --nodelist=bentley
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@180
set -euo pipefail
export TZ=Europe/Paris
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
campaign="${1:?Missing campaign directory}"
project="${2:?Missing project directory}"
cd "$campaign/code"
exec 9>"$campaign/.campaign.lock"
flock -n 9 || { echo 'Campaign already running'; exit 1; }
# Also prevent concurrent training of the existing language run.
language_run="$("$project/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["language_run"])' "$campaign/manifest.json")"
exec 8>"$language_run/.training.lock"
flock -n 8 || { echo 'Language run already active'; exit 1; }
remaining="$("$project/.venv/bin/python" -m src.language_schedule budget)"
allocation_seconds=$(( ${3:?Missing allocation minutes} * 60 ))
if (( remaining > allocation_seconds )); then remaining="$allocation_seconds"; fi
remaining=$((remaining - 360))
if (( remaining < 240 )); then exit 1; fi
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv --loop=30 >> "$campaign/gpu_${SLURM_JOB_ID}.csv" &
monitor_pid=$!
child_pid=''
cleanup() { kill "$monitor_pid" 2>/dev/null || true; }
forward_stop() { if [[ -n "$child_pid" ]]; then kill -USR1 -- "-$child_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap forward_stop USR1 TERM
"$project/.venv/bin/python" -m pip freeze > "$campaign/environment.txt"
# A separate process group lets Slurm signals reach trainer and controller.
setsid timeout --signal=USR1 --kill-after=120s "$((remaining + 120))s" \
    "$project/.venv/bin/python" -u scripts/run_vision_campaign.py "$campaign" --seconds "$remaining" &
child_pid=$!
set +e
wait "$child_pid"
result=$?
# wait can be interrupted by the signal trap while the child saves its checkpoint.
if kill -0 "$child_pid" 2>/dev/null; then wait "$child_pid"; result=$?; fi
set -e
if (( result == 75 )); then
    "$project/.venv/bin/python" scripts/submit_vision_campaign.py "$campaign" --continue-campaign
elif (( result != 0 )); then
    echo "Campaign failed (exit $result); no further job submitted"
    exit "$result"
fi
