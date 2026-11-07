#!/usr/bin/env bash
# Submit four jobs with afterok dependencies to allow at most 8 allocated hours.
#SBATCH --job-name=language96m
#SBATCH --partition=SallesInfo
#SBATCH --nodelist=bentley
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@180
#SBATCH --output=logs/language_%j.log

set -euo pipefail
export TZ=Europe/Paris
# Refuse even setup/preflight if a job is accidentally invoked during the day.
hour="$(date +%H)"
if (( 10#$hour >= 8 && 10#$hour < 18 )); then
    echo 'Execution forbidden between 08:00 and 18:00 Europe/Paris'
    exit 1
fi
segment_start="$(date +%s)"
night_end="${2:?Missing the absolute night deadline}"
segment_end=$((segment_start + 7200))
if (( night_end < segment_end )); then segment_end="$night_end"; fi
project_root="${SLURM_SUBMIT_DIR:?Submit from the project root}"
cd "$project_root"
run_dir="${1:?Usage: sbatch scripts/train_language_bentley.sbatch runs/unique_run}"
mkdir -p "$run_dir"
run_dir="$(realpath "$run_dir")"
python="$project_root/.venv/bin/python"
data_root="$project_root/data/language"
cd "$run_dir/code"
exec 9>"$run_dir/.training.lock"
flock -n 9 || { echo 'This run is already active'; exit 1; }
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
remaining="$("$python" -m src.language_schedule budget --end-epoch "$segment_end")"

# Never train on incomplete prepared data.
"$python" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] == "complete"' "$data_root/manifest.json"
if [[ ! -f "$run_dir/resume_verified" ]]; then
    # Bound the preflight too, including its child processes.
    preflight_seconds=$((remaining - 310))
    if (( preflight_seconds > 180 )); then preflight_seconds=180; fi
    timeout --kill-after=10s "${preflight_seconds}s" "$python" scripts/check_language_resume_cuda.py \
        --data-root "$data_root" > "$run_dir/resume_check.log" 2>&1
    touch "$run_dir/resume_verified"
fi
resume=()
if [[ -f "$run_dir/last.pt" ]]; then
    resume=(--resume "$run_dir/last.pt")
fi
remaining="$("$python" -m src.language_schedule budget --end-epoch "$segment_end")"
echo "Night deadline: $(date -d "@$night_end" --iso-8601=seconds); remaining allocation: ${remaining}s"
# Graceful stop five minutes early; independent hard kill no later than 08:00.
exec timeout --signal=USR1 --kill-after=120s "$((remaining - 120))s" \
    "$python" -u -m src.train_language \
    --data-root "$data_root" --out-dir "$run_dir" \
    --n-layer 8 --n-head 8 --n-embd 768 --block-size 1024 \
    --batch-size 4 --total-batch-size 32768 \
    --max-steps 58629 --warmup-steps 100 \
    --eval-every 250 --eval-steps 20 \
    --save-every 250 --save-seconds 900 --keep-checkpoints 3 \
    --max-seconds "$((remaining - 300))" --night-only --compile "${resume[@]}"
