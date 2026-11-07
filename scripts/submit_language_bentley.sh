#!/usr/bin/env bash
# Run ON Bentley, from the project root. Four allocations, at most 8 hours total.
set -euo pipefail
export TZ=Europe/Paris
read -r night_begin night_deadline night_end < <(.venv/bin/python -m src.language_schedule schedule)
mkdir -p runs logs
run_dir="$(mktemp -d "$PWD/runs/language96m_fineweb_8h_XXXXXX")"
mkdir -p "$run_dir/code/src/models/language" "$run_dir/code/scripts"
cp src/train_language.py src/language_schedule.py "$run_dir/code/src/"
cp src/models/__init__.py src/models/blocks.py "$run_dir/code/src/models/"
cp src/models/language/*.py "$run_dir/code/src/models/language/"
cp scripts/check_language_resume_cuda.py "$run_dir/code/scripts/"
cp scripts/train_language_bentley.sbatch "$run_dir/"
cp data/language/manifest.json "$run_dir/dataset_manifest.json"
printf 'Timezone: Europe/Paris\nBegin: %s\nDeadline: %s\n' "$night_begin" "$night_deadline" > "$run_dir/schedule.txt"
printf '%s\n' "$run_dir" > logs/latest_language_run.txt
previous=''
jobs=()
rollback() {
    if [[ ${#jobs[@]} -gt 0 ]]; then scancel "${jobs[@]}"; fi
}
trap rollback ERR
for segment in 1 2 3 4; do
    dependency=()
    if [[ -n "$previous" ]]; then dependency=(--dependency="afterok:$previous"); fi
    job="$(sbatch --parsable --kill-on-invalid-dep=yes \
        --begin="$night_begin" --deadline="$night_deadline" "${dependency[@]}" \
        "$run_dir/train_language_bentley.sbatch" "$run_dir" "$night_end")"
    job="${job%%;*}"
    jobs+=("$job")
    printf '%s %s\n' "$segment" "$job" >> "$run_dir/jobs.txt"
    previous="$job"
done
trap - ERR
printf 'Run: %s\nJobs: %s\n' "$run_dir" "${jobs[*]}"
