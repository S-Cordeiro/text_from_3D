"""Submit bounded 2h night allocations, continuing only after a clean pause."""
import argparse
from datetime import datetime, timedelta, time
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--language-run', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--continue-campaign', action='store_true')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    manifest_path = campaign / 'manifest.json'
    if not args.continue_campaign:
        if manifest_path.exists():
            raise ValueError('Campaign already exists; refusing duplicate submission')
        if not all((args.project, args.language_run, args.dataset)):
            parser.error('New campaigns require project, language-run and dataset')
        experiments = []
        for name, width, depth in [('baseline', 256, 4), ('depth6', 256, 6), ('width384', 384, 4)]:
            experiments.append(dict(name=name, d_model=width, mv_layers=depth, epochs=80,
                unfreeze=0, lr=1e-4, min_lr=3e-6, dino_lr=1e-6, warmup=750))
        for blocks in (0, 1, 2):
            experiments.append(dict(name=f'finetune_dino{blocks}', d_model=256, mv_layers=4,
                epochs=10, unfreeze=blocks, lr=2e-5, min_lr=2e-6, dino_lr=1e-6,
                warmup=100, initialize_from='baseline'))
        hashes = {str(p.relative_to(campaign / 'code')): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((campaign / 'code').rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
        manifest = dict(project=str(args.project.resolve()), python=str(args.project.resolve() / '.venv/bin/python'),
            language_run=str(args.language_run.resolve()), dataset=str(args.dataset.resolve()),
            experiments=experiments, code_sha256=hashes, max_allocations=32,
            protocol='One seed (1337); same effective batch 16, microbatch 4; 80 epochs from scratch; '
                     'three matched 10-epoch branches from baseline best (frozen control, last 1/2 DINO blocks); '
                     '15M trainable parameter limit; nights 18:00-08:00 Europe/Paris.')
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    manifest = json.loads(manifest_path.read_text())
    jobs_file = campaign / 'jobs.jsonl'
    previous = [json.loads(line) for line in jobs_file.read_text().splitlines()] if jobs_file.exists() else []
    if len(previous) >= manifest['max_allocations']:
        raise RuntimeError('Allocation cap reached; campaign paused for review')
    paris = ZoneInfo('Europe/Paris')
    now = datetime.now(paris)
    if now.hour < 8:
        begin, end = now, datetime.combine(now.date(), time(8), paris)
    elif now.hour < 18:
        begin = datetime.combine(now.date(), time(18), paris)
        end = datetime.combine(now.date() + timedelta(days=1), time(8), paris)
    else:
        begin, end = now, datetime.combine(now.date() + timedelta(days=1), time(8), paris)
    minutes = min(120, int((end - begin).total_seconds() // 60) - 1)
    if minutes < 20:
        begin = datetime.combine(now.date(), time(18), paris)
        end = datetime.combine(now.date() + timedelta(days=1), time(8), paris)
        minutes = 120
    command = ['sbatch', '--parsable', f'--begin={begin:%Y-%m-%dT%H:%M:%S}',
        f'--deadline={end:%Y-%m-%dT%H:%M:%S}', f'--time={minutes}',
        f'--output={campaign}/slurm_%j.log', '--kill-on-invalid-dep=yes']
    if previous:
        command += [f'--dependency=afterok:{previous[-1]["job_id"]}']
    command += [str(campaign / 'code/scripts/vision_campaign_bentley.sbatch'), str(campaign), manifest['project'], str(minutes)]
    job = subprocess.check_output(command, text=True, cwd=manifest['project']).strip().split(';')[0]
    with jobs_file.open('a') as f:
        f.write(json.dumps(dict(job_id=job, begin=begin.isoformat(), deadline=end.isoformat(), minutes=minutes)) + '\n')
    print(f'Campaign: {campaign}\nSubmitted job: {job}\nBegin: {begin.isoformat()}\nDeadline: {end.isoformat()}', flush=True)


if __name__ == '__main__':
    main()
