"""Resumable night campaign: finish the existing LLM, then controlled vision ablations."""
import argparse
import json
from pathlib import Path
import subprocess
import signal
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def llm_state(python, checkpoint):
    code = ('import json,torch,sys; c=torch.load(sys.argv[1],map_location="cpu",weights_only=True); '
            'print(json.dumps({"step":c["step"],"target":c["args"]["max_steps"],'
            '"tokens_per_step":c["args"]["total_batch_size"],"val_loss":c["val_loss"]}))')
    return json.loads(subprocess.check_output([python, '-c', code, str(checkpoint)], text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--seconds', type=int, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    manifest = read_json(campaign / 'manifest.json')
    python = manifest['python']
    end = time.monotonic() + args.seconds
    def request_stop(signum, frame):
        nonlocal end
        end = time.monotonic()
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, request_stop)
    def remaining():
        return int(end - time.monotonic())
    def run(command, cwd, logfile):
        print('RUN', command, flush=True)
        with Path(logfile).open('a') as output:
            subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, check=True)
    state = llm_state(python, Path(manifest['language_run']) / 'last.pt')
    write_json(campaign / 'llm_status.json', state)
    if state['target'] != 58629:
        raise ValueError('Unexpected LLM target; refusing to change the original schedule')
    if state['step'] < state['target']:
        if remaining() < 600:
            return 75
        language_run = Path(manifest['language_run'])
        run([python, '-u', '-m', 'src.train_language', '--resume', str(language_run / 'last.pt'),
             '--out-dir', str(language_run), '--max-seconds', str(remaining() - 120),
             '--night-only', '--keep-checkpoints', '2', '--save-every', '250', '--save-seconds', '900'],
            language_run / 'code', campaign / 'llm_resume.log')
        state = llm_state(python, language_run / 'last.pt')
        write_json(campaign / 'llm_status.json', state)
        if state['step'] < state['target']:
            return 75
    audit = campaign / 'dataset_audit.json'
    if not audit.exists():
        run([python, '-m', 'src.vision_audit', manifest['dataset'], '--output', str(audit),
             '--historic-runs', str(Path(manifest['project']) / 'runs')],
            campaign / 'code', campaign / 'audit.log')
    for experiment in manifest['experiments']:
        directory = campaign / experiment['name']
        status = directory / 'status.json'
        if status.exists() and read_json(status)['completed']:
            continue
        if remaining() < 600:
            return 75
        command = [python, '-u', '-m', 'src.train', '--dataset', manifest['dataset'],
            '--dataset-audit', str(audit), '--run-dir', str(directory),
            '--epochs', str(experiment['epochs']), '--d-model', str(experiment['d_model']),
            '--mv-layers', str(experiment['mv_layers']), '--dino-unfreeze-blocks', str(experiment['unfreeze']),
            '--lr', str(experiment['lr']), '--min-lr', str(experiment['min_lr']),
            '--dino-lr', str(experiment['dino_lr']), '--warmup-steps', str(experiment['warmup']),
            '--batch-size', '16', '--micro-batch-size', '4', '--seed', '1337',
            '--eval-every', '300', '--log-every', '25', '--save-seconds', '900',
            '--max-seconds', str(remaining() - 120)]
        if (directory / 'last.pt').exists():
            command += ['--resume', str(directory / 'last.pt')]
        elif experiment.get('initialize_from'):
            command += ['--init-checkpoint', str(campaign / experiment['initialize_from'] / 'best.pt')]
        write_json(campaign / 'active_experiment.json', experiment)
        run(command, campaign / 'code', campaign / f"{experiment['name']}.console.log")
        if not status.exists() or not read_json(status)['completed']:
            return 75
    for experiment in manifest['experiments']:
        directory = campaign / experiment['name']
        if (directory / 'best_validation_predictions.jsonl.gz').exists():
            continue
        if remaining() < 600:
            return 75
        command = [python, '-m', 'scripts.analyze_vision_checkpoint', str(directory),
                   '--max-seconds', str(remaining() - 120)]
        with (campaign / 'analysis.log').open('a') as output:
            result = subprocess.run(command, cwd=campaign / 'code', stdout=output, stderr=subprocess.STDOUT)
        if result.returncode == 75:
            return 75
        result.check_returncode()
    run([python, 'scripts/summarize_vision_campaign.py', str(campaign)], campaign / 'code', campaign / 'summary.log')
    write_json(campaign / 'status.json', {'completed': True, 'finished_at_unix': time.time()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
