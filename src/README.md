# Training

Run these commands **from the project root** (the directory containing
`src/`, `.venv/`, and `data/`). Use `-m src.train` to resolve imports.
The datasets must already be generated and dependencies installed in `.venv`.

## Two cubes — 80 epochs

```bash
.venv/bin/python -u -m src.train \
  --dataset data/cube_sphere_cylinder \
  --epochs 80 \
  --eval-every 300 --log-every 25 \
  --runs-dir runs/cube_sphere_cylinder \
  --no-checkpoints
```

Training uses batches of 16 scenes, shuffled at each epoch, and four data-loading
workers. Validation uses batches of one scene. `config.json` records the training
loader's batch size.

The learning rate uses a linear warmup over 750 optimizer steps, starting at
`1e-4 / 750` and reaching `1e-4`, followed by cosine decay to `3e-6` over the
remaining training steps. Runs of at most 750 steps only use warmup.
With 5,000 training scenes, each epoch contains 313 optimizer steps, so 20 epochs
correspond to 6,260 steps. These settings are defined in `src/train.py`.

## Limit execution time to 6 hours

```bash
timeout --signal=TERM --kill-after=60s 6h \
  .venv/bin/python -u -m src.train \
    --dataset data/two_cubes \
    --epochs 20 \
    --eval-every 600 --log-every 25 \
    --runs-dir runs/two_cubes_cosine \
```

## Results and checkpoints

Each launch creates a new `run_N` directory under `--runs-dir`, containing
`config.json`, `train.log`, and `metrics.jsonl`. `--eval-every` and `--log-every`
are measured in steps.

The first example disables checkpoints with `--no-checkpoints` to save disk
space; model weights are lost when training stops. The six-hour example saves
`best.pt` and `last.pt` by default, so allow enough space for both files and a
temporary checkpoint. The training script does not currently provide a resume
option.

Show all options:

```bash
.venv/bin/python -m src.train --help
```
