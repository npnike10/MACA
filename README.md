# MACA

## Overview

The repository for "Multi-level Advantage Credit Assignment for Cooperative Multi-Agent Reinforcement Learning" at AISTATS 2025.

## Quick setup (Python 3.8)

Recommended: start from a HARL-ready environment, then install MACA extras.

```shell
conda create -n harl python=3.8
conda activate harl
git clone git@github.com:npnike10/HARL.git
cd HARL
pip install -e .
cd ..
# now go to this MACA repo root
# Install pytorch manually first (matching your CUDA/CPU setup)
pip install sacred wandb
pip install gym==0.26.2 gymnasium==1.1.1 lbforaging==2.0.0 "vmas[gymnasium]==1.4.3"
```

For Python 3.8, VMAS import compatibility is handled in `harl/envs/gym/vmas_wrapper.py` (no VMAS env logic is changed).

## Train examples (`examples/train.py`)

Run from the MACA repo root.

```shell
# MAPPO_T on LBF v3
python -m examples.train --algo mappo_t --env gym --exp_name mappo_t_lbf \
  --scenario lbforaging:Foraging-8x8-2p-1f-v3 --n_rollout_threads 1 \
  --n_eval_rollout_threads 1 --num_env_steps 200 --episode_length 50 --cuda False

# MAPPO_T on VMAS
python -m examples.train --algo mappo_t --env gym --exp_name mappo_t_vmas \
  --scenario vmas-balance --n_agents 5 --n_rollout_threads 1 \
  --n_eval_rollout_threads 1 --num_env_steps 200 --episode_length 50 --cuda False

# HAPPO on LBF v3
python -m examples.train --algo happo --env gym --exp_name happo_lbf \
  --scenario lbforaging:Foraging-8x8-2p-1f-v3 --n_rollout_threads 1 \
  --n_eval_rollout_threads 1 --num_env_steps 200 --episode_length 50 --cuda False
```

For `gym` scenarios starting with `lbforaging:` or `vmas-`, MACA now applies DISSCv2-aligned evaluation/logging defaults unless you override them:
`eval_episodes=100`, `log_interval_steps=50000`, `eval_interval_steps=50000`.

## WandB for training runs (`mappo_t`)

`harl/configs/algos_cfgs/mappo_t.yaml` contains the logger fields used by WandB:

```yaml
logger:
  use_wandb: False
  wandb_project: project-name
  wandb_entity: entity-name
  wandb_mode: offline
```

For real WandB logging, set those values accordingly (at minimum: `use_wandb=True`, your `wandb_project`, and your `wandb_entity`), either by editing the YAML or by CLI overrides:

```shell
python -m examples.train --algo mappo_t --env gym --exp_name mappo_t_lbf_wandb \
  --scenario lbforaging:Foraging-8x8-2p-1f-v3 \
  --algo_args.logger.use_wandb=True \
  --algo_args.logger.wandb_project=<your_project> \
  --algo_args.logger.wandb_entity=<your_team_or_username> \
  --algo_args.logger.wandb_mode=online
```

## WandB multi-seed sweep templates

Two DISSC-style sweep templates are provided:

- `examples/search.config.wandb.mappo_t_lbf.yaml`
- `examples/search.config.wandb.mappo_t_vmas.yaml`

Run:

```shell
wandb sweep examples/search.config.wandb.mappo_t_lbf.yaml
SWEEP_NUM_SEEDS=3 wandb agent <entity>/<project>/<sweep_id>
```

The wrapper (`examples/wandb_sweep_wrapper_multi_seed.py`) translates DISSC-style keys to MACA keys and aggregates the sweep metric (`eval_episode_rewards_mean`) across seeds.

Wrapper notes:
- Non-equivalent DISSC keys (`standardise_rewards`, `q_nstep`, `target_update_interval_or_tau`) fail fast with a clear error.
- `save_model=False` is accepted for DISSC-style compatibility and ignored with a warning.
- Optional env vars:
  - `SWEEP_NUM_SEEDS` (default `3`)
  - `SWEEP_METRIC_NAME` (fallback metric if not detected from sweep)
  - `SWEEP_WRAPPER_WANDB_MODE` / `SWEEP_CHILD_WANDB_MODE` (default `online`)
  - `SWEEP_ENABLE_LOG_FALLBACK` (default `1`, parse child logs if API metric fetch fails)
  - `SWEEP_LOG_DIR` (default `results/wandb_wrapper_logs`)
