# MACA

## Overview

The repository for "Multi-level Advantage Credit Assignment for Cooperative Multi-Agent Reinforcement Learning" at AISTATS 2025.

## Quick setup (Python 3.8)

```shell
conda create -n harl python=3.8
conda activate harl
# Install pytorch manually first (matching your CUDA/CPU setup)
pip install sacred
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
