# PPO-9-turbine-case

Train a **Proximal Policy Optimization (PPO)** agent to control **yaw angles for a 9‑turbine wind farm** using the **WFSim** MATLAB simulator through the **MATLAB Engine API for Python**.

This repository contains:
- A Python PPO implementation for continuous control
- A Python environment wrapper that steps a MATLAB/WFSim simulation
- MATLAB scripts used by the simulator interface (`Initial_9.m`, `Timestep_9.m`)

## Repository contents

- `ppo_9turbine.py` — Main training script + MATLAB-engine environment wrapper (`WindFarmEnv`)
- `ppo_agent_9turbine.py` — PPO agent (actor/critic), replay buffer, normalization utilities
- `Initial_9.m` — MATLAB initialization entry point (called at reset / startup)
- `Timestep_9.m` — MATLAB timestep entry point (called every step; returns turbine power)

## How it works (high level)

### State
The environment uses a 10‑D state:
- 9 yaw values (one per turbine)
- 1 scalar: total farm power (MW)

### Action
The agent outputs a 9‑D continuous action in `[-1, 1]`, which is scaled to yaw degrees:
- `action ∈ [-1, 1]  ->  yaw ∈ [-30°, 30°]`

### Reward
Reward is the total wind farm power output (in MW) returned by the MATLAB simulation.

## Requirements

### System
- MATLAB installed
- WFSim available locally (the code currently assumes a local WFSim folder)
- A Python environment that can import:
  - `numpy`
  - `torch`
  - `matlab.engine` (MATLAB Engine API for Python)

### Python packages
You’ll typically need:
- `numpy`
- `torch`

`matlab.engine` is installed via MATLAB (not pip).

## Setup

1. **Install the MATLAB Engine API for Python**  
   In your MATLAB installation, run the engine installer (exact steps depend on OS/MATLAB version).

2. **Clone / download WFSim locally**

3. **Update WFSim path in `ppo_9turbine.py`**  
   In `WindFarmEnv.__init__`, update:
   ```python
   base_path = r'/Users/akhilpatel/Desktop/Dissertation/WFSim-master'
   ```
   to point to your WFSim directory.

   The script adds these subfolders to MATLAB’s path:
   - `layoutDefinitions`
   - `controlDefinitions`
   - `solverDefinitions`

## Run training

From the repo root:

```bash
python ppo_9turbine.py
```

Common arguments (defaults are in the script):
- `--seed`
- `--max_train_steps`
- `--save_freq`
- `--batch_size`
- `--mini_batch_size`
- PPO hyperparameters: `--gamma`, `--lamda`, `--epsilon`, `--K_epochs`, `--entropy_coef`
- `--use_state_norm`
- `--use_lr_decay`

Example:

```bash
python ppo_9turbine.py --max_train_steps 50000 --save_freq 1000 --seed 10
```

## Outputs

During training, the script prints progress every 100 steps.

Model checkpoints are saved as:
- `actor_step_<total_steps>.pth`
- `critic_step_<total_steps>.pth`

## Notes / assumptions

- The environment currently uses a **fixed** thrust coefficient setting:
  - `CT_prime = 2 * ones(9)`
- Episode termination is currently time-limit only (`max_steps`), not a failure condition.
- This repo expects MATLAB scripts `Initial_9` and `Timestep_9` to be callable from MATLAB’s path.

## Citation

WFSIM: https://github.com/TUDelft-DataDrivenControl/WFSim
Based on a 2-turbine case provided to me by Yuhan Su

