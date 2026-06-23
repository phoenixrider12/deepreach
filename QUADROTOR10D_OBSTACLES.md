# Quadrotor10D obstacle settings (mid_gate & backroom)

The `Quadrotor10D` dynamics (`dynamics/dynamics.py`) is a 10-D quaternion rate model
(matching SousVide's `carl` frame, world frame z-DOWN, metric). It ships with **two
obstacle environments**, selected by the `--env` flag, plus gate-specific shape
options. Both are derived from the corresponding Gaussian-splat scene.

## Selecting the obstacle setting

`run_experiment.py` auto-exposes every `Quadrotor10D.__init__` argument as a
**required** CLI flag, so you must pass all three of `--env`, `--stand_shape`,
`--top_shape` on every run (the latter two are ignored when `--env backroom`).

| flag | values | applies to | meaning |
|---|---|---|---|
| `--env` | `gate` \| `backroom` | both | **which scene's obstacle set to use** |
| `--stand_shape` | `prism` \| `cuboid` | gate only | triangular stands, or their axis-aligned bbox (smoother to learn) |
| `--top_shape` | `bar` \| `roof` | gate only | top crossbar as a thin box, or a roof half-space the drone must stay below |

### `--env gate` — mid_gate
Two triangular stands + a middle crossbar + the top (bar or roof) + floor half-space.
The drone passes through the gate holes.
```bash
python run_experiment.py --mode train --experiment_name Quadrotor_gate \
  --dynamics_class Quadrotor10D --env gate --stand_shape cuboid --top_shape roof \
  --tMax 1 --pretrain --pretrain_iters 10000 --num_epochs 110000 --counter_end 100000 \
  --num_nl 512 --lr 2e-5 --num_MPC_batches 30 \
  --use_wandb --wandb_entity <ENTITY> --wandb_project deepreachMPC \
  --wandb_name Quadrotor_gate --wandb_group Quadrotor_gate
```

### `--env backroom` — backroom
Perimeter walls + central divider + lower block + a triangular top-right corner +
floor/ceiling half-spaces (from the edited occupancy map). `--stand_shape` /
`--top_shape` are required by the CLI but ignored here — pass any valid value.
```bash
python run_experiment.py --mode train --experiment_name Quadrotor_backroom \
  --dynamics_class Quadrotor10D --env backroom --stand_shape cuboid --top_shape roof \
  --tMax 1 --pretrain --pretrain_iters 10000 --num_epochs 110000 --counter_end 100000 \
  --num_nl 512 --lr 2e-5 --num_MPC_batches 30 \
  --use_wandb --wandb_entity <ENTITY> --wandb_project deepreachMPC \
  --wandb_name Quadrotor_backroom --wandb_group Quadrotor_backroom
```

## What changes with `--env`
`env` switches the **obstacle SDF** (`obstacle_sdf` dispatches to `gate_obstacle_sdf`
or `backroom_obstacle_sdf`), the **compute domain** (`state_range_`), the
**target-state sampler** (`sample_target_state`), and the **validation plot slices**.
The drone model, control bounds, and Hamiltonian are identical across envs.
`boundary_fn = obstacle_sdf(pos) - collisionR`, with `l(x) > 0` safe and `l(x) < 0`
in the failure set (avoid BRT).

## Validation plots
Each checkpoint logs three velocity-swept top-down BRT slices to wandb: `val_plot`
(vx), `val_plot_vy`, `val_plot_vz`, at an env-appropriate flight-height slice.

## Programmatic use
```python
from dynamics.dynamics import Quadrotor10D
dyn = Quadrotor10D(env='backroom')                       # backroom obstacles
dyn = Quadrotor10D(env='gate', stand_shape='cuboid', top_shape='roof')  # mid_gate
```
See `../ANALYSIS_TOOLS.md` (parent SousVide repo) for how the obstacle geometry was
derived from the splats, and for the occupancy-map / obstacle-editing tools.
