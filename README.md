# OCHRE Gym

All authors are with the National Renewable Energy Laboratory (NREL).

## Overview

OCHRE (pronounced "Oh-ker") Gym is a Gymnasium environment based on the purely Python-based [OCHRE](https://github.com/NREL/OCHRE) residential energy building simulator. OCHRE is a high-fidelity, high-resolution residential building model developed by NREL with behind-the-meter DERs and flexible load models that integrates with controllers and distribution models in building-to-grid co-simulation platforms. It has been benchmarked against EnergyPlus to quantify the tradeoff between fidelity and speed. Read more about OCHRE [here](https://www.sciencedirect.com/science/article/pii/S0306261921002464).

OCHRE Gym features:

- No EnergyPlus; each Dwelling consists of multiple RC circuits implemented in pure Python
- Works with any building that OCHRE supports: get building models from NREL End-Use Load Profiles, ResStock, BEopt, etc.
- Flexible control of building equipment (HVAC, Water Heater)--support coming for DERs (PV, Battery, EV)
- Customizable observation space with equipment-level, building-level, and building metadata
- Simple reward: minimize cost of energy use while maintaining comfort
- 3 different demand response (DR) cost functions: Real-Time Pricing (RTP), Time-of-Use (TOU), and Power Constraints (PC)

[Read our docs](https://nrel.github.io/ochre_gym) to get started quickly.

## Installation

Install from PyPI `pip install ochre_gym`.

Or,

Install in editable mode with `pip install -e .` from the root of this repo.

1. Using `conda` or `venv`, create an environment with `python >= 3.9`: `conda create -n ochre_gym python=3.9`.
1. Clone this repo: `git clone git@github.com/NREL/ochre_gym.git`
2. `cd ochre_gym`
2. `pip install -e .`

Test your installation with `unittest` by running `python3 -m unittest` from the root of this repo.

## Quick Start

Init one of the provided buildings (e.g., `basic-v0`) with `ochre_gym.load()`:

```python
import ochre_gym

env = ochre_gym.load(
    env_name="basic-v0",
)

for step in range(1000):

    # Sample an action from the action space
    action = env.action_space.sample()

    # Step the environment with the sampled action
    obs, rew, terminated, truncated, info = env.step(action)
    
    # Check if the episode is done       
    if terminated:
        print("Episode finished after {} timesteps".format(step+1))
        break
```

The `ochre_gym.load()` function will handle creating the OCHRE building simulator instance using the properties, schedule, and weather files located in `ochre_gym/buildings/basic-v0`. 
Keyword arguments passed to `load` can be used to override the defaults in the `ochre_gym/buildings/defaults.toml` config file. 

## Funding Acknowledgement

This work was authored by the National Renewable Energy Laboratory (NREL), operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported by the Laboratory Directed Research and Development (LDRD) Program at NREL. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.

