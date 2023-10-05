# Welcome to the OCHRE Gym Docs

OCHRE (pronounced "Oh-ker") Gym is a Gymnasium environment based on the purely Python-based [OCHRE](https://github.com/NREL/OCHRE) residential energy building simulator. OCHRE is a high-fidelity, high-resolution residential building model developed by NREL with behind-the-meter DERs and flexible load models that integrates with controllers and distribution models in building-to-grid co-simulation platforms. It has been benchmarked against EnergyPlus to quantify the tradeoff between fidelity and speed. Read more about OCHRE [here](https://www.sciencedirect.com/science/article/pii/S0306261921002464).

OCHRE Gym features:

- No EnergyPlus; each Dwelling consists of multiple RC circuits implemented in pure Python
- Works with any building that OCHRE supports: get building models from NREL End-Use Load Profiles, ResStock, BEopt, etc.
- Flexible control of building equipment (HVAC, Water Heater)--support coming for DERs (PV, Battery, EV)
- Customizable observation space with equipment-level, building-level, and building metadata
- Simple reward: minimize cost of energy use while maintaining comfort
- 3 different demand response (DR) cost functions: Real-Time Pricing (RTP), Time-of-Use (TOU), and Power Constraints (PC)


## Installation

Install in editable mode with `pip install -e .` from the root of this repo.

1. Using `conda` or `venv`, create an environment with `python >= 3.9`: `conda create -n ochre_gym python=3.9`.
1. Clone this repo: `git clone git@github.com/NREL/ochre_gym.git`
2. `cd ochre_gym`
2. `pip install -e .`

Test your installation with `unittest` by running `python3 -m unittest` from the root of this repo.

Support for installation via PyPI is coming soon.

## Getting Started

- [Quick start](https://nrel.github.io/ochre_gym/Getting%20Started/basics/)
- [Stable Baselines3 integration](https://nrel.github.io/ochre_gym/Getting%20Started/stable_baselines/)
