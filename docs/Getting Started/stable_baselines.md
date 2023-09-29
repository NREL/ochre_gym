# Stable Baselines3

You can use an OchreEnv with [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) as a SB3 vector env. Simply make sure to set `vectorize_observations=True` and `vectorize_actions=True` when loading the environment.

```python

import ochre_gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

vec_env = ochre_gym.load(
    env_name="basic-v0",
    vectorize_actions=True,      # for vec_env
    vectorize_observations=True, # for vec_env
    override_equipment_controls={
        'HVAC Cooling': ['Setpoint'],
        'HVAC Heating': ['Setpoint']
    },
    override_ochre_observations_with_keys = [
        'Energy Price ($)',
        'Temperature - Indoor (C)',
        'Total Electric Power (kW)'
    ],
    dr_type = 'RTP',
    start_time = '2018-06-03 12:00:00',
    time_res = '00:30',
    episode_duration = '30 days',
    seed = 1,
    log_to_file = False,
    log_to_console = False,
)

model = SAC("MlpPolicy", vec_env, verbose=0)
model.learn(total_timesteps=80000)
model.save("sac_ochre")
```

This uses the default Soft Actor Critic (SAC) model with a Multi-Layer Perceptron (MLP) policy network and trains it for 80,000 environment steps. The trained agent is saved to `sac_ochre.zip` in the current directory.