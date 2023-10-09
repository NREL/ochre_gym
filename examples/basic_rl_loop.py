import ochre_gym
import numpy as np

env = ochre_gym.load(
    env_name="basic-v0",
    seed=42,
    override_equipment_controls={
        'HVAC Cooling': ['Setpoint'],
        'HVAC Heating': ['Setpoint'],
    },
    disable_uncontrollable_loads=True,
    vectorize_observations=True,
    use_all_ochre_observations=True,
    vectorize_actions=True,
    start_time = '2018-07-01 01:00:00',
    end_time = '2018-07-31 23:00:00',
    time_res = '00:30',
    lookahead = '01:00',
    reward_normalization=True,
    thermal_comfort_unit_penalty=10.0,
    thermal_comfort_band=[20, 23],
    verbosity = 7,
    dr_type='RTP',  # TOU, RTP, PC
    log_to_console=False,
    log_to_file=True
)

print(env.action_space)
print('Action space flattened shape: ', env.action_space.shape)

print(env.observation_space)
print('Observation space flattened shape: ', env.observation_space.shape)

done = False
total_reward = 0.0


for i in range(300):
    
    action = env.action_space.sample()

    obs, rew, terminated, truncated, info = env.step(action)
    
    total_reward += rew
    if terminated:
        break

print('OCHRE control results')
for k,v in info.items():
    print(k,v)
print(rew, terminated, truncated)
print()
print('Total reward is %f' % total_reward)

env.close()
