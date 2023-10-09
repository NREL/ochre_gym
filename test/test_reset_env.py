import unittest 
import ochre_gym
import numpy as np


class TestResetEnv(unittest.TestCase): 

    def setUp(self):
        self.env = ochre_gym.load(
            env_name="basic-v0",
            seed=42,
            override_equipment_controls={
                'HVAC Cooling': ['Setpoint', 'Load Fraction'],
                'HVAC Heating': ['Duty Cycle', 'Setpoint', 'Deadband']
            },
            start_time = '2018-06-03 12:00:00',
            end_time = '2018-06-04 12:00:00',
            time_res = '00:15',
            reward_normalization=True,
            vectorize_observations=True,
            thermal_comfort_unit_penalty=10.0,
            thermal_comfort_band=[20, 23],
            dr_type='TOU',
            log_to_console=False,
            log_to_file=False
        )


    def test_same_reward(self):
        """Test that the same action applied to the same environment state after reset results in the same reward.
        """
        actions_taken = []
        # run once till termination with random actions
        terminated = False
        reward = 0
        while not terminated:
            action = self.env.action_space.sample()
            actions_taken.append(action)
            obs, rew, terminated, truncated, info = self.env.step(action)
            reward += rew

        # reset and run again with the same actions
        self.env.reset()
        reward2 = 0
        for action in actions_taken:
            obs, rew, terminated, truncated, info = self.env.step(action)
            reward2 += rew
        
        self.assertEqual(reward, reward2)
        self.env.close()


    def test_same_observation(self):
        """Test that the same action applied to the same environment state after reset results in the same outcome.
        """
        # take one random action
        action = self.env.action_space.sample()
        obs, rew, terminated, truncated, info = self.env.step(action)
        # reset
        self.env.reset()
        # take the same action
        obs2, rew2, terminated2, truncated2, info2 = self.env.step(action)
        # check that the observations are the same
        self.assertTrue(np.allclose(obs, obs2, atol=1e-6))
        self.env.close()


    def test_reset_step_count(self):
        """Test that the step count is reset to 0 after a reset.
        """
        # take one random action
        action = self.env.action_space.sample()
        obs, rew, terminated, truncated, info = self.env.step(action)
        # check that the step count is 1
        self.assertEqual(self.env.step_count, 1)
        # reset
        self.env.reset()
        # check that the step count is 0
        self.assertEqual(self.env.step_count, 0)
        self.env.close()
