import unittest
import ochre_gym
import numpy as np


class TestActSpaces(unittest.TestCase):

    def test_vector_actions(self):
        """Test that a vector action space is clipped and processed correctly.
        """
        env = ochre_gym.load(
            env_name="basic-v0",
            override_equipment_controls={
                'HVAC Cooling': ['Load Fraction', 'Setpoint'],
                'HVAC Heating': ['Duty Cycle', 'Setpoint', 'Deadband'],
                'Water Heating': ['Duty Cycle', 'Setpoint']
            },
            vectorize_actions=True,
            clip_actions=True,
            env_seed = 1,
            log_to_file = False,
            log_to_console = False,
        )

        self.assertTrue(env.action_space.shape == (7,))

        test_action = np.array([
            1, 20, 0.5, 20, 5, 0.1, 59
        ])  
                
        test_action = env.action(test_action)
        self.assertTrue(test_action[0] == 1)   # HVAC Cooling - Load Fraction
        self.assertTrue(test_action[1] == 20)  # HVAC Cooling - Setpoint
        self.assertTrue(test_action[2] == 0.5) # HVAC Heating - Duty Cycle
        self.assertTrue(test_action[3] == 20)  # HVAC Heating - Setpoint
        self.assertTrue(test_action[4] == 5)   # HVAC Heating - Deadband
        self.assertTrue(test_action[5] == 0.1) # Water Heating - Duty Cycle
        self.assertTrue(test_action[6] == 59)  # Water Heating - Setpoint

        env.step(test_action)
        env.close()

    def test_vector_action_wrong_shape(self):
        """Test that an exception is raised when vector actions with the wrong shape are passed."""
        env = ochre_gym.load(
            env_name="basic-v0",
            override_equipment_controls={
                'HVAC Cooling': ['Load Fraction', 'Setpoint'],
                'HVAC Heating': ['Duty Cycle', 'Setpoint', 'Deadband'],
                'Water Heating': ['Duty Cycle', 'Setpoint']
            },
            vectorize_actions=True,
            clip_actions=True,
            env_seed = 1,
            log_to_file = False,
            log_to_console = False,
        )

        test_action = np.array([
            1, 20, 0.5, 20, 5, 0.1
        ])  
        with self.assertRaises(ValueError):
            env.step(test_action)
        env.close()


    def test_action_with_dictionary_inputs(self):
        """Test composite actions."""
        env = ochre_gym.load(
            env_name="basic-v0",
            override_equipment_controls={
                'HVAC Cooling': ['Load Fraction', 'Setpoint'],
                'HVAC Heating': ['Duty Cycle', 'Setpoint', 'Deadband'],
                'Water Heating': ['Duty Cycle', 'Setpoint']
            },
            vectorize_actions=False,
            env_seed = 1,
            log_to_file = False,
            log_to_console = False,
        )

        test_action = {
            'HVAC Cooling': {
                'Setpoint': 20,
                'Load Fraction': 1
            },
            'HVAC Heating': {
                'Duty Cycle': 0.5,
                'Setpoint': 20,
                'Deadband': 5
            },
            'Water Heating': {
                'Setpoint': 59,
                'Duty Cycle': 0.1
            }
        }
        env.step(test_action)
        env.close()


    def test_empty_action(self):
        env = ochre_gym.load(
            env_name="basic-v0",
            override_equipment_controls={
                'HVAC Cooling': ['Setpoint', 'Load Fraction'],
                'HVAC Heating': ['Duty Cycle', 'Setpoint', 'Deadband'],
                'Water Heating': ['Setpoint', 'Duty Cycle']
            },
            vectorize_actions=False,
            env_seed = 1,
            log_to_file = False,
            log_to_console = False,
        )
        test_action = {}
        with self.assertRaises(ValueError):
            env.step(test_action)
            
        env.close()

    def test_action_with_missing_control_type(self):
        env = ochre_gym.load(
            env_name="basic-v0",
            override_equipment_controls={
                'HVAC Cooling': ['Setpoint', 'Load Fraction'],
                'HVAC Heating': ['Duty Cycle', 'Setpoint', 'Deadband'],
                'Water Heating': ['Setpoint', 'Duty Cycle']
            },
            vectorize_actions=False,
            env_seed = 1,
            log_to_file = False,
            log_to_console = False,
        )

        # Missing HVAC Cooling - Setpoint
        test_action = {
            'HVAC Cooling': {
                'Load Fraction': 1
            },
            'HVAC Heating': {
                'Duty Cycle': 0.5,
                'Setpoint': 20,
                'Deadband': 5
            },
            'Water Heating': {
                'Setpoint': 59,
                'Duty Cycle': 0.1
            }
        }

        with self.assertRaises(ValueError):
            env.step(test_action)
        env.close()


    def test_action_with_missing_equipment(self):
        env = ochre_gym.load(
            env_name="basic-v0",
            override_equipment_controls={
                'HVAC Cooling': ['Setpoint', 'Load Fraction'],
                'HVAC Heating': ['Duty Cycle', 'Setpoint', 'Deadband'],
                'Water Heating': ['Setpoint', 'Duty Cycle']
            },
            vectorize_actions=False,
            env_seed = 1,
            log_to_file = False,
            log_to_console = False,
        )

        # Missing HVAC Cooling
        test_action = {
            'HVAC Heating': {
                'Duty Cycle': 0.5,
                'Setpoint': 20,
                'Deadband': 5
            },
            'Water Heating': {
                'Setpoint': 59,
                'Duty Cycle': 0.1
            }
        }

        with self.assertRaises(ValueError):
            env.step(test_action)
        env.close()


    def test_action_with_extra_control_type(self):
        env = ochre_gym.load(
            env_name="basic-v0",
            override_equipment_controls={
                'HVAC Cooling': ['Setpoint', 'Load Fraction'],
                'HVAC Heating': ['Duty Cycle', 'Setpoint', 'Deadband'],
                'Water Heating': ['Setpoint', 'Duty Cycle']
            },
            vectorize_actions=False,
            env_seed = 1,
            log_to_file = False,
            log_to_console = False,
        )

        # Extra HVAC Cooling - Duty Cycle
        test_action = {
            'HVAC Cooling': {
                'Setpoint': 20,
                'Load Fraction': 1,
                'Duty Cycle': 0.5
            },
            'HVAC Heating': {
                'Duty Cycle': 0.5,
                'Setpoint': 20,
                'Deadband': 5
            },
            'Water Heating': {
                'Setpoint': 59,
                'Duty Cycle': 0.1
            }
        }

        with self.assertRaises(ValueError):
            env.step(test_action)
        env.close()

    def test_action_with_extra_equipment(self):
        env = ochre_gym.load(
            env_name="basic-v0",
            override_equipment_controls={
                'HVAC Cooling': ['Setpoint', 'Load Fraction'],
                'HVAC Heating': ['Duty Cycle', 'Setpoint', 'Deadband'],
                'Water Heating': ['Setpoint', 'Duty Cycle']
            },
            vectorize_actions=False,
            env_seed = 1,
            log_to_file = False,
            log_to_console = False,
        )

        # Extra Equipment - EV
        test_action = {
            'HVAC Cooling': {
                'Load Fraction': 1
            },
            'HVAC Heating': {
                'Duty Cycle': 0.5,
                'Setpoint': 20,
                'Deadband': 5
            },
            'Water Heating': {
                'Setpoint': 59,
                'Duty Cycle': 0.1
            },
            'EV': {
                'Duty Cycle': 0.1
            }
        }

        with self.assertRaises(ValueError):
            env.step(test_action)
        env.close()


    def test_ClipActionComposite(self):

        env = ochre_gym.load(
            env_name="basic-v0",
            override_equipment_controls={
                'HVAC Cooling': ['Load Fraction', 'Setpoint'],
                'HVAC Heating': ['Duty Cycle', 'Setpoint', 'Deadband'],
                'Water Heating': ['Duty Cycle', 'Setpoint']
            },
            vectorize_actions=False,
            clip_actions=True,
            env_seed = 1,
            log_to_file = False,
            log_to_console = False,
        )

        test_action = {
            'HVAC Cooling': {
                'Setpoint': -100, # Should be clipped to low
                'Load Fraction': 2 # Should be clipped to 1
            },
            'HVAC Heating': {
                'Duty Cycle': 100, # Should be clipped to high
                'Setpoint': 20,
                'Deadband': 5
            },
            'Water Heating': {
                'Setpoint': 59,
                'Duty Cycle': -3  # Should be clipped to low
            }
        }

        test_action = env.action(test_action)
    
        self.assertEqual(test_action['HVAC Cooling']['Setpoint'], env.action_space['HVAC Cooling']['Setpoint'].low)
        self.assertEqual(test_action['HVAC Cooling']['Load Fraction'], 1)
        self.assertEqual(test_action['HVAC Heating']['Duty Cycle'], env.action_space['HVAC Heating']['Duty Cycle'].high)
        self.assertEqual(test_action['Water Heating']['Duty Cycle'], env.action_space['Water Heating']['Duty Cycle'].low)
        env.close()

if __name__ == '__main__':
    unittest.main()