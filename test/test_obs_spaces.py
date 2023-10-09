import unittest
from pathlib import Path
import tomli
from ochre import Dwelling
from ochre.utils import hpxml
import ochre_gym
from ochre_gym.spaces.obs_spaces import TimeAndEnergyPriceObservationSpaceConfig
from ochre_gym.spaces.obs_spaces import OchreObservationSpace
import numpy as np


class TestObsSpace(unittest.TestCase):

    def setUp(self) -> None:
        THIS_PATH = Path(__file__).parent.absolute()
        config_dir = THIS_PATH / '..' / 'ochre_gym' / 'buildings' 
        with open(config_dir / 'defaults.toml', 'rb') as f:
            dwelling_args = tomli.load(f)['defaults']
        building_dir = config_dir / 'basic-v0'
        dwelling_args['weather_file'] = 'G0600650.epw'
        dwelling_args = ochre_gym.parse_dwelling_args(dwelling_args, building_dir)
        dwelling_args['verbosity'] = 7

        self.dwelling = Dwelling('test', **dwelling_args)       
        self.dwelling_metadata, _ = hpxml.load_hpxml(**dwelling_args)
        
        self.keys = [
            "Air Changes per Hour - Indoor (1/hour)",
            "Air Density - Indoor (kg/m^3)",
            "Day of month",
            "Day of week",
            "Energy Price ($)",
            "Forced Ventilation Flow Rate - Indoor (m^3/s)",
            "Forced Ventilation Heat Gain - Indoor (W)",
            "Grid Voltage (-)",
            "HVAC Cooling Electric Power (kW)",
            "HVAC Cooling Reactive Power (kVAR)",
            "HVAC Heating Electric Power (kW)",
            "HVAC Heating Gas Power (therms/hour)",
            "HVAC Heating Reactive Power (kVAR)",
            "Hour of day",
            "Humidity Ratio - Indoor (-)",
            "Infiltration Flow Rate - Attic (m^3/s)",
            "Infiltration Flow Rate - Garage (m^3/s)",
            "Infiltration Flow Rate - Indoor (m^3/s)",
            "Infiltration Heat Gain - Attic (W)",
            "Infiltration Heat Gain - Garage (W)",
            "Infiltration Heat Gain - Indoor (W)",
            "Internal Heat Gain - Indoor (W)",
            "Lighting Electric Power (kW)",
            "Lighting Reactive Power (kVAR)",
            "Natural Ventilation Flow Rate - Indoor (m^3/s)",
            "Natural Ventilation Heat Gain - Indoor (W)",
            "Net Latent Heat Gain - Indoor (W)",
            "Net Sensible Heat Gain - Attic (W)",
            "Net Sensible Heat Gain - Garage (W)",
            "Net Sensible Heat Gain - Indoor (W)",
            "Occupancy (Persons)",
            "Occupancy Heat Gain - Indoor (W)",
            "Other Electric Power (kW)",
            "Other Gas Power (therms/hour)",
            "Other Reactive Power (kVAR)",
            "Radiation Heat Gain - Attic (W)",
            "Radiation Heat Gain - Garage (W)",
            "Radiation Heat Gain - Indoor (W)",
            "Relative Humidity - Indoor (-)",
            "Temperature - Attic (C)",
            "Temperature - Garage (C)",
            "Temperature - Ground (C)",
            "Temperature - Indoor (C)",
            "Temperature - Outdoor (C)",
            "Total Electric Energy (kWh)",
            "Total Electric Power (kW)",
            "Total Gas Energy (therms)",
            "Total Gas Power (therms/hour)",
            "Total Reactive Energy (kVARh)",
            "Total Reactive Power (kVAR)",
            "Unmet HVAC Load (C)",
            "Water Heating Gas Power (therms/hour)",
            "Wet Bulb - Indoor (C)",
            "Window Transmitted Solar Gain (W)"
        ]
    
    def test_create_default_observation_space(self):
        observation_space_config = TimeAndEnergyPriceObservationSpaceConfig(
                observation_args={
                    'reward_args': {
                        'dr_type': 'RTP',
                    },
                    'lookahead_steps': 1
                },
                vectorize_observations=True,
                use_all_ochre_observations=True,
                override_ochre_observations_with_keys=None
            )
        observations_manager = OchreObservationSpace(
            observation_space_config,
            self.dwelling, 
            self.dwelling_metadata
        )
        observation_space, observation_keys = \
            observations_manager.get_obs_space_from_ochre()
        self.assertEqual(observation_keys, self.keys)

    def test_observation_space_shape_given_keys(self):

        env = ochre_gym.load(
            env_name="basic-v0",
            vectorize_observations=False,
            override_ochre_observations_with_keys = ['Temperature - Outdoor (C)'],
            log_to_file = False,
            log_to_console = False,
        )
        self.assertTrue(env.observation_keys[0] == 'Temperature - Outdoor (C)', 
                        f"Observation key is {env.observation_keys[0]}, expected 'Temperature - Outdoor (C)'.")
        self.assertTrue(len(env.observation_keys) == 1)
        env.close()

    def test_observation_space_demand_response_programs(self):
        # test RTP
        env = ochre_gym.load(
            env_name="basic-v0",
            override_ochre_observations_with_keys = [
                'Energy Price ($)',
                'Temperature - Indoor (C)',
                'Total Electric Power (kW)'
            ],
            dr_type = 'RTP',
            start_time = "2018-06-01 00:00:00",
            episode_duration = "3 days",
            time_res = "00:15",
            lookahead = "00:30",  # 2 lookahead steps
            vectorize_observations=True,
            env_seed = 1,
            log_to_file = False,
            log_to_console = False,
        )
        self.assertTrue(env.observation_space.shape == (4,), 
                        f"Observation space shape is {env.observation_space.shape}, expected (4,).")
        self.assertTrue(env.observation_keys == [
                'Energy Price ($)', 'Temperature - Indoor (C)', 'Total Electric Power (kW)'
        ])
        env.close()

        # test PC
        env = ochre_gym.load(
            env_name="basic-v0",
            override_ochre_observations_with_keys = [
                'Power Limit (kW)',
                'Temperature - Indoor (C)',
                'Total Electric Power (kW)'
            ],
            dr_type = 'PC',
            start_time = "2018-06-01 00:00:00",
            episode_duration = "3 days",
            time_res = "00:15",
            lookahead = "00:30",  # 2 lookahead steps
            vectorize_observations=True,
            env_seed = 1,
            log_to_file = False,
            log_to_console = False,
        )
        self.assertTrue(env.observation_space.shape == (4,), 
                        f"Observation space shape is {env.observation_space.shape}, expected (4,).")
        self.assertTrue(env.observation_keys == [
                'Power Limit (kW)', 'Temperature - Indoor (C)', 'Total Electric Power (kW)'
        ])
       
    def test_observation_space_env_reset(self):
        env = ochre_gym.load(
            env_name="basic-v0",
            vectorize_observations=True,
            override_ochre_observations_with_keys = ['Temperature - Outdoor (C)'],
            log_to_file = False,
            log_to_console = False,
        )
         
        new_obs, _= env.reset()

        # test observation space shape after reset
        self.assertTrue(new_obs.shape == (1,),
                        f"Observation shape is {new_obs.shape}, expected (1,).")
        env.close()

    def test_observation_space_env_step(self):
        env = ochre_gym.load(
            env_name="basic-v0",
            vectorize_observations=True,
            log_to_file = False,
            log_to_console = False,
        )

        new_obs, _, _, _, _ = env.step(env.action_space.sample())

        self.assertTrue(env.observation_keys == self.keys,
                        f"Observation keys are {env.observation_keys}, expected {self.keys}.")
        # test observation space shape after step
        self.assertTrue(new_obs.shape == (len(self.keys),),
                        f"Observation shape is {new_obs.shape}, expected ({len(self.keys)},).")
        env.close()

    def test_unflatten_observation(self):
        env = ochre_gym.load(
            env_name="basic-v0",
            vectorize_observations=True,
            log_to_file = False,
            log_to_console = False,
        )

        new_obs, _, _, _, _ = env.step(env.action_space.sample())
        
        # assert new_obs is type np.array
        self.assertTrue(isinstance(new_obs, type(np.array([]))),
                        f"Observation type is {type(new_obs)}, expected {type(np.array([]))}.")
        dict_obs = env.observation_vector_to_dict(new_obs)
        # assert dict_obs is type dict
        self.assertTrue(isinstance(dict_obs, type({})),
                        f"Observation type is {type(dict_obs)}, expected {type({})}.")
        for k in dict_obs.keys():
            self.assertTrue(k in self.keys,
                            f"Observation key {k} is not in the expected keys.")
        env.close()
