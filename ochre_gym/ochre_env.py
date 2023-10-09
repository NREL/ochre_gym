import datetime
import os

import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Union, Optional, List
from collections import OrderedDict
import numpy as np
import random
import pandas as pd
from ochre import Dwelling
from ochre.utils import hpxml
from ochre.Equipment import ALL_END_USES
from ochre.Models.StateSpaceModel import ModelException
from ochre_gym.spaces.obs_spaces import OchreObservationSpace
from ochre_gym.spaces.obs_spaces import OchreObservationSpaceBaseConfig
from ochre_gym.spaces.obs_spaces import TimeAndEnergyPriceObservationSpaceConfig
from ochre_gym.spaces.act_spaces import get_action_space_from_ochre
from ochre_gym.spaces.act_spaces import vector_action_to_ochre_control
from ochre_gym import log
import logging
from copy import deepcopy


class OchreEnv(gym.Env):
    """The OCHRE Gym Environment.

    This is a wrapper for an OCHRE Dwelling simulator, which is a building energy simulation
    tool. The environment is designed to be used with the Gymnasium
    interface. 

    """
    metadata = {'large_penalty': -1000}

    @log.redirect_print_statements(logging.getLogger(__name__))
    def __init__(self,
                 env_name: str,
                 dwelling_args: Dict[str, Any],
                 actions: Dict[str, List[str]],
                 vectorize_actions: bool,
                 lookahead: str,
                 reward_args: Dict[str, Any],
                 disable_uncontrollable_loads: bool,
                 vectorize_observations: bool,
                 use_all_ochre_observations: bool,
                 override_ochre_observations_with_keys: Optional[List[str]],
                 observation_space_config: Optional[OchreObservationSpaceBaseConfig] = None,
                 logger: logging.Logger = None):
        """Initialize the OCHRE Gym Environment.

        Args:
            env_name (str): Name of the environment.
            dwelling_args (Dict): Dictionary of OCHRE Dwelling arguments for
                the OCHRE simulator. See https://ochre-docs-final.readthedocs.io/en/latest/InputsAndArguments.html#dwelling-arguments.
            actions (Dict): Dictionary with keys given by equipment types and values 
                given by the equipment control types. Sets the actions
                for the environment.
            lookahead (str): Length of price lookahead provided as part of observation, in "hour:minute" format.
            reward_args (Dict): Reward configuration. See ochre_env.Reward for more info.
            disable_uncontrollable_loads (bool): Disable load due to uncontrolled appliances.
            vectorize_observations (bool): Vectorize the observation space. If False, the observation space
                is a composite spaces.Dict. If True, it is a spaces.Box.
            use_all_ochre_observations (bool): Whether to use all OCHRE observations or
                a reduced set of defaults. Default: True.
            override_ochre_observations_with_keys (List[str]): Only take these observations from OCHRE.
            observation_space_config (Optional[OchreObservationSpaceBaseConfig]): Observation space configuration.
                Optionally override the default observation space configuration/args by directly passing
                a subclass of OchreObservationSpaceBaseConfig.
            logger (logging.Logger): Logger object. Default: None.        
        """
        # fix the random seed. OCHRE will use the same seed internally.
        self.seed = dwelling_args['seed']
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.dwelling_args = dwelling_args
        self.logger = logger
        self.dwelling = Dwelling(env_name, **dwelling_args)       
        self.dwelling_metadata, _ = hpxml.load_hpxml(**dwelling_args)

        # Remove controllable equipment and appliances from dwelling
        # not specified in the experiment_args
        new_equipment_by_end_use = OrderedDict()
        new_equipment = OrderedDict()
        for equipment_category in actions.keys():
            if equipment_category not in ALL_END_USES:
                raise ValueError(
                    f"Equipment category {equipment_category} not supported by OCHRE.")
            # keep the equipment_by_end_use 
            new_equipment_by_end_use[equipment_category] = \
                self.dwelling.equipment_by_end_use[equipment_category]
            
            # all equipment names mapped to this category
            eq_names = [x.name for x in self.dwelling.equipment_by_end_use[equipment_category]]
            # if any are found
            if len(eq_names) > 0:
                for eq in eq_names:
                    # keep them
                    new_equipment[eq] = self.dwelling.equipment[eq]
 
        if not disable_uncontrollable_loads: # 'Other' and 'Lighting'
            for uncontrollable_eq in ['Other', 'Lighting']:
                new_equipment_by_end_use[uncontrollable_eq] = self.dwelling.equipment_by_end_use[uncontrollable_eq]
                eq_names = [x.name for x in self.dwelling.equipment_by_end_use[uncontrollable_eq]]
                if len(eq_names) > 0:
                    for eq in eq_names:
                        new_equipment[eq] = self.dwelling.equipment[eq]
                
        self.dwelling.equipment = new_equipment
        self.dwelling.equipment_by_end_use = new_equipment_by_end_use
        self.equipment_by_end_use = self.dwelling.equipment_by_end_use

        self._save_initial_states()

        # flexible episode length, user can set episode len in config.
        simulation_timestamps = pd.date_range(
            dwelling_args['start_time'],
            dwelling_args['start_time'] + dwelling_args['duration'],
            freq=dwelling_args['time_res'],
            inclusive='left')
        self.steps_per_episode = len(simulation_timestamps)

        price_lookahead = datetime.datetime.strptime(
            lookahead, "%H:%M")
        
        # convert datetime.timedelta dwelling_args['time_res'] to minutes
        time_res_minutes = dwelling_args['time_res'].seconds / 60
        lookahead_minutes = datetime.timedelta(hours=price_lookahead.hour,
                               minutes=price_lookahead.minute).seconds / 60
        if lookahead_minutes % time_res_minutes != 0:
            raise ValueError(
                f"A price lookahead of {int(lookahead_minutes)} mins is less than or is " \
                 " not a multiple of the control interval " \
                f"{int(time_res_minutes)} mins.")

        self.lookahead_steps = int(lookahead_minutes / time_res_minutes)

        if not observation_space_config:
            observation_space_config = TimeAndEnergyPriceObservationSpaceConfig(
                observation_args={
                    'reward_args': reward_args,
                    'lookahead_steps': self.lookahead_steps
                },
                vectorize_observations=vectorize_observations,
                use_all_ochre_observations=use_all_ochre_observations,
                override_ochre_observations_with_keys=override_ochre_observations_with_keys
            )
        self.observations_manager = OchreObservationSpace(
            observation_space_config,
            self.dwelling, 
            self.dwelling_metadata
        )
        self.observation_space, self.observation_keys = \
            self.observations_manager.get_obs_space_from_ochre()
        
        # N.b. self.action_space.shape is None
        # if the action_space is a nested dict.
        self.vectorize_actions = vectorize_actions
        self.action_space, self.action_keys = get_action_space_from_ochre(
            self.equipment_by_end_use,
            actions,
            self.vectorize_actions)

        self.reward = Reward(reward_args,
                             simulation_timestamps,
                             dwelling_args['time_res'])

        self.step_count = 0

    def _save_initial_states(self):
        """Save the initial states of the OCHRE Dwelling. 
        """
        self._initial_dwelling_cache = deepcopy(self.dwelling)

    @log.redirect_print_statements(logging.getLogger(__name__))
    def reset(self, seed=None, options=None):
        """Reset the environment.

        Rolls back the OCHRE Dwelling to the state after
        the initialization period using `copy.deepcopy`.
        The decorator is used to redirect OCHRE's print statements 
        to the logger. 

        Args:
            seed (int): seed for the random number generator.
            options (Dict): options for the OCHRE Dwelling.

        Returns:
            obs (np.array): a numpy array of observations.
            control_result (Dict): a dictionary of OCHRE control results.
        """
        self.dwelling = deepcopy(self._initial_dwelling_cache)
        self.observations_manager.dwelling = self.dwelling
        
        self.equipment_by_end_use = self.dwelling.equipment_by_end_use

        self.step_count = 0

        control_result = self.dwelling.generate_results()
        control_result.update(self.dwelling.envelope.generate_results())
        obs = self.get_obs(control_result)

        return obs, control_result


    @log.redirect_print_statements(logging.getLogger(__name__))
    def step(self, action: Union[Dict, np.array]):
        """Take a step in the environment. 

        The decorator is used to redirect OCHRE's print statements to the logger.
        Currently, every time step, an action must be provided 
        for every equipment. 

        Args:
            action (Union[Dict,np.array]): a numpy array of actions, with shape (n,)
                if vectorize_actions is True, otherwise a dictionary of actions.

        Returns:
            obs (np.array): a numpy array of observations.
            rew (float): the single step reward.
            terminated (bool): a flag indicating if the episode is terminated.
            truncated (bool): a flag indicating if the episode is truncated.
            info (Dict): extra information about the step.

        Raises:
            ValueError:  If the dictionary action is malformed.
            ModelException:  Internally, OCHRE may throw a ModelException if the Dwelling tries to
                    do something "un-physical". Our current way of handling this is to stop
                    the episode and return a large negative reward.
            AssertionError:  Same as above, but for an assertion error.
        """
        if self.step_count >= self.steps_per_episode:
            raise ValueError(
                'The episode has ended. Please reset the environment.')

        terminated = False
        truncated = False

        if self.vectorize_actions:
            # Check that an action has been provided
            # for every equipment in the action space.
            assert action.shape[0] == self.action_space.shape[0], \
                f"The action has dim {action.shape[0]} but dim {self.action_space.shape[0]}" \
                "was expected. An action must be provided for every equipment control type " \
                " in the OCHRE Dwelling."
            
            # Convert the action to a dict.
            control_signal = vector_action_to_ochre_control(action, self.action_keys)
        else:
            if not set(action.keys()) == set(self.action_keys.keys()):
                raise ValueError(
                    f"The provided equipment types {list(action.keys())} " \
                    f" do not match the environment equipment types {list(self.action_space.keys())}"
                )
            for equipment in action.keys():
                # check for extra/missing control types 
                if not set(list(action[equipment].keys())) == set(self.action_keys[equipment]):
                    raise ValueError(f"The provided control types " \
                        f"{list(action[equipment].keys())} for equipment {equipment} do not match " \
                        f"the expected control types {self.action_keys[equipment]}")
                
            for dev_type, c_types in self.action_space.items():
                for ct, action_subspace in c_types.items():
                    if isinstance(action_subspace, spaces.Box):
                        action[dev_type][ct] = float(np.squeeze(action[dev_type][ct]))
                    elif isinstance(action_subspace, spaces.MultiBinary):
                        action[dev_type][ct] = int(np.squeeze(action[dev_type][ct]))

            control_signal = action

        # Implement control in OCHRE.
        try:
            # N.b. deepcopy is used to avoid modifying the original action Dict 
            control_results = self.dwelling.update(deepcopy(control_signal))
            control_results.update(self.dwelling.envelope.generate_results())
            obs = self.get_obs(control_results)
            rew = self.reward(control_results, self.step_count)
        except ModelException as e:
            # OCHRE simulation will terminates if poor control is implemented.
            # just randomly generate an obs
            obs = self.observation_space.sample()
            rew = OchreEnv.metadata['large_penalty']
            control_results = {'status': f'OCHRE ModelException: {e}'}
            terminated = True
            truncated = True
        except AssertionError as e:
            obs = self.observation_space.sample()
            rew = OchreEnv.metadata['large_penalty']
            control_results = {'status': f'OCHRE internal AssertionError: {e}'}
            terminated = True
            truncated = True

        self.step_count += 1

        if self.step_count >= self.steps_per_episode:
            terminated = True
            truncated = True

        return obs, rew, terminated, truncated, control_results

    def get_obs(self, control_results):
        """Obtain observation from the Dwelling control results.

        Args:
            control_results (Dict): the control results from OCHRE.

        Returns:
            obs (np.array): a numpy array for the flattened observation.
        """
        lookahead_values, current_energy_price = None, None
        if self.reward.dr_type == 'RTP':
            lookahead_values = self.reward.energy_price_predict
        elif self.reward.dr_type == 'PC':
            lookahead_values = self.reward.power_limit
        elif self.reward.dr_type == 'TOU':
            current_energy_price = self.reward.energy_price[self.step_count]
            
        observation_args = {
            'dwelling_metadata': self.dwelling_metadata,
            'episode_step': self.step_count,
            'current_energy_price': current_energy_price,
            'lookahead_values': lookahead_values,
            'lookahead_steps': self.lookahead_steps
        }
        return self.observations_manager.ochre_control_result_to_observation(
                                               control_results, observation_args)

    def observation_vector_to_dict(self, observation_vector: np.array) -> OrderedDict:
        """Convert the observation vector to a dictionary.

        Args:
            observation_vector (np.array): a numpy array of observations, with shape (n,)

        Returns:
            observation_dict (OrderedDict): The observation dict.
        """
        return self.observations_manager.unflatten_observation(observation_vector)
    

    def action_vector_to_dict(self, action_vector: np.array) -> OrderedDict:
        """Convert the action vector to a dictionary.

        Args:
            action_vector (np.array): a numpy array of actions, with shape (n,)

        Returns:
            action_dict (OrderedDict): The action dict.
        """
        return vector_action_to_ochre_control(action_vector, self.action_keys)


    def render(self, mode='human'):
        pass

    def close(self):
        if self.logger:
            self.logger.info('Closing OCHRE environment.')
            # close all logger file handlers
            for handler in self.logger.handlers[:]:
                handler.close()


class Reward:
    """ Reward function for OCHRE Gym environment.
    """
    DR_TYPES_AND_DEFAULT_OBS = {
        'TOU': 'Energy Price ($)',
        'PC': 'Power Limit (kW)',
        'RTP': 'Energy Price ($)'
    }

    def __init__(self, 
                 reward_args: Dict,
                 simulation_steps: pd.core.indexes.datetimes.DatetimeIndex,
                 time_resolution: datetime.timedelta):
        """Initialize the reward function from the given configuration.

        Args:
            reward_args (Dict): reward configuration. See below.
            simulation_steps (DatetimeIndex): simulation steps in the control episode.
            time_resolution (datetime.timedelta): control interval.

        reward_args dictionary - key name (value type)
        ---------------------------
            thermal_comfort_band_high (float): upper bound of the thermal comfort band.
            thermal_comfort_band_low (float): lower bound of the thermal comfort band.
            thermal_discocomfort_unit_cost (float): unit cost of thermal discomfort.
            reward_scale (float): reward scale (Default is 1.)
            dr_type (string): types of DR programs: 'TOU', 'PC' and 'RTP'.
            dr_subfolder (string): The name of the subfolder in `ochre_gym/energy_price` containing the DR files.
            flat_energy_price (float): energy price in $/kWh.
            tou_price_file (string): name of the file in which TOU daily series is
              stored.
            rtp_price_file (string): name of the file in which RTP historical series
              is stored.
            pc_power_file (string): name of the file in which DR power limit time
              series is stored.
            pc_unit_penalty (float): unit cost of power limit violation.        
        """
        self.thermal_comfort_band_high = reward_args['thermal_comfort_band_high']
        self.thermal_comfort_band_low = reward_args['thermal_comfort_band_low']
        self.thermal_discomfort_unit_cost = reward_args['thermal_comfort_unit_penalty']
        self.reward_scale = reward_args['reward_scale']
        self.dr_type = reward_args['dr_type']
        self.pc_unit_penalty = reward_args['pc_unit_penalty']

        def serialize_by_time(daily_series: pd.DataFrame, series_name: str):
            """ Convert the daily series to values over the control horizon.

            Converting the 24-hour daily series (i.e., 0:00 - 24:00) to value
            series over the specified control horizon (e.g., 8:00 - 8:00 3 days
            later). Using the mod operator (%) to go back to the beginning of
            the day.

            Args:
              daily_series (Pandas dataframe): Daily price/power limit series
                starting from 0:00.
              series_name (string): The name of the value series to be processed.
            """
            # format string for date 6/1/18 00:10
            
            daily_series['Datetime'] = pd.to_datetime(daily_series['Datetime'],
                                                      format='%m/%d/%y %H:%M')
            daily_series = daily_series.resample(time_resolution,
                                                 on='Datetime').first()
            daily_series = daily_series[series_name].to_numpy()

            steps_per_hour = 3600 / time_resolution.seconds
            assert int(steps_per_hour) == steps_per_hour
            idx = (simulation_steps[0].hour * steps_per_hour
                   + simulation_steps[0].minute * 60 / time_resolution.seconds)
            idx = int(idx)

            new_series = []
            for _ in range(len(simulation_steps)):
                new_series.append(daily_series[idx % len(daily_series)])
                idx += 1

            return np.array(new_series)

        if self.dr_type == 'RTP':
            rtp_data = pd.read_csv(os.path.join(reward_args['dr_subfolder'],
                                                reward_args['rtp_price_file']))
            rtp_data['Datetime'] = pd.to_datetime(rtp_data['Datetime'],
                                                  format='%Y-%m-%d %H:%M:%S')
            rtp_data = rtp_data[(rtp_data['Datetime'] > simulation_steps[0])
                                & (rtp_data['Datetime'] <= simulation_steps[-1])]
            assert len(rtp_data) != 0, "RTP data should cover the simulation period."
            rtp_data = rtp_data.resample(time_resolution,
                                         on='Datetime').first()
            self.energy_price = rtp_data['RTP'].to_numpy()
            self.energy_price_predict = rtp_data['DAP'].to_numpy()
        elif self.dr_type == 'TOU':
            tou_data = pd.read_csv(os.path.join(reward_args['dr_subfolder'],
                                                reward_args['tou_price_file']))
            self.energy_price = serialize_by_time(tou_data, 'TOU')
        elif self.dr_type == 'PC':
            self.energy_price = ([reward_args['flat_energy_price']]
                                 * len(simulation_steps))
            power_data = pd.read_csv(
                os.path.join(reward_args['dr_subfolder'],
                             reward_args['pc_power_file']))
            self.power_limit = serialize_by_time(power_data, 'power_limit')


    def __call__(self, control_results: Dict[str, Any],
                 step_idx: int) -> float:
        """Calculate single step control reward based on the
          control results.

        Args:
            control_results (Dict): control results from OCHRE.
            step_idx (int): Current step index.

        Returns:
            reward (float): reward for the current control step scaled by reward_scale.
        """
        reward = 0.0
        t = control_results['Time']

        # Energy cost
        energy_used = control_results['Total Electric Energy (kWh)']
        # energy_used = control_results['HVAC Heating Electric Power (kW)']

        energy_price = self.energy_price[step_idx]
        reward -= energy_used * energy_price

        # Thermal discomfort cost
        indoor_temp = control_results['Temperature - Indoor (C)']

        deviation = max(max(0.0, indoor_temp - self.thermal_comfort_band_high),
                        max(0.0, self.thermal_comfort_band_low - indoor_temp),
                        0.0)
        reward -= self.thermal_discomfort_unit_cost * deviation ** 2

        if self.dr_type == 'PC':
            power_consumption = control_results['Total Electric Power (kW)']
            violation = max(0.0, power_consumption
                            - self.power_limit[step_idx])
            reward -= self.pc_unit_penalty * violation ** 2

        return reward * self.reward_scale
