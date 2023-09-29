from collections.abc import Iterable
from gymnasium import spaces
import numpy as np
from ochre import Dwelling
from collections import OrderedDict
from typing import Dict, Union, List, Optional, Tuple
from abc import ABC, abstractmethod


class OchreObservationSpaceBaseConfig(ABC):
    """Base class for OCHRE observation space configuration.
    Also provides two hooks for customizing the observation space and 
    customizing the observations from OCHRE.

    A reduced set of OCHRE observations is defined in
    parameter OCHRE_GYM_REDUCED_KEYS:

    ```python
    'Temperature - Indoor (C)',  # Envelope
    'Temperature - Outdoor (C)',  # Envelope
    'Total Electric Power (kW)',  # Dwelling
    'Total Gas Power (therms/hour)', # Dwelling
    'HVAC Heating Electric Power (kW)', # HVAC Heating
    'HVAC Heating Gas Power (therms/hour)', # HVAC Heating
    'HVAC Cooling Electric Power (kW)', # HVAC Cooling,
    'Water Heating Gas Power (therms/hour)', # Water Heating
    'Water Heating Electric Power (kW)', # Water Heating,
    'EV Electric Power', # EV
    'PV Electric Power', # PV
    'Battery Electric Power', # Battery,
    'Gas Generator Electric Power', # Gas Generator
    'Gas Generator Gas Power', # Gas Generator
    ```
    """

    # Not all keys will be used if the Dwelling is missing equipment, e.g., no EV.
    OCHRE_GYM_REDUCED_KEYS = sorted([
        'Temperature - Indoor (C)',  # Envelope
        'Temperature - Outdoor (C)',  # Envelope
        'Total Electric Power (kW)',  # Dwelling
        'Total Gas Power (therms/hour)', # Dwelling
        'HVAC Heating Electric Power (kW)', # HVAC Heating
        'HVAC Heating Gas Power (therms/hour)', # HVAC Heating
        'HVAC Cooling Electric Power (kW)', # HVAC Cooling,
        'Water Heating Gas Power (therms/hour)', # Water Heating
        'Water Heating Electric Power (kW)', # Water Heating,
        'EV Electric Power', # EV
        'PV Electric Power', # PV
        'Battery Electric Power', # Battery,
        'Gas Generator Electric Power', # Gas Generator
        'Gas Generator Gas Power', # Gas Generator
    ])
    def __init__(self, 
                 observation_args: Dict,
                 vectorize_observations: bool,
                 use_all_ochre_observations: bool,
                 override_ochre_observations_with_keys: Optional[List[str]]):
        """Initialize the OCHRE observation space configuration.
        
        Args:
            observation_args (Dict): A dictionary of keyword arguments.
            vectorize_observations (bool): Whether to vectorize observations.
            use_all_ochre_observations (bool): Whether to use all OCHRE observations or not.
            override_ochre_observations_with_keys (Optional[List[str]]): A list of keys to override
                the OCHRE observations with. If None, then use all OCHRE observations.
        """
        self.observation_args = observation_args
        self.vectorize_observations = vectorize_observations
        self.use_all_ochre_observations = use_all_ochre_observations
        self.override_ochre_observations_with_keys = override_ochre_observations_with_keys

    @abstractmethod
    def customize_observation_space_hook(self, observations: Dict) -> Dict:
        """A hook to specify a custom observation space for OCHRE Gym.
        If vectorizing observations, performed beforehand. By default,
        does nothing. Override this in subclass.

        Args:
            observations (Dict): A dictionary of observations.
        
        Returns:
            observations (Dict): A dictionary of observations.
        """
        pass    

    @abstractmethod
    def customize_observations_from_ochre_hook(self,
                                               observations: Dict,
                                               ochre_control_result: Dict,
                                               args: Dict) -> Dict:
        """A hook enabling a user to augment a given observation based
        on the OCHRE control result and extra info.

        Args:
            observations (Dict): A dictionary of observations.
            ochre_control_result (Dict): A dictionary of control results from OCHRE.
            args (Dict): A dictionary of extra info to help create the desired observation.
        
        Returns:
            observations (Dict): The customized dictionary of observations.
        """
        pass


class TimeAndEnergyPriceObservationSpaceConfig(OchreObservationSpaceBaseConfig):
    """A custom configuration for the observation space that adds time and energy price, 
    which are not included in the OCHRE observations by default.
    """
    def __init__(self,
                 observation_args,
                 vectorize_observations,
                 use_all_ochre_observations,
                 override_ochre_observations_with_keys):
        """Initialize the observation space configuration.

        Expects that the observation_args dictionary has a 'reward_args' key
        with a 'dr_type' sub-key. If 'dr_type' is 'RTP', then the observation space
        will include a vector of energy prices. If 'dr_type' is 'PC', then the
        observation space will include a vector of power limits.

        Expects that the observation_args dictionary has a 'lookahead_steps' key
        with a value of the number of steps to look ahead. If 'lookahead_steps' is
        None, then the observation space will include a single energy price or power
        limit value.

        Args:
            observation_args (Dict): A dictionary of keyword arguments.
            vectorize_observations (bool): Whether to vectorize observations.
            use_all_ochre_observations (bool): Whether to use all OCHRE observations or a default
                reduced set.
            override_ochre_observations_with_keys (Optional[List[str]]): A list of keys to override
                the OCHRE observations with. If None, then use all OCHRE observations.
        """
        super().__init__(
                observation_args,
                vectorize_observations,
                use_all_ochre_observations,
                override_ochre_observations_with_keys)
        assert 'reward_args' in self.observation_args, \
            'reward_args must be provided in observation_args'
        assert 'dr_type' in self.observation_args['reward_args'], \
            'dr_type must be provided in observation_args["reward_args"]'
        assert 'lookahead_steps' in observation_args, \
            'lookahead_steps must be provided in observation_args'
        if self.observation_args['lookahead_steps'] is None:
            self.observation_args['lookahead_steps'] = 1

    def customize_observation_space_hook(self, observations: Dict) -> Dict:
        """Customize the observation space by adding time and energy price observations.

        Args:
            observations (Dict): A dictionary of observations to customize.
        Returns:
            observations (Dict): The customized dictionary of observations.
        """

        if self.observation_args['reward_args']['dr_type'] == 'RTP':
            observations['Energy Price ($)'] = spaces.Box(low=0., high=15.,
                                                    shape=(self.observation_args['lookahead_steps'],))   
        else: # TOU and PC
            observations['Energy Price ($)'] = spaces.Box(low=0., high=15., shape=(1,))
            
        if self.observation_args['reward_args']['dr_type'] == 'PC':
            observations['Power Limit (kW)'] = spaces.Box(low=0., high=50.,
                                            shape=(self.observation_args['lookahead_steps'],))
        
        observations['Hour of day'] = spaces.Box(low=0., high=23., shape=(1,))
        observations['Day of week'] = spaces.Box(low=0., high=6., shape=(1,))
        observations['Day of month'] = spaces.Box(low=0., high=30., shape=(1,))
        return observations
    
    def customize_observations_from_ochre_hook(self,
                                               observations: Dict,
                                               ochre_control_result: Dict,
                                               args: Dict) -> Dict:
        """Add time and energy price observations from the OCHRE control result.

        Args:
            observations (Dict): A dictionary of observations.
            ochre_control_result (Dict): A dictionary of control results from OCHRE.
            args (Dict): A dictionary of extra info to help create the desired observation.
        Returns:
            observations (Dict): The customized dictionary of observations.
        """
        dt = ochre_control_result['Time']
        hour = dt.hour
        day_of_week = dt.weekday()
        day_of_month = dt.day
        observations['Hour of day'] = hour
        observations['Day of week'] = day_of_week
        observations['Day of month'] = day_of_month  

        if args['lookahead_values'] is not None:
            lookahead_ = args['lookahead_values'][
                    args['episode_step'] : args['episode_step'] + args['lookahead_steps']]
            # Extend to the fixed length
            if len(lookahead_) < args['lookahead_steps']:
                short_fall = args['lookahead_steps'] - len(lookahead_)
                lookahead_ = np.append(
                    lookahead_, np.array([lookahead_[-1]] * short_fall))
                        
            if self.observation_args['reward_args']['dr_type'] == 'RTP':
                observations['Energy Price ($)'] = lookahead_
            
            if self.observation_args['reward_args']['dr_type'] == 'PC':
                observations['Power Limit (kW)'] = lookahead_

        if self.observation_args['reward_args']['dr_type'] == 'TOU':
            observations['Energy Price ($)'] = args['current_energy_price']   

        return observations 
    


class MetadataObservationSpaceConfig(TimeAndEnergyPriceObservationSpaceConfig):
    """
    A configuration for the observation space that:

    - Adds time and energy price observations
    - Adds metadata:
        - Has Attic (binary)
        - Has Garage (binary)
        - Square Footage (float)
        - Number of Bedrooms (int)
    """

    def __init__(self, 
                 observation_args,
                 vectorize_observations,
                 use_all_ochre_observations,
                 override_ochre_observations_with_keys):
        """Initialize the OCHRE observation space configuration.
        
        Args:
            observation_args (Dict): A dictionary of keyword arguments.
            vectorize_observations (bool): Whether to vectorize observations.
            use_all_ochre_observations (bool): Whether to use all OCHRE observations or not.
            override_ochre_observations_with_keys (Optional[List[str]]): A list of keys to override
                the OCHRE observations with. If None, then use all OCHRE observations.
        """
        assert 'dwelling_metadata' in observation_args, '"dwelling_metadata" must be a key in observation_args'
        super().__init__(observation_args,
                         vectorize_observations,
                         use_all_ochre_observations,
                         override_ochre_observations_with_keys)

    def customize_observation_space_hook(self, observations: Dict) -> Dict:
        """Customize the observation space by adding metadata observations.
        Args:
            observations (Dict): A dictionary of observations to customize.
        
        Returns:
            observations (Dict): The customized dictionary of observations.
        """
        observations = super().customize_observation_space_hook(observations)
        if 'Garage' in self.observation_args['dwelling_metadata']['zones']:
            observations['Has Garage'] = spaces.MultiBinary(1)
        if 'Attic' in self.observation_args['dwelling_metadata']['zones']:
            observations['Has Attic'] = spaces.MultiBinary(1)

        # Altitude, Avg. wind speed, avg. ambient temp, average ground temp
        metadata_observations = {
            'Square Footage': spaces.Box(low=0., high=10000., shape=(1,)),
            'Number of Bedrooms': spaces.Discrete(10),
        }
        observations = {**observations, **metadata_observations}
        return observations
    

    def customize_observations_from_ochre_hook(self, observations, control_result, args):
        """Customize the observations from OCHRE by adding metadata observations.

        Args:
            observations (Dict): A dictionary of observations.
        
        Returns:
            observations (Dict): The customized dictionary of observations.
        """
        observations = super().customize_observations_from_ochre_hook(observations, control_result, args)
        has_attic = 0
        has_garage = 0
        total_sq_ft = args['dwelling_metadata']['zones']['Indoor']['Zone Area (m^2)']
        if 'Garage' in args['dwelling_metadata']['zones']:
            has_garage = 1
            total_sq_ft += args['dwelling_metadata']['zones']['Garage']['Zone Area (m^2)']
        if 'Attic' in args['dwelling_metadata']['zones']:
            has_attic = 1
            total_sq_ft += args['dwelling_metadata']['zones']['Attic']['Zone Area (m^2)']

        observations['Square Footage'] = total_sq_ft
        observations['Has Attic'] = has_attic
        observations['Has Garage'] = has_garage
        observations['Number of Bedrooms'] = args['dwelling_metadata'][
            'construction']['Number of Bedrooms (-)']
        
        return observations
    

class OchreObservationSpace:
    """A class for creating an OCHRE Gym observation space and managing it.
    Expects an OchreObservationSpaceBaseConfig object to be passed in the constructor
    that specifies how to configure the observation space. Users should subclass 
    the OchreObservationSpaceBaseConfig class to create their own custom observation if desired.

    Keys in the observation space are ordered alphabetically.
    """
    def __init__(self,  
                 observation_space_config: OchreObservationSpaceBaseConfig,
                 dwelling: Dwelling,
                 dwelling_metadata: Dict):
        """Initialize the OchreObservationSpace.

        Args:
            observation_space_config (OchreObservationSpaceBaseConfig): A configuration object
                for the observation space.
            dwelling (Dwelling): An OCHRE dwelling object.
            dwelling_metadata (Dict): A dictionary of dwelling metadata.
        """
        self.dwelling = dwelling
        self.dwelling_metadata = dwelling_metadata
        self.observation_space_config = observation_space_config
        self.observation_keys = None

    def get_obs_space_from_ochre(self) -> Tuple[Union[spaces.Dict, spaces.Box], List[str]]:
        """Obtain observation space using an OCHRE dwelling simulation.

        Either returns a composite (Dict) or flattened (vector) observation space.
        This is decided by the observation_space_config.vectorize_observations parameter.

        Returns:
            observation_space (Union[spaces.Dict, spaces.Box]): The observation space.
            observation_keys (List[str]): The observation keys. Sets the class parameter 
                self.observation_keys.
        """
        control_result = self.dwelling.generate_results()
        control_result.update(self.dwelling.envelope.generate_results())
        observations = OrderedDict({})

        ################### OCHRE OBSERVATIONS ###################
        if self.observation_space_config.use_all_ochre_observations:
            for key, value in control_result.items():
                if key == 'Time':
                    continue
                if 'Mode' in key:
                    observations[key] = spaces.MultiBinary(1)
                if 'Occupancy' in key:
                    observations[key] = spaces.Discrete(10)
                elif isinstance(value, (float, np.floating, int, np.integer)):
                    observations[key] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
            
        else:
            for key in OchreObservationSpaceBaseConfig.OCHRE_GYM_REDUCED_KEYS:
                if key in control_result.keys():
                    observations[key] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        
        # Hook for customizing observations
        observations = self.observation_space_config.customize_observation_space_hook(observations)

        if self.observation_space_config.override_ochre_observations_with_keys:
            # filter out keys not in observation_keys
            observations = {k: v for k, v in observations.items() \
                            if k in self.observation_space_config.override_ochre_observations_with_keys}

        # SORT ALPHABETICALLY
        obs_dict = spaces.Dict(OrderedDict(sorted(observations.items())))
        self.observation_keys = list(obs_dict.spaces.keys())

        if self.observation_space_config.vectorize_observations:
            # Flatten the observation space
            obs_dict = spaces.utils.flatten_space(obs_dict)
        
        return obs_dict, self.observation_keys


    def ochre_control_result_to_observation(self,
                                            control_result: Dict,
                                            observation_args: Dict) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Convert an OCHRE control result to an observation.

        Args:
            control_result (Dict): A dictionary of control results from OCHRE.
            observation_args (Dict): A dictionary of keyword arguments.

        Returns:
            A numpy array or a dictionary of observations.
        """
        if self.observation_keys is None:
            raise ValueError('Observation keys are not set.')
                
        observation = OrderedDict({})

        for key in self.observation_keys:
            if key in control_result.keys():
                observation[key] = control_result[key]

        observation = self.observation_space_config.customize_observations_from_ochre_hook(
            observation, control_result, observation_args)
        
        if self.observation_space_config.override_ochre_observations_with_keys:
            # filter out keys not in observation_keys
            observation = {k: v for k, v in observation.items() \
                            if k in self.observation_space_config.override_ochre_observations_with_keys}

        # sort by key alphabetically
        observation = OrderedDict(sorted(observation.items()))

        assert len(observation.keys()) == len(self.observation_keys)
        
        if self.observation_space_config.vectorize_observations:
            obs_vec = []
            for item in observation.values():
                if isinstance(item, (float, np.floating, int, np.integer)):
                    obs_vec.append(item)
                elif isinstance(item, Iterable):
                    obs_vec += list(item)
            return np.array(obs_vec)
        else:
            return observation

    def unflatten_observation(self, vector_observation: np.array) -> OrderedDict:
        """Given a vector observation, unflatten into a dict.
        
        Args:
            vector_observation (np.array): A vector observation.

        Returns:
            dict_obs (OrderedDict): A dictionary of observations.
        """
        if self.observation_keys is None:
            raise ValueError('Observation keys are not set.')
        dict_obs = OrderedDict({})
        for idx,key in enumerate(self.observation_keys):
            dict_obs[key] = vector_observation[idx]
        return dict_obs


