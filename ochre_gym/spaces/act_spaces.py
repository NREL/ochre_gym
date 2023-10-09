import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Union, Tuple
from collections import OrderedDict
import numpy as np
from ochre_gym.spaces import OchreEquipmentSubSpace, SubSpaceType
from ochre_gym.spaces import register_equipment_subspace


HVAC_COOLING_ACTIONS = register_equipment_subspace([
    OchreEquipmentSubSpace(
        name='Load Fraction',
        shape=(1,),
        type=SubSpaceType.DISCRETE,
        n=1
    ).register(),
    OchreEquipmentSubSpace(
        name='Duty Cycle',
        shape=(1,),
        type=SubSpaceType.CONTINUOUS,
        upper_bound=np.array([0.2], dtype=np.float32),
        lower_bound=np.array([0.0], dtype=np.float32)
    ).register(),
    OchreEquipmentSubSpace(
        name='Setpoint',
        shape=(1,),
        type=SubSpaceType.CONTINUOUS,
        upper_bound=np.array([35.], dtype=np.float32),
        lower_bound=np.array([18.], dtype=np.float32)
    ).register(),
    OchreEquipmentSubSpace(
        name='Deadband',
        shape=(1,),
        type=SubSpaceType.CONTINUOUS,
        upper_bound=np.array([10.0], dtype=np.float32),
        lower_bound=np.array([0.1], dtype=np.float32)
    ).register()
])

HVAC_HEATING_ACTIONS = register_equipment_subspace([
    OchreEquipmentSubSpace(
        name='Load Fraction',
        shape=(1,),
        type=SubSpaceType.DISCRETE,
        n=1
    ).register(),
    OchreEquipmentSubSpace(
        name='Duty Cycle',
        shape=(1,),
        type=SubSpaceType.CONTINUOUS,
        upper_bound=np.array([1.0], dtype=np.float32),
        lower_bound=np.array([0.0], dtype=np.float32)
    ).register(),
    OchreEquipmentSubSpace(
        name='Setpoint',
        shape=(1,),
        type=SubSpaceType.CONTINUOUS,
        upper_bound=np.array([35.], dtype=np.float32),
        lower_bound=np.array([18.], dtype=np.float32)
    ).register(),
    OchreEquipmentSubSpace(
        name='Deadband',
        shape=(1,),
        type=SubSpaceType.CONTINUOUS,
        upper_bound=np.array([10.0], dtype=np.float32),
        lower_bound=np.array([0.1], dtype=np.float32)
    ).register()
])


WATER_HEATING_ACTIONS = register_equipment_subspace([
    OchreEquipmentSubSpace(
        name='Load Fraction',
        shape=(1,),
        type=SubSpaceType.DISCRETE,
        n=1
    ).register(),
    OchreEquipmentSubSpace(
        name='Duty Cycle',
        shape=(1,),
        type=SubSpaceType.CONTINUOUS,
        upper_bound=np.array([1.0], dtype=np.float32),
        lower_bound=np.array([0.0], dtype=np.float32)
    ).register(),
    OchreEquipmentSubSpace(
        name='Setpoint',
        shape=(1,),
        type=SubSpaceType.CONTINUOUS,
        upper_bound=np.array([62.], dtype=np.float32),
        lower_bound=np.array([25.], dtype=np.float32),
    ).register(),
    OchreEquipmentSubSpace(
        name='Deadband',
        shape=(1,),
        type=SubSpaceType.CONTINUOUS,
        upper_bound=np.array([10.0], dtype=np.float32),
        lower_bound=np.array([0.5], dtype=np.float32)
    ).register()    
])


# EV_ACTIONS = {
# }

# PV_ACTIONS = {
# }

# GAS_GENERATOR_ACTIONS= {
# }

# BATTERY_ACTIONS= {
# }

EQUIPMENT_SUBSPACES = OrderedDict({
    'HVAC Cooling': HVAC_COOLING_ACTIONS,
    'HVAC Heating': HVAC_HEATING_ACTIONS,
    'Water Heating': WATER_HEATING_ACTIONS
    # TODO: Adding support in future version
    #'EV': EV_ACTIONS,  
    #'PV': PV_ACTIONS,
    #'Gas Generator': GAS_GENERATOR_ACTIONS,
    #'Battery': BATTERY_ACTIONS
})

CONTROL_TYPES_PRECEDENCE_ORDER = [
    'Load Fraction',
    'Duty Cycle',
    'Setpoint',
    'Deadband'
]

def get_action_space_from_ochre(equipment_by_end_use: Dict,
                                user_config: Dict,
                                vectorize_actions: bool) -> Tuple[Union[spaces.Dict, spaces.Box], OrderedDict]:
    """Obtain the action space for an OCHRE dwelling simulation.

    Args:
        equipment_by_end_use (Dict): A dictionary of equipment by end use for
            the OCHRE Dwelling
        user_config (Dict): A dictionary specifying which control types to use 
            for each device in a Dwelling
        vectorize_actions (bool): Whether to vectorize the actions

    Returns:
        action_space (Union[spaces.Dict, spaces.Box]): The action space for an OCHRE Dwelling.
        action_keys (OrderedDict): The keys of the action dict, keys are equipment category 
            and values are control types.
    """
    device_level_actions = {}
    action_keys = OrderedDict({})

    # iterate through the device types in fixed order
    for dev_type in EQUIPMENT_SUBSPACES.keys():
        if not dev_type in equipment_by_end_use.keys():
            continue 
        device_level_actions[dev_type] = spaces.Dict()
        action_keys[dev_type] = []
        equipment_controls = user_config[dev_type]
        for ec in CONTROL_TYPES_PRECEDENCE_ORDER:
            # only add the control type if it is in the user config
            if not ec in equipment_controls:
                continue
            dev_act_space = EQUIPMENT_SUBSPACES[dev_type][ec]
            action_keys[dev_type].append(ec)
            if dev_act_space['type'] == SubSpaceType.DISCRETE:
                device_level_actions[dev_type][ec] = \
                    spaces.MultiBinary(dev_act_space['n'])
            elif dev_act_space['type'] == SubSpaceType.CONTINUOUS:
                device_level_actions[dev_type][ec] = \
                    spaces.Box(low=dev_act_space['lower_bound'],
                        high=dev_act_space['upper_bound'],
                        shape=dev_act_space['shape'])
    action_space = spaces.Dict(device_level_actions)

    if vectorize_actions:
        action_space = spaces.utils.flatten_space(action_space)
    return action_space, action_keys


def vector_action_to_ochre_control(action: np.ndarray, action_keys: OrderedDict) -> OrderedDict:
    """Unflatten a numpy array into a dict of actions.

    Args:
        action (np.ndarray): The flattened action array.
        action_keys (OrderedDict): The keys of the action dict.

    Returns:
        action_dict (OrderedDict): The action dict.
    """
    action_dict = OrderedDict({})
    start_idx = 0
    for dev_type, control_types in action_keys.items():
        action_dict[dev_type] = OrderedDict({})
        for ct in control_types:
            end_idx = start_idx + EQUIPMENT_SUBSPACES[dev_type][ct]['shape'][0]
            if EQUIPMENT_SUBSPACES[dev_type][ct]['type'] == SubSpaceType.DISCRETE:
                action_dict[dev_type][ct] = int(np.squeeze(action[start_idx:end_idx]))
            elif EQUIPMENT_SUBSPACES[dev_type][ct]['type'] == SubSpaceType.CONTINUOUS:
                action_dict[dev_type][ct] = float(np.squeeze(action[start_idx:end_idx]))
            start_idx = end_idx
    return action_dict
   

class ClipActionComposite(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Clip the spaces.Box action spaces of a composite
        action space to their bounds.
    """
    
    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions in a composite action space.

        Args:
            env (Env): The environment to wrap.
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)
    

    def _clip_subspace(self, 
                       action_subspace: Union[spaces.Dict, spaces.Box, spaces.MultiBinary],
                       action: Dict) -> Dict:
        """Clips the continuous actions in a composite action subspace.

        Args:
            action_subspace (spaces.Dict | spaces.spaces.Box | spaces.MultiBinary): The action space to clip.
            action (Dict): The action subspace to clip.

        Returns:
            action (Dict): The clipped action subspace.
        """
        if isinstance(action_subspace, spaces.Box):
            action = np.clip(action, action_subspace.low, action_subspace.high)
        if isinstance(action_subspace, spaces.MultiBinary):
            action = np.clip(action, 0, 1)
        elif isinstance(action_subspace, spaces.Dict):
            for action_key, action_subsubspace_ in action_subspace.items():
                # updates the action and iter
                action[action_key] = self._clip_subspace(action_subsubspace_, action[action_key])
        return action
    

    def action(self, action_: Dict) -> Dict:
        """Recursively clip the flattened, continuous actions in a composite action space.

        Args:
            action_ (Dict): The action to clip.

        Returns:
            action_ (Dict): The clipped composite action.
        """
        if self.vectorize_actions:
            raise ValueError("Vectorized actions are not supported with ClipActionComposite."
                             "Set vectorized_actions to False.")

        for action_key, action_subspace in self.action_space.items():
            action_[action_key] = self._clip_subspace(action_subspace, action_[action_key])
        return action_
